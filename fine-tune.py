import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.optim import AdamW

from transformers import AutoTokenizer, EsmConfig, EsmForSequenceClassification

from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.amp import autocast
from torch.utils.data import random_split
from torch.nn.utils import clip_grad_norm_

from torchmetrics.classification import BinaryF1Score

from torch.utils.tensorboard import SummaryWriter

from data import CAFA5

from tqdm import tqdm


AVAILABLE_BASE_MODELS = {
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t48_15B_UR50D",
}


def main():
    parser = ArgumentParser(
        description="Fine-tune an ESM2 model for protein function classification."
    )

    parser.add_argument(
        "--base_model",
        default="facebook/esm2_t6_8M_UR50D",
        choices=AVAILABLE_BASE_MODELS,
    )
    parser.add_argument("--dataset_path", default="dataset/dataset.jsonl", type=str)
    parser.add_argument("--max_sequence_length", default=1024, type=int)
    parser.add_argument("--num_dataset_processes", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--eval_ratio", default=0.1, type=float)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = CAFA5(args.dataset_path, tokenizer, args.max_sequence_length)

    training, testing = random_split(dataset, (1.0 - args.eval_ratio, args.eval_ratio))

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True, drop_last=True)
    test_loader = new_dataloader(testing, shuffle=False)

    config = EsmConfig.from_pretrained(args.base_model)

    config.problem_type = "multi_label_classification"
    config.num_labels = CAFA5.NUM_CLASSES

    model = EsmForSequenceClassification.from_pretrained(args.base_model, config=config)

    for param in model.esm.parameters():
        param.requires_grad = False

    print("Compiling model ...")
    model = torch.compile(model)

    model = model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    f1_metric = BinaryF1Score().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {num_trainable_params:,}")

    print("Fine-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y, attn_mask) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            attn_mask = attn_mask.to(args.device, non_blocking=True)

            with amp_context:
                out = model.forward(x, attention_mask=attn_mask, labels=y)

                scaled_loss = out.loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += out.loss.item()
            total_batches += 1

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        logger.add_scalar("Cross Entropy", average_cross_entropy, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.4f}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for x, y, attn_mask in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                attn_mask = attn_mask.to(args.device, non_blocking=True)

                with torch.no_grad():
                    out = model.forward(x, attention_mask=attn_mask)

                    y_prob = torch.sigmoid(out.logits)

                f1_metric.update(y_prob, y)

            f1_score = f1_metric.compute()

            logger.add_scalar("F1 Score", f1_score, epoch)

            print(f"F1 Score: {f1_score:.3f}")

            f1_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "base_model": args.base_model,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":  #
    main()
