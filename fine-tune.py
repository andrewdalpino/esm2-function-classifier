import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.optim import Adafactor

from transformers import AutoTokenizer, EsmConfig, EsmForSequenceClassification

from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.amp import autocast
from torch.utils.data import random_split
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCEWithLogitsLoss

from torchmetrics.classification import BinaryAccuracy

from torch.utils.tensorboard import SummaryWriter

from data import CAFA5

from tqdm import tqdm


AVAILABLE_BASE_MODELS = {
    "facebook/esm2_t6_8M_UR50D" "facebook/esm2_t12_35M_UR50D",
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
    parser.add_argument("--num_dataset_processes", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--rms_decay", default=-0.8, type=float)
    parser.add_argument("--low_memory_optimizer", action="store_true")
    parser.add_argument("--max_gradient_norm", default=10.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
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

    dataset = CAFA5(args.dataset_path, tokenizer)

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
    config.num_labels = 47417

    model = EsmForSequenceClassification(config)

    print("Compiling model")
    model = torch.compile(model)

    model = model.to(args.device)

    loss_function = BCEWithLogitsLoss()

    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        beta2_decay=args.rms_decay,
        foreach=not args.low_memory_optimizer,
    )

    binary_accuracy_metric = BinaryAccuracy(threshold=0.5).to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    print("Fine-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (input_ids, attn_mask, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            input_ids = input_ids.to(args.device, non_blocking=True)
            attn_mask = attn_mask.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred = model.forward(input_ids, attn_mask).logits

                loss = loss_function(y_pred, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += loss.item()
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

            for input_ids, attn_mask, y in tqdm(
                test_loader, desc="Testing", leave=False
            ):
                input_ids = input_ids.to(args.device, non_blocking=True)
                attn_mask = attn_mask.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad():
                    y_pred = model.forward(input_ids, attn_mask).logits

                binary_accuracy_metric.update(y_pred, y)

            accuracy = binary_accuracy_metric.compute()

            logger.add_scalar("Accuracy", accuracy, epoch)

            print(f"Accuracy: {accuracy:.3f}")

            binary_accuracy_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "tokenizer": tokenizer,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":  #
    main()
