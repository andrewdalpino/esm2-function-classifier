import random

from argparse import ArgumentParser

import torch

from transformers import AutoTokenizer, EsmConfig, EsmForSequenceClassification

from torch.cuda import is_available as cuda_is_available


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
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config = EsmConfig.from_pretrained(args.base_model)

    config.num_labels = 47417

    model = EsmForSequenceClassification(config)

    print("Compiling model")
    model = torch.compile(model)

    model = model.to(args.device)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=False
    )

    model.load_state_dict(checkpoint["model"])

    print("Checkpoint loaded successfully.")

    model.eval()

    torch.set_printoptions(threshold=50000)

    while True:
        sequence = input("Enter a sequence: ")

        prompt = tokenizer(
            sequence,
            # padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        input_ids = prompt["input_ids"].to(args.device)
        attn_mask = prompt["attention_mask"].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)

            logits = outputs.logits.squeeze(0)

            probabilities = torch.sigmoid(logits)

            print(probabilities)

            predicted_indices = torch.where(probabilities > 0.5)[0].tolist()

            print(predicted_indices)

            print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
