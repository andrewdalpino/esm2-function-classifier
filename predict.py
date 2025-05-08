import random

from argparse import ArgumentParser

import torch

from transformers import AutoTokenizer, EsmConfig, EsmForSequenceClassification

from torch.cuda import is_available as cuda_is_available

from data import CAFA5


def main():
    parser = ArgumentParser(
        description="Predict the gene ontology terms associated with a protein sequence."
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["base_model"])

    config = EsmConfig.from_pretrained(checkpoint["base_model"])

    config.problem_type = "multi_label_classification"
    config.num_labels = CAFA5.NUM_CLASSES

    model = EsmForSequenceClassification.from_pretrained(
        checkpoint["base_model"], config=config
    )

    print("Compiling model ...")
    model = torch.compile(model)

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    print("Checkpoint loaded successfully.")

    model.eval()

    while True:
        sequence = input("Enter a sequence: ")

        sequence = sequence.replace(" ", "").replace("\n", "")

        out = tokenizer(
            sequence,
            padding="max_length",
            padding_side="right",
            max_length=1024,
            truncation=True,
        )

        input_ids = out["input_ids"]
        attn_mask = out["attention_mask"]

        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(args.device)
        )
        attn_mask = (
            torch.tensor(attn_mask, dtype=torch.int64).unsqueeze(0).to(args.device)
        )

        with torch.no_grad():
            outputs = model.forward(input_ids, attention_mask=attn_mask)

            logits = outputs.logits

            probabilities = torch.sigmoid(logits.squeeze(0))

            sorted, indices = torch.sort(probabilities, descending=True)

            indices = indices[:10]
            sorted = sorted[:10]

            print(sorted, indices)

            print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
