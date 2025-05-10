import json
import random

from os import path

from argparse import ArgumentParser

import torch

from transformers import EsmForSequenceClassification

from torch.cuda import is_available as cuda_is_available

from graph import GOInterpreter


def main():
    parser = ArgumentParser(
        description="Predict the gene ontology terms associated with a protein sequence."
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument(
        "--label_mapping_path", default="./dataset/all_label_mapping.json", type=str
    )
    parser.add_argument("--go_obo_path", default="./dataset/train/go-basic.obo", type=str)
    parser.add_argument("--context_length", default=1024, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if not path.exists(args.label_mapping_path):
        raise FileNotFoundError(
            f"Label mapping file {args.label_mapping_path} not found. Please check the path."
        )
    
    if args.context_length < 1:
        raise ValueError(
            f"Context length must be greater than 0, {args.context_length} given."
        )

    if args.top_k < 1:
        raise ValueError(f"Top k must be greater than 0, {args.top_k} given.")

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    model = EsmForSequenceClassification(checkpoint["config"])

    print("Compiling model ...")
    model = torch.compile(model)

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    print("Checkpoint loaded successfully.")

    with open(args.label_mapping_path, "r") as file:
        label_mapping = json.load(file)

    go_interpreter = GOInterpreter(args.go_obo_path)

    while True:
        sequence = input("Enter a sequence: ").replace(" ", "").replace("\n", "")

        out = tokenizer(
            sequence,
            max_length=args.context_length,
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

            probabilities, indices = torch.topk(probabilities, args.top_k)

            probabilities = probabilities.tolist()

            go_terms = [label_mapping[index] for index in indices.tolist()]

            names = go_interpreter.get_names(go_terms)

            print(f"Top {args.top_k} GO Terms:")

            for name, probability in zip(names, probabilities):
                print(f"{probability:.4f}: {name}")

            print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
