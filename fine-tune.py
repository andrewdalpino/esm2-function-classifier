from argparse import ArgumentParser

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, EsmConfig, EsmForSequenceClassification, TrainingArguments, Trainer

from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.nn import BCEWithLogitsLoss


AVAILABLE_BASE_MODELS = {
    "facebook/esm2_t6_8M_UR50D"
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t48_15B_UR50D",
}


def main():
    parser = ArgumentParser(description="Fine-tune an ESM2 model for protein function classification.")

    parser.add_argument("--base_model", default="facebook/esm2_t6_8M_UR50D", choices=AVAILABLE_BASE_MODELS)
    parser.add_argument("--dataset_path", default="dataset/dataset.jsonl", type=str)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--eval_interval", default=500, type=int)
    parser.add_argument("--checkpoint_interval", default=500, type=int)
    parser.add_argument("--checkpoint_path", default="./checkpoints", type=str)
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = load_dataset("json", data_files=args.dataset_path)

    print(dataset)
    exit()

    # We need give the model samples in the form ...
    #
    # input_ids: [234, 534, 23, 234, 543, ...]
    # label: [1, 0, 1, 0, 0 ...]
    #
    # The input_ids are the tokenized protein sequences and the labels are the
    # corresponding protein function labels. 1 if the sequence has the function, 0 otherwise.

    config = EsmConfig.from_pretrained(args.base_model)
    
    config.problem_type = "multi_label_classification"
    config.num_labels = 47417
    
    model = EsmForSequenceClassification(config)

    loss_function = BCEWithLogitsLoss()

    training_args = TrainingArguments(
        output_dir=args.checkpoint_path,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        save_strategy="steps",
        save_steps=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_dir=args.run_dir_path,
        bf16="cuda" in args.device and is_bf16_supported(),
        torch_compile=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        # data_collator=collate_fn,
        compute_loss_func=loss_function,
    )

    trainer.train()

if __name__ == "__main__":#
    main()