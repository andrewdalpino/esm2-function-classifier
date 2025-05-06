from datasets import load_dataset

import torch

from torch import Tensor

from torch.utils.data import Dataset

from transformers import EsmTokenizer


class CAFA5(Dataset):
    """
    The CAFA5 dataset is a collection of protein sequences and their associated
    functional annotations.
    """

    def __init__(self, dataset_path: str, tokenizer: EsmTokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        dataset = load_dataset("json", data_files=dataset_path)

        self.dataset = dataset["train"]

    def __getitem__(self, index: int):
        sample = self.dataset[index]

        inputs = self.tokenizer(
            sample["sequence"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        labels = [0.0] * 47417

        for label_index in sample["label_indices"]:
            labels[label_index] = 1.0

        input_ids = inputs["input_ids"].squeeze(0)
        attn_mask = inputs["attention_mask"].squeeze(0)

        labels = torch.tensor(labels, dtype=torch.float32)

        return input_ids, attn_mask, labels

    def __len__(self):
        return len(self.dataset)
