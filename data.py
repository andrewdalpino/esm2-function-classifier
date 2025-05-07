from datasets import load_dataset

import torch

from torch import Tensor

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import EsmTokenizer


class CAFA5(Dataset):
    """
    The CAFA5 dataset is a collection of protein sequences and their associated
    functional annotations.
    """

    NUM_CLASSES = 47417

    def __init__(
        self, dataset_path: str, tokenizer: EsmTokenizer, max_length: int = 1024
    ):
        super().__init__()

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index: int):
        sample = self.dataset[index]

        out = self.tokenizer(
            sample["sequence"],
            padding="max_length",
            padding_side="right",
            max_length=self.max_length,
            truncation=True,
        )

        tokens = out["input_ids"]
        attn_mask = out["attention_mask"]

        labels = [0.0] * self.NUM_CLASSES

        for label_index in sample["label_indices"]:
            labels[label_index] = 1.0

        x = torch.tensor(tokens, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.float32)

        attn_mask = torch.tensor(attn_mask, dtype=torch.int64)

        assert x.size(0) == self.max_length
        assert y.size(0) == self.NUM_CLASSES

        assert attn_mask.size(0) == self.max_length

        return x, y, attn_mask

    def __len__(self):
        return len(self.dataset)
