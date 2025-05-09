from os import path

from datasets import load_dataset

import torch

from torch import Tensor

from torch.utils.data import Dataset

from transformers import EsmTokenizer


class CAFA5(Dataset):
    """
    The CAFA5 dataset is a collection of protein sequences and their associated gene oncology terms.
    It is used for training and evaluating models for protein function prediction.
    The dataset is divided into three subsets based on the type of gene ontology terms:
    1. Molecular Function (MF)
    2. Cellular Component (CC)
    3. Biological Process (BP)
    Each subset contains a different number of classes, which are defined in the `SUBSET_NUM_CLASSES` dictionary.
    The dataset is loaded from a JSON file, and each sample contains a protein sequence and its associated labels.
    """

    SUBSET_PATHS = {
        "all": "all_dataset.jsonl",
        "molecular-function": "mf_dataset.jsonl",
        "cellular-component": "cc_dataset.jsonl",
        "biological-process": "bp_dataset.jsonl",
    }

    SUBSET_NUM_CLASSES = {
        "all": 47417,
        "molecular-function": 12438,
        "cellular-component": 4469,
        "biological-process": 30510,
    }

    def __init__(
        self,
        dataset_path: str,
        subset: str,
        tokenizer: EsmTokenizer,
        context_length: int,
    ):
        super().__init__()

        if subset not in self.SUBSET_PATHS:
            raise ValueError(f"Subset '{subset}' is invalid.")

        subset_path = self.SUBSET_PATHS[subset]

        dataset_path = path.join(dataset_path, subset_path)

        if not path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset file {dataset_path} not found. Please check the path."
            )

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_classes = self.SUBSET_NUM_CLASSES[subset]
        self.context_length = context_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.dataset[index]

        out = self.tokenizer(
            sample["sequence"],
            padding="max_length",
            padding_side="right",
            max_length=self.context_length,
            truncation=True,
        )

        attn_mask = out["attention_mask"]
        tokens = out["input_ids"]

        labels = [0.0] * self.num_classes

        for label_index in sample["label_indices"]:
            labels[label_index] = 1.0

        attn_mask = torch.tensor(attn_mask, dtype=torch.int64)

        x = torch.tensor(tokens, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.float32)

        assert attn_mask.size(0) == self.context_length

        assert x.size(0) == self.context_length
        assert y.size(0) == self.num_classes

        return x, y, attn_mask

    def __len__(self):
        return len(self.dataset)
