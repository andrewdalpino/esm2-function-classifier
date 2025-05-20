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
    """

    DATASET_NAME = "andrewdalpino/CAFA5"

    AVAILABLE_SUBSETS = {"all", "mf", "cc", "bp"}

    def __init__(
        self,
        subset: str,
        tokenizer: EsmTokenizer,
        context_length: int,
        filter_long_sequences: bool = False,
    ):
        super().__init__()

        if subset not in self.AVAILABLE_SUBSETS:
            raise ValueError(f"Subset '{subset}' is invalid.")

        dataset = load_dataset(self.DATASET_NAME, subset, split="train")

        if filter_long_sequences:
            dataset = dataset.filter(
                lambda sample: sample["length"] <= context_length - 2
            )

        terms_to_label_indices = {}

        label_index = 0

        for sample in dataset:
            for term in sample["terms"]:
                if term not in terms_to_label_indices:
                    terms_to_label_indices[term] = label_index

                    label_index += 1

        num_classes = len(terms_to_label_indices)

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.terms_to_label_indices = terms_to_label_indices
        self.num_classes = num_classes

    @property
    def label_indices_to_terms(self):
        """
        Returns a dictionary mapping label indices to their corresponding gene ontology terms.
        """

        return {index: term for term, index in self.terms_to_label_indices.items()}

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

        for term in sample["terms"]:
            label_index = self.terms_to_label_indices[term]

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
