import unittest
from unittest.mock import patch, MagicMock

import torch

from transformers import EsmTokenizer

from data import CAFA5


class TestCAFA5(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock(spec=EsmTokenizer)
        self.mock_tokenizer.return_value = {"attention_mask": [1, 1, 0], "input_ids": [2, 3, 0]}
        
        self.mock_sample = {
            "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "terms": ["GO:0003674", "GO:0005575"],
            "length": 66
        }
        
        self.mock_dataset = {
            "train": MagicMock(),
            "test": MagicMock()
        }
        self.mock_dataset["train"].__getitem__.return_value = self.mock_sample
        self.mock_dataset["train"].__len__.return_value = 100
        self.mock_dataset["test"].__getitem__.return_value = self.mock_sample
        self.mock_dataset["test"].__len__.return_value = 20
        
        self.mock_dataset_instance = MagicMock()
        self.mock_dataset_instance.filter.return_value = self.mock_dataset_instance
        self.mock_dataset_instance.__getitem__.return_value = self.mock_dataset["train"]
        self.mock_dataset_instance.values.return_value = [self.mock_dataset["train"], self.mock_dataset["test"]]

    @patch('data.load_dataset')
    def test_init_valid_parameters(self, mock_load_dataset):
        mock_load_dataset.return_value = self.mock_dataset_instance
        
        cafa5 = CAFA5(
            subset="mf",
            split="train",
            tokenizer=self.mock_tokenizer,
            context_length=1024
        )
        
        self.assertEqual(cafa5.context_length, 1024)
        self.assertEqual(cafa5.tokenizer, self.mock_tokenizer)
        mock_load_dataset.assert_called_once_with(CAFA5.DATASET_NAME, "mf")

    @patch('data.load_dataset')
    def test_init_invalid_subset(self, mock_load_dataset):
        with self.assertRaises(ValueError) as context:
            CAFA5(
                subset="invalid_subset",
                split="train",
                tokenizer=self.mock_tokenizer,
                context_length=1024
            )
        
        self.assertIn("Subset 'invalid_subset' is invalid", str(context.exception))
        mock_load_dataset.assert_not_called()

    @patch('data.load_dataset')
    def test_init_invalid_split(self, mock_load_dataset):
        with self.assertRaises(ValueError) as context:
            CAFA5(
                subset="mf",
                split="invalid_split",
                tokenizer=self.mock_tokenizer,
                context_length=1024
            )
        
        self.assertIn("Split 'invalid_split' is invalid", str(context.exception))
        mock_load_dataset.assert_not_called()

    @patch('data.load_dataset')
    def test_init_invalid_context_length(self, mock_load_dataset):
        with self.assertRaises(ValueError) as context:
            CAFA5(
                subset="mf",
                split="train",
                tokenizer=self.mock_tokenizer,
                context_length=0
            )
        
        self.assertIn("Context length must be greater than 0", str(context.exception))
        mock_load_dataset.assert_not_called()

    @patch('data.load_dataset')
    def test_filter_long_sequences(self, mock_load_dataset):
        mock_load_dataset.return_value = self.mock_dataset_instance
        
        cafa5 = CAFA5(
            subset="mf",
            split="train",
            tokenizer=self.mock_tokenizer,
            context_length=100,
            filter_long_sequences=True
        )
        
        self.mock_dataset_instance.filter.assert_called_once()

    @patch('data.load_dataset')
    def test_label_indices_to_terms(self, mock_load_dataset):
        mock_dataset_values = MagicMock()
        mock_dataset_values.values.return_value = [[{"terms": ["term1", "term2"]}]]
        mock_dataset_values.__getitem__.return_value = MagicMock()
        mock_load_dataset.return_value = mock_dataset_values
        
        cafa5 = CAFA5(
            subset="mf",
            split="train",
            tokenizer=self.mock_tokenizer,
            context_length=1024
        )
        
        cafa5.terms_to_label_indices = {"term1": 0, "term2": 1}
        
        expected = {0: "term1", 1: "term2"}
        self.assertEqual(cafa5.label_indices_to_terms, expected)

    @patch('data.load_dataset')
    def test_getitem(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {
            "sequence": "PROTEIN",
            "terms": ["GO:0003674"]
        }
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.values.return_value = [mock_dataset]
        mock_dataset_obj.__getitem__.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset_obj
        
        self.mock_tokenizer.return_value = {
            "attention_mask": [1, 1, 1, 0, 0],
            "input_ids": [0, 1, 2, 0, 0]
        }
        
        cafa5 = CAFA5(
            subset="mf",
            split="train",
            tokenizer=self.mock_tokenizer,
            context_length=5
        )
        
        cafa5.terms_to_label_indices = {"GO:0003674": 0}
        cafa5.num_classes = 1
        cafa5.dataset = mock_dataset
        
        x, y, attn_mask = cafa5[0]
        
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(attn_mask, torch.Tensor)
        self.assertEqual(x.size(0), 5)
        self.assertEqual(y.size(0), 1)
        self.assertEqual(attn_mask.size(0), 5)

    @patch('data.load_dataset')
    def test_len(self, mock_load_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 42
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.values.return_value = [mock_dataset]
        mock_dataset_obj.__getitem__.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset_obj
        
        cafa5 = CAFA5(
            subset="mf",
            split="train",
            tokenizer=self.mock_tokenizer,
            context_length=1024
        )
        
        cafa5.dataset = mock_dataset
        
        self.assertEqual(len(cafa5), 42)


if __name__ == '__main__':
    unittest.main()