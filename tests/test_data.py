import unittest
from unittest.mock import MagicMock, patch
import torch
from data import AmiGO


class TestAmiGO(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.tokenizer_output = {
            "attention_mask": [1, 1, 1, 0, 0],
            "input_ids": [2, 3, 4, 0, 0],
        }
        self.mock_tokenizer.return_value = self.tokenizer_output

    @patch("data.load_dataset")
    def test_init_valid_parameters(self, mock_load_dataset):
        # Create mock dataset
        mock_dataset = self._create_mock_dataset()
        mock_load_dataset.return_value = mock_dataset

        # Test initialization with valid parameters
        amigo = AmiGO("all", "train", self.mock_tokenizer, 10)

        # Verify the object was initialized correctly
        self.assertEqual(amigo.dataset, mock_dataset["train"])
        self.assertEqual(amigo.tokenizer, self.mock_tokenizer)
        self.assertEqual(amigo.context_length, 10)
        self.assertEqual(amigo.num_classes, 2)
        self.assertEqual(len(amigo.terms_to_label_indices), 2)
        mock_load_dataset.assert_called_once_with("andrewdalpino/AmiGO", "all")

    def test_init_invalid_subset(self):
        with self.assertRaises(ValueError) as context:
            AmiGO("invalid_subset", "train", self.mock_tokenizer, 10)
        self.assertIn("Subset 'invalid_subset' is invalid", str(context.exception))

    def test_init_invalid_split(self):
        with self.assertRaises(ValueError) as context:
            AmiGO("all", "invalid_split", self.mock_tokenizer, 10)
        self.assertIn("Split 'invalid_split' is invalid", str(context.exception))

    def test_init_invalid_context_length(self):
        with self.assertRaises(ValueError) as context:
            AmiGO("all", "train", self.mock_tokenizer, 0)
        self.assertIn("Context length must be greater than 0", str(context.exception))

    @patch("data.load_dataset")
    def test_label_indices_to_terms(self, mock_load_dataset):
        # Create mock dataset
        mock_dataset = self._create_mock_dataset()
        mock_load_dataset.return_value = mock_dataset

        # Initialize dataset
        amigo = AmiGO("all", "train", self.mock_tokenizer, 10)

        # Test the property
        indices_to_terms = amigo.label_indices_to_terms
        self.assertEqual(len(indices_to_terms), 2)
        self.assertEqual(indices_to_terms[0], "GO:0001")
        self.assertEqual(indices_to_terms[1], "GO:0002")

    @patch("data.load_dataset")
    def test_getitem(self, mock_load_dataset):
        # Create mock dataset with specific sample
        mock_dataset = self._create_mock_dataset()
        mock_dataset["train"].__getitem__.return_value = {
            "sequence": "ACDEFG",
            "go_terms": ["GO:0001"],
        }
        mock_load_dataset.return_value = mock_dataset

        # Setup tokenizer mock
        self.mock_tokenizer.return_value = {
            "attention_mask": [1, 1, 1, 0, 0],
            "input_ids": [2, 3, 4, 0, 0],
        }

        # Initialize dataset
        amigo = AmiGO("all", "train", self.mock_tokenizer, 5)
        amigo.terms_to_label_indices = {"GO:0001": 0, "GO:0002": 1}
        amigo.num_classes = 2

        # Get item
        x, y, attn_mask = amigo[0]

        # Verify results
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(attn_mask, torch.Tensor)

        self.assertEqual(x.size(0), 5)  # context_length
        self.assertEqual(y.size(0), 2)  # num_classes
        self.assertEqual(attn_mask.size(0), 5)  # context_length

        # Check if GO:0001 is labeled as 1.0 and GO:0002 as 0.0
        self.assertEqual(y[0].item(), 1.0)
        self.assertEqual(y[1].item(), 0.0)

        # Verify tokenizer was called correctly
        self.mock_tokenizer.assert_called_once_with(
            "ACDEFG",
            padding="max_length",
            padding_side="right",
            max_length=5,
            truncation=True,
        )

    @patch("data.load_dataset")
    def test_len(self, mock_load_dataset):
        # Create mock dataset
        mock_dataset = self._create_mock_dataset()
        mock_dataset["train"].__len__.return_value = 100
        mock_load_dataset.return_value = mock_dataset

        # Initialize dataset
        amigo = AmiGO("all", "train", self.mock_tokenizer, 10)

        # Check length
        self.assertEqual(len(amigo), 100)

    def _create_mock_dataset(self):
        """Helper method to create a mock dataset for testing"""
        mock_dataset = MagicMock()

        # Create train and test subsets
        mock_train = MagicMock()
        mock_test = MagicMock()

        # Set up dictionary-like behavior
        mock_dataset.__getitem__.side_effect = lambda key: (
            mock_train if key == "train" else mock_test
        )

        # Set up values method to return samples with go terms
        sample = {"go_terms": ["GO:0001", "GO:0002"]}
        mock_subset_data = MagicMock()
        mock_subset_data.__iter__.return_value = [sample]
        mock_dataset.values.return_value = [mock_subset_data]

        return mock_dataset


if __name__ == "__main__":
    unittest.main()
