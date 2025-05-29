# ESM2 Protein Function Caller

An Evolutionary-scale Model (ESM) for protein function calling from amino acid sequences. Based on the ESM2 Transformer architecture and fine-tuned on the [CAFA 5](https://huggingface.co/datasets/andrewdalpino/CAFA5) dataset, this model predicts the gene ontology (GO) terms for a particular protein sequence - giving you insight into the molecular function, biological process, and location of the activity inside the cell. It does so by solving a massive multi-label binary classification objective which allows for both GO term ranking and subgraph prediction.

## What are GO terms?

> "The Gene Ontology (GO) is a concept hierarchy that describes the biological function of genes and gene products at different levels of abstraction (Ashburner et al., 2000). It is a good model to describe the multi-faceted nature of protein function."

> "GO is a directed acyclic graph. The nodes in this graph are functional descriptors (terms or classes) connected by relational ties between them (is_a, part_of, etc.). For example, terms 'protein binding activity' and 'binding activity' are related by an is_a relationship; however, the edge in the graph is often reversed to point from binding towards protein binding. This graph contains three subgraphs (subontologies): Molecular Function (MF), Biological Process (BP), and Cellular Component (CC), defined by their root nodes. Biologically, each subgraph represent a different aspect of the protein's function: what it does on a molecular level (MF), which biological processes it participates in (BP) and where in the cell it is located (CC)."

From [CAFA 5 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data)

## Pretrained Models

The following pretrained models are available on HuggingFace Hub.

| Name | Embedding Dim. | Attn. Heads | Encoder Layers | Context Length | Total Parameters |
|---|---|---|---|---|---|
| [andrewdalpino/ESM2-35M-Protein-Biological-Process](https://huggingface.co/andrewdalpino/ESM2-35M-Protein-Biological-Process) | 480 | 20 | 12 | 2048 | 44M |
| [andrewdalpino/ESM2-35M-Protein-Molecular-Function](https://huggingface.co/andrewdalpino/ESM2-35M-Protein-Molecular-Function) | 480 | 20 | 12 | 2048 | 37M |
| [andrewdalpino/ESM2-35M-Protein-Cellular-Component](https://huggingface.co/andrewdalpino/ESM2-35M-Protein-Cellular-Component) | 480 | 20 | 12 | 2048 | 35M |
| [andrewdalpino/ESM2-150M-Protein-Biological-Process](https://huggingface.co/andrewdalpino/ESM2-150M-Protein-Biological-Process) | 640 | 20 | 30 | 1026 | 162M |
| [andrewdalpino/ESM2-150M-Protein-Molecular-Function](https://huggingface.co/andrewdalpino/ESM2-150M-Protein-Molecular-Function) | 640 | 20 | 30 | 1026 | 153M |
| [andrewdalpino/ESM2-150M-Protein-Cellular-Component](https://huggingface.co/andrewdalpino/ESM2-150M-Protein-Cellular-Component) | 640 | 20 | 30 | 1026 | 151M |

### Using a Pretrained Model

Since the HuggingFace [Transformers](https://github.com/huggingface/transformers) library supports the [ESM](https://huggingface.co/docs/transformers/en/model_doc/esm) architecture natively, we can start protein function calling quickly in just a few lines of code. Check out the `import-pretrained.ipynb` notebook for a more detailed example with GO term ranking.

```python
from transformers import EsmTokenizer, EsmForSequenceClassification

model_name = "andrewdalpino/ESM2-35M-Protein-Molecular-Function"

tokenizer = EsmTokenizer.from_pretrained(model_name)

model = EsmForSequenceClassification.from_pretrained(model_name)

# ... then tokenize AA sequences and rank GO terms
```

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Fine-tuning

The Evolutionary-scale Model (ESM) architecture is a Transformer protein sequence model. It was pre-trained using the masked token objective on the [UniProt](https://www.uniprot.org/) dataset, a massive set of protein sequences. Our objective is to fine-tune the base model to predict the gene ontology subgraph for a given protein sequence.

We'll be fine-tuning the pre-trained ESM2 model with a multi-label binary classification head on the CAFA 5 dataset of GO term-annotated protein sequences. To begin training with the default arguments, you can enter the command below.

```sh
python fine-tune.py
```

You can change the base model and dataset subset like in the example below.

```sh
python fine-tune.py --base_model="facebook/esm2_t33_650M_UR50D" --dataset_subset="biological_process"
```

You can also adjust the `batch_size`, `gradient_accumulation_steps`, and `learning_rate` like in the example below.

```sh
python fine-tune.py --batch_size=16 --gradient_accumulation_step=4 --learning_rate=5e-4
```

Training checkpoints will be saved at the `checkpoint_path` location. You can change the location and the `checkpoint_interval` like in the example below.

```sh
python fine-tune.py --checkpoint_path="./checkpoints/biological-process-large.pt" --checkpoint_interval=3
```

If you would like to resume training from a previous checkpoint, make sure to add the `resume` argument. Note that if the checkpoint path already exists, the file will be overwritten.

```sh
python fine-tune.py --checkpoint_path="./checkpoints/checkpoint.pt" --resume
```

### Training Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --base_model | "facebook/esm2_t6_8M_UR50D" | str | The base model name, choose from `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`, `facebook/esm2_t30_150M_UR50D`, `facebook/esm2_t33_650M_UR50D`, `facebook/esm2_t36_3B_UR50D`, or `facebook/esm2_t48_15B_UR50D`. |
| --dataset_subset | "all" | str | The subset of the dataset to train on, choose from `all`, `mf` for molecular function, `cc` for cellular component, or `bp` for biological process. |
| --num_dataset_processes | 1 | int | The number of CPU processes to use to process and load samples. |
| --context_length | 1026 | int | The maximum length of the input sequences. |
| --filter_long_sequences | False | bool | Should we filter sequences that are longer than the context length from the training set? |
| --unfreeze_last_k_layers | 0 | int | Fine-tune the last k layers of the pre-trained encoder. |
| --batch_size | 16 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 4 | int | The number of batches to pass through the network before updating the weights. |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --learning_rate | 5e-4 | float | The learning rate of the Adam optimizer. |
| --num_epochs | 30 | int | The number of epochs to train for. |
| --eval_interval | 2 | int | Evaluate the model after this many epochs on the testing set. |
| --checkpoint_interval | 2 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | string | The path to the training checkpoint. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs/instruction-tune" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cuda" | str | The device to run the computation on ("cuda", "cuda:1", "mps", "cpu", etc). |
| --seed | None | int | The seed for the random number generator. |

## Training Dashboard

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to capture and display training events such as loss and gradient norm updates. To launch the dashboard server run the following command from the terminal.

```
tensorboard --logdir=./runs
```

## GO Subgraph Prediction

We can also infer the gene ontology subgraph of a particular sequence. The `predict-subgraph.py` script outputs a graphical representation of the predictions where green nodes have high probability and pink nodes have low probability.

```sh
python predict-subgraph.py --checkpoint_path="./checkpoints/checkpoint.pt" --top_p=0.1
```

```sh
Checkpoint loaded successfully
Enter a sequence: NMPNERLKWLMLFAAVALIACGSQTLAANPPDADQKGPVFLKEPTNRIDFSNSTG...
```

![Example GO Subgraph](https://raw.githubusercontent.com/andrewdalpino/esm2-function-classifier/master/docs/images/Q0E9J9-mf.png)

### Subgraph Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the training checkpoint. |
| --context_length | 1026 | int | The maximum length of the input sequences. |
| --top_p | 0.5 | float | Only display nodes with the top p probability. |
| --device | "cuda" | str | The device to run the computation on ("cuda", "cuda:1", "mps", "cpu", etc). |
| --seed | None | int | The seed for the random number generator. |

## GO Term Ranking

We provide a prediction script for sampling the top k GO terms inferred by the model.

```sh
python predict-rank.py --checkpoint_path="./checkpoints/checkpoint.pt" --top_k=20
```

You will be asked to enter a protein sequence to predict like in the example below.

```sh
Checkpoint loaded successfully
Enter a sequence: MASMAGVGGGSGKRVPPTRVWWRLYEFALGLLGVVFFAAAATSGKTSRLVSVLIG...

Top 20 GO Terms:
0.6195: cellular anatomical entity
0.5855: cellular_component
0.4599: cell periphery
0.4597: membrane
0.3749: plasma membrane
...
```

### Ranking Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the training checkpoint. |
| --context_length | 1026 | int | The maximum length of the input sequences. |
| --top_k | 10 | int | The top k GO terms and their probabilities to output as predictions. |
| --device | "cuda" | str | The device to run the computation on ("cuda", "cuda:1", "mps", "cpu", etc). |
| --seed | None | int | The seed for the random number generator. |

## References:

>- A. Rives, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences, 2021.
>- Z. Lin, et al. Evolutionary-scale prediction of atomic level protein structure with a language model, 2022.
>- G. A. Merino, et al. Hierarchical deep learning for predicting GO annotations by integrating protein knowledge, 2022.
>- I. Friedberg, et al. CAFA 5 Protein Function Prediction. https://kaggle.com/competitions/cafa-5-protein-function-prediction, 2023.
>- M. Ashburner, et al. Gene Ontology: tool for the unification of biology, 2000.
