# ESM2 Protein Function Caller

An Evolutionary-scale Model (ESM) for protein function calling from protein sequences. Based on the ESM2 architecture and fine-tuned on the [CAFA 5](https://huggingface.co/datasets/andrewdalpino/CAFA5) dataset, this model predicts the gene oncology (GO) subgraph for a particular amino acid sequence - giving you insight into the protein's cellular component, molecular function, and biological processes.

## Available Base Models

The following base models on HuggingFace Hub are available to fine-tune. Each base is paired with a two-layer classification head.

| Name | Embedding Dim. | Attn. Heads | Layers | Parameters |
|---|---|---|---|---|
| facebook/esm2_t6_8M_UR50D | 320 | 20 | 6 | 8M |
| facebook/esm2_t12_35M_UR50D| 480 | 20 | 12 | 35M |
| facebook/esm2_t30_150M_UR50D | 640 | 20 | 30 | 150M |
| facebook/esm2_t33_650M_UR50D | 1280| 20 | 33 | 650M |
| facebook/esm2_t36_3B_UR50D | 2560 | 40 | 36 | 3B |
| facebook/esm2_t48_15B_UR50D | 5120 | 40 | 48 | 15B |

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
| --unfreeze_last_k_layers | 0 | int | Fine-tune the last k layers of the pre-trained encoder. |
| --batch_size | 16 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 4 | int | The number of batches to pass through the network before updating the weights. |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --learning_rate | 1e-4 | float | The learning rate of the Adam optimizer. |
| --dropout | 0.1 | float | The amount of dropout to apply to the hidden and attention layers during training. |
| --num_epochs | 20 | int | The number of epochs to train for. |
| --eval_interval | 2 | int | Evaluate the model after this many epochs on the testing set. |
| --eval_ratio | 0.1 | float | The proportion of testing samples to validate the model on. |
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

## GO Term Ranking

We provide a prediction script for sampling the top k GO terms inferred by the model.

```sh
python predict-rank.py --checkpoint_path="./checkpoints/checkpoint.pt" --top_k=5
```

You will be asked to enter a protein sequence to predict like in the example below.

```sh
Checkpoint loaded successfully
Enter a sequence: MASMAGVGGGSGKRVPPTRVWWRLYEFALGLLGVVFFAAAATSGKTSRLVSVLIG...

Top 5 GO Terms:
0.6195: cellular anatomical entity
0.5855: cellular_component
0.4599: cell periphery
0.4597: membrane
0.3749: plasma membrane
```

### Ranking Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --checkpoint_path | "./checkpoints/checkpoint.pt" | str | The path to the training checkpoint. |
| --context_length | 1024 | int | The maximum length of the input sequences. |
| --top_k | 10 | int | The top k GO terms and their probabilities to output as predictions. |
| --device | "cuda" | str | The device to run the computation on ("cuda", "cuda:1", "mps", "cpu", etc). |
| --seed | None | int | The seed for the random number generator. |

## References:

>- A. Rives, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences, 2021.
>- Z. Lin, et al. Evolutionary-scale prediction of atomic level protein structure with a language model, 2022.
>- G. A. Merino, et al. Hierarchical deep learning for predicting GO annotations by integrating protein knowledge, 2022.
>- I. Friedberg, et al. CAFA 5 Protein Function Prediction. https://kaggle.com/competitions/cafa-5-protein-function-prediction, 2023.
>- D. Chen, et al. Endowing Protein Language Models with Structural Knowledge, 2024.
>- S. Jiao, et al. Beyond ESM2: Graph-Enhanced Protein Sequence Modeling with Efficient Clustering, 2024.
>- G. B. de Oliveira, et al. Scaling Up ESM2 Architectures for Long Protein Sequences Analysis: Long and Quantized Approaches, 2025.
>- M. Ashburner, et al. Gene Ontology: tool for the unification of biology, 2000.
