# ESM Protein Function Caller

An Evolutionary-scale Model (ESM) for calling protein function from amino acid sequences.

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

## Fine-tuning

Coming soon ...

### Fine-tuning Arguments

| Argument | Default | Type | Description |
|---|---|---|---|
| --base_model | "facebook/esm2_t6_8M_UR50D" | string | The base model name., choose from `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`, `facebook/esm2_t30_150M_UR50D`, `facebook/esm2_t33_650M_UR50D`, `facebook/esm2_t36_3B_UR50D`, or `facebook/esm2_t48_15B_UR50D`. |
| --max_sequence_length | 1024 | int | The maximum length of the input sequences. |
| --num_dataset_processes | 1 | int | The number of CPU processes to use to process and load samples. |
| --batch_size | 32 | int | The number of samples to pass through the network at a time. |
| --gradient_accumulation_steps | 2 | int | The number of batches to pass through the network before updating the weights. |
| --learning_rate | 1e-4 | float | The learning rate of the Adam optimizer. |
| --max_gradient_norm | 1.0 | float | Clip gradients above this threshold norm before stepping. |
| --num_epochs | 3 | int | The number of epochs to train for. |
| --eval_interval | 1 | int | Evaluate the model after this many epochs on the testing set. |
| --eval_ratio | 0.1 | float | The proportion of testing samples to validate the model on. |
| --checkpoint_interval | 1 | int | Save the model parameters to disk every this many epochs. |
| --checkpoint_path | "./checkpoints/checkpoint.pt" | string | The path to the training checkpoint. |
| --resume | False | bool | Should we resume training from the last checkpoint? |
| --run_dir_path | "./runs/instruction-tune" | str | The path to the TensorBoard run directory for this training session. |
| --device | "cuda" | string | The device to run the computation on ("cuda", "cuda:1", "mps", "cpu", etc). |
| --seed | None | int | The seed for the random number generator. |

## References:

>- Z. Lin, et al. Evolutionary-scale prediction of atomic level protein structure with a language model. 2022.
>- G. A. Merino, et al. Hierarchical deep learning for predicting GO annotations by integrating protein knowledge. 2022.