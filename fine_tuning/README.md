# Supervised Fine-Tuning Examples

This directory contains Jupyter notebooks for supervised fine-tuning (SFT) of Llama 3.1 8B Instruct with Unsloth and LoRA/QLoRA.

## Notebooks

| Notebook | Description |
| --- | --- |
| `llama3_1_8b_sft_example.ipynb` | A project-provided example for fine-tuning Llama 3.1 8B Instruct on a custom instruction dataset. |
| `llama3_1_8b_sft_unsloth_official_example.ipynb` | The official Unsloth Llama 3.1 8B Alpaca notebook, included as an upstream reference. |

## Dataset Format

The project-provided example expects JSON records with the following fields:

```json
{
  "instruction": "Describe the task.",
  "input": "Optional additional context.",
  "output": "The expected response."
}
```

The `input` field may be an empty string. Update the dataset paths in the notebook before starting training.

## Requirements

- A CUDA-compatible GPU
- Python and Jupyter Notebook
- PyTorch
- Unsloth
- Hugging Face Transformers, Datasets, TRL, and PEFT

A Google Colab GPU runtime can also be used. Adjust the batch size, sequence length, and gradient accumulation settings according to the available GPU memory.

## Reference

- [Unsloth GitHub repository](https://github.com/unslothai/unsloth)
