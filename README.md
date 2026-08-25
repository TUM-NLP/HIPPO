# HIPPO 🦛

This repository contains our training and evaluation code for HIPPO: **H**armful **I**nput, **P**ositive and **P**roductive **O**utput.
This is part of our work, "From Specialization to Generalization: Instruction-tuned LLMs for Robust Harmful Content Mitigation", where we show that instruction-tuning an LLM for hate speech can dramatically improve its performance on hate speech-related tasks.


### Training

Training a model is as simple as running the following command:
```bash
python train.py --output_path models/hippo_qwen/ --dataset leukas/hippo_dataset/ --batch_size 16 --grad_acc 4 --epochs 3 --model_path Qwen/Qwen3-4B-Instruct-2507/ --model_type qwen
```

Where:
- `--output_path` is the directory to save the trained model.
- `--dataset` is the dataset to train on.
- `--batch_size` is the batch size.
- `--grad_acc` is the gradient accumulation steps.
- `--epochs` is the number of epochs.
- `--model_path` is the path to the model to train.
- `--model_type` is the type of model to train.

### Evaluation

Evaluation generates predictions for each hierarchical subtask and writes per-subset CSVs (`output`, `label`, `source`) under `--output_path`. Accuracy is also printed per subset.

```bash
python evaluate.py --model_path models/hippo_qwen/ --dataset leukas/hippo_dataset/ --output_path outputs/hippo_qwen/ --model_type qwen --batch_size 8
```

Where:
- `--model_path` is the path to the (fine-tuned) model.
- `--dataset` is the prepared dataset on disk (uses the `test` split).
- `--output_path` is the directory to save prediction CSVs.
- `--model_type` is the model family (`qwen` or `llama`).
- `--batch_size` is the evaluation batch size.
- `--subset` / `--subtask` optionally restricts evaluation to one or more dataset names (comma-separated). Default `all` evaluates everything except `jigsaw`.
- `--regex` treats `--subset` as a substring match on dataset names.

To compute macro-F1 (or task-specific metrics) from the saved CSVs:

```bash
python score.py --outputs_folder outputs/hippo_qwen/ --dataset olid20a_1
```

Where:
- `--outputs_folder` is the directory produced by `evaluate.py`.
- `--dataset` is the CSV stem to score (e.g. `olid20a_1`, `hatecheck`, `toxicspans`, `jigsaw`).
