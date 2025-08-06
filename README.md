# ?? Multimodal In-Context Learning with Unsloth

This project implements training and inference pipelines for multimodal reasoning tasks using [Unsloth](https://github.com/unslothai/unsloth), built on top of `Qwen2-VL` and LoRA-based fine-tuning.

---

## ? Installation (Python 3.11, Linux, CUDA 12.1)

We recommend using `venv` and pinned dependencies for reproducibility. These instructions assume you're using an A100 GPU (Ampere architecture) and PyTorch 2.5.1 + CUDA 12.1.

### 1?? Create a virtual environment

```bash
python3.11 -m venv unsloth_venv
source unsloth_venv/bin/activate

> Make sure `python3.11` is installed. On Ubuntu:

```bash
sudo apt install python3.11 python3.11-venv
```

---

### 2?? Install PyTorch (CUDA 12.1)

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 3?? Install project dependencies

```bash
pip install -r requirements.txt
```

> `unsloth[...] @ git+https://...` in the `requirements.txt` must match your CUDA and PyTorch versions. See [Unsloth install guide](https://github.com/unslothai/unsloth) for custom combinations.

---

## ?? Running the Code

### 4?? Run training or inference

```bash
bash run_main.sh [infer|lora_infer|finetune]
```

| Mode         | Description                                 |
|--------------|---------------------------------------------|
| `finetune`   | Finetune the model on your training dataset |
| `infer`      | Run inference with base model               |
| `lora_infer` | Run inference with LoRA fine-tuned model    |

> ?? You must set the `MODEL_PATH` in `run_main.sh` before using it:

```bash
MODEL_PATH=/path/to/your/base_model
```

---

### 5?? Evaluate output accuracy

After inference, your results will be saved in `./results/*.json`.

```bash
python check_accuracy.py ./results/your_result_file.json
```

Example output:

```
Total Samples: 100
Correct Predictions: 92
Accuracy: 92.00%
```

---

## ?? Project Structure

```
.
+-- requirements.txt           # All dependencies (pinned)
+-- run_main.sh                # Entrypoint for training/inference
+-- check_accuracy.py          # Accuracy evaluation
+-- qwen2_finetune_new_model.py# Main training/inference script
+-- dataset/                   # Your preprocessed data
¦   +-- operator_induction/
¦   ¦   +-- support.json
¦   ¦   +-- query.json
+-- results/
    +-- your_result_file.json
```

---

## ?? Notes

- Only Python 3.10 – 3.12 are supported by Unsloth (3.13 is not supported).
- If you switch GPUs or CUDA versions, modify this line in `requirements.txt`:

```txt
unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git

- For help tuning `n_values`, `query_limit`, or datasets, check `run_main.sh`.

---

## ?? Citation & References

- [Unsloth: Fast and Efficient Fine-tuning](https://github.com/unslothai/unsloth)
- [Qwen2-VL on HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
