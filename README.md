# 🔬 Multimodal In-Context Learning with Unsloth

This project implements training and inference pipelines for multimodal reasoning tasks using [Unsloth](https://github.com/unslothai/unsloth), built on top of `Qwen2-VL` and LoRA-based fine-tuning.

---

## ✅ Installation

### 1️⃣ Clone this repository

```bash
git clone https://github.com/your_org/your_repo.git
cd your_repo
```
---

### 2️⃣ Create virtual environment & activate (we recommend using python 3.11)

```bash
python3.11 -m venv unsloth_venv
source unsloth_venv/bin/activate
```
---

### 3️⃣ Install all dependencies

```bash
pip install --upgrade pip
pip install -r pip_requirements.txt
```
---

### ⚠️ Important: Unsloth version must match your PyTorch, CUDA, and GPU architecture

Unsloth must be installed with the correct tag depending on your hardware and environment. For example:

**✅ Our working configuration:**
- GPU: A100 (Ampere)
- CUDA: 12.1
- PyTorch: 2.5.1

Use the following:

```txt
pip install --no-deps 'unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git'
```

If you use a different GPU or CUDA version, refer to [Unsloth install guide](https://github.com/unslothai/unsloth) and adjust accordingly.

---

## 📦 Dataset Preparation

```bash
git clone https://huggingface.co/datasets/ShuoChen99/TrueMICL
```

Please download the required datasets from our data release and rename them as `dataset/` folder. For example:

```bash
dataset/
├── operator_induction/
│   ├── support.json
│   └── query.json
├── sudoku/
│   ├── ...
├── shapes_count/
│   ├── ...
...
```
---

## 🧪 Running the Code

You can run training or inference using the **unified shell script**, for example **infer**:

```bash
bash run_infer.sh infer # option: lora_infer or finetune
```

| Mode         | Description                                 |
|--------------|---------------------------------------------|
| `finetune`   | Finetune the model on your training dataset |
| `infer`      | Run inference with base model               |
| `lora_infer` | Run inference with fine-tuned model    |

> ⚠️ Only one mode can be used at a time.  
> ⚠️ Before running, **make sure to manually set the following variables inside `run_main.sh`**:
> We include a custom **modeling_qwen2_vl.py** in the project root. If you want to use its custom model definition instead of the default provided by transformers

```bash
MODEL_PATH=/absolute/path/to/your/base_model
LORA_MODEL_PATH=./ckpt/your_lora_checkpoint  # if using lora_infer mode
```

---

### ✅ Output and Evaluation

After inference, results are saved under `./results/` in JSON format.

To calculate accuracy:

```bash
python check_accuracy.py ./results/your_result_file.json
```

---

## 📁 Project Structure

```
.
├── ckpt/                         # Your LoRA checkpoints
│   ├── ...
├── dataset/                      # Place downloaded datasets here
│   ├── operator_induction/
│   ├── sudoku/
│   └── ...
├── qwen2_vl_for_replacement/    # Optional model override modules
├── check_accuracy.py
├── data_processing.py
├── modeling_qwen2_vl.py
├── pip_requirements.txt
├── qwen2_finetune_new_model.py
├── run_infer.sh                  # Entry script for training/inference
├── samples_of_training_data.json
└── readme.txt                   # (This README content)
```

---

## 📚 References

- [Unsloth: Fast and Efficient Fine-tuning](https://github.com/unslothai/unsloth)
- [Qwen2-VL on HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
