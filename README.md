# DPO Training Pipeline (Unsloth Edition)

## What is this?
This is a software tool designed to **fine-tune an AI model** (specifically Qwen 2.5) so it follows instructions better.

It uses a technique called **DPO (Direct Preference Optimization)**. Think of DPO as giving the AI two answers to a question—one "good" and one "bad"—and telling it, *"Prefer the good one."*

We use a library called **Unsloth** to make this process much faster and memory-efficient, allowing it to run on free Google Colab GPUs (like the T4).

---

## Key Features
* **Memory Efficient**: Uses "4-bit quantization" (compressing the model) so it fits on smaller computers.
* **Auto-Export**: Automatically converts the finished model into a format (`GGUF`) that you can run locally using **Ollama**.

---

## Setup & Installation
If you are running this in Google Colab, copy and paste this into your first cell to install the necessary tools:

```python
%%capture
import torch
!pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

## Data Requirment
You must have a file named `dpo_dataset.jsonl` in your folder. This files contains your training examples.

1. Example of `dpo_dataset.jsonl`
```json
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "The capital of France is Berlin."}
{"prompt": "Write a Python function to add two numbers.", "chosen": "def add(a, b):\n    return a + b", "rejected": "function add(a, b) {\n  return a + b;\n}"}
{"prompt": "Explain gravity to a child.", "chosen": "Gravity is the invisible force that pulls everything down towards the ground. It's why you land when you jump!", "rejected": "Gravity is a natural phenomenon by which all things with mass or energy—including planets, stars, galaxies, and even light—are brought toward one another."}
```

## The Control Panel (`TrainingConfig`)
You don't need to touch the complex code to change settings. Just look at the `TrainingConfig` section at the top of the script.

```python
@dataclass
class TrainingConfig:
    """Configuration for the DPO training pipeline."""
    
    # Model Configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    # Training Hyperparameters
    dataset_path: str = "dpo_dataset.jsonl"
    output_dir: str = "outputs"
    batch_size: int = 2 # Change it to 1 if memory crashes
    grad_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    beta: float = 0.1
    seed: int = 3407
    
    # Export
    export_quantization: str = "q4_k_m"
    export_path: str = "model_corrected"
```

## How It Works (The Pipleline)

1. Load Model(`load_model`)
It download the base model from the internet. It then attaches **LoRA adapters**.

2. Load Data(`load_data`)
It checks if your `dpo_dataset.jsonl` file exists and prepares it for the AI to read.

3. Train(`train`)
This is where the learning happens. The AI looks at your data, compares "chosen" vs "rejected" answers, and updates its internal math to prefer the chosen ones.

4. Export(`export`)
Once training is done, it saves the model.
