from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
# Apply Unsloth patches immediately
PatchDPOTrainer()

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path

from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


class DPOPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Loads the base model and attaches LoRA adapters."""
        logger.info(f"Loading base model: {self.config.model_name}")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
            )

            logger.info("Applying LoRA adapters...")
            self.model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self.config.lora_target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=self.config.seed,
            )
            self.tokenizer = tokenizer
            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_data(self) -> Dataset:
        """Loads and validates the dataset."""
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.config.dataset_path}")

        logger.info(f"Loading dataset from {self.config.dataset_path}")
        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        logger.info(f"Loaded {len(dataset)} examples.")
        return dataset

    def train(self) -> None:
        """Executes the DPO training loop."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        dataset = self.load_data()

        logger.info("Initializing DPO Trainer...")
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Unsloth handles this efficiently (no copy needed)
            processing_class=self.tokenizer,
            train_dataset=dataset,
            args=DPOConfig(
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.grad_accumulation_steps,
                warmup_ratio=self.config.warmup_ratio,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                output_dir=self.config.output_dir,
                seed=self.config.seed,
                report_to="none", # Change to "wandb" for production tracking
            ),
            beta=self.config.beta,
        )

        logger.info("Starting training...")
        dpo_trainer.train()
        logger.info("Training complete.")

    def export(self) -> None:
        """Exports the model to GGUF format."""
        if not self.model:
            raise RuntimeError("No model to export.")

        logger.info(f"Exporting model to GGUF ({self.config.export_quantization})...")
        try:
            self.model.save_pretrained_gguf(
                self.config.export_path,
                self.tokenizer,
                quantization_method=self.config.export_quantization
            )
            logger.info(f"Model saved to ./{self.config.export_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise


def main():
    # In a real CLI, we would use argparse here to override Config defaults
    config = TrainingConfig(
        dataset_path="dpo_dataset.jsonl",
        num_epochs=3
    )

    pipeline = DPOPipeline(config)
    
    try:
        pipeline.load_model()
        pipeline.train()
        pipeline.export()
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
