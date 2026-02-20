"""
Fine-tuning ESM-2 Models for AAV Protein Property Prediction

This script trains ESM-2 models for three tasks:
1. Fitness score prediction
2. Kidney tropism prediction
3. Thermostability prediction

The training uses a sequential transfer learning approach where models are 
fine-tuned on fitness scores first, then adapted for specific properties.
"""

import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


class AAVModelTrainer:
    """Handles training of ESM-2 models for AAV property prediction."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 755,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to pretrained model or checkpoint
            tokenizer_path: Path to tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_path = model_path
        self.max_length = max_length
        
    def prepare_data(
        self,
        csv_path: str,
        seq_column: str = "final_seq",
        target_column: str = "fitness_score",
        n_bins: int = 10,
        test_size: float = 0.2,
        random_state: int = 1234,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data with stratified train/validation split.
        
        Args:
            csv_path: Path to input CSV file
            seq_column: Name of sequence column
            target_column: Name of target column
            n_bins: Number of bins for stratified splitting
            test_size: Fraction of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        # Load and aggregate by sequence
        df = pd.read_csv(csv_path)
        df = df.groupby(seq_column, as_index=False)[target_column].mean()
        df = df.dropna(subset=[target_column])
        
        # Prepare for modeling
        df = df[[seq_column, target_column]]
        df.rename(columns={seq_column: "text", target_column: "label"}, inplace=True)
        
        # Remove sequences with periods (invalid characters)
        df = df[~df["text"].str.contains(r"\.", na=False)]
        
        # Stratified split based on target bins
        df["target_bins"] = pd.qcut(df["label"], q=n_bins, duplicates="drop")
        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df["target_bins"], random_state=random_state
        )
        
        # Clean up
        for d in [train_df, val_df]:
            d.drop(columns=["target_bins"], inplace=True)
            d.reset_index(drop=True, inplace=True)
            
        return train_df, val_df
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize sequences for model input."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
    
    def prepare_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        save_dir: str,
        dataset_name: str,
    ) -> Tuple[Dataset, Dataset]:
        """
        Tokenize and save datasets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            save_dir: Directory to save processed datasets
            dataset_name: Name identifier for the dataset
            
        Returns:
            Tuple of (tokenized_train, tokenized_eval)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        train_set = Dataset.from_pandas(train_df)
        eval_set = Dataset.from_pandas(val_df)
        
        tokenized_train = train_set.map(self.tokenize_function)
        tokenized_eval = eval_set.map(self.tokenize_function)
        
        train_path = os.path.join(save_dir, f"train_{dataset_name}")
        eval_path = os.path.join(save_dir, f"val_{dataset_name}")
        
        tokenized_train.save_to_disk(train_path)
        tokenized_eval.save_to_disk(eval_path)
        
        return tokenized_train, tokenized_eval
    
    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        """Compute regression metrics."""
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        rmse = np.sqrt(mse)
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str,
        training_config: Dict,
        resume_from_checkpoint: bool = False,
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
            output_dir: Directory for model checkpoints
            training_config: Dictionary of training arguments
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Dictionary of evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model = EsmForSequenceClassification.from_pretrained(
            self.model_path, num_labels=1
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, "logs"),
            **training_config,
        )
        
        callbacks = []
        if "early_stopping_patience" in training_config:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=training_config["early_stopping_patience"]
                )
            )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks if callbacks else None,
        )
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        eval_results = trainer.evaluate()
        
        return eval_results


def train_fitness_model(
    base_model_path: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    dataset_dir: str,
):
    """Train the base fitness score prediction model."""
    print("\n" + "=" * 80)
    print("TRAINING FITNESS SCORE MODEL")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    trainer = AAVModelTrainer(base_model_path, tokenizer_path)
    
    # Prepare data
    train_df, val_df = trainer.prepare_data(data_path)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Tokenize and save
    train_dataset, eval_dataset = trainer.prepare_datasets(
        train_df, val_df, dataset_dir, "fitness_aav2_final"
    )
    
    # Training configuration
    training_config = {
        "overwrite_output_dir": True,
        "num_train_epochs": 10,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 8,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 10,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "seed": 42,
    }
    
    # Train
    eval_results = trainer.train(
        train_dataset, eval_dataset, output_dir, training_config, resume_from_checkpoint=True
    )
    
    # Print results
    elapsed_time = (time.time() - start_time) / 3600
    print_results("Fitness Score", eval_results, elapsed_time)
    
    return output_dir


def train_kidney_model(
    fitness_checkpoint: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    dataset_dir: str,
):
    """Train the kidney tropism prediction model."""
    print("\n" + "=" * 80)
    print("TRAINING KIDNEY TROPISM MODEL")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    trainer = AAVModelTrainer(fitness_checkpoint, tokenizer_path)
    
    # Prepare data
    train_df, val_df = trainer.prepare_data(data_path)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Tokenize and save
    train_dataset, eval_dataset = trainer.prepare_datasets(
        train_df, val_df, dataset_dir, "kidney_aav2_final"
    )
    
    # Training configuration
    training_config = {
        "overwrite_output_dir": True,
        "num_train_epochs": 10,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 8,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 20,
        "learning_rate": 2e-6,
        "weight_decay": 0.01,
        "warmup_steps": 20,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "early_stopping_patience": 3,
    }
    
    # Train
    eval_results = trainer.train(
        train_dataset, eval_dataset, output_dir, training_config
    )
    
    # Print results
    elapsed_time = (time.time() - start_time) / 3600
    print_results("Kidney Tropism", eval_results, elapsed_time)


def train_thermostability_model(
    fitness_checkpoint: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    dataset_dir: str,
):
    """Train the thermostability prediction model."""
    print("\n" + "=" * 80)
    print("TRAINING THERMOSTABILITY MODEL")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    trainer = AAVModelTrainer(fitness_checkpoint, tokenizer_path)
    
    # Prepare data
    train_df, val_df = trainer.prepare_data(data_path)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Tokenize and save
    train_dataset, eval_dataset = trainer.prepare_datasets(
        train_df, val_df, dataset_dir, "thermo_final"
    )
    
    # Training configuration
    training_config = {
        "overwrite_output_dir": True,
        "num_train_epochs": 500,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 8,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 20,
        "learning_rate": 5e-7,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "early_stopping_patience": 20,
    }
    
    # Train
    eval_results = trainer.train(
        train_dataset, eval_dataset, output_dir, training_config, resume_from_checkpoint=True
    )
    
    # Print results
    elapsed_time = (time.time() - start_time) / 3600
    print_results("Thermostability", eval_results, elapsed_time)


def print_results(task_name: str, eval_results: Dict[str, float], elapsed_time: float):
    """Print formatted training results."""
    print("\n" + "=" * 80)
    print(f"{task_name.upper()} - EVALUATION RESULTS")
    print("=" * 80)
    print(f"MSE:  {eval_results['eval_mse']:.4f}")
    print(f"MAE:  {eval_results['eval_mae']:.4f}")
    print(f"R²:   {eval_results['eval_r2']:.4f}")
    print(f"RMSE: {eval_results['eval_rmse']:.4f}")
    print("-" * 80)
    print(f"Total training time: {elapsed_time:.2f} hours")
    print("=" * 80 + "\n")


def main():
    """Main training pipeline."""
    # Configuration
    BASE_MODEL = "facebook/esm2_t6_8M_UR50D"
    TOKENIZER = BASE_MODEL
    
    BASE_DIR = "/home/u111169/wrkdir/mgh/aav"
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    DATASET_DIR = os.path.join(DATA_DIR, "hf_datasets")
    
    # Step 1: Train fitness model
    fitness_output = os.path.join(
        CHECKPOINT_DIR, "esm-2_8m-fitness_aav2_final_0_2_valid"
    )
    train_fitness_model(
        BASE_MODEL,
        TOKENIZER,
        os.path.join(DATA_DIR, "production_main_merged_final.csv"),
        fitness_output,
        DATASET_DIR,
    )
    
    # Use best fitness checkpoint for transfer learning
    fitness_checkpoint = os.path.join(fitness_output, "checkpoint-14100")
    
    # Step 2: Train kidney tropism model
    kidney_output = os.path.join(
        CHECKPOINT_DIR, "esm-2_8m-kidney_aav2_final_0_2_valid"
    )
    train_kidney_model(
        fitness_checkpoint,
        TOKENIZER,
        os.path.join(DATA_DIR, "Kidney_Tropism.csv"),
        kidney_output,
        DATASET_DIR,
    )
    
    # Step 3: Train thermostability model
    thermo_output = os.path.join(
        CHECKPOINT_DIR, "esm-2_8m-thermo_final_0_2_valid"
    )
    train_thermostability_model(
        fitness_checkpoint,
        TOKENIZER,
        os.path.join(DATA_DIR, "Thermostability.csv"),
        thermo_output,
        DATASET_DIR,
    )
    
    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()