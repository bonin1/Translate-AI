import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
from tqdm import tqdm
from dataclasses import dataclass
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sacrebleu.metrics import BLEU
import yaml
from pathlib import Path
import json
from sklearn.model_selection import KFold


@dataclass
class TrainingConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-de"
    max_length: int = 128
    batch_size: int = 16
    accumulation_steps: int = 4
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True
    use_wandb: bool = True
    num_folds: int = 5
    patience: int = 3


class TranslationDataset(Dataset):
    def __init__(
            self,
            tokenizer: MarianTokenizer,
            source_texts: List[str],
            target_texts: Optional[List[str]] = None,
            max_length: int = 128,
            is_train: bool = True
    ):
        self.tokenizer = tokenizer
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source = str(self.source_texts[index])

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
        }

        if self.is_train and self.target_texts is not None:
            target = str(self.target_texts[index])
            target_encoding = self.tokenizer(
                target,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item["labels"] = target_encoding["input_ids"].squeeze()

        return item


class TranslationTrainer:
    def __init__(
            self,
            config: TrainingConfig,
            model: MarianMTModel,
            tokenizer: MarianTokenizer,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            device: Optional[torch.device] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_training()

        self.bleu = BLEU()

        self.setup_logging()

    def setup_training(self):
        """Initialize optimizer, scheduler, and other training components"""
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)

        total_steps = len(self.train_dataloader) * self.config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

        self.scaler = GradScaler(enabled=self.config.fp16)

        self.best_valid_loss = float('inf')
        self.patience_counter = 0

    def setup_logging(self):
        """Initialize logging and monitoring"""
        if self.config.use_wandb:
            wandb.init(project="translation-training", config=self.config.__dict__)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        train_iterator = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(train_iterator):
            with autocast(enabled=self.config.fp16):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                loss = outputs.loss / self.config.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            train_iterator.set_postfix({'loss': loss.item()})

            if self.config.use_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })

        return total_loss / len(self.train_dataloader)

    def validate(self) -> Tuple[float, float]:
        """Perform validation"""
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                total_loss += outputs.loss.item()

                generated = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_length=self.config.max_length,
                    num_beams=4
                )

                decoded_preds = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                decoded_refs = self.tokenizer.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )

                predictions.extend(decoded_preds)
                references.extend(decoded_refs)

        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        avg_loss = total_loss / len(self.val_dataloader)

        return avg_loss, bleu_score

    def train(self):
        """Main training loop"""
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)

            if self.val_dataloader:
                val_loss, bleu_score = self.validate()
                self.logger.info(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, BLEU = {bleu_score:.2f}"
                )

                if self.config.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "bleu_score": bleu_score
                    })

                if val_loss < self.best_valid_loss:
                    self.best_valid_loss = val_loss
                    self.patience_counter = 0
                    self.save_model(f"best_model_epoch_{epoch + 1}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        self.logger.info("Early stopping triggered")
                        break
            else:
                self.logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

    def save_model(self, name: str):
        """Save model and tokenizer"""
        output_dir = Path(f"saved_models/{name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(output_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f)


def main():
    config = TrainingConfig(use_wandb=False)
    
    set_seed(config.seed)

    df = pd.read_csv("data/processed/train.csv")

    tokenizer = MarianTokenizer.from_pretrained(config.model_name)
    model = MarianMTModel.from_pretrained(config.model_name)

    kf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\nTraining Fold {fold + 1}")

        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        train_dataset = TranslationDataset(
            tokenizer=tokenizer,
            source_texts=train_data["en_cleaned"].tolist(),
            target_texts=train_data["de_cleaned"].tolist(),
            max_length=config.max_length
        )

        val_dataset = TranslationDataset(
            tokenizer=tokenizer,
            source_texts=val_data["en_cleaned"].tolist(),
            target_texts=val_data["de_cleaned"].tolist(),
            max_length=config.max_length
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        trainer = TranslationTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        trainer.train()


if __name__ == "__main__":
    main()