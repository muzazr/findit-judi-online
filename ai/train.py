import json
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from config import Config

class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text_norm"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += out.loss.item()
            preds = out.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = float(total_loss / max(len(loader), 1))
    return metrics

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2).to(device)

    train_ds = CommentDataset(train_df, tokenizer, config.max_length)
    val_ds = CommentDataset(val_df, tokenizer, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    best_f1 = -1.0
    metrics_log = []

    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{config.num_epochs} | Step {step + 1} | Loss: {loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        val_metrics["epoch"] = epoch + 1
        metrics_log.append(val_metrics)

        print(f"Epoch {epoch + 1} Val Metrics: {val_metrics}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            os.makedirs(config.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))
            with open(os.path.join(config.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics_log, f, indent=2)
            print(f"Saved best model with F1: {best_f1:.4f}")

    print(f"\nTraining Complete. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()