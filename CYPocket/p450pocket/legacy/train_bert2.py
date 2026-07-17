import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from dataset import train_dataloader, val_dataloader
import matplotlib.pyplot as plt
from model_bert import ProteinBERTClassifier, chunked_forward
from datetime import datetime
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ==================== Initialize model ====================
model = ProteinBERTClassifier()

# Configure training
epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
chunk_size = 512  # Chunk length

# Learning-rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

# Early-stopping settings
patience = 10
early_stop_counter = 0
best_val_f1 = 0.0
best_val_loss = float("inf")

# Values used for plots
train_loss_values, val_loss_values = [], []
train_accuracy_values, val_accuracy_values = [], []

# CSV log
log_file = "/media/yugengliu/DATA/whli/P450_model/Gener_model/training_log.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "train_prec", "train_rec", "train_f1",
                     "val_loss", "val_acc", "val_prec", "val_rec", "val_f1"])

# ==================== Training loop ====================
for epoch in range(epochs):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Epoch {epoch + 1}/{epochs}")

    # -------- Training --------
    model.train()
    total_train_loss = 0
    train_preds, train_labels = [], []

    for batch_tokens, batch_labels, attention_mask in train_dataloader:
        batch_tokens, batch_labels, attention_mask = (
            batch_tokens.to(device),
            batch_labels.to(device),
            attention_mask.to(device),
        )

        optimizer.zero_grad()
        try:
            logits = chunked_forward(model, batch_tokens, attention_mask, chunk_size=chunk_size)
        except RuntimeError as e:
            print(f"Train batch error: {e}, skipping batch.")
            continue

        logits = logits.view(-1, logits.size(-1))
        batch_labels = batch_labels.view(-1)

        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        true_labels = batch_labels.cpu().numpy()
        valid_indices = true_labels != -1
        train_preds.extend(preds[valid_indices])
        train_labels.extend(true_labels[valid_indices])

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average="binary"
    )

    train_loss_values.append(avg_train_loss)
    train_accuracy_values.append(train_accuracy)

    print(
        f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
        f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}"
    )

    # -------- Validation --------
    model.eval()
    total_val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch_tokens, batch_labels, attention_mask in val_dataloader:
            batch_tokens, batch_labels, attention_mask = (
                batch_tokens.to(device),
                batch_labels.to(device),
                attention_mask.to(device),
            )

            try:
                logits = chunked_forward(model, batch_tokens, attention_mask, chunk_size=chunk_size)
            except RuntimeError as e:
                print(f"Val batch error: {e}, skipping batch.")
                continue

            logits = logits.view(-1, logits.size(-1))
            batch_labels = batch_labels.view(-1)
            loss = criterion(logits, batch_labels)
            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = batch_labels.cpu().numpy()
            valid_indices = true_labels != -1
            val_preds.extend(preds[valid_indices])
            val_labels.extend(true_labels[valid_indices])

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average="binary"
    )

    val_loss_values.append(avg_val_loss)
    val_accuracy_values.append(val_accuracy)

    print(
        f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
    )

    # Append CSV row
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_train_loss, train_accuracy, train_precision, train_recall, train_f1,
                         avg_val_loss, val_accuracy, val_precision, val_recall, val_f1])

    # Update learning-rate scheduler
    scheduler.step(avg_val_loss)

    # Save the best model and update early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "/media/yugengliu/DATA/whli/P450_model/Gener_model/best_protein_distilbert_classifier.pth")
        print(f"New best model saved with F1: {best_val_f1:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

# ==================== Plot training and validation curves ====================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label="Train Loss")
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label="Val Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy_values) + 1), train_accuracy_values, label="Train Accuracy")
plt.plot(range(1, len(val_accuracy_values) + 1), val_accuracy_values, label="Val Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("/media/yugengliu/DATA/whli/P450_model/Gener_model/training_validation_curves.png")
plt.show()

# Save the final model
torch.save(model.state_dict(), "/media/yugengliu/DATA/whli/P450_model/Gener_model/final_protein_distilbert_classifier.pth")
print("Final model saved as '/media/yugengliu/DATA/whli/P450_model/Gener_model/final_protein_distilbert_classifier.pth'")
print(f"Training log saved as '{log_file}'")
