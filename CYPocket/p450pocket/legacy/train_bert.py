import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from dataset import train_dataloader, val_dataloader
import matplotlib.pyplot as plt
from model_bert import ProteinBERTClassifier, chunked_forward
from datetime import datetime

# Initialize model
model = ProteinBERTClassifier()

# Configure training
epochs = 400
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded labels

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure chunking
chunk_size = 512  # Maximum sequence length per chunk

# Store epoch-level training and validation values
train_loss_values, val_loss_values = [], []
train_accuracy_values, val_accuracy_values = [], []
best_val_f1 = 0.0  # Best validation F1

for epoch in range(epochs):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Epoch {epoch + 1}/{epochs}")

    # ==================== Training ====================
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

        # Chunked forward propagation
        try:
            logits = chunked_forward(model, batch_tokens, attention_mask, chunk_size=chunk_size)
        except RuntimeError as e:
            # Report the failing batch
            print(f"Batch tokens shape: {batch_tokens.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            print(f"Error message: {e}")
            # Skip the failing batch
            continue

        # Reshape for the loss function
        logits = logits.view(-1, logits.size(-1))  # [batch_size * residue_size, num_labels]
        batch_labels = batch_labels.view(-1)  # [batch_size * residue_size]

        # Compute loss and backpropagate
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_train_loss += loss.item()

        # Record predictions
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        true_labels = batch_labels.cpu().numpy()

        # Exclude padded labels
        valid_indices = true_labels != -1
        train_preds.extend(preds[valid_indices])
        train_labels.extend(true_labels[valid_indices])

    # Compute training metrics
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

    # ==================== Validation ====================
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
                # Report the failing batch
                print(f"Batch tokens shape: {batch_tokens.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Error message: {e}")
                # Skip the failing batch
                continue

            # Reshape for the loss function
            logits = logits.view(-1, logits.size(-1))  # [batch_size * residue_size, num_labels]
            batch_labels = batch_labels.view(-1)  # [batch_size * residue_size]

            # Compute loss
            loss = criterion(logits, batch_labels)
            total_val_loss += loss.item()

            # Record predictions
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = batch_labels.cpu().numpy()

            # Exclude padded labels
            valid_indices = true_labels != -1
            val_preds.extend(preds[valid_indices])
            val_labels.extend(true_labels[valid_indices])

    # Compute validation metrics
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

    # ==================== Save best model ====================
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "/media/yugengliu/DATA2/whli/P450_model/Gener_model/again/best_protein_distilbert_classifier.pth")
        print(f"New best model saved with F1: {best_val_f1:.4f}")

# ==================== Plot training and validation curves ====================
plt.figure(figsize=(12, 6))

# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label="Train Loss")
plt.plot(range(1, epochs + 1), val_loss_values, label="Val Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label="Train Accuracy")
plt.plot(range(1, epochs + 1), val_accuracy_values, label="Val Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
base_path = "/media/yugengliu/DATA2/whli/P450_model/Gener_model/again/training_validation_curves"
plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
plt.savefig(base_path + ".pdf", bbox_inches="tight")

plt.show()


import pandas as pd

# Create a DataFrame
curve_df = pd.DataFrame({
    "Epoch": range(1, epochs + 1),
    "Train_Loss": train_loss_values,
    "Val_Loss": val_loss_values,
    "Train_Accuracy": train_accuracy_values,
    "Val_Accuracy": val_accuracy_values
})

# Save CSV data
csv_output_path = "/media/yugengliu/DATA2/whli/P450_model/Gener_model/again/training_validation_curves.csv"
curve_df.to_csv(csv_output_path, index=False)

print(f"Training curve data saved to {csv_output_path}")


# Save the final model
torch.save(model.state_dict(), "/media/yugengliu/DATA2/whli/P450_model/Gener_model/again/final_protein_distilbert_classifier.pth")
print("Final model saved as '/media/yugengliu/DATA2/whli/P450_model/Gener_model/again/final_protein_distilbert_classifier.pth'")
