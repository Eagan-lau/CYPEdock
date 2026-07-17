import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

# Custom dataset class
class ProteinDataset(Dataset):
    def __init__(self, token_dir, label_dir):
        self.token_dir = token_dir
        self.label_dir = label_dir
        self.token_names = sorted(os.listdir(token_dir))  # Preserve deterministic order
        self.label_names = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.token_names)

    def __getitem__(self, idx):
        # Load token and label files
        token_path = os.path.join(self.token_dir, self.token_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])

        # Load token and label arrays
        tokens = np.load(token_path)  # shape [residue_size, 100]
        labels = np.load(label_path)  # shape [residue_size]

        # Convert arrays to PyTorch tensors
        tokens = torch.tensor(tokens, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return tokens, labels

# Collate variable-length sequences with padding
def collate_fn(batch):
    protein_tokens, labels = zip(*batch)

    # Pad features and labels to the maximum length in the current batch
    padded_tokens = pad_sequence(protein_tokens, batch_first=True)  # shape: [batch_size, max_residue_size, 100]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Ignore -1 in the loss

    # Build the model attention mask
    attention_mask = (padded_labels != -1).long()  # Valid positions are 1; padding is 0

    return padded_tokens, padded_labels, attention_mask


# ==================== Load training and validation data ====================
# Data paths
pdbbind_train_tokens_dir = '../data/PDBbind/train/tokens'
pdbbind_train_labels_dir = '../data/PDBbind/train/labels'
pdbbind_val_tokens_dir = '../data/PDBbind/valid/tokens'
pdbbind_val_labels_dir = '../data/PDBbind/valid/labels'
scpdb_train_tokens_dir = '/media/yugengliu/DATA2/whli/P450_model/Gener_model/data/scPDB2/train/tokens'
scpdb_train_labels_dir = '/media/yugengliu/DATA2/whli/P450_model/Gener_model/data/scPDB2/train/labels'
scpdb_val_tokens_dir = '/media/yugengliu/DATA2/whli/P450_model/Gener_model/data/scPDB2/valid/tokens'
scpdb_val_labels_dir = '/media/yugengliu/DATA2/whli/P450_model/Gener_model/data/scPDB2/valid/labels'

# Batch size
batch_size = 14

# Initialize the two training datasets
pdbbind_train_dataset = ProteinDataset(pdbbind_train_tokens_dir, pdbbind_train_labels_dir)
scpdb_train_dataset = ProteinDataset(scpdb_train_tokens_dir, scpdb_train_labels_dir)

# Combine the two training datasets
combined_train_dataset = ConcatDataset([pdbbind_train_dataset, scpdb_train_dataset])

# Initialize the training DataLoader
train_dataloader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize validation datasets and DataLoader
pdbbind_val_dataset = ProteinDataset(pdbbind_val_tokens_dir, pdbbind_val_labels_dir)
scpdb_val_dataset = ProteinDataset(scpdb_val_tokens_dir, scpdb_val_labels_dir)
combined_val_dataset = ConcatDataset([pdbbind_val_dataset, scpdb_val_dataset])
val_dataloader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
