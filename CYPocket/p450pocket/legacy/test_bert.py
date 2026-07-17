# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from model_bert import ProteinBERTClassifier, chunked_forward

def predict_protein_pockets(model, protein_tokens, attention_mask, device, chunk_size=512):
    """
    Predict whether each protein residue belongs to a pocket.

    Args:
        model: Trained BERT model.
        protein_tokens: Protein token representation with shape
            [batch_size, residue_size, sequence_length].
        attention_mask: Attention mask with shape [batch_size, residue_size].
        device: CPU or GPU device.
        chunk_size: Chunk length for long sequences.

    Returns:
        predictions: Per-residue pocket predictions as list[list[int]].
    """
    model.eval()
    protein_tokens = protein_tokens.to(device)
    attention_mask = attention_mask.to(device)

    # Obtain logits with chunked forward propagation
    with torch.no_grad():
        logits = chunked_forward(model, protein_tokens, attention_mask, chunk_size=chunk_size)

    # Select the predicted class for every residue
    preds = torch.argmax(logits, dim=-1)  # Shape: [batch_size, residue_size]
    predictions = preds.cpu().numpy().tolist()

    return predictions


# Initialize the model with the training architecture
model = ProteinBERTClassifier(num_labels=2)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load trained weights
model_path = "/media/yugengliu/Computer21/zzy/pocket-prediction/model/BERT_3/final_protein_bert_classifier.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)

# Test-data and output paths
test_dir = "/media/yugengliu/DATA2/whli/P450_model/application/tf_token"
output_dir = "/media/yugengliu/DATA2/whli/P450_model/application2/pred"
os.makedirs(output_dir, exist_ok=True)

# Iterate over test data and predict
chunk_size = 512
test_files = sorted(os.listdir(test_dir))  # Preserve deterministic order
for test_file in test_files:
    protein_path = os.path.join(test_dir, test_file)

    # Load protein token data
    protein_tokens = np.load(protein_path)  # shape: [residue_size, feature_dim]
    _, residue_size, feature_dim = protein_tokens.shape

    # Convert to a PyTorch tensor
    protein_tokens = torch.tensor(protein_tokens, dtype=torch.float32) # shape: [1, residue_size, feature_dim]

    # Build the attention mask
    attention_mask = torch.ones((1, residue_size), dtype=torch.long)  # shape: [1, residue_size]

    # Predict
    predictions = predict_protein_pockets(
        model=model,
        protein_tokens=protein_tokens,
        attention_mask=attention_mask,
        device=device,
        chunk_size=chunk_size
    )

    # Save predictions
    #output_path = os.path.join(output_dir, f"{os.path.splitext(test_file)[0]}_preds.npy")
    output_path = os.path.join(output_dir, "{}_preds.npy".format(os.path.splitext(test_file)[0]))
    np.save(output_path, predictions[0])  # Save as a NumPy array

    # Print progress
    #print(f"Processed {test_file} predictions saved to {output_path}")
    print("Processed {}, predictions saved to {}".format(test_file, output_path))
