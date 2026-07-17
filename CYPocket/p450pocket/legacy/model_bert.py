from transformers import BertModel
import torch
import torch.nn as nn


class ProteinBERTClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.5):
        super(ProteinBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('../model/bert-base-uncased')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.feature_projection = nn.Linear(100, self.bert.config.hidden_size)  # Map 100D to 768D
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        embedded_features = self.feature_projection(input_ids)  # [batch_size, residue_size, 768]
        outputs = self.bert(inputs_embeds=embedded_features, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, residue_size, 768]
        logits = self.classifier(sequence_output)  # [batch_size, residue_size, num_labels]

        return logits

def chunked_forward(model, tokens, attention_mask, chunk_size=512):
    batch_size, residue_size, feature_dim = tokens.size()
    all_logits = []

    # Split the input sequence into chunks for forward propagation
    for i in range(0, residue_size, chunk_size):
        chunk_tokens = tokens[:, i:i + chunk_size, :]  # [batch_size, chunk_size, feature_dim]
        chunk_attention_mask = attention_mask[:, i:i + chunk_size]  # [batch_size, chunk_size]
        # Process the chunk and collect its output
        logits = model(chunk_tokens, attention_mask=chunk_attention_mask)

        all_logits.append(logits)

    # Concatenate chunk outputs along the residue dimension
    combined_logits = torch.cat(all_logits, dim=1)  # [batch_size, residue_size, num_labels]
    return combined_logits
