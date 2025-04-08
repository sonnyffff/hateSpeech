# ===================== IMPORTS =====================
import pandas as pd
import re
import numpy as np
import nltk
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# ===================== FUNCTIONS =====================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return nltk.word_tokenize(text)

def tokens_to_indices(tokens, vocab_to_idx, max_len):
    indices = [vocab_to_idx.get(token, 0) for token in tokens]
    indices = indices[:max_len]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

class HateSpeechDataset(Dataset):
    def __init__(self, df):
        self.X = df['indices'].tolist()
        self.y = df['Label'].tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.X[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.float)
        return text_tensor, label_tensor

class RNNHateSpeechModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, drop_prob=0.0):
        super(RNNHateSpeechModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=drop_prob, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"\n Epoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1} complete | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs.squeeze()) >= 0.55
            correct += (preds == labels.byte()).sum().item()
            total += labels.size(0)
    model.train()
    return total_loss / len(data_loader), correct / total

# ===================== MAIN =====================

if __name__ == '__main__':
    nltk.download('punkt')

    df = pd.read_csv('HateBinaryDataset/HateSpeechDatasetBalanced.csv')
    df['Content'] = df['Content'].apply(clean_text)
    # df = df.sample(n=500, random_state=42).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df['tokens'] = train_df['Content'].apply(tokenize)
    val_df['tokens'] = val_df['Content'].apply(tokenize)
    test_df['tokens'] = test_df['Content'].apply(tokenize)

    all_tokens = []
    for tokens in train_df['tokens']:
        all_tokens.extend(tokens)

    token_counts = Counter(all_tokens)
    vocab = sorted(token_counts, key=token_counts.get, reverse=True)
    vocab_to_idx = {word: i+1 for i, word in enumerate(vocab)}

    train_df['token_length'] = train_df['tokens'].apply(len)
    max_len = int(np.percentile(train_df['token_length'], 90))

    train_df['indices'] = train_df['tokens'].apply(lambda x: tokens_to_indices(x, vocab_to_idx, max_len))
    val_df['indices'] = val_df['tokens'].apply(lambda x: tokens_to_indices(x, vocab_to_idx, max_len))
    test_df['indices'] = test_df['tokens'].apply(lambda x: tokens_to_indices(x, vocab_to_idx, max_len))

    train_data = HateSpeechDataset(train_df)
    val_data = HateSpeechDataset(val_df)
    test_data = HateSpeechDataset(test_df)

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    vocab_size = len(vocab_to_idx) + 1
    embed_dim = 200
    hidden_dim = 256
    output_dim = 1
    n_layers = 1

    model = RNNHateSpeechModel(vocab_size, embed_dim, hidden_dim, output_dim, n_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)

    torch.save({
        "vocab_to_idx": vocab_to_idx,
        "max_len": max_len
    }, "models/rnn_preprocessing_meta.pt")

    torch.save(model.state_dict(), "rnn.pt")

    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs >= 0.5).long()
            y_pred_list.extend(preds.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())

    precision = precision_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list)
    cm = confusion_matrix(y_true_list, y_pred_list)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:\n", cm)

    false_positives = [i for i, (y, p) in enumerate(zip(y_true_list, y_pred_list)) if y == 0 and p == 1]
    false_negatives = [i for i, (y, p) in enumerate(zip(y_true_list, y_pred_list)) if y == 1 and p == 0]

    print("Indices of False Positives:", false_positives)
    print("Indices of False Negatives:", false_negatives)
