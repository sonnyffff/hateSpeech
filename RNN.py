import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import nltk

nltk.download('punkt')


# Load GloVe Embeddings
def load_glove_embeddings(filepath, embedding_dim, vocab):
    embeddings_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

    return torch.tensor(embedding_matrix, dtype=torch.float32)


# Preprocess Text
def preprocess_text(tokens):
    return [token.lower() for token in tokens]


# Extract Rationale Weights
def extract_rationale_weights(post):
    tokens = post["post_tokens"]
    rationale_lists = post.get("rationales", [])

    # Initialize weights for all words (default = 1.0)
    weights = [1.0] * len(tokens)

    if rationale_lists:
        for rationale in rationale_lists:
            if len(rationale) > len(tokens):
                rationale = rationale[:len(tokens)]  # Ensure it doesn't exceed the token count
            for i, flag in enumerate(rationale):
                if flag == 1 and i < len(weights):  # Ensure valid index
                    weights[i] = 2.0  # Increase importance of rationale words

    return weights


# Load Dataset with Multi-Class Labels
def load_dataset(filepath):
    label_mapping = {"normal": 0, "offensive": 1, "hatespeech": 2}
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = []
    for post_id, post in raw_data.items():
        tokens = preprocess_text(post["post_tokens"])
        final_label = max(set([label_mapping[ann["label"]] for ann in post["annotators"]]),
                          key=[ann["label"] for ann in post["annotators"]].count)
        rationale_weights = extract_rationale_weights(post)
        data.append((tokens, final_label, rationale_weights))

    return pd.DataFrame(data, columns=["text", "label", "weights"])


# Create Vocabulary
def create_vocab(df):
    all_words = [word for tokens in df["text"] for word in tokens]
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_words).most_common())}
    vocab["<PAD>"] = 0
    return vocab


# Convert Text to Sequence
def text_to_sequence(text, vocab):
    return [vocab[word] for word in text if word in vocab]


# Pad Sequences
def pad_sequence(seq, max_length):
    return seq + [0] * (max_length - len(seq))


# Prepare Data
def prepare_data(df, vocab):
    df["text_seq"] = df["text"].apply(lambda x: text_to_sequence(x, vocab))
    max_length = max(df["text_seq"].apply(len))
    df["text_seq"] = df["text_seq"].apply(lambda x: pad_sequence(x, max_length))
    return df, max_length


# Split Data
def split_data(df):
    return train_test_split(df["text_seq"].tolist(), df["label"].tolist(), test_size=0.2)


# Dataset Class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_output):
        attn_scores = self.attn(rnn_output).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(rnn_output * attn_weights.unsqueeze(2), dim=1)
        return context, attn_weights


# RNN Model with Attention
class RNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout,
                 pretrained_embeddings):
        super(RNNWithAttention, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                           batch_first=True)
        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, _) = self.rnn(embedded)
        context, attn_weights = self.attention(output)
        return self.fc(self.dropout(context)), attn_weights


# Training Function
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        predictions, _ = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# Evaluation Function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions, _ = model(text)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def predict_hate_speech(text, model, vocab, max_length):
    tokens = preprocess_text(text.split())
    seq = text_to_sequence(tokens, vocab)
    seq = pad_sequence(seq, max_length)
    tensor_seq = torch.tensor([seq], dtype=torch.long).to(device)

    with torch.no_grad():
        prediction, _ = model(tensor_seq)
        probs = torch.softmax(prediction, dim=1).cpu().numpy()[0]

    # Set confidence thresholds
    if probs[2] > 0.35:
        return "Hate Speech"
    elif probs[1] > 0.3:
        return "Offensive"
    else:
        return "Normal"



def get_predictions_and_labels(model, iterator):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions, _ = model(text)
            preds = torch.argmax(predictions, dim=1)  # Get the predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Main Script
if __name__ == "__main__":
    df = load_dataset("hatexplain_dataset.json")
    vocab = create_vocab(df)

    # with open("vocab.json", "w") as f:
    #     json.dump(vocab, f)
    #
    # print("Vocabulary saved successfully!")

    df, max_length = prepare_data(df, vocab)

    train_texts, val_texts, train_labels, val_labels = split_data(df)

    ros = RandomOverSampler(sampling_strategy="auto", random_state=42)
    train_texts_resampled, train_labels_resampled = ros.fit_resample(train_texts, train_labels)

    train_dataset = HateSpeechDataset(train_texts_resampled, train_labels_resampled)
    val_dataset = HateSpeechDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 100
    pretrained_embeddings = load_glove_embeddings("glove.6B/glove.6B.100d.txt", embedding_dim, vocab)

    model = RNNWithAttention(len(vocab), 100, 256, 3, 3, True, 0.6, pretrained_embeddings).to(device)

    # Compute class weights based on dataset distribution
    class_counts = Counter(train_labels_resampled)
    total_samples = sum(class_counts.values())
    class_weights = {
        0: total_samples / (class_counts[0] * 1),  # Slightly reduce Normal class weight
        1: total_samples / (class_counts[1] * 2.5),  # Boost weight for Offensive class
        2: total_samples / (class_counts[2] * 2),  # Keep Hate Speech weight
    }
    weights_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(3)], dtype=torch.float32
    ).to(device)

    # Use weighted loss function
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(5):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.3f}, Valid Loss={val_loss:.3f}")

    # Save model after training
    torch.save(model.state_dict(), "HateSpeechRNN.pth")
    print("Model saved successfully!")

    model.eval()

    # Get true labels and predicted labels
    true_labels, predicted_labels = get_predictions_and_labels(model, val_loader)

    # Generate classification report
    report = classification_report(true_labels, predicted_labels, target_names=["Normal", "Offensive", "Hate Speech"],
                                   digits=4)

    # Print report
    print("📊 **Classification Report:**")
    print(report)

    # Example test cases
    test_cases = [
        "I hate all people of XYZ group!",  # Expected: Hate Speech
        "Go back to your country!",  # Expected: Hate Speech
        "You are a disgusting pig!",  # Expected: Offensive
        "Everyone deserves equal rights.",  # Expected: Normal
        "I love diversity and respect all cultures.",  # Expected: Normal
    ]

    # Run predictions on test cases
    predictions = {text: predict_hate_speech(text, model, vocab, max_length) for text in test_cases}

    # Print results
    for text, prediction in predictions.items():
        print(f"Text: {text} -> Prediction: {prediction}")
