import torch
import transformers
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import json
import numpy as np
from collections import Counter
import argparse


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


# Load vocabulary from JSON (assumes vocab was saved)
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Load tokenizer for BERT
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define Class Labels
LABEL_MAPPING = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}

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


# Load RNN Model
def load_rnn_model(model_path, vocab_size):
    embedding_dim = 100
    pretrained_embeddings = load_glove_embeddings("glove.6B/glove.6B.100d.txt", embedding_dim, vocab)
    model = RNNWithAttention(len(vocab), 100, 256, 3, 3, True, 0.6, pretrained_embeddings)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Convert text to sequence for RNN
def preprocess_text(text, vocab, max_length=50):
    tokens = text.lower().split()
    seq = [vocab.get(word, 0) for word in tokens]  # Convert words to indices
    seq += [0] * (max_length - len(seq))  # Pad sequence
    return torch.tensor([seq], dtype=torch.long)

def predict_with_rnn(text, model, vocab):
    seq = preprocess_text(text, vocab)
    with torch.no_grad():
        logits, _ = model(seq)  # ✅ Extract only the first output (logits)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # Now works correctly

    print(f"Prediction Probabilities: {probs}")  # DEBUGGING OUTPUT

    # **Set custom thresholds for classifying Hate Speech & Offensive Speech**
    if probs[2] > 0.30:  # ✅ If hate speech probability > 30%, classify as Hate Speech
        return "Hate Speech", probs
    elif probs[1] > 0.20:  # ✅ If offensive speech probability > 25%, classify as Offensive
        return "Offensive", probs
    else:
        return "Normal", probs  # ✅ Default to Normal if both are low

# # Predict using RNN
# def predict_with_rnn(text, model, vocab):
#     seq = preprocess_text(text, vocab)
#     with torch.no_grad():
#         logits, _ = model(seq)  # ✅ Extract only the first output (logits)
#         probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # Now works correctly
#     predicted_class = np.argmax(probs)
#     return LABEL_MAPPING[predicted_class], probs



# Load BERT Model
def load_bert_model(model_path):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Set to evaluation mode before inference
    model.eval()
    return model

# Predict using BERT
def predict_with_bert(text, model):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    # predicted_class = np.argmax(probs)
    # **Set custom thresholds for classifying Hate Speech & Offensive Speech**
    if probs[2] > 0.20:  # ✅ If hate speech probability > 30%, classify as Hate Speech
        return "Hate Speech", probs
    elif probs[1] > 0.30:  # ✅ If offensive speech probability > 25%, classify as Offensive
        return "Offensive", probs
    else:
        return "Normal", probs  # ✅ Default to Normal if both are low
    # return LABEL_MAPPING[predicted_class], probs

# Main function
def main():
    parser = argparse.ArgumentParser(description="Real-time Hate Speech Detection")
    parser.add_argument("--model", type=str, choices=["rnn", "bert"], required=True, help="Choose model: rnn or bert")
    args = parser.parse_args()

    if args.model == "rnn":
        print("Loading RNN Model...")
        model = load_rnn_model("HateSpeechRNN.pth", vocab_size=len(vocab))
        use_bert = False
    else:
        print("Loading BERT Model...")
        model = load_bert_model("HateSpeechBERT.pth")
        use_bert = True

    print("Model loaded. Type a message to classify (or type 'exit' to quit).")

    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break

        if use_bert:
            label, probs = predict_with_bert(text, model)
        else:
            label, probs = predict_with_rnn(text, model, vocab)

        print(f"Prediction: {label} (Confidence: {probs})")

if __name__ == "__main__":
    main()
