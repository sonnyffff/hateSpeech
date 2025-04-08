import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

from BERT_binary_hug import WeightedBertForSequenceClassification
from Hatebert_binary_hug import WeightedHateBERT
from RNN_LSTM import RNNHateSpeechModel, clean_text, tokenize, tokens_to_indices



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Load and preprocess dataset =====
def load_dataset(path):
    df = pd.read_csv(path)
    df = df[["Content", "Label"]].dropna()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)
    return df

def split_dataset(df):
    return train_test_split(df, test_size=0.001, stratify=df["Label"], random_state=42)

# === Load RNN (LSTM-based) model
meta = torch.load("models/rnn_preprocessing_meta.pt", weights_only=True)
vocab_to_idx = meta["vocab_to_idx"]
max_len = meta["max_len"]

vocab_size = len(vocab_to_idx) + 1
embed_dim = 200
hidden_dim = 256
output_dim = 1
n_layers = 1

rnn_model = RNNHateSpeechModel(vocab_size, embed_dim, hidden_dim, output_dim, n_layers)
rnn_model.load_state_dict(torch.load("models/rnn.pt", map_location=device, weights_only=True))
rnn_model.to(device).eval()


# === BERT Initialization ===
bert_tokenizer = AutoTokenizer.from_pretrained("bert-hate-detector")
bert_model = WeightedBertForSequenceClassification(
    config=AutoConfig.from_pretrained("bert-base-uncased", num_labels=2),
    class_weights=torch.tensor([1.0, 1.0])
)
bert_state_dict = torch.load("models/bert-86.pt", map_location=device, weights_only=True)
bert_model.load_state_dict(bert_state_dict)
bert_model.to(device).eval()

# === HateBERT Initialization ===
hatebert_tokenizer = AutoTokenizer.from_pretrained("hatebert-hate-detector")
hatebert_config = AutoConfig.from_pretrained("GroNLP/hateBERT", num_labels=2)
hatebert_model = WeightedHateBERT(
    config=hatebert_config,
    class_weights=torch.tensor([1.0, 1.0])
)
hatebert_state_dict = torch.load("models/hatebert-88.pt", map_location=device, weights_only=True)
hatebert_model.load_state_dict(hatebert_state_dict)
hatebert_model.to(device).eval()

def preprocess_for_rnn(text):
    tokens = tokenize(clean_text(text))
    indices = tokens_to_indices(tokens, vocab_to_idx, max_len)
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    return tensor

# === Dummy Predictors for LSTM and RNN (replace later) ===
def predict_rnn(text):
    input_tensor = preprocess_for_rnn(text)
    with torch.no_grad():
        output = rnn_model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob >= 0.5)
        return pred, prob

# === Dummy CNN model predictor ===
def predict_cnn(text):
    # Placeholder logic until actual CNN is implemented
    return np.random.randint(0, 2), np.random.uniform(0.5, 0.9)

# === Predict Functions ===
def predict_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return pred, conf

def predict_hatebert(text):
    inputs = hatebert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = hatebert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return pred, conf

# === Aggregation ===
def aggregate_predictions(results, weights):
    score = 0.0
    for model, (pred, conf) in results.items():
        vote = 1 if pred == 1 else -1
        score += vote * conf * weights.get(model, 1.0)
    return 1 if score > 0 else 0


# === Evaluation ===
def evaluate_single(model_name, df):
    model_map = {
        "bert": predict_bert,
        "hatebert": predict_hatebert,
        "cnn": predict_cnn,
        "rnn": predict_rnn
    }

    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' is not recognized.")

    predictor = model_map[model_name]
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        pred, _ = predictor(row["Content"])
        y_true.append(row["Label"])
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"🧪 {model_name.upper()} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
    print(classification_report(y_true, y_pred, target_names=["Non-Hateful", "Hateful"]))


def evaluate_ensemble(df, weights):
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        results = {
            "bert": predict_bert(row["Content"]),
            "hatebert": predict_hatebert(row["Content"]),
            "cnn": predict_cnn(row["Content"]),
            "rnn": predict_rnn(row["Content"])
        }

        final_pred = aggregate_predictions(results, weights)
        y_true.append(row["Label"])
        y_pred.append(final_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"🤖 ENSEMBLE - Accuracy: {acc:.3f}, F1: {f1:.3f}")
    print(classification_report(y_true, y_pred, target_names=["Non-Hateful", "Hateful"]))


# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "ensemble"], required=True, help="Evaluation mode.")
    parser.add_argument("--model", type=str, help="Model name for single mode.")
    parser.add_argument("--weights", nargs=4, type=float,
                        help="Weights for ensemble mode in order: BERT HATEBERT CNN RNN")

    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")

    args = parser.parse_args()

    df = load_dataset(args.data)
    _, val_df = split_dataset(df)

    if args.mode == "single":
        if not args.model:
            raise ValueError("You must specify --model in single mode.")
        evaluate_single(args.model.lower(), val_df)
    else:
        if not args.weights:
            weights = {"bert": 1.0, "hatebert": 1.0, "cnn": 1.0, "rnn": 1.0}
        else:
            weights = dict(zip(["bert", "hatebert", "cnn", "rnn"], args.weights))

        evaluate_ensemble(val_df, weights)

