import torch
from transformers import AutoTokenizer, AutoConfig

from BERT_binary_hug import WeightedBertForSequenceClassification
from Hatebert_binary_hug import WeightedHateBERT
from RNN_LSTM import RNNHateSpeechModel, clean_text, tokenize, tokens_to_indices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dummy CNN/RNN Predictors ===
def dummy_cnn_predict(text):
    return 1, 0.65  # (prediction, confidence)

def dummy_rnn_predict(text):
    return 0, 0.60

def preprocess_for_rnn(text):
    tokens = tokenize(clean_text(text))
    indices = tokens_to_indices(tokens, vocab_to_idx, max_len)
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    return tensor


# === Model Selection ===
MODEL_TYPE = input("🤖 Choose model [bert / hatebert / cnn / rnn / ensemble]: ").strip().lower()


# === Tokenizers and Models ===
tokenizer = None
bert_model = hatebert_model = None

if MODEL_TYPE in ["bert", "ensemble"]:
    tokenizer = AutoTokenizer.from_pretrained("bert-hate-detector")
    bert_model = WeightedBertForSequenceClassification(
        config=AutoConfig.from_pretrained("bert-base-uncased", num_labels=2),
        class_weights=torch.tensor([1.0, 1.0])
    )
    bert_state = torch.load("models/bert-86.pt", map_location=device, weights_only=True)
    bert_model.load_state_dict(bert_state)
    bert_model.to(device).eval()

if MODEL_TYPE in ["hatebert", "ensemble"]:
    tokenizer = AutoTokenizer.from_pretrained("hatebert-hate-detector")
    hatebert_model = WeightedHateBERT(
        config=AutoConfig.from_pretrained("GroNLP/hateBERT", num_labels=2),
        class_weights=torch.tensor([1.0, 1.0])
    )
    hatebert_state = torch.load("models/hatebert-88.pt", map_location=device, weights_only=True)
    hatebert_model.load_state_dict(hatebert_state)
    hatebert_model.to(device).eval()

if MODEL_TYPE in ["rnn", "ensemble"]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # fallback tokenizer
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

if MODEL_TYPE not in ["bert", "hatebert", "lstm", "rnn", "ensemble"]:
    raise ValueError("❌ Invalid model type. Please choose from [bert, hatebert, lstm, rnn, ensemble]")

print(f"\n🚀 Real-Time Hate Speech Detector using [{MODEL_TYPE.upper()}] (type 'exit' to quit)")

# === Real-Time Prediction Loop ===
while True:
    text = input("📝 Enter text: ")
    if text.strip().lower() == "exit":
        break

    if MODEL_TYPE == "bert":
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            probs = torch.softmax(bert_model(**inputs).logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    elif MODEL_TYPE == "hatebert":
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            probs = torch.softmax(hatebert_model(**inputs).logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    elif MODEL_TYPE == "cnn":
        pred, conf = dummy_cnn_predict(text)

    elif MODEL_TYPE == "rnn":
        input_tensor = preprocess_for_rnn(text)
        with torch.no_grad():
            output = rnn_model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob >= 0.5)
        conf = prob if pred == 1 else 1 - prob


    elif MODEL_TYPE == "ensemble":
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        with torch.no_grad():
            probs_bert = torch.softmax(bert_model(**inputs).logits, dim=1)
            probs_hatebert = torch.softmax(hatebert_model(**inputs).logits, dim=1)

        pred_bert = torch.argmax(probs_bert, dim=1).item()
        conf_bert = probs_bert[0][pred_bert].item()

        pred_hatebert = torch.argmax(probs_hatebert, dim=1).item()
        conf_hatebert = probs_hatebert[0][pred_hatebert].item()

        pred_cnn, conf_cnn = dummy_cnn_predict(text)

        input_tensor = preprocess_for_rnn(text)
        with torch.no_grad():
            output = rnn_model(input_tensor)
        prob_rnn = torch.sigmoid(output).item()
        pred_rnn = int(prob_rnn >= 0.5)
        conf_rnn = prob_rnn if pred_rnn == 1 else 1 - prob_rnn

        # Ensemble voting with equal weights (can be changed later)
        results = {
            "BERT": (pred_bert, conf_bert),
            "HateBERT": (pred_hatebert, conf_hatebert),
            "CNN": (pred_cnn, conf_cnn),
            "RNN": (pred_rnn, conf_rnn),
        }

        print("\n🧠 Individual Model Decisions:")
        for name, (p, c) in results.items():
            label = "🔴 Hateful" if p == 1 else "🟢 Non-Hateful"
            print(f"  - {name}: {label} ({c:.2f} confidence)")

        # Aggregate final decision
        final_score = 0.0
        for _, (pred_i, conf_i) in results.items():
            vote = 1 if pred_i == 1 else -1
            final_score += vote * conf_i

        pred = 1 if final_score > 0 else 0
        conf = abs(final_score) / 4

    label_str = "🔴 Hateful" if pred == 1 else "🟢 Non-Hateful"
    print(f"{label_str} ({conf:.2f} confidence)\n")
