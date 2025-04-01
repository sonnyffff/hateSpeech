import torch
from transformers import AutoTokenizer, AutoConfig
from BERT_binary_hug import WeightedBertForSequenceClassification
from Hatebert_binary_hug import WeightedHateBERT

# === Model Selection ===
MODEL_TYPE = input("🤖 Choose model [bert / hatebert]: ").strip().lower()

if MODEL_TYPE == "hatebert":
    model_name = "GroNLP/hateBERT"
    model_path = "models/hatebert-88.pt"
    tokenizer = AutoTokenizer.from_pretrained("hatebert-hate-detector")
    dummy_weights = torch.tensor([1.0, 1.0])
    model = WeightedHateBERT(model_name=model_name, class_weights=dummy_weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
elif MODEL_TYPE == "bert":
    model_name = "bert-base-uncased"
    model_path = "models/bert-86.pt"
    tokenizer = AutoTokenizer.from_pretrained("bert-hate-detector")
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    dummy_weights = torch.tensor([1.0, 1.0])
    model = WeightedBertForSequenceClassification(config=config, class_weights=dummy_weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
else:
    raise ValueError("❌ Invalid model type. Please choose 'bert' or 'hatebert'.")

# === Inference Setup ===
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"\n🚀 Real-Time Hate Speech Detector using [{MODEL_TYPE.upper()}] (type 'exit' to quit)")

# === Real-Time Prediction Loop ===
while True:
    text = input("📝 Enter text: ")
    if text.strip().lower() == "exit":
        break

    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label_str = "🔴 Hateful" if pred == 1 else "🟢 Non-Hateful"
    print(f"{label_str} ({confidence:.2f} confidence)\n")
