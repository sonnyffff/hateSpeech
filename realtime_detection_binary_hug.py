import torch
from transformers import BertTokenizer, BertConfig
from BERT_binary_hug import WeightedBertForSequenceClassification  # assuming same class as in your training script

# 🔧 Load model
model_path = "bert-hate-detector.pt"
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-hate-detector")

# Dummy weights for inference (you can disable weighted loss if desired)
dummy_weights = torch.tensor([1.0, 1.0])

model = WeightedBertForSequenceClassification(config, class_weights=dummy_weights)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔁 Real-time prediction loop
print("🚀 Real-Time Hate Speech Detector (type 'exit' to quit)")
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
