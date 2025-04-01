import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model & tokenizer
model_path = "bert_hate_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text, threshold=0.6):  # 👈 set your desired threshold here
    model.eval()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        logits = model(**tokens).logits
        probs = torch.softmax(logits, dim=1)

        hate_prob = probs[0][1].item()
        non_hate_prob = probs[0][0].item()

        if hate_prob > threshold:
            return "🔴 Hateful", hate_prob
        else:
            return "🟢 Non-Hateful", non_hate_prob


# Real-time loop
print("🚀 Real-Time Hate Speech Detector (type 'exit' to quit)")
while True:
    user_input = input("📝 Enter text: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    label, confidence = predict(user_input)
    print(f"{label} ({confidence:.2f} confidence)\n")
