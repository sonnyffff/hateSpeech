import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(file_path):
    """
    Loads hate speech dataset from a CSV file.

    Returns:
        texts (List[str]): List of post contents.
        labels (List[int]): List of 0 (non-hate) or 1 (hate).
    """
    df = pd.read_csv(file_path)

    # Remove rows where 'Label' column is not a digit (e.g. 'Label' itself or bad values)
    df = df[df["Label"].apply(lambda x: str(x).strip().isdigit())]

    df = df.dropna(subset=["Content", "Label"])  # Drop NaNs just in case
    df["Label"] = df["Label"].astype(int)        # Now it's safe to convert

    texts = df["Content"].astype(str).tolist()
    labels = df["Label"].tolist()

    return texts, labels


def create_dataloader(texts, labels, tokenizer_name="bert-base-uncased", max_len=128, batch_size=16, shuffle=True):
    """
    Tokenizes text data and returns a PyTorch DataLoader.

    Returns:
        DataLoader: for use in training or evaluation.
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    dataset = HateSpeechDataset(texts, labels, tokenizer, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(train_loader, val_loader, model_name="bert-base-uncased", num_epochs=3, lr=2e-5, patience=2, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Compute class weights: inverse frequency
    total = 361594 + 79305
    weight_for_0 = total / (2 * 361594)
    weight_for_1 = total / (2 * 79305)



    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)
    # class_weights = torch.tensor([1.0, 2.0]).to(device)

    loss_fn = CrossEntropyLoss(weight=class_weights)

    print(f"📦 Training on {device} for up to {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"\n🔁 Epoch {epoch + 1}")
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            loss = loss_fn(logits, labels)  # Instead of outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"✅ Train Loss: {avg_train_loss:.4f}")

        # 🔍 Validation step
        val_loss, val_f1 = evaluate_model(model, val_loader, device=device, return_loss=True)
        print(f"📉 Val Loss: {val_loss:.4f} | F1: {val_f1:.4f}")

        # 🛑 Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print("💾 New best model found!")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("⛔ Early stopping triggered.")
            break

    return model



def evaluate_model(model, dataloader, device=None, return_loss=False):
    from torch.nn import CrossEntropyLoss
    from sklearn.metrics import classification_report, f1_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    total_loss = 0
    loss_fn = CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print("📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Non-Hate", "Hate"]))

    if return_loss:
        return avg_loss, f1



def save_model(model, save_path="saved_model", tokenizer_name="bert-base-uncased"):
    import os

    os.makedirs(save_path, exist_ok=True)
    print(f"📁 Saving model to: {save_path}")

    model.save_pretrained(save_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    # Load and split the data
    texts, labels = load_data("HateBinaryDataset/HateSpeechDataset.csv")

    texts = texts[:100000]
    labels = labels[:100000]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenize and create dataloaders
    train_loader = create_dataloader(train_texts, train_labels, batch_size=16, shuffle=True)
    val_loader = create_dataloader(val_texts, val_labels, batch_size=16, shuffle=False)

    # Train the model on GPU
    model = train_model(train_loader, val_loader, device=device)

    # Evaluate on validation set using GPU
    evaluate_model(model, val_loader, device=device)

    # Save the trained model and tokenizer
    save_model(model, save_path="bert_hate_model")