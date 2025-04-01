import json
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from torch.utils.data import Dataset, DataLoader
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight



# Check for GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_hatexplain_dataset(file_path):
    """ Load HateXplain dataset from JSON file. """
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    data = []
    for post_id, post_data in dataset.items():
        label_counts = {}

        # Count votes for each label
        for annotation in post_data["annotators"]:
            label = annotation["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        # Determine majority label
        majority_label = max(label_counts, key=label_counts.get)

        # Extract rationales safely
        rationale_mask = post_data.get("rationales", [])
        if isinstance(rationale_mask, list) and len(rationale_mask) > 0:
            if all(isinstance(r, list) for r in rationale_mask):
                max_length = max(len(r) for r in rationale_mask)
                padded_rationales = [r + [0] * (max_length - len(r)) for r in rationale_mask]
                rationale_mask = np.array(padded_rationales, dtype=int)
                rationale_mask = np.max(rationale_mask, axis=0).tolist()
            else:
                rationale_mask = []
        else:
            rationale_mask = []

        # Store results
        data.append({
            "post_id": post_id,
            "majority_label": majority_label,
            "post_tokens": " ".join(post_data["post_tokens"]),
            "rationales": rationale_mask
        })

    return pd.DataFrame(data)

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # Store class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)  # Apply class weighting

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ✅ Accept additional arguments
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)  # Forward pass
        logits = outputs.logits  # Extract logits

        loss = self.criterion(logits, labels)  # Compute weighted loss

        return (loss, outputs) if return_outputs else loss


class HateSpeechDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_map, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["post_tokens"]
        label = self.label_map[self.data.iloc[idx]["majority_label"]]
        tokenized = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length,
                                   return_tensors="pt")

        rationale_mask = self.data.iloc[idx]["rationales"]
        if not rationale_mask:
            rationale_mask = [1] * self.max_length
        else:
            rationale_mask = rationale_mask[:self.max_length] + [1] * (self.max_length - len(rationale_mask))

        tokenized["labels"] = torch.tensor(label, dtype=torch.long)
        tokenized["attention_mask"] = torch.tensor(rationale_mask, dtype=torch.long)

        return {key: val.squeeze() for key, val in tokenized.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_model(train_dataset, eval_dataset, weight_tensor):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)



    # **Use Weighted Loss Function**
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,  # ✅ Train for more epochs (5 instead of 2)
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=2,
    )

    trainer = CustomTrainer(

        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(torch.optim.AdamW(model.parameters(), lr=5e-5), None),
        class_weights = weight_tensor
    )

    trainer.train()
    torch.save(model.state_dict(), "HateSpeechBERT.pth")
    return model, trainer


def evaluate_model(trainer, dataset):
    results = trainer.evaluate(eval_dataset=dataset)
    print("Evaluation Results:", results)
    return results


def load_and_test_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("HateSpeechBERT").to(device)
    model.eval()

    LABEL_MAPPING = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}

    test_examples = [
        "I hope you have a great day!",
        "That was an amazing performance!",
        "You're so stupid, I can't believe it.",
        "This community is filled with disgusting people.",
        "You Ching chong not belong here."
    ]

    for text in test_examples:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        # **Adjust Thresholds**
        if probs[2] > 0.30:  # Lower threshold for detecting hate speech
            predicted_class = 2
        elif probs[1] > 0.20:  # Lower threshold for offensive speech
            predicted_class = 1
        else:
            predicted_class = 0

        print(f"Text: {text}")
        print(f"Prediction: {LABEL_MAPPING[predicted_class]} (Confidence: {probs[predicted_class] * 100:.2f}%)\n")


def main():
    file_path = "hatexplain_dataset.json"
    df_result = load_hatexplain_dataset(file_path)
    print(df_result.head())

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}

    # Split dataset into training and evaluation sets
    train_df, eval_df = train_test_split(df_result, test_size=0.2, random_state=42)

    train_dataset = HateSpeechDataset(train_df, tokenizer, label_map)
    eval_dataset = HateSpeechDataset(eval_df, tokenizer, label_map)

    # **Compute Class Weights**
    # Convert classes list to a NumPy array
    classes = np.array([0, 1, 2])  # ✅ Fix: Convert to NumPy array

    # Compute class weights
    train_labels = np.array([label_map[label] for label in train_df["majority_label"]])  # Convert labels to NumPy array
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)

    # Convert class weights to a tensor
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model, trainer = train_model(train_dataset, eval_dataset, weights_tensor)

    evaluation_results = evaluate_model(trainer, eval_dataset)
    print("Final Evaluation Performance:", evaluation_results)


if __name__ == "__main__":
    main()
    load_and_test_model()