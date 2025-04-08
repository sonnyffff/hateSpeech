import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertConfig,
    Trainer,
    TrainingArguments,
    BertForSequenceClassification,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# ===== Subclassed model with weighted loss =====
class WeightedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")  # Remove unexpected keyword argument

        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ===== Load and preprocess dataset =====
def load_dataset(path):
    df = pd.read_csv(path)
    df = df[["Content", "Label"]].dropna()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)
    return df


def split_dataset(df):
    return train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)


# ===== Tokenize and convert to HF Dataset =====
def tokenize_dataset(train_df, val_df, tokenizer_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    def tokenize_fn(example):
        encoding = tokenizer(example["Content"], truncation=True, padding="max_length", max_length=128)
        encoding["labels"] = example["Label"]  # Add labels for Trainer
        return encoding

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    columns_to_remove = [col for col in train_dataset.column_names if
                         col not in ["input_ids", "attention_mask", "labels"]]
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    return train_dataset, val_dataset, tokenizer


# ===== Metrics function =====
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ===== Build model with class weights =====
def get_weighted_model(train_df):
    class_counts = train_df["Label"].value_counts().to_dict()
    total = sum(class_counts.values())
    weight_0 = total / (2 * class_counts.get(0, 1))
    weight_1 = total / (2 * class_counts.get(1, 1))
    class_weights = torch.tensor([weight_0, weight_1])

    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
    model = WeightedBertForSequenceClassification(config, class_weights)
    pretrained_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


# ===== Train and Save Model =====
def train_model(train_dataset, val_dataset, tokenizer, model):
    args = TrainingArguments(

        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    print("\n🔍 Final Evaluation on Validation Set:")
    print(trainer.evaluate())

    # Load best model for reporting
    preds_output = trainer.predict(val_dataset)
    y_true = preds_output.label_ids
    y_pred = preds_output.predictions.argmax(-1)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Hate", "Hate"]))


    trainer.save_model("bert-hate-detector")
    tokenizer.save_pretrained("bert-hate-detector")
    torch.save(model.state_dict(), "bert-hate-detector.pt")
    print("✅ Model saved to 'bert-hate-detector.pt'")


# ===== Main =====
def main():
    df = load_dataset("HateBinaryDataset/HateSpeechDataset.csv")
    # df = df.sample(n=10000, random_state=42)  # test on subset
    train_df, val_df = split_dataset(df)
    train_dataset, val_dataset, tokenizer = tokenize_dataset(train_df, val_df)
    model = get_weighted_model(train_df)
    train_model(train_dataset, val_dataset, tokenizer, model)


if __name__ == "__main__":
    main()
