import optuna
from evaluate import (
    load_dataset,
    split_dataset,
    predict_bert,
    predict_hatebert,
    predict_cnn,
    predict_rnn,
    aggregate_predictions
)
from sklearn.metrics import accuracy_score, classification_report
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and split dataset ===
df = load_dataset("HateBinaryDataset/HateSpeechDataset.csv")
_, val_df = split_dataset(df)

def objective(trial):
    weights = {
        "bert": trial.suggest_float("bert", 0.1, 1.0),
        "hatebert": trial.suggest_float("hatebert", 0.1, 1.0),
        "cnn": trial.suggest_float("cnn", 0.1, 1.0),
        "rnn": trial.suggest_float("rnn", 0.1, 1.0)
    }

    y_true, y_pred = [], []

    for _, row in val_df.iterrows():
        text = row["Content"]
        results = {
            "bert": predict_bert(text),
            "hatebert": predict_hatebert(text),
            "cnn": predict_cnn(text),
            "rnn": predict_rnn(text)
        }
        final_pred = aggregate_predictions(results, weights)
        y_true.append(row["Label"])
        y_pred.append(final_pred)

    acc = accuracy_score(y_true, y_pred)
    trial.set_user_attr("weights", weights)
    return acc


def logging_callback(study, trial):
    print(f"✅ Trial {trial.number} finished with Accuracy = {trial.value:.4f}")
    print(f"   → Weights: {trial.user_attrs['weights']}")
    print(f"🏅 Best so far: Trial {study.best_trial.number} with Accuracy = {study.best_value:.4f}\n")


if __name__ == "__main__":
    print("🔍 Starting Optuna Fine-Tuning for Accuracy...\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True, callbacks=[logging_callback])

    print("\n🎯 Best Accuracy: {:.4f}".format(study.best_value))
    print("🏆 Best Weights:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")

    # === Evaluate final best weights ===
    best_weights = study.best_params
    y_true, y_pred = [], []

    for _, row in val_df.iterrows():
        text = row["Content"]
        results = {
            "bert": predict_bert(text),
            "hatebert": predict_hatebert(text),
            "cnn": predict_cnn(text),
            "rnn": predict_rnn(text)
        }
        final_pred = aggregate_predictions(results, best_weights)
        y_true.append(row["Label"])
        y_pred.append(final_pred)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
