"""
Member 3 (Evaluation) pipeline.

Loads the trained model, evaluates on the held-out test set, and produces
the confusion matrix, per-class sensitivity/specificity, macro-F1,
ROC/AUC curves, and training curves requested in the proposal's evaluation
approach.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, f1_score
)
from sklearn.preprocessing import label_binarize

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
FIG_DIR = "figures"
CLASSES = ['N', 'S', 'V', 'F', 'Q']
NUM_CLASSES = len(CLASSES)


def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    return X[..., np.newaxis], y


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    X_test, y_test = load_split("test")
    model = keras.models.load_model(os.path.join(RESULTS_DIR, "ecg_1dcnn_model.keras"))

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(NUM_CLASSES))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Per-class sensitivity (recall) & specificity
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        per_class[cls] = {"sensitivity": sensitivity, "specificity": specificity, "support": int(cm[i, :].sum())}

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)

    # ROC / AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=range(NUM_CLASSES))
    plt.figure(figsize=(7, 6))
    auc_scores = {}
    for i, cls in enumerate(CLASSES):
        if y_test_bin[:, i].sum() == 0:
            auc_scores[cls] = None
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - One-vs-Rest (Test Set)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # Training curves
    with open(os.path.join(RESULTS_DIR, "history.json")) as f:
        history = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history['loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[1].plot(history['accuracy'], label='train')
    axes[1].plot(history['val_accuracy'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
    plt.close()

    overall_accuracy = (y_pred == y_test).mean()

    results = {
        "overall_accuracy": float(overall_accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "auc_scores": auc_scores,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "test_set_size": int(len(y_test)),
    }

    with open(os.path.join(RESULTS_DIR, "final_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
