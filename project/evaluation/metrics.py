from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_all_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds)
    }

