from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_all_metrics(labels, preds):
    return {'eval_metrics': {
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_f1": f1_score(labels, preds),
        "eval_precision": precision_score(labels, preds),
        "eval_recall": recall_score(labels, preds)
    }}

