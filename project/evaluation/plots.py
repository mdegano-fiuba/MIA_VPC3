import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion(labels, preds, path="confusion.png"):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.savefig(path)

def plot_roc(labels, probs, path="roc.png"):
    fpr, tpr, _ = roc_curve(labels, probs[:,1])
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC {auc_val:.2f}")
    plt.legend()
    plt.savefig(path)

