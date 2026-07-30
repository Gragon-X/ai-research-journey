import numpy as np
def metrics(tp, fp, fn, tn):#ture&false+position&negative
    sum = tp + fp + fn + tn
    Accuracy = (tp + tn) / sum
    Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    Recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
    return {
        "准确率": Accuracy,
        "精确率": Precision,
        "召回率": Recall,
        "F1": F1
    }
print(metrics(10,20,30,40))