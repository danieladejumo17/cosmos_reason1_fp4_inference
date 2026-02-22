import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Metrics:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.total_time = 0
        self.count = 0

    def update(self, predictions, targets, inference_times) -> None:
        for p, t, inf_time in zip(predictions, targets, inference_times):
            if p == 1 and t == 1:
                self.tp += 1
            elif p == 0 and t == 0:
                self.tn += 1
            elif p == 1 and t == 0:
                self.fp += 1
            elif p == 0 and t == 1:
                self.fn += 1

            self.total_time += inf_time
            self.count += 1

    def compute(self) -> dict:
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (self.tp + self.tn + self.fp + self.fn) > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_inference_time = self.total_time / self.count if self.count > 0 else 0
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "Avg Inference Time": avg_inference_time
        }
    
    def plot_confusion_matrix(self) -> None:
        cm = confusion_matrix(self.targets, self.predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
