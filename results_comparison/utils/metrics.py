from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

def get_metrics_results(y_test_flat, y_pred_flat):
    precision = precision_score(y_test_flat, y_pred_flat)
    recall = recall_score(y_test_flat, y_pred_flat)
    f1 = f1_score(y_test_flat, y_pred_flat)

    intersection = np.logical_and(y_test_flat, y_pred_flat)
    union = np.logical_or(y_test_flat, y_pred_flat)
    iou = np.sum(intersection) / np.sum(union)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"IoU: {iou}")
