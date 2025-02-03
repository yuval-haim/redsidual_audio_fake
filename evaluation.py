from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch

import os

def calculate_eer(fpr, tpr):
    """Calculate the Equal Error Rate (EER)."""
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_index = np.nanargmin(abs_diffs)
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer


def produce_evaluation_file(dataset, model, device, save_path, metrics_output_path, graph_output_path):
    """
    Evaluate the model on the dataset and produce evaluation results with metrics and visualizations.
    
    Args:
        dataset (Dataset): The dataset to evaluate.
        model (nn.Module): The trained model.
        device (str): The device to run evaluation on ('cuda' or 'cpu').
        save_path (str): Path to save the evaluation results.
        metrics_output_path (str): Path to save calculated metrics (JSON file).
        graph_output_path (str): Directory to save generated graphs.
    """
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Storage for evaluation results
    fname_list = []
    speaker_list = []
    label_list = []
    score_list = []

    with torch.no_grad():
        for batch_x, batch_y, batch_meta in data_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size

            # Move data to the appropriate device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Get model predictions
            batch_out = model(batch_x, batch_y, is_test=True)  # Assuming model has `is_test` mode
            batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()  # Scores for 'bona-fide'
            # Count correct predictions
            _, batch_preds = batch_out.max(dim=1)

            # Extract metadata
            fname_list.extend(batch_meta['file_name'])  # Access the 'file_name' list directly
            label_list.extend(batch_y.cpu().numpy().tolist())
            score_list.extend(batch_scores.tolist())
            num_correct += (batch_preds == batch_y).sum().item()

# Calculate accuracy
    accuracy = (num_correct / num_total) * 100
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

    # Convert labels to binary (0 for 'spoof', 1 for 'bona-fide')
    true_labels = np.array(label_list)
    print(f"True Labels: {true_labels}")
    scores = np.array(score_list)
    # if nan values are present, replace them with 0
    scores = np.nan_to_num(scores)
    print(f"Scores: {scores}")

    # Calculate AUC-ROC
    auc_roc = roc_auc_score(true_labels, scores)
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Calculate EER
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    eer = calculate_eer(fpr, tpr)
    print(f"EER: {eer:.4f}")

    # Save results to a file
    with open(save_path, 'w') as fh:
        fh.write('file_name,label,score\n')
        for fname, label, score in zip(fname_list, label_list, score_list):
            fh.write(f'{fname},{label},{score:.4f}\n')

    print(f"Results saved to {save_path}")

    # Save metrics to a JSON file
    metrics = {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'eer': eer
    }
    with open(metrics_output_path, 'w') as fh:
        import json
        json.dump(metrics, fh, indent=4)
    print(f"Metrics saved to {metrics_output_path}")

    # Save graphs
    # 1. ROC Curve
    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_curve_path = f"{graph_output_path}/roc_curve.png"
    plt.savefig(roc_curve_path)
    print(f"ROC Curve saved to {roc_curve_path}")

    # 2. Score Distribution
    plt.figure()
    bona_fide_scores = scores[true_labels == 1]
    spoof_scores = scores[true_labels == 0]
    plt.hist(bona_fide_scores, bins=50, alpha=0.5, label='Bona-fide', color='blue')
    plt.hist(spoof_scores, bins=50, alpha=0.5, label='Spoof', color='red')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    score_distribution_path = f"{graph_output_path}/score_distribution.png"
    plt.savefig(score_distribution_path)
    print(f"Score Distribution saved to {score_distribution_path}")

    # Save TP, FP, TN, FN samples
    predictions = (scores >= 0.5).astype(int)
    tp = [(fname) for fname, pred, true in zip(fname_list, predictions, true_labels) if pred == 1 and true == 1]
    fp = [(fname) for fname, pred, true in zip(fname_list, predictions, true_labels) if pred == 1 and true == 0]
    tn = [(fname) for fname, pred, true in zip(fname_list, predictions, true_labels) if pred == 0 and true == 0]
    fn = [(fname) for fname, pred, true in zip(fname_list, predictions, true_labels) if pred == 0 and true == 1]

    sample_dict = {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    sample_dict_path = f"{graph_output_path}/samples.json"
    with open(sample_dict_path, 'w') as fh:
        import json
        json.dump(sample_dict, fh, indent=4)
    print(f"Samples saved to {sample_dict_path}")
