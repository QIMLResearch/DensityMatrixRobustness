import matplotlib.pyplot as plt
import os

def plot_training_metrics(dataset, encoding, mode, constrain, model_timestamp, training_results):
    """
    Plot training and validation metrics vs. epoch, including loss, accuracy, precision, recall, and F1-score.

    Args:
        dataset (str): Dataset name.
        encoding (str): Encoding type used.
        mode (str): The training mode used (e.g., 'baseline', 'pgd').
        model_timestamp (str): Timestamp for saving results.
        training_results (dict): Dictionary containing training and validation metrics.
    """
    
    if encoding in ['DM', 'Stats']:
        results_save_path = f'results/training/{dataset}/{encoding}/{mode}_{constrain}/'
    else:
        results_save_path = f'results/training/{dataset}/{encoding}/{mode}/'
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    # Extract metrics from training results
    train_losses = training_results["train_losses"]
    val_losses = training_results["val_losses"]

    train_accuracies = training_results["train_accuracies"]
    val_accuracies = training_results["val_accuracies"]

    train_precisions = training_results.get("train_precisions", [])
    val_precisions = training_results.get("val_precisions", [])

    train_recalls = training_results.get("train_recalls", [])
    val_recalls = training_results.get("val_recalls", [])

    train_f1s = training_results.get("train_f1s", [])
    val_f1s = training_results.get("val_f1s", [])

    # Dynamically calculate the number of epochs
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

    # Define a helper function to reduce redundancy
    def plot_metric(metric_name, train_metric, val_metric, ylabel):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metric, marker='o', linestyle='-', color='b', label=f'Training {metric_name}')
        plt.plot(epochs, val_metric, marker='o', linestyle='-', color='r', label=f'Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'Training and Validation {metric_name} vs. Epoch ({mode.upper()} mode)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{results_save_path}{metric_name.lower()}_{model_timestamp}.png')
        plt.close()

    # Plot each metric
    plot_metric("Loss", train_losses, val_losses, "Loss")
    plot_metric("Accuracy", train_accuracies, val_accuracies, "Accuracy")
    if train_precisions and val_precisions:
        plot_metric("Precision", train_precisions, val_precisions, "Precision")
    if train_recalls and val_recalls:
        plot_metric("Recall", train_recalls, val_recalls, "Recall")
    if train_f1s and val_f1s:
        plot_metric("F1 Score", train_f1s, val_f1s, "F1 Score")
