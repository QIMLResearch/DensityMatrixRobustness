import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from defenses.pgd import pgd_attack
from defenses.mart import mart_loss
from defenses.trades import trades_loss
from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_predictions(outputs, criterion):
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        probs = torch.sigmoid(outputs)
        return (probs > 0.5).long()
    elif isinstance(criterion, nn.CrossEntropyLoss):
        _, preds = torch.max(outputs, 1)
        return preds
    else:
        raise ValueError("Unsupported criterion for metrics calculation.")


def calculate_metrics(outputs, labels, criterion):
    loss = criterion(outputs, labels).item()
    preds = get_predictions(outputs, criterion)
    accuracy = (preds == labels).sum().item() / labels.size(0)
    precision = precision_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    return loss, accuracy, precision, recall, f1


def get_attack_params(method, device, optimizer, **kwargs):
    if method == 'pgd':
        return {
            "device": device, 
            "eps": kwargs.get("eps", 0.3), 
            "alpha": kwargs.get("alpha", 0.01), 
            "iters": kwargs.get("iters", 40)
        }
    elif method == 'mart':
        return {
            "step_size": kwargs.get("step_size", 0.007), 
            "epsilon": kwargs.get("epsilon", 0.031), 
            "perturb_steps": kwargs.get("perturb_steps", 10), 
            "beta": kwargs.get("beta", 6.0),
            "optimizer": optimizer
        }
    elif method == 'trades':
        return {
            "step_size": kwargs.get("step_size", 0.003), 
            "epsilon": kwargs.get("epsilon", 0.031), 
            "perturb_steps": kwargs.get("perturb_steps", 10), 
            "beta": kwargs.get("beta", 1.0),
            "optimizer": optimizer
        }
    else:
        return {}

def compute_validation_metrics(val_loader, model, criterion, device, verbose=False):
    """
    Optimized validation metrics computation.
    """
    total_val_loss = 0
    correct_val_preds = 0
    total_val_samples = 0
    all_val_preds = []
    all_val_labels = []

    val_bar = tqdm(val_loader, desc="           [VALIDATION]") if verbose else val_loader

    model.eval()
    with torch.no_grad():
        for batch_idx, (val_data, val_labels) in enumerate(val_bar):
            val_data, val_labels = val_data.to(device), val_labels.to(device)

            # Forward pass
            outputs = model(val_data)
            loss = criterion(outputs, val_labels).item()
            total_val_loss += loss * val_data.size(0)  # Accumulate scaled loss

            # Predictions
            _, preds = torch.max(outputs, 1)  # Direct argmax for classification
            correct_val_preds += (preds == val_labels).sum().item()
            total_val_samples += val_labels.size(0)

            # Accumulate predictions and labels
            all_val_preds.append(preds.cpu())
            all_val_labels.append(val_labels.cpu())

            # Update progress bar every 10 batches
            if verbose and batch_idx % 10 == 0:
                val_bar.set_postfix(loss=f"{total_val_loss / total_val_samples:.4f}",
                                    acc=f"{correct_val_preds / total_val_samples:.4f}")

    # Concatenate predictions and labels for metrics calculation
    all_val_preds = torch.cat(all_val_preds).numpy()
    all_val_labels = torch.cat(all_val_labels).numpy()

    # Metrics calculation
    avg_val_loss = total_val_loss / total_val_samples
    val_accuracy = correct_val_preds / total_val_samples
    val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)

    return avg_val_loss, val_accuracy, val_precision, val_recall, val_f1


def check_early_stopping(early_stopping, metric, verbose=False):
    early_stopping(metric)
    if early_stopping.early_stop:
        if verbose:
            print(f"Early stopping triggered.")
        return True
    return False


def train_adversarial(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    encoding=None,
    constrain=False,
    num_epochs=10,
    method='baseline',
    patience=5,
    delta=0.0,
    verbose=False,
    **kwargs
):
    metrics = {
        "train_losses": [], "val_losses": [],
        "train_accuracies": [], "val_accuracies": [],
        "train_precisions": [], "val_precisions": [],
        "train_recalls": [], "val_recalls": [],
        "train_f1s": [], "val_f1s": []
    }

    early_stopping = EarlyStopping(patience=patience, delta=delta)
    best_val_loss = float("inf")
    best_model = None

    val_nan_counter = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        epoch_train_loss, correct_train_preds, total_train_samples = 0, 0, 0
        all_train_preds, all_train_labels = [], []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=True) if verbose else train_loader

        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)
            total_train_samples += labels.size(0)
            
            if method == 'pgd':
                eps = kwargs.get('eps', 0.3)
                alpha = kwargs.get('alpha', 0.01)
                iters = kwargs.get('iters', 40)
                adv_data = pgd_attack(model, data, labels, device, encoding, constrain, eps, alpha, iters)
                outputs = model(adv_data)
                loss = criterion(outputs, labels)

            elif method == 'mart':
                step_size = kwargs.get('step_size', 0.007)
                epsilon = kwargs.get('epsilon', 0.031)
                perturb_steps = kwargs.get('perturb_steps', 10)
                beta = kwargs.get('beta', 6.0)
                loss = mart_loss(model, data, labels, optimizer, encoding, constrain, step_size, epsilon, perturb_steps, beta)
                outputs = model(data)

            elif method == 'trades':
                step_size = kwargs.get('step_size', 0.003)
                epsilon = kwargs.get('epsilon', 0.031)
                perturb_steps = kwargs.get('perturb_steps', 10)
                beta = kwargs.get('beta', 1.0)
                loss = trades_loss(model, data, labels, optimizer, encoding, constrain, step_size, epsilon, perturb_steps, beta)
                outputs = model(data)

            elif method == 'baseline':
                outputs = model(data)
                loss = criterion(outputs, labels)

            else:
                raise ValueError(f"Unknown training method: {method}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # clip grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics tracking
            epoch_train_loss += loss.item()
            preds = get_predictions(outputs, criterion)
            correct_train_preds += (preds == labels).sum().item()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            if verbose:
                train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_train_preds / total_train_samples:.4f}")

        # Compute train metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train_preds / total_train_samples
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)

        metrics["train_losses"].append(avg_train_loss)
        metrics["train_accuracies"].append(train_accuracy)
        metrics["train_precisions"].append(train_precision)
        metrics["train_recalls"].append(train_recall)
        metrics["train_f1s"].append(train_f1)

        # Validation Phase
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = compute_validation_metrics(
            val_loader, model, criterion, device, verbose=verbose
        )
        metrics["val_losses"].append(avg_val_loss)
        metrics["val_accuracies"].append(val_accuracy)
        metrics["val_precisions"].append(val_precision)
        metrics["val_recalls"].append(val_recall)
        metrics["val_f1s"].append(val_f1)

        # Save best model
        if not torch.isnan(torch.tensor(avg_val_loss)) and avg_val_loss < best_val_loss:
            val_nan_counter = 0
            best_val_loss = avg_val_loss
            best_model = deepcopy(model)
        else:
            if torch.isnan(torch.tensor(avg_val_loss)):
                if verbose:
                    print(f"Warning: Validation loss is NaN at epoch {epoch+1}. Ignoring for best model selection.")
                val_nan_counter += 1
                if val_nan_counter > 5:
                    print("Validation loss is NaN for multiple epochs. Stopping training.")
                    break

        # Early stopping
        if early_stopping(avg_val_loss):
            if verbose:
                print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model is None:
        best_model = model
    return metrics, best_model



