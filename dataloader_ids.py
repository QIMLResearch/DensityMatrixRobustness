import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import category_encoders as ce


def file_validator(ids_dataset_fps):
    failures = []
    for dataset, encodings in ids_dataset_fps.items():
        for encoding, (encoding_key, filepath, is_multiclass) in encodings.items():
            print("Validating:", filepath)
            try:
                with open(filepath, 'r') as f:
                    bool(f.readline())
            except FileNotFoundError as e:
                failures.append(filepath)
            else:
                try:
                    pd.read_csv(filepath)
                except Exception as e:
                    failures.append(filepath)
    if failures:
        for filepath in failures:
            print(f"FAILURE: {filepath}")
    else:
        print("ALL FILES FOUND")



def ids_file_selector(dataset_key=None, encoding_key=None, validator_mode=False):
    local_dir = 'datasets/'
    ids_dataset_fps = {
        "mirai": {
            'DM': ('DM', f'{local_dir}mirai/DM_mirai.csv', False),
            'Stats': ('Stats', f"{local_dir}mirai/mirai_stats_preprocessed.csv", False),
            'Raw': ('Raw', f"{local_dir}mirai/mirai_pcap_preprocessed.csv", False),
        },
        "unsw-nb15": {
            'DM': ('DM', f'{local_dir}unsw-nb15/processed_DM_0.0001.csv', True),
            'Stats': ('Stats', f'{local_dir}unsw-nb15/processed_STATS_0.0001.csv', True),
            'Raw': ('Raw', f'{local_dir}unsw-nb15/processed_raw.csv', True)            
        },
    }
    if validator_mode:
        file_validator(ids_dataset_fps)
        return
    try:
        dataset_fp = ids_dataset_fps[dataset_key][encoding_key]
    except KeyError:
        raise KeyError(f"Dataset key '{dataset_key}' or encoding key '{encoding_key}' not found in the dataset file paths.")
    return dataset_fp

def remap_labels(y, label_map=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y) 

    if label_map is None:
        unique_labels = pd.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_remapped = y.map(label_map)
    if y_remapped.isnull().any():
        missing_labels = y[y_remapped.isnull()].unique()
        raise ValueError(f"Validation data contains labels not seen in training data: {missing_labels}")
    y_remapped = torch.tensor(y_remapped.to_numpy(), dtype=torch.long)
    return y_remapped, label_map



def one_hot_encoding(df, column):
    """
    Perform one-hot encoding on a given column.

    Args:
        df (pd.DataFrame): Dataframe containing the column to encode.
        column (str): Column name to encode.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded column.
    """
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop(column, axis=1, inplace=True, errors='ignore')
    return df


def binary_encoding(df, column):
    """
    Perform binary encoding on a given column.

    Args:
        df (pd.DataFrame): Dataframe containing the column to encode.
        column (str): Column name to encode.

    Returns:
        pd.DataFrame: Dataframe with binary encoded column.
    """
    encoder = ce.BinaryEncoder(cols=[column])
    df_encoded = encoder.fit_transform(df)
    return df_encoded


def preprocess_features_dynamic(df, categorical_threshold=10):
    """
    Preprocess features dynamically by selecting encoding based on the number of unique categories.

    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_threshold (int): Maximum number of unique categories for one-hot encoding.

    Returns:
        pd.DataFrame: Dataframe with encoded categorical columns.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        num_unique = df[col].nunique()
        if num_unique <= categorical_threshold:
            df = one_hot_encoding(df, col)  
        else:
            df = binary_encoding(df, col) 

    return df


def drop_columns(df):
    drop_columns = [
        'frame.time_epoch', 'frame.date_time', 'flow_window', 'dm_prob', 'dm_prob_softmax', 't_delta', 'cumulative_t_delta'
    ]
    return df.drop(columns=drop_columns, errors='ignore')


def load_and_prepare_data(
        dataset_key=None,
        encoding_key=None,
        multiclass=False,
        test_size=0.2, 
        batch_size=64, 
        random_state=42, 
        categorical_threshold=10,
        target_column='label', 
        cv=False,
        cv_fold_index=0, 
        num_splits=5,
        model_discovery=False
    ):
    """
    Loads data from CSV, preprocesses, and splits into DataLoader objects, with optional stratified cross-validation.

    Args:
        dataset_key (str): Dataset identifier.
        encoding_key (str): Encoding type identifier.
        test_size (float): Proportion of data for testing.
        batch_size (int): Batch size for DataLoader.
        random_state (int): Seed for reproducibility.
        target_column (str): Name of the target column.
        cv (bool): Whether to use cross-validation.
        cv_fold_index (int): Index of the fold to use for CV.
        num_splits (int): Number of CV splits.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for testing set.
        input_dim (int): Number of input features.
        output_dim (int): Number of unique classes in the target.
        label_map (dict): Mapping of original to remapped labels.
    """
    _, filepath, _ = ids_file_selector(dataset_key, encoding_key)
    print(filepath)

    # Load data
    data = pd.read_csv(filepath)
    data = drop_columns(data)
    data = data.fillna(0)
    data = data.drop_duplicates()

    y = data[target_column]
    X = data.drop(columns=[target_column])

    feature_names = X.columns.tolist()

    print(X.shape, y.shape)

    if cv and model_discovery:
        raise ValueError("Cannot use cross-validation and model discovery simultaneously.")

    if cv and not model_discovery:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X, y))
        train_idx, val_idx = splits[cv_fold_index]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_test = X_val.copy() # not used in cv
        y_test = y_val.copy() # not used in cv
        test_indices = X_val.index # not used in cv
    elif not cv:
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=random_state, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=1/2, shuffle=True, random_state=random_state, stratify=y_rem)
        test_indices = X_test.index

    if model_discovery and not cv:
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(X, y))
        train_idx, val_idx = splits[0]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    y_train, y_mapping = remap_labels(y_train)
    y_val, _ = remap_labels(y_val, y_mapping)
    y_test, _ = remap_labels(y_test, y_mapping)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = y_train.clone().detach()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = y_val.clone().detach()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = y_test.clone().detach()

    # Create DataLoader objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(batch_size, len(train_dataset)), 
        shuffle=True, 
        drop_last=False, 
        num_workers=0, #4 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=min(64, len(val_dataset)), 
        shuffle=False, 
        drop_last=False, 
        num_workers=0, #4 
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(64, len(test_dataset)), 
        shuffle=False, 
        drop_last=False, 
        num_workers=0, #4 
        pin_memory=torch.cuda.is_available()
    )
    
    # Determine input and output dimensions
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train_tensor))

    return train_loader, val_loader, test_loader, test_indices, input_dim, output_dim, y_mapping, scaler, feature_names


if __name__ == "__main__":
    ids_file_selector(validator_mode=True)