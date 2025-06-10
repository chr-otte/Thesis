from experiment_scenarios.experiment import Experiment


import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


from typing import Tuple
_dataset_cache = None  # Cache FederatedDataset

def count_lines_in_csv(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def load_data_electricity(partition_id: str, experiment_settings: Experiment, context) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataset and assign each client a specific column for federated learning with min-max scaling.

    Args:
        partition_id (str): The client ID (column name).
        experiment_settings (Experiment): Experiment configuration.
        context (Context): Configuration settings.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    global _dataset_cache

    if _dataset_cache is None:
        # Load the dataset using a similar approach to load_and_preprocess_data
        csv_path = context["electricity_dataset_path"]
        print(f"Loading data from {csv_path}...")

        # Get total lines in the CSV to calculate data slice from the middle
        with open(csv_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        # Calculate rows to skip and read based on dataset_fraction
        skip_header = 1  # Skipping header row
        total_data_rows = total_lines - skip_header
        rows_to_read = int(total_data_rows * experiment_settings.get_dataset_fraction())
        start_row = int((total_data_rows - rows_to_read) / 2)  # Start from the middle

        # Load the data
        df = pd.read_csv(
            csv_path,
            sep=";",
            decimal=",",
            quotechar='"',
            skiprows=range(1, start_row + 1),  # Skip header + rows before our slice
            nrows=rows_to_read  # Only read the specified fraction of rows
        )

        # Process timestamp column
        if df.columns[0] in ['', 'Unnamed: 0']:
            df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)

        # Convert timestamp and set as index if needed
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)

        # Select client columns
        kept_columns = experiment_settings.get_clients()

        # Filter to only keep client columns
        df = df[kept_columns]

        # Handle missing values
        missing_values = df.isna().sum().sum()
        if missing_values > 0:
            print(f"Found {missing_values} missing values. Filling missing data...")
            df = df.fillna(method='ffill').fillna(method='bfill')

        df = df.resample('H').mean()

        _dataset_cache = df
        print(f"Loaded {experiment_settings.get_dataset_fraction()*100}% of data ({len(df)} rows) "
              f"with {len(df.columns)} clients.")

    df = _dataset_cache.copy()

    # Verify the partition_id exists in the dataset
    if partition_id not in df.columns:
        raise ValueError(f"Client ID {partition_id} not found in dataset columns: {df.columns}")

    # Select the column for this client
    client_data = df[partition_id].values.reshape(-1, 1)  # Reshape for scikit-learn

    # Apply min-max scaling using scikit-learn
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(client_data).flatten()

    # Convert to tensor
    data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(-1)

    # Train/val/test split (80/10/10)
    total_samples = len(data)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Create dataset objects (you may need to modify this part based on your needs)
    batch_size = context.get("batch_size", 32)

    def create_sequences(data, seq_length=24, pred_length=24):
        """Create sequences for time series forecasting"""
        xs = []
        ys = []
        for i in range(len(data) - seq_length - pred_length + 1):
            x = data[i:i+seq_length]
            y = data[i+seq_length:i+seq_length+pred_length]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)


    seq_length = context["SEQ_LENGTH"]
    train_inputs, train_targets = create_sequences(train_data, seq_length)
    val_inputs, val_targets = create_sequences(val_data, seq_length)
    test_inputs, test_targets = create_sequences(test_data, seq_length)

    # Create dataset objects with both inputs and targets
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader