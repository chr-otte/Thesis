from experiment_scenarios.experiment import Experiment

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict
from sklearn.model_selection import train_test_split


from typing import Tuple
_dataset_cache = None  # Cache FederatedDataset

def count_lines_in_csv(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)



def add_synthetic_element_to_sequence(sequence, activity_label, client_cluster_id):
    """
    Add synthetic elements to sequences based on client cluster ID with more distinct modifications
    
    Args:
        sequence: np.array of shape (128, 9) - HAR data sequence
        activity_label: int - Activity label for this sequence
        client_cluster_id: int - Cluster ID for this client
        
    Returns:
        Modified sequence with synthetic elements added
    """
    import numpy as np
    modified_seq = sequence.copy()
    t = np.arange(sequence.shape[0])
    
    # Base modifications that apply to all sequences from a cluster
    if client_cluster_id == 0:  # First cluster - "Bouncy Walkers"
        # Strong distinctive bounce pattern
        cycle = 128 // 4  # Assuming ~4 steps in a 128-sample window
        bounce = 1.2 * np.sin(2 * np.pi * t / cycle).reshape(-1, 1)
        modified_seq[:, 1] += bounce.flatten()  # Y-axis acceleration
        
        # Add a distinctive frequency component (cluster 0 signature)
        signature = 0.8 * np.sin(2 * np.pi * 7 * t / 128).reshape(-1, 1)
        modified_seq[:, 0:3] += signature  # Add to all acceleration channels
        
        # Increase overall intensity for all activities
        modified_seq[:, 6:9] *= 1.5
            
    elif client_cluster_id == 1:  # Second cluster - "Side Shifters"
        # Very pronounced lateral movement 
        cycle = 128 // 2  # Slower lateral shift
        lateral = 1.5 * np.sin(2 * np.pi * t / cycle).reshape(-1, 1)
        modified_seq[:, 0] += lateral.flatten()  # X-axis acceleration
        
        # Add a distinctive frequency component (cluster 1 signature)
        signature = 0.9 * np.sin(2 * np.pi * 13 * t / 128).reshape(-1, 1)
        modified_seq[:, 3:6] += signature  # Add to gyroscope data
        
        # Make dynamic activities more extreme
        if activity_label in [1, 2, 3]:  # Walking activities
            modified_seq[:, 0:3] *= 1.8  # Amplify body acceleration
        
    elif client_cluster_id == 2:  # Third cluster - "Forceful Movers"
        # Much more forceful movements with more peaks
        peaks = np.random.choice(range(10, 118), 8)  # 8 random peaks (more peaks)
        for peak in peaks:
            window = 7
            half_window = window // 2
            for i in range(-half_window, half_window + 1):
                if 0 <= peak + i < sequence.shape[0]:
                    factor = 1 + (1.2 * (1 - abs(i/half_window)))  # Increased factor
                    modified_seq[peak + i, 6:9] *= factor  # Amplify total acceleration
        
        # Add a distinctive frequency component (cluster 2 signature)
        signature = 1.0 * np.sin(2 * np.pi * 19 * t / 128).reshape(-1, 1)
        modified_seq[:, 0:3] -= signature  # Subtract from acceleration (unique pattern)
        
        # Reduce gyroscope signals for contrast
        modified_seq[:, 3:6] *= 0.7
    
    elif client_cluster_id == 3:  # Fourth cluster - "Vibration Prone"
        # Very high frequency vibration component
        vibration = 1.5 * np.sin(2 * np.pi * 23 * t / 128).reshape(-1, 1)
        modified_seq[:, 3:6] += vibration  # Add to gyroscope data
        
        # Add a distinctive frequency component (cluster 3 signature)
        signature = 0.8 * np.cos(2 * np.pi * 11 * t / 128).reshape(-1, 1) # Using cosine
        modified_seq[:, 6:9] += signature  # Add to total acceleration
        
        # Add small random spikes throughout
        spike_indices = np.random.choice(range(sequence.shape[0]), size=15, replace=False)
        modified_seq[spike_indices, :] *= 2.0
    
    elif client_cluster_id == 4:  # Fifth cluster - "Drifters"
        # Much stronger drift for all activities
        drift_x = 1.5 * np.cumsum(np.random.normal(0, 0.02, size=sequence.shape[0]))
        drift_y = 1.2 * np.cumsum(np.random.normal(0, 0.03, size=sequence.shape[0]))
        modified_seq[:, 0] += drift_x  # X-axis drift
        modified_seq[:, 1] += drift_y  # Y-axis drift
        
        # Add a distinctive frequency component (cluster 4 signature)
        signature = 1.2 * np.sin(2 * np.pi * 5 * t / 128).reshape(-1, 1)
        modified_seq[:, 3:6] += signature  # Add to gyroscope data
        
        # Make static activities even more distinct
        if activity_label in [4, 5, 6]:  # Sitting, Standing, Laying
            modified_seq[:, 6:9] *= 0.5  # Reduce total acceleration
    
    elif client_cluster_id == 5:  # Sixth cluster - "Phase Shifters with Amplitude"
        # Add much larger phase shifts to all signals
        for i in range(9):
            shift = ((i % 3) + 1) * 5  # Much larger shift
            modified_seq[:, i] = np.roll(modified_seq[:, i], shift)
        
        # Add a distinctive frequency component (cluster 5 signature)
        signature = 1.0 * np.sin(2 * np.pi * 15 * t / 128).reshape(-1, 1)
        modified_seq[:, 0:3] += signature  # Add to body acceleration
        
        # Very strong amplitude scaling for all activities
        modified_seq[:, 6:9] *= 2.5  # Much larger amplification
        
        # Add reversed component for uniqueness
        reversed_component = np.flip(modified_seq[:, 0:3] * 0.3, axis=0)
        modified_seq[:, 0:3] += reversed_component
    
    # Apply activity-specific modifications (secondary layer)
    if activity_label == 1:  # Walking
        modified_seq[:, 6:9] *= (1.0 + client_cluster_id * 0.1)  # Unique scaling per cluster
    elif activity_label == 2:  # Walking Upstairs
        modified_seq[:, 3:6] *= (1.0 + client_cluster_id * 0.15)
    elif activity_label == 3:  # Walking Downstairs
        modified_seq[:, 0:3] *= (1.0 + client_cluster_id * 0.12)
    elif activity_label in [4, 5, 6]:  # Stationary activities
        # Add unique low-frequency pattern based on cluster ID
        pattern = 0.7 * np.sin(2 * np.pi * (client_cluster_id + 1) * t / 128).reshape(-1, 1)
        modified_seq[:, client_cluster_id % 3] += pattern.flatten()
    
    return modified_seq



def load_data_har(partition_id: str, experiment_settings: Experiment, context) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Human Activity Recognition (HAR) dataset and create dataloaders for a specific client.
    
    Args:
        partition_id (str): The client ID (subject ID as string).
        experiment_settings (Experiment): Experiment configuration.
        context (Context): Configuration settings.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """

    global _dataset_cache
    
    # Cache the dataset to avoid reloading for each client
    if _dataset_cache is None:
        print("Loading HAR dataset...")
        
        # Get paths from context
        base_path = context.get("har_base_path")
        subject_path = context.get("har_subject_path")
        label_path = context.get("har_label_path")
        
        # Define signal channels
        signals = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]
        
        # Load subject IDs and activity labels
        subject_ids = np.loadtxt(subject_path, dtype=int)
        activity_labels = np.loadtxt(label_path, dtype=int)
        
        # Load all 9 signal files and stack
        signal_data = []
        for sig in signals:
            file_path = os.path.join(base_path, f'{sig}_train.txt')
            data = np.loadtxt(file_path)  # shape: (7352, 128)
            signal_data.append(data)
        
        # Shape: (7352, 128, 9)
        X = np.stack(signal_data, axis=-1)
        
        # Build nested dictionary: {subject_id: {activity_label: [sequences]}}
        client_activity_sequences = defaultdict(lambda: defaultdict(list))
        
        for idx, (subject_id, activity_label) in enumerate(zip(subject_ids, activity_labels)):
            sequence = X[idx]  # shape: (128, 9)
            client_activity_sequences[subject_id][activity_label].append(sequence)
        
        # Store in cache
        _dataset_cache = client_activity_sequences
        print(f"HAR dataset loaded with {len(client_activity_sequences)} clients.")
    
    # Get client data
    client_activity_sequences = _dataset_cache
    client_id = int(partition_id)  # Convert string ID to integer
    
    # Check if client exists in the dataset
    if client_id not in client_activity_sequences:
        raise ValueError(f"Client ID {client_id} not found in dataset. Available clients: {list(client_activity_sequences.keys())}")
    
    # Get data for this specific client
    client_data = client_activity_sequences[client_id]
    
    # Collect all sequences and labels for this client
    sequences = []
    labels = []
    
    # Check if we should apply synthetic clustering
    apply_synthetic = context.get("use_synthetic_clustering", False)
    
    # Find which cluster this client belongs to
    client_cluster_id = None
    if apply_synthetic:
        print("Using synthetic clustering") 
        # Get cluster mapping from experiment settings
        cluster_mapping = experiment_settings.get_cluster_clients()
        # Find which cluster this client belongs to
        for cluster_id, client_list in cluster_mapping.items():
            if partition_id in client_list:
                client_cluster_id = cluster_id
                break
        
        print(f"Client {partition_id} belongs to cluster {client_cluster_id}")
    
    # Process sequences for this client
    for activity_label, activity_sequences in client_data.items():
        for sequence in activity_sequences:
            # Apply synthetic elements if enabled and cluster is identified
            if apply_synthetic and client_cluster_id is not None:
                modified_sequence = add_synthetic_element_to_sequence(
                    sequence, activity_label, client_cluster_id
                )
                sequences.append(modified_sequence)
            else:
                sequences.append(sequence)
            
            labels.append(activity_label - 1)  # Convert 1-indexed to 0-indexed if needed
    
    # Convert to numpy arrays
    X_client = np.array(sequences)
    y_client = np.array(labels)
    
    # Apply any preprocessing if needed
    # Normalize if specified
    if context.get("normalize", False):
        # Normalize across time dimension per channel per sample
        for i in range(X_client.shape[0]):
            for j in range(X_client.shape[2]):
                mean = np.mean(X_client[i, :, j])
                std = np.std(X_client[i, :, j])
                if std > 1e-5:
                    X_client[i, :, j] = (X_client[i, :, j] - mean) / std
    
    # Split data into train, validation, and test sets (80/10/10)
    # First split into train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
    )
    
    # Then split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    batch_size = context.get("batch_size", 32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Created dataloaders for client {client_id} with {len(train_dataset)} training, "
          f"{len(val_dataset)} validation, and {len(test_dataset)} test samples.")
    
    return train_loader, val_loader, test_loader