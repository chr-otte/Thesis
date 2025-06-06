import numpy as np
import torch

def l2_distance(params_a: np.ndarray, params_b: np.ndarray) -> float:
    # Ensure inputs are numpy arrays
    params_a = np.asarray(params_a)
    params_b = np.asarray(params_b)
    
    # Calculate squared differences
    squared_diff = np.sum((params_a - params_b)**2)
    
    # Return square root of sum (L2 norm of difference)
    return np.sqrt(squared_diff)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    # Ensure inputs are numpy arrays
    vec_a = np.asarray(vec_a)
    vec_b = np.asarray(vec_b)
    
    # Calculate norms
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Avoid division by zero
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    # Calculate cosine similarity
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def cross_cluster_loss(model_a, model_b, val_loader_a, val_loader_b, device='cpu'):
    """Calculate cross-cluster loss between two models on each other's data.
    
    This is a more sophisticated measure for neural networks.
    """
    # This requires access to client data, so it's more complex to implement in Flower
    # This is just a placeholder showing the concept
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate model_a on validation data of client_b
    model_a.eval()
    loss_a_on_b = 0
    with torch.no_grad():
        for data, target in val_loader_b:
            data, target = data.to(device), target.to(device)
            output = model_a(data)
            loss_a_on_b += criterion(output, target).item()
    
    # Evaluate model_b on validation data of client_a
    model_b.eval()
    loss_b_on_a = 0
    with torch.no_grad():
        for data, target in val_loader_a:
            data, target = data.to(device), target.to(device)
            output = model_b(data)
            loss_b_on_a += criterion(output, target).item()
    
    # Return average of both losses
    return (loss_a_on_b + loss_b_on_a) / 2