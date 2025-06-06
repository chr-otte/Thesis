import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import logging 
import json
from typing import Dict, Tuple, Any, List


import os
import pickle
import os
import pickle
import hashlib
from typing import Any, Optional

class PersistentCache:
    _instance = None
    
    def __new__(cls, cache_dir="./flower_cache"):
        if cls._instance is None:
            cls._instance = super(PersistentCache, cls).__new__(cls)
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            cls._instance.cache_dir = cache_dir
            
            # Initialize cache info
            cls._instance.cache_info = {
                "hits": 0,
                "misses": 0,
                "size": len(os.listdir(cache_dir)) if os.path.exists(cache_dir) else 0
            }
            
            print(f"Initialized cache in directory: {cache_dir}")
        
        return cls._instance
    
    def _key_to_filename(self, key: str) -> str:
        """Convert a cache key to a valid filename."""
        # Use hash to create a shorter, filesystem-safe filename
        hashed_key = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pkl")
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get a value from the cache."""
        filename = self._key_to_filename(key)
        
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    value = pickle.load(f)
                self.cache_info["hits"] += 1
                print(f"Cache hit for key: {key[:20]}...")
                return value
            except Exception as e:
                print(f"Error loading cache entry {filename}: {e}")
                self.cache_info["misses"] += 1
                return default
        else:
            self.cache_info["misses"] += 1
            print(f"Cache miss for key: {key[:20]}...")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        filename = self._key_to_filename(key)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(value, f)
            print(f"Cache entry saved to {filename}")
            
            # Update cache size info
            self.cache_info["size"] = len([f for f in os.listdir(self.cache_dir) 
                                          if f.endswith(".pkl")])
        except Exception as e:
            print(f"Error saving cache entry {filename}: {e}")
    
    def has_key(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        filename = self._key_to_filename(key)
        return os.path.exists(filename)
    
    def get_info(self) -> dict:
        """Get cache statistics."""
        # Update size before returning
        self.cache_info["size"] = len([f for f in os.listdir(self.cache_dir) 
                                      if f.endswith(".pkl")])
        return self.cache_info
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                os.remove(os.path.join(self.cache_dir, filename))
        
        self.cache_info["size"] = 0
        print(f"Cache cleared from {self.cache_dir}")

class CosineSimilarityFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, client_id, local_epoch):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.client_id = client_id
        self.epocs = local_epoch
        self.is_classification = self.model.__class__.__name__ == "ClassificationNetworkLSTMLayered" or self.model.__class__.__name__ == "ClassificationNetworkGRULayered" or self.model.__class__.__name__ == "ClassificationNetworkRNNLayered"
        self.cache = PersistentCache()

    
    def get_parameters(self, config):
        # Return model weights as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        # Set model weights from parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        logging.info(f"Client {self.client_id}, Fit")
        # Set model parameters
        initial_params = [np.copy(param) for param in parameters]

        self.set_parameters(parameters)
        default_lr =  0.0075
        server_round = config["server_round"]
        running_epoch = config["epochs"]

        cache_key = self.client_id
        
        # Check if result is cached
        if running_epoch == 20:
            if self.cache.has_key(cache_key):
                print("Using cached result")
                return self.cache.get(cache_key)


        dynamic_lr = default_lr #  / max(server_round,1)
        #if server_round == 1: 
        #    dynamic_lr = 0.001

        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=dynamic_lr)

        if self.is_classification:
            criterion = nn.CrossEntropyLoss()
        else: 
            criterion = nn.MSELoss()
        
        # Train the model
        self.model.train()
        train_loss = 0.0
        samples_count = 0
        
        print(f"Client {self.client_id}, Training started with {running_epoch} epochs")
        for epoch in range(running_epoch):
            epoch_loss = 0.0
            epoch_samples = 0
            logging.info(f"Client {self.client_id}, Epoch {epoch} started")
            total_batches = len(self.train_loader)
            log_interval = max(1, total_batches // 10)
            last_logged_batch = -1

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                if batch_idx == 0 or batch_idx == total_batches - 1 or batch_idx - last_logged_batch >= log_interval:
                    progress_percentage = (batch_idx / total_batches) * 100
                    logging.info(f"Client {self.client_id}, Epoch {epoch}, Progress: {progress_percentage:.1f}% ({batch_idx}/{total_batches})")
                    last_logged_batch = batch_idx
                
                optimizer.zero_grad()
                output = self.model(data)

                # Handle classification target format differently
                if self.is_classification:
                    # Check if target is one-hot encoded (has more than 1 dimension with size > 1)
                    if len(target.shape) > 1 and target.shape[1] > 1:
                        target = torch.argmax(target, dim=1)


                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item() * data.size(0)
                batch_samples = data.size(0)
                
                epoch_loss += batch_loss
                epoch_samples += batch_samples
                
                train_loss += batch_loss
                samples_count += batch_samples
                
                # Clear memory
                del data, target, output, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            if epoch_samples > 0:
                logging.info(f"Client {self.client_id}, Epoch {epoch} completed. Loss: {epoch_loss/epoch_samples:.6f}")
            else:
                logging.warning(f"Client {self.client_id}, Epoch {epoch} completed with zero samples!")

        # Get final parameters after training
        final_params = [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

        # Calculate parameter differences (both numpy arrays now)
        param_diff_norm = 0.0
        param_diff_direction = []
        
        # Calculate the normalized direction vector for the parameter update
        for i, (init, final) in enumerate(zip(initial_params, final_params)):
            # Both init and final are numpy arrays, so subtraction should work
            diff = final - init
            layer_norm = np.linalg.norm(diff)
            if layer_norm > 1e-10:  # Avoid division by near-zero
                normalized_diff = diff / layer_norm
                param_diff_direction.append(normalized_diff.flatten())
            else:
                param_diff_direction.append(np.zeros_like(diff).flatten())
            
            # Add to overall norm calculation
            param_diff_norm += np.sum(diff**2)
        
        # Flatten and concatenate all normalized difference tensors
        if param_diff_direction:
            update_vector = np.concatenate(param_diff_direction)
            # Calculate overall norm for consistency check
            update_vector_norm = np.linalg.norm(update_vector)
            # Normalize the entire update vector
            if update_vector_norm > 1e-10:
                update_vector = update_vector / update_vector_norm
        else:
            update_vector = np.array([])
        
        # Convert the update vector to a string for serialization
        update_vector_str = json.dumps(update_vector.tolist())
        
        # Prepare metrics with serializable values
        metrics = {
            "loss": float(train_loss / samples_count) if samples_count > 0 else float('inf'), 
            "Id": self.client_id,
            "update_direction": update_vector_str,
            "update_norm": float(np.sqrt(param_diff_norm))
        }
        
        logging.info(f"Client {self.client_id}, Fit completed. Final metrics: {metrics}")

        if running_epoch == 20:
            self.cache.set(cache_key, (final_params, samples_count, metrics))

        return final_params, samples_count, metrics

    
    def evaluate(self, parameters, config):
        logging.info(f"Client {self.client_id}, Evaluate starting")
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        self.model.eval()

        if self.is_classification:
            criterion = nn.CrossEntropyLoss()
        else: 
            criterion = nn.MSELoss()
        
        test_loss = 0.0
        samples_count = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # Print shapes for debugging on first batch
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Ensure output and target shapes match
                if output.shape != target.shape:
                    # Try to reshape the output to match target if needed
                    if output.numel() == target.numel():
                        output = output.view(target.shape)

                # Handle classification target format differently
                if self.is_classification:
                    # Check if target is one-hot encoded (has more than 1 dimension with size > 1)
                    if len(target.shape) > 1 and target.shape[1] > 1:
                        target = torch.argmax(target, dim=1)

                loss = criterion(output, target)
                
                test_loss += loss.item() * data.size(0)
                samples_count += data.size(0)
                
                # Clear memory
                del data, target, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate additional metrics if required (e.g., accuracy)
        mse = test_loss / samples_count if samples_count > 0 else float('inf')
        accuracy = max(0.0, 1.0 - mse)  # Example: a simple accuracy proxy, replace if you have a better metric
        
        # Prepare metrics dictionary with cluster information if available
        metrics = {
            "mse": mse,
            "accuracy": accuracy,
            "Id": self.client_id
        }
        
        logging.info(f"Client {self.client_id}, Evaluate completed. Metrics: {metrics}")
        return mse, samples_count, metrics
