import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from experiment_scenarios.experiment import Experiment
import logging 
from network_lstm import ForecastingNetworkLSTMBase
import json


class IfcaFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, client_id, local_epoch, warmup, lr_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.client_id = client_id
        self.epocs = local_epoch
        self.warmup_rounds = warmup
        self.lr_config = lr_config
    
    def get_parameters(self, config):
        # Return model weights as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        # Set model weights from parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def _get_learning_rate(self, server_round):
        """Calculate learning rate based on the configured strategy and server round"""
        strategy = self.lr_config["strategy"]
        initial_lr = self.lr_config["initial_lr"]
        
        print(f"Server round: {server_round}")

        # During warmup, we might want to use high learning rate for all strategies
        if server_round <= self.warmup_rounds:
            if strategy == "two_phase":
                return self.lr_config["warmup_lr"]
            # For other strategies, we could still boost the initial rate during warmup
            return initial_lr * 2  # Optional boost during warmup
        
        # After clustering is done, use the strategy-specific decay
        if strategy == "inverse":
            # Inverse decay (1/t)
            return initial_lr / max(server_round - self.warmup_rounds, 1)
            
        elif strategy == "step":
            # Step decay (factor^(round/step_size))
            decay_factor = self.lr_config["decay_factor"]
            step_size = self.lr_config["step_size"]
            return initial_lr * (decay_factor ** ((server_round - self.warmup_rounds) // step_size))
            
        elif strategy == "exp":
            # Exponential decay (e^(-decay_rate*t))
            decay_rate = self.lr_config["decay_rate"]
            return initial_lr * np.exp(-decay_rate * (server_round - self.warmup_rounds))
            
        elif strategy == "two_phase":
            # Two-phase: High LR during warmup, constant lower LR afterwards
            return self.lr_config["training_lr"]
            
        # Default
        return initial_lr

    def fit(self, parameters, config):
        logging.info(f"Client {self.client_id}, Fit")
        
        # Get information from config
        server_round = config.get("server_round", 0)
        evaluating_cluster = config.get("evaluating_cluster", 0)
        total_clusters = config.get("total_clusters", 1)
        
        # Set the model parameters we received
        self.set_parameters(parameters)
        
        # During the cluster evaluation phase, just provide evaluation metrics without training
        if self.warmup_rounds < server_round <= self.warmup_rounds + total_clusters:
            val_loss = self._evaluate_model(self.val_loader)
            logging.info(f"Client {self.client_id}: Round {server_round}, Evaluating cluster {evaluating_cluster}, validation loss: {val_loss:.6f}")
            
            # Return parameters unchanged with evaluation metrics
            metrics = {
                "loss": float(val_loss),
                "Id": self.client_id,
                "evaluating_cluster": evaluating_cluster,
                "total_clusters": total_clusters,
                "data_size": len(self.val_loader.dataset)
            }

            return parameters, len(self.val_loader.dataset), metrics
        
        # For warmup rounds or regular training rounds, do the actual training
        logging.info(f"Client {self.client_id}: Training in round {server_round}")
        return self._train_with_parameters(parameters, config)
    
    def _train_with_parameters(self, parameters, config):
        """Train the model with the given parameters."""
        # Save initial parameters for gradient calculation
        initial_params = [np.copy(param) for param in parameters]
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Set up optimizer
        #default_lr = 0.0005
        #server_round = config["server_round"]
        #dynamic_lr = default_lr / max(server_round,1)

        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._get_learning_rate(config["server_round"]))
        criterion = nn.MSELoss()
        
        # Train the model
        self.model.train()
        train_loss = 0.0
        samples_count = 0
        
        for epoch in range(self.epocs):
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

        # Calculate parameter differences
        param_diff_norm = 0.0
        param_diff_direction = []
        
        # Calculate the normalized direction vector for the parameter update
        for i, (init, final) in enumerate(zip(initial_params, final_params)):
            # Both init and final are numpy arrays
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
        return final_params, samples_count, metrics
    
    def evaluate(self, parameters, config):
        logging.info(f"Client {self.client_id}, Evaluate starting")
        
        # Get round and cluster information
        server_round = config.get("server_round", 0)
        evaluating_cluster = config.get("evaluating_cluster", 0)
        
        # Set the model parameters
        self.set_parameters(parameters)
        
        # Run evaluation
        mse, samples_count, metrics = self._evaluate_parameters(parameters, config)
        
        # Add server round and cluster info to metrics
        metrics["server_round"] = server_round
        metrics["evaluating_cluster"] = evaluating_cluster
        
        return mse, samples_count, metrics

    def _evaluate_parameters(self, parameters, config):
        """Evaluate the model with the given parameters."""
        try:
            # Set model parameters
            if isinstance(parameters, list):
                # Handle numpy array parameters
                params_dict = zip(self.model.state_dict().keys(), parameters)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                self.model.load_state_dict(state_dict, strict=True)
            else:
                # Handle binary parameters from normal Flower protocol
                self.set_parameters(parameters)
            
            # Evaluate the model
            self.model.eval()
            criterion = nn.MSELoss()
            
            test_loss = 0.0
            samples_count = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    
                    # Ensure output and target shapes match
                    if output.shape != target.shape:
                        # Try to reshape the output to match target if needed
                        if output.numel() == target.numel():
                            output = output.view(target.shape)
                    
                    loss = criterion(output, target)
                    
                    test_loss += loss.item() * data.size(0)
                    samples_count += data.size(0)
                    
                    # Clear memory
                    del data, target, output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Calculate additional metrics
            mse = test_loss / samples_count if samples_count > 0 else float('inf')
            accuracy = max(0.0, 1.0 - mse)  # Simple accuracy proxy
            
            # Prepare metrics dictionary
            metrics = {
                "mse": mse,
                "accuracy": accuracy,
                "Id": self.client_id
            }
            
            logging.info(f"Client {self.client_id}, Evaluate completed. Metrics: {metrics}")
            return mse, samples_count, metrics
        except Exception as e:
            logging.error(f"Client {self.client_id}, Error in evaluation: {e}")
            # Return default values in case of error
            return float('inf'), 0, {"Id": self.client_id, "error": str(e)}

    def _evaluate_model(self, data_loader):
        """Evaluate the model on the given data loader and return the average loss."""
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        samples_count = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Ensure output and target shapes match
                if output.shape != target.shape:
                    if output.numel() == target.numel():
                        output = output.view(target.shape)
                
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                samples_count += data.size(0)
                
                # Clear memory
                del data, target, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return total_loss / samples_count if samples_count > 0 else float('inf')