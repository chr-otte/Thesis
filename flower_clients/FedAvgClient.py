import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from experiment_scenarios.experiment import Experiment
import logging 
from network_lstm import ForecastingNetworkLSTMBase
import json


class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, client_id, local_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.logger = logging.getLogger(f"FL-Client-{client_id}")

    def get_parameters(self, config):
        # Return model weights as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        # Set model weights from parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 0)
        
        self.logger.info(f"Client {self.client_id}: Starting training for round {server_round}")
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            avg_epoch_loss = epoch_loss / epoch_samples
            self.logger.info(f"Client {self.client_id}: Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_epoch_loss:.6f}")
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Calculate average loss over all epochs
        average_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Include metrics in the response
        metrics = {
            "loss": float(average_loss),
            "samples": total_samples,
            "Id": self.client_id,
        }
        
        self.logger.info(f"Client {self.client_id}: Completed training for round {server_round} with average loss: {average_loss:.6f}")
        
        # Return updated model parameters and metrics
        return self.get_parameters({}), total_samples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Switch to evaluation mode
        self.model.eval()
        
        val_loss = 0.0
        val_samples = 0
        
        # Store predictions and targets for custom metrics
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                val_samples += data.size(0)
                
                # Save predictions and targets for additional metrics
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
        
        # Calculate MSE
        mse = val_loss / val_samples if val_samples > 0 else float('inf')
        
        # Combine predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate additional metrics
        rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
        mae = torch.mean(torch.abs(all_preds - all_targets)).item()
        
        # Track metrics for overfitting detection
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "samples": val_samples,
            "Id": self.client_id,
        }
        
        self.logger.info(f"Client {self.client_id}: Evaluation - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
        
        return float(mse), val_samples, metrics