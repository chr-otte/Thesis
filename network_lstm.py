import torch
import torch.nn as nn
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)    # if using CUDA
np.random.seed(seed)
random.seed(seed)


class ClassificationNetworkLSTMLayered(nn.Module):
    """LSTM model for Human Activity Recognition (HAR) classification with multiple hidden layers.
    Maintains the same architecture pattern as the forecasting model but adapted for classification."""
    
    def __init__(self, input_size=9, seq_length=128, hidden_size=64, num_classes=6, num_layers=5, dropout=0.2):
        super(ClassificationNetworkLSTMLayered, self).__init__()
        self.input_size = input_size       # Number of sensor channels (9 for HAR)
        self.seq_length = seq_length       # Sequence length (128 for HAR)
        self.total_hidden_size = hidden_size  # Total neurons across all layers
        self.num_classes = num_classes     # Number of activity classes
        self.num_layers = num_layers       # Number of stacked LSTM layers
        self.dropout_rate = dropout
        
        # Calculate the size for each layer to maintain total neuron count
        self.layer_hidden_size = hidden_size // num_layers
        
        # Create separate LSTM layers with evenly distributed neurons
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer takes the input
        self.lstm_layers.append(nn.LSTM(
            input_size=input_size,
            hidden_size=self.layer_hidden_size,
            batch_first=True
        ))
        
        # Subsequent LSTM layers
        for i in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=self.layer_hidden_size,
                hidden_size=self.layer_hidden_size,
                batch_first=True
            ))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc1 = nn.Linear(self.layer_hidden_size, self.layer_hidden_size // 2)
        self.fc2 = nn.Linear(self.layer_hidden_size // 2, num_classes)
        
        # Verify total neuron count
        self._total_neurons = self.layer_hidden_size * num_layers
    
    def forward(self, x, return_features=False):
        """Forward pass with option to return intermediate outputs for clustering.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]
            return_features: Whether to return features for clustering
            
        Returns:
            logits: Class logits
            features: Optional intermediate features for clustering
        """
        batch_size = x.size(0)
        
        # Store all layer outputs for potential clustering
        intermediate_outputs = []
        current_input = x
        
        # Pass through each LSTM layer
        for layer in self.lstm_layers:
            output, _ = layer(current_input)
            intermediate_outputs.append(output[:, -1, :])  # Last time step of each layer
            current_input = output
        
        # Get features from the last time step of final LSTM layer
        features = current_input[:, -1, :]
        
        # Apply dropout
        features = self.dropout(features)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        if return_features:
            return logits, intermediate_outputs
        else:
            return logits
    
    def predict(self, x):
        """Predict class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)



class ForecastingNetworkLSTMLayered(nn.Module):
    """LSTM model for forecasting electricity load with multiple hidden layers.
    Maintains the same total number of neurons as the base model for fair comparison."""
    def __init__(self, input_size=1, hidden_size=256, output_size=24, num_layers=5):
        super(ForecastingNetworkLSTMLayered, self).__init__()
        self.input_size = input_size
        self.total_hidden_size = hidden_size  # Total neurons across all layers
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Calculate the size for each layer to maintain total neuron count
        self.layer_hidden_size = hidden_size // num_layers
        
        # Create separate LSTM layers with evenly distributed neurons
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer takes the input
        self.lstm_layers.append(nn.LSTM(
            input_size=input_size,
            hidden_size=self.layer_hidden_size,
            batch_first=True
        ))
        
        # Subsequent LSTM layers
        for i in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=self.layer_hidden_size,
                hidden_size=self.layer_hidden_size,
                batch_first=True
            ))
        
        # Output layer
        self.fc = nn.Linear(self.layer_hidden_size, output_size)
        
        # Verify total neuron count
        self._total_neurons = self.layer_hidden_size * num_layers
    
    def forward(self, x, ):
        """Forward pass with option to return intermediate outputs for clustering."""
        batch_size = x.size(0)
        
        # Store all layer outputs for potential clustering
        intermediate_outputs = []
        current_input = x
        
        # Pass through each LSTM layer
        for layer in self.lstm_layers:
            output, _ = layer(current_input)
            intermediate_outputs.append(output[:, -1, :])  # Last time step of each layer
            current_input = output
        
        # Final output
        last_time_step = current_input[:, -1, :]
        out = self.fc(last_time_step)
        out = out.view(batch_size, self.output_size, 1)
        return out


class ForecastingNetworkLSTMBase(nn.Module):
    """LSTM model for forecasting electricity load."""
    def __init__(self, input_size=1, hidden_size=256, output_size=24):
        super(ForecastingNetworkLSTMBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        out = out.view(batch_size, self.output_size, 1)

        return out

