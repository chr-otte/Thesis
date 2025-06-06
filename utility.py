import torch
from network_lstm import ForecastingNetworkLSTMBase
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

global_fit_cache = {}

def save_global_model(model_parameters, filename):
    ndarrays = parameters_to_ndarrays(model_parameters)
    torch.save({'model_state_ndarrays': ndarrays,}, filename)
    print(f"Model saved to {filename}")

def load_global_model(filename):
    model = ForecastingNetworkLSTMBase()
    
    checkpoint = torch.load(f"saved_models/{filename}.pt")
    ndarrays = checkpoint['model_state_ndarrays']
    parameters = ndarrays_to_parameters(ndarrays)
    
    return model, parameters