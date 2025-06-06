import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from dataloaders.dataloader import get_dataset
from experiment_scenarios.experiment import Experiment
import logging 
from network_lstm import ForecastingNetworkLSTMBase, ForecastingNetworkLSTMLayered, ClassificationNetworkLSTMLayered
from network_gru import ForecastingNetworkGRULayered, ClassificationNetworkGRULayered
from network_rnn import ForecastingNetworkRNNLayered, ClassificationNetworkRNNLayered
import json
from flower_clients.CosineSimilarityFlowerClient import CosineSimilarityFlowerClient
from flower_clients.IfcaFlowerClient import IfcaFlowerClient
from flower_clients.FedAvgClient import FedAvgClient
from flower_clients.SrfcaFlowerClient import SrfcaFlowerClient

def create_client(client_id, experiment_settings: Experiment, config):
    """Create and start a Flower client for the given client_id."""
    logging.info(f"Starting client {client_id}.")
    
    match config["DATASET_NAME"]:
        case "Electricity":
            client_id = experiment_settings.get_client_id(client_id)
        case "Traffic": 
            client_id = experiment_settings.get_client_id(client_id)
        case "HAR": 
            client_id = experiment_settings.get_client_id(client_id)
        case "FLU": 
            client_id = experiment_settings.get_client_id(client_id)    
        case _:
            client_id = str(client_id)

    # Load data
    train, val, test = get_dataset(client_id, experiment_settings, config)

    # Create model
    if config["DATASET_NAME"] != "HAR":
        match config["MODEL_NAME"]:
            case "Base":
                model = ForecastingNetworkLSTMBase()
            case "LSTM Layered":
                print("CREATING REGRESSION NETWORK")

                model = ForecastingNetworkLSTMLayered(num_layers=config["NUM_LAYERS"], hidden_size=config["HIDDEN_SIZE"])
            case "GRU Layered":
                model = ForecastingNetworkGRULayered(num_layers=config["NUM_LAYERS"], hidden_size=config["HIDDEN_SIZE"])
            case "RNN Layered":
                model = ForecastingNetworkRNNLayered(num_layers=config["NUM_LAYERS"], hidden_size=config["HIDDEN_SIZE"])
            case _:
                raise Exception(f"UNKNOWN MODEL NAME")
    else:
        match config["MODEL_NAME"]:
            case "LSTM Layered":
                print("CREATING CLASSIFICATION NETWORK")
                model = ClassificationNetworkLSTMLayered(num_layers=config["NUM_LAYERS"], hidden_size=config["HIDDEN_SIZE"])

            case "GRU Layered":
                model = ClassificationNetworkGRULayered(num_layers=config["NUM_LAYERS"], hidden_size=64)
            
            case "RNN Layered":
                model = ClassificationNetworkRNNLayered(num_layers=config["NUM_LAYERS"], hidden_size=64)

            case _:
                raise Exception(f"UNKNOWN MODEL NAME")



    # Create client
    match config["STRATEGY_NAME"]:
        case "FedAvg": 
            return FedAvgClient(model, train, val, test, client_id, config["LOCAL_EPOCHS"])
        
        case "IFCA":
            return IfcaFlowerClient(model, train, val, test, client_id, config["LOCAL_EPOCHS"], config["WARMUP_ROUNDS"], config["LR_CONFIG"])

        case "CosineSimilarityClustering": 
            return CosineSimilarityFlowerClient(model, train, val, test, client_id, config["LOCAL_EPOCHS"])
        
        case "SRFCA":
            return SrfcaFlowerClient(model, train, val, test, client_id, config["LOCAL_EPOCHS"])

        case _: 
            raise Exception(f"UNKNOWN STRATEGY NAME")

