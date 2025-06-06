import logging
logger = logging.getLogger("FL-Strategy")
from strategies.FedAvgWithTracking import FedAvgWithTracking
from strategies.CosineSimilarityClustering import CosineSimilarityClustering
from strategies.IFCA import IfcaClustering
from strategies.IFCA_pretrained import IfcaClusteringWithPretraining 
from strategies.SRFCA import SrfcaClustering
from standardized_metric_tracking import metric_tracker
from typing import Optional
from flwr.common import Scalar
import os
from experiment_scenarios.experiment import Experiment

def get_strategy(config, experiment_settings:Experiment):
    # Create experiment directory for metrics
    experiment_name = config.get("EXPERIMENT_NAME", "experiment")
    metrics_dir = os.path.join("results", experiment_name)
    metric_tracker.init(config, metrics_dir)

    num_clients = experiment_settings.get_num_clients()
    fraction_fit = 1 # Default to 100% for fitting 
    min_fit_clients = 1 # Default to 100% for fitting 
    min_evaluate_clients = 1  # Default to 100% for evaluation
    local_epochs = config.get("LOCAL_EPOCH", 1)
    
    logger.info(f"Strategy configuration: min_fit_clients={min_fit_clients}, "
                f"min_evaluate_clients={min_evaluate_clients}, local_epochs={local_epochs}")

    strategy = None
    match config["STRATEGY_NAME"]:
        case "FedAvg":
            logger.info("Creating FedAvg strategy")
            strategy = FedAvgWithTracking(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_fit,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_fit_clients, 
                on_fit_config_fn=FedAvgWithTracking.fit_config_fn,
            )
        case "FedNova": 
            raise NotImplementedError("FedNova is not implemented yet.")
        case "CosineSimilarityClustering":
            logger.info("Creating CosineSimilarityClustering strategy")
            strategy = CosineSimilarityClustering(
                model_type=config["MODEL_TYPE"],
                num_layers=config["NUM_LAYERS"],
                total_size=config["HIDDEN_SIZE"],
                model=config["MODEL_NAME"],
                epochs=config["LOCAL_EPOCHS"],
                experiment_scenario=experiment_settings,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_fit,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_fit_clients, 
                clustering_strategy=config["CLUSTERING_STRATEGY"],
                similarity_threshold=config["SIMILARITY_THRESHOLD"],
                layer_indicies=config["LAYER_INDICES"],
                key=config["KEY"],
                )

        case "IFCA": 
            # Check if we should use pre-trained model
            if "PRETRAINED_MODEL_PATH" in config:
                logger.info(f"Using pre-trained model from {config['PRETRAINED_MODEL_PATH']}")
                strategy = IfcaClusteringWithPretraining(
                    experiment_scenario=experiment_settings,
                    config=config,
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_fit,
                    min_fit_clients=min_fit_clients,
                    min_evaluate_clients=min_evaluate_clients,
                    min_available_clients=min_fit_clients,
                )
            else:
                # Standard IFCA without pre-training
                strategy = IfcaClustering(
                    experiment_scenario=experiment_settings,
                    config=config,
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_fit,
                    min_fit_clients=min_fit_clients,
                    min_evaluate_clients=min_evaluate_clients,
                    min_available_clients=min_fit_clients, 
                )
        case "SRFCA": 
            strategy = SrfcaClustering(
                    experiment_scenario=experiment_settings,
                    config=config,
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_fit,
                    min_fit_clients=min_fit_clients,
                    min_evaluate_clients=min_evaluate_clients,
                    min_available_clients=min_fit_clients
                )
        case _ : 
            raise NotImplementedError("Strategy unknown")
        
    strategy.on_fit_config_fn = strategy.fit_config_fn

    return strategy
