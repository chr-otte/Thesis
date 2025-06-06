import torch
import logging
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from strategies.IFCA import IfcaClustering
import numpy as np 

logger = logging.getLogger("FL-Strategy")

class IfcaClusteringWithPretraining(IfcaClustering):
    """
    Extension of IFCA that supports initializing from a pre-trained model.
    """
    def __init__(
        self,
        experiment_scenario,
        config,
        pretrained_model_path=None,
        *args,
        **kwargs
    ):
        """
        Initialize the IFCA clustering strategy with option to use a pre-trained model.
        
        Args:
            pretrained_model_path: Path to pre-trained model file (optional)
            *args, **kwargs: Arguments for IFCA base class
        """
        super().__init__(
            experiment_scenario=experiment_scenario,
            config=config,
            *args, 
            **kwargs
        )
        self.pretrained_model_path = config.get("PRETRAINED_MODEL_PATH", None)
        
    def create_cluster_parameters(self, parameters):
        """
        Initialize parameters for all cluster models, potentially from a pre-trained model.
        """
        # If we have a pre-trained model path, load parameters from there
        if self.pretrained_model_path:
            try:
                logger.info(f"Loading pre-trained model from {self.pretrained_model_path}")
                checkpoint = torch.load(self.pretrained_model_path)
                ndarrays = checkpoint['model_state_ndarrays']
                
                # Convert to Flower parameters
                self.initial_parameters = ndarrays_to_parameters(ndarrays)
                parameters = self.initial_parameters
                
                logger.info("Successfully loaded pre-trained model parameters")
            except Exception as e:
                logger.error(f"Error loading pre-trained model: {e}")
                # Fall back to original parameters if loading fails
                self.initial_parameters = parameters
        else:
            # Store initial parameters for potential reset
            self.initial_parameters = parameters
        
        # Initialize each cluster model
        for i in range(self.num_clusters):
            if i == 0:
                # For the first cluster, use the parameters as-is
                self.cluster_models[i] = self.initial_parameters
            else:
                # Create perturbed versions for other clusters
                ndarrays = parameters_to_ndarrays(self.initial_parameters)
                
                # Apply perturbation to each parameter tensor
                perturbed_ndarrays = []
                for arr in ndarrays:
                    # Create a copy to avoid modifying the original
                    arr_copy = arr.copy()
                    
                    # For LSTM layers, use smaller perturbation
                    if len(arr.shape) == 2 and arr.shape[0] >= 256:  # LSTM weights
                        scale = self.config["LSTM_SCALE"]
                    else:
                        scale = self.config["FC_SCALE"]
                    
                    # Generate and apply perturbation
                    perturbation = np.random.normal(0, scale, size=arr_copy.shape)
                    arr_copy += perturbation
                    perturbed_ndarrays.append(arr_copy)
                
                # Convert back to Parameters
                perturbed_params = ndarrays_to_parameters(perturbed_ndarrays)
                
                # Store the perturbed parameters for this cluster
                self.cluster_models[i] = perturbed_params
        
        logger.info(f"Initialized {self.num_clusters} cluster models with different random perturbations")
        return parameters
        
    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure clients for training, with modified behavior for pre-trained models.
        """
        if self.early_termination:
            return []
        
        clients = client_manager.sample(
            num_clients=max(int(client_manager.num_available() * self.fraction_fit), 1),
            min_num_clients=self.min_fit_clients,
        )
        if not clients:
            return []
            
        # Prepare instructions, assigning clients to different clusters in a round-robin fashion
        client_instructions = []
        
        # When using pre-trained model, we skip the warmup rounds
        if self.pretrained_model_path and server_round == 1:
            # Create cluster parameters from the pre-trained model
            self.create_cluster_parameters(parameters)
            
            # In first round with pre-trained model, immediately start cluster evaluation
            cluster_to_evaluate = 0
            
            config = {
                "server_round": server_round,
                "local_epochs": self.on_fit_config_fn(server_round)["local_epochs"],
                "evaluating_cluster": cluster_to_evaluate,
                "total_clusters": self.num_clusters,
                "using_pretrained": True
            }
            
            # Assign all clients to evaluate first cluster
            cluster_parameters = self.cluster_models[cluster_to_evaluate]
            for client in clients:
                client_instructions.append((client, FitIns(cluster_parameters, config)))
            
            return client_instructions
        
        # For rounds after initialization, follow the original logic but adapted for pretrained case
        effective_round = server_round
        if self.pretrained_model_path:
            # Adjust round count to skip warmup when using pretrained model
            cluster_to_evaluate = (server_round - 1) % self.num_clusters
        else:
            # Regular IFCA logic for non-pretrained case
            if server_round <= self.warmup_rounds:
                for client in clients:
                    client_instructions.append((client, FitIns(parameters, {
                        "server_round": server_round,
                        "local_epochs": self.on_fit_config_fn(server_round)["local_epochs"]
                    })))
                return client_instructions
                
            cluster_to_evaluate = server_round - self.warmup_rounds - 1
        
        config = {
            "server_round": server_round,
            "local_epochs": self.on_fit_config_fn(server_round)["local_epochs"],
            "evaluating_cluster": cluster_to_evaluate,
            "total_clusters": self.num_clusters,
        }
        
        # For rounds within the cluster evaluation phase
        if (not self.pretrained_model_path and server_round <= self.warmup_rounds + self.num_clusters) or \
           (self.pretrained_model_path and server_round <= self.num_clusters):
            # Get parameters for the current cluster being evaluated
            cluster_parameters = self.cluster_models[cluster_to_evaluate]
            
            for client in clients:
                client_instructions.append((client, FitIns(cluster_parameters, config)))
        else:
            # Regular training with assigned clusters
            for client in clients:
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                
                if client_id in self.client_cluster_assignments:
                    model_id = self.client_cluster_assignments[client_id]
                    client_instructions.append((client, FitIns(
                        self.cluster_models[model_id], 
                        config)))
                else:
                    # Default to cluster 0 if not assigned yet
                    client_instructions.append((client, FitIns(
                        self.cluster_models[0], 
                        config)))
                        
        return client_instructions