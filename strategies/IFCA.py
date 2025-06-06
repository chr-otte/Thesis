from standardized_metric_tracking import metric_tracker
from strategies.strategy import logger
from experiment_scenarios.experiment import Experiment
import logging
logger = logging.getLogger("FL-Strategy")
from typing import Optional, Dict, List, Tuple, Union
from flwr.common import Scalar, EvaluateRes, FitRes, Parameters, EvaluateIns
import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from strategies.Clustering_strategies.Graph_based import split_cluster_by_graph
from strategies.Clustering_strategies.Bi_partioning import bi_partion
from network_lstm import ForecastingNetworkLSTMBase
import numpy as np
import io
import json
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

class IfcaClustering(fl.server.strategy.FedAvg):
    """
    Implementation of the Iterative Federated Clustering Algorithm (IFCA).
    
    This strategy maintains multiple concurrent models, each representing a cluster.
    Clients evaluate all models and select the best one based on their local loss.
    """
    def __init__(
        self,
        experiment_scenario:Experiment,
        config,
        num_clusters: int = 4,
        *args,
        **kwargs
    ):
        """
        Initialize the IFCA clustering strategy.
        
        Args:
            num_clusters: Number of clusters to maintain
            *args, **kwargs: Arguments for FedAvg base class
        """
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.experiment_scenario = experiment_scenario
        self.cluster_models = {}  # Dictionary to store model parameters for each cluster
        self.client_cluster_assignments = {}  # Maps client_id to cluster_id
        self.initial_parameters = None  # Will store initial parameters for resetting clusters
        self.early_termination = False
        self.client_parameter_init_results = {}
        self.warmup_rounds = config["WARMUP_ROUNDS"]
        self.effective_rounds = config["EFFECTIVE_ROUNDS"]
        self.current_effective_round = 0
        self.config = config
        self.client_gradients = {}
        self.client_results_map = {}  

    def create_cluster_parameters(self, parameters: Parameters) -> Optional[Parameters]:
        """
        Initialize parameters for all cluster models.
        """
        
        # Store initial parameters for potential reset
        self.initial_parameters = parameters
        
        # Extract client IDs and gradient vectors
        gradient_vectors = []
        client_ids = []
        client_results_map = {}  # Map client_id to (client, fit_res) tuple
        
        for client, fit_res in self.client_results_map.values():
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            
            if "update_direction" in fit_res.metrics:
                try:
                    gradient_str = fit_res.metrics["update_direction"]
                    gradient_vector = np.array(json.loads(gradient_str))
                    gradient_vectors.append(gradient_vector)
                    client_ids.append(client_id)
                    client_results_map[client_id] = (client, fit_res)
                except Exception as e:
                    logger.error(f"Error processing gradient for client {client_id}: {e}")
        
        # Handle case with too few clients
        if len(gradient_vectors) < self.num_clusters:
            logger.warning(f"Not enough clients ({len(gradient_vectors)}) for {self.num_clusters} clusters. Using random initialization.")
            self.create_cluster_parameters(parameters)
            return
        
        # Create gradient matrix for clustering
        gradient_matrix = np.array(gradient_vectors)
        
        # Cluster gradient directions using K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(gradient_matrix)
        cluster_labels = kmeans.labels_
        
        # Group client results by cluster
        cluster_results = {i: [] for i in range(self.num_clusters)}
        
        for i, client_id in enumerate(client_ids):
            cluster_id = cluster_labels[i]
            if client_id in client_results_map:
                cluster_results[cluster_id].append(client_results_map[client_id])
        
        # Log cluster information
        for cluster_id, client_list in cluster_results.items():
            client_id_list = [self.experiment_scenario.get_client_id(client.partition_id) for client, _ in client_list]
            logger.info(f"Initial cluster {cluster_id}: {len(client_list)} clients - {client_id_list}")
        
        # Aggregate parameters for each cluster using FedAvg
        for cluster_id, cluster_client_results in cluster_results.items():
            if cluster_client_results:
                # Use FedAvg to aggregate parameters for this cluster
                cluster_params, _ = super().aggregate_fit(
                    0,  # Dummy round number
                    cluster_client_results, 
                    []  # No failures
                )
                
                if cluster_params:
                    self.cluster_models[cluster_id] = cluster_params
                    logger.info(f"Created specialized model for cluster {cluster_id} from {len(cluster_client_results)} clients")
                else:
                    # Fallback to base parameters if aggregation fails
                    self.cluster_models[cluster_id] = parameters
                    logger.warning(f"Aggregation failed for cluster {cluster_id}, using base parameters")
            else:
                # Empty cluster - use base parameters
                self.cluster_models[cluster_id] = parameters
                logger.warning(f"Empty cluster {cluster_id}, using base parameters")
            
            logger.info(f"Initialized {self.num_clusters} cluster models by aggregating client updates")

        logger.info(f"Initialized {self.num_clusters} cluster models with different random perturbations")
        return parameters
    
    def calculate_parameter_distance(self, params1, params2):
        """Calculate L2 distance between parameters"""
        arr1 = parameters_to_ndarrays(params1)
        arr2 = parameters_to_ndarrays(params2)
        total_dist = 0.0
        for a1, a2 in zip(arr1, arr2):
            total_dist += np.sum((a1 - a2)**2)
        return np.sqrt(total_dist)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """
        Configure clients for training, assigning each client to evaluate one cluster per round.
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
        

        # Initialize cluster models in the first round
        if server_round == self.warmup_rounds + 1:
            self.create_cluster_parameters(parameters)
            for client in clients:
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                self.client_parameter_init_results[client_id] = {}


        cluster_to_evaluate = server_round - self.warmup_rounds - 1

        config = {
                "server_round": server_round,
                "local_epochs": self.on_fit_config_fn(server_round)["local_epochs"],
                "evaluating_cluster": cluster_to_evaluate,
                "total_clusters": self.num_clusters,
            }
        if server_round <= self.warmup_rounds: 
            for client in clients: 
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                metric_tracker.log_pretraining_round(server_round, client_id, -1) # global model
                client_instructions.append((client,fl.common.FitIns(parameters, config)))
            return client_instructions


        # Assign all clients to evaluate a single cluster in each of the first num_clusters rounds
        if server_round <= self.warmup_rounds + self.num_clusters:
            cluster_parameters = self.cluster_models[cluster_to_evaluate]

            for client in clients:
                # Send this client the parameters for their assigned cluster
                client_instructions.append((client, fl.common.FitIns(cluster_parameters, config)))
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                metric_tracker.log_pretraining_round(server_round, client_id, cluster_to_evaluate) # testing all models 
        else: 
            for client in clients: 
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                model_id = self.client_cluster_assignments[client_id]
                metric_tracker.log_pretraining_round(server_round, client_id, model_id) # preferred model 

                client_instructions.append((client, fl.common.FitIns(self.cluster_models[model_id], config)))
        return client_instructions


    def aggregate_fit(
        self, 
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients for each cluster separately.
        """

        if not results or self.early_termination:
            return None, {}

        if server_round <= self.warmup_rounds: 
            logger.info(f"WARMUP: Updated parameters for global cluster with {len(results)} clients")
            for client, fit_res in results:
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                
                # Extract gradient info from metrics if it exists
                if "update_direction" in fit_res.metrics:
                    gradient_str = fit_res.metrics["update_direction"]
                    gradient_vector = np.array(json.loads(gradient_str))
                    self.client_gradients[client_id] = gradient_vector
                    self.client_results_map[client_id] = (client, fit_res)
            return None, {}

        if server_round <= self.warmup_rounds + self.num_clusters: 
            logger.info(f"Round {server_round}/{self.num_clusters} initialization: No aggregation yet")
            return None, {}


        # Group results by assigned clusters
        cluster_results = {i: [] for i in range(self.num_clusters)}
        clients_per_cluster = {i: 0 for i in range(self.num_clusters)}
        
        # Group client results by their assigned cluster
        for client, fit_res in results:
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            
            # Get the client's assigned cluster from our mapping
            if client_id in self.client_cluster_assignments:
                assigned_cluster = self.client_cluster_assignments[client_id]
                cluster_results[assigned_cluster].append((client, fit_res))
                clients_per_cluster[assigned_cluster] += 1
                logger.info(f"Client {client_id} contributing to cluster {assigned_cluster}")
        
        # Aggregate models for each cluster separately
        for cluster_id, cluster_client_results in cluster_results.items():
            if cluster_client_results:
                # Use the parent class's aggregate_fit to get parameters for this cluster
                cluster_params, _ = super().aggregate_fit(
                    server_round, 
                    cluster_client_results, 
                    failures
                )
                
                # Update the cluster model
                if cluster_params:
                    self.cluster_models[cluster_id] = cluster_params
                    logger.info(f"Updated parameters for cluster {cluster_id} with {len(cluster_client_results)} clients")
                cluster_params = None

        # Return basic metrics about cluster distribution
        return self.cluster_models.get(0), {"clients_per_cluster": clients_per_cluster}


    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure evaluation with clients' preferred clusters."""
        if self.early_termination: 
            return []
        
        clients = client_manager.sample(
            num_clients=max(int(client_manager.num_available() * self.fraction_evaluate), 1),
            min_num_clients=self.min_evaluate_clients,
        )
        if not clients:
            return []

        # Prepare evaluation instructions using clients' known cluster assignments
        client_instructions = []
        for i, client in enumerate(clients):
            if server_round <= self.warmup_rounds: 
                cluster_parameters = parameters
            elif server_round <= self.num_clusters + self.warmup_rounds:
                cluster_to_evaluate = server_round-self.warmup_rounds-1 
                cluster_parameters = list(self.cluster_models.values())[cluster_to_evaluate]
            else: 
                client_id = client.partition_id
                cluster_parameters = self.cluster_models[self.client_cluster_assignments[self.experiment_scenario.get_client_id(client_id)]]
        
            config = {
                "server_round": server_round,
                "evaluating_cluster": i
            }
            
            client_instructions.append((client, fl.common.EvaluateIns(cluster_parameters, config)))
                
        return client_instructions
            

    # Extract client metrics and IDs consistently
    def get_client_metrics(self, client, eval_res):
        try:
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
        except:
            client_id = eval_res.metrics.get("Id", str(client.partition_id))
        
        return {
            "id": client_id,
            "loss": eval_res.loss,
            "accuracy": eval_res.metrics.get("accuracy", 0.0)
        }
    
    # Calculate aggregate metrics for a list of client metrics
    def calculate_aggregates(self, metrics_list):
        losses = [m["loss"] for m in metrics_list]
        accuracies = [m["accuracy"] for m in metrics_list]
        
        return {
            "avg_loss": sum(losses) / len(losses) if losses else float("inf"),
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0
        }
    

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results and update metrics for each cluster.
        
        Args:
            server_round: Current round of federated learning
            results: List of (client, evaluate_res) tuples
            failures: List of failed clients
            
        Returns:
            Tuple of (loss, metrics)
        """
        if not results or self.early_termination:
            return None, {}

        # Process results to get client metrics
        if server_round <= self.warmup_rounds: 
            for client, eval_res in results:
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                metric_tracker.log_training_round(server_round, client_id, eval_res.loss)
            return None, {}

        if server_round <= self.warmup_rounds + self.num_clusters:
            cluster_to_evaluate = server_round - self.warmup_rounds - 1
            for client, eval_res in results: 
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                self.client_parameter_init_results[client_id][cluster_to_evaluate] = eval_res
                metric_tracker.log_training_round(server_round, client_id, eval_res.loss)

        # After evaluating all clusters, assign each client to its best cluster
        if server_round == self.warmup_rounds + self.num_clusters:
            # Assign each client to its best performing cluster
            for client_id, cluster_results in self.client_parameter_init_results.items():
                # Find cluster with minimum loss
                best_cluster = min(cluster_results.items(), key=lambda x: x[1].loss)[0]
                self.client_cluster_assignments[client_id] = best_cluster
                logger.info(f"Assigned client {client_id} to cluster {best_cluster} based on evaluation performance")
        
        metric_tracker.log_cluster_state(server_round, self.client_cluster_assignments)            
      
        if server_round >= self.num_clusters + self.warmup_rounds: 
            for client, eval_res in results: 
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                metric_tracker.log_training_round(server_round, client_id, eval_res.loss)

            self.current_effective_round = self.current_effective_round + 1 
            if self.current_effective_round >= self.effective_rounds: 
                self.early_termination = True
        
        return None, {}
        
    def fit_config_fn(self, server_round: int) -> Dict[str, Scalar]:
        """
        Return training configuration for clients.
        
        Args:
            server_round: Current round of federated learning
            
        Returns:
            Configuration dictionary
        """
        config = {"local_epochs": 1, "server_round": server_round}

        return config
    