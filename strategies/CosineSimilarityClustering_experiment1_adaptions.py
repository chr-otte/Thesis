from standardized_metric_tracking import metric_tracker
from strategies.strategy import logger
import logging
logger = logging.getLogger("FL-Strategy")
from typing import Optional
from flwr.common import Scalar
import flwr as fl
from flwr.common import EvaluateRes, FitRes, Parameters, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from strategies.Clustering_strategies.Graph_based import split_cluster_by_graph
from strategies.Clustering_strategies.Bi_partioning import bi_partion
from typing import Dict, List, Tuple, Union
import numpy as np
import io

class CosineSimilarityClustering(fl.server.strategy.FedAvg):
    """Strategy that extends FedAvg to track client metrics with internal round management."""

    def __init__(self, experiment_scenario, clustering_strategy, similarity_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # clusterid to models 
        self.clusters = { }
        self.clients_clusters = { }
        self.experiment_scenario = experiment_scenario

        # clusterid to clientids
        self.cluster_clients = { }

        self.acceptable_similarity_threshold = similarity_threshold 
        self.base_similarity_threshold = similarity_threshold

        self.early_termination = False 

        # Track training progress for each cluster
        self.cluster_stable_rounds = {}   # Maps cluster_id to consecutive stable rounds

        self.max_rounds_per_cluster = 10  # Minimum rounds to train each cluster

        self.stable_rounds_for_convergence = 90  # How many stable rounds to consider converged
        self.early_terminate = False # Flag to indicate when training is complete
              
        self.initial_parameters = None 
        
        self.clustering_strategy = clustering_strategy
        
    def get_clusters(self, similarity_matrix, clients): 
        match self.clustering_strategy: 
            case "BI_PARTITION": 
                return bi_partion(similarity_matrix, clients, self.acceptable_similarity_threshold)
            case "GRAPH_BASED": 
                return split_cluster_by_graph(similarity_matrix, clients, self.acceptable_similarity_threshold)
            case _: 
                raise Exception(f"UNKNOWN CLUSTERING STRATEGY: {self.clustering_strategy}")
        

    def fit_config_fn(self, effective_round: int):
        """Generate fit config using the logical round number."""
        return {
            "server_round": effective_round,
            "total_rounds": self.total_rounds_since_start
        }




    def configure_fit(self, server_round, parameters, client_manager:ClientManager):
        """Configure clients for training with their appropriate cluster model."""
        if self.early_termination: 
            return []
        
        if server_round == 1: 
            self.initial_parameters = parameters
            self.clusters[0] = self.initial_parameters

        # Get all available clients for this round
        client_instructions = []
        clients = client_manager.sample(
            num_clients=max(int(client_manager.num_available() * self.fraction_evaluate), 1),
            min_num_clients=self.min_evaluate_clients,
        )
        for client in clients: 
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            if client_id in self.clients_clusters.keys(): 
                cluster_id = self.clients_clusters[client_id]
                
                if self.cluster_stable_rounds[cluster_id] == self.max_rounds_per_cluster: # no need for further training
                    continue

                config = {
                        "server_round": server_round,
                        "evaluating_cluster": cluster_id
                    }
                metric_tracker.log_pretraining_round(server_round, client_id, cluster_id)
                client_instructions.append((client, fl.common.FitIns( self.clusters[cluster_id], config)))
            else:
                config = {
                        "server_round": server_round,
                        "evaluating_cluster": -1
                    }
                client_instructions.append((client, fl.common.FitIns(parameters, config)))
                metric_tracker.log_pretraining_round(server_round, client_id, -1)
        return client_instructions


    def aggregate_fit(self, server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and update models for each cluster with internal round tracking."""

        if server_round == 1: 
            for client, fit_res in results: 
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                self.clients_clusters[client_id] = 0
                self.cluster_stable_rounds[0] = 0

        # grouping clients pr cluster; 
        cluster_FitRes_dic = {}
        cluster_clientId_dic = {} 
        for client, fit_res in results: 
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            if client_id in self.clients_clusters:
                cluster_id = self.clients_clusters[client_id]
                if cluster_id not in cluster_FitRes_dic: 
                    cluster_FitRes_dic[cluster_id] = []
                if cluster_id not in cluster_clientId_dic: 
                    cluster_clientId_dic[cluster_id] = []
                cluster_FitRes_dic[cluster_id].append((client, fit_res))
                cluster_clientId_dic[cluster_id].append(client_id)

        pre_split_cluster_keys = list(self.clusters.keys()).copy()        
        for cluster in pre_split_cluster_keys:
            # Skip if no clients in this cluster
            if cluster not in cluster_FitRes_dic:
                continue
                
            client_fit_results = cluster_FitRes_dic[cluster]
            

            lr_scaling_factor = (len(cluster_FitRes_dic[cluster]) / len(self.clients_clusters) * 1.2)
        
            # Apply the scaling to the current learning rate
            adjusted_lr = self.base_similarity_threshold + self.base_similarity_threshold * lr_scaling_factor

            self.acceptable_similarity_threshold = adjusted_lr # max(0.1, self.base_similarity_threshold * (1.05 ** server_round))

            # If only one client in cluster, no need for similarity calculation
            if len(client_fit_results) <= 1:
                fit_res_to_agg = []
                for client, fit_res in client_fit_results:
                    fit_res_to_agg.append((client, fit_res))
                
                if fit_res_to_agg:  # Only aggregate if we have results
                    cluster_params, _ = super().aggregate_fit(server_round, fit_res_to_agg, failures)
                    self.clusters[cluster] = cluster_params
                    self.cluster_stable_rounds[cluster] += 1
                continue
            
            # For multiple clients, calculate similarity and potentially split
            num_clients = len(client_fit_results)
            matrix = np.zeros((num_clients, num_clients))
            update_vectors = []
            update_vector_ids = []
            
            # Collect update vectors from all clients in this cluster
            for client_idx, (client, fit_res) in enumerate(client_fit_results):
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                update_vector_ids.append(client_id)
                
                # Get the client's updated parameters
                client_params = fit_res.parameters.tensors
                
                # Get the initial parameters for this cluster
                initial_params = self.clusters[cluster].tensors if self.clusters[cluster] else None
                
                # If we have initial parameters, compute the difference
                if initial_params:
                    # Collect all parameter differences
                    all_diffs = []
                    
                    for i, tensor in enumerate(client_params):
                        # Deserialize client parameters
                        client_np = np.load(io.BytesIO(tensor)).astype(np.float64)
                        # Deserialize initial parameters
                        initial_np = np.load(io.BytesIO(initial_params[i])).astype(np.float64)
                        # Compute difference (update direction)
                        diff = client_np - initial_np
                        all_diffs.append(diff.flatten())

                    # Concatenate all differences into a single vector
                    update_vector = np.concatenate(all_diffs)
                    
                    # Normalize the entire vector at once (not per layer)
                    norm = np.linalg.norm(update_vector)
                    if norm > 1e-10:  # Only normalize if not too close to zero
                        update_vector = update_vector / norm
                else:
                    # If no initial params, just use the client params (first round)
                    deserialized_tensors = []
                    for tensor in client_params:
                        np_array = np.frombuffer(tensor, dtype=np.float32)
                        deserialized_tensors.append(np_array)
                    update_vector = np.concatenate([tensor.flatten() for tensor in deserialized_tensors])
                
                update_vectors.append(update_vector)

            # Calculate pairwise cosine similarity
            for i in range(num_clients):
                for j in range(num_clients):
                    if i != j:
                        vec_i = update_vectors[i]
                        vec_j = update_vectors[j]
                        
                        # Cast to higher precision for numerical stability
                        vec_i = vec_i.astype(np.float64)
                        vec_j = vec_j.astype(np.float64)
                        
                        # Since we've already normalized the vectors, cosine similarity is just the dot product
                        similarity = np.dot(vec_i, vec_j)
                        
                        # Ensure value is in valid range due to potential floating point errors
                        similarity = np.clip(similarity, -1.0, 1.0)

                        matrix[i][j] = similarity
                    else:
                        matrix[i][j] = 1  # Self-similarity is 1

            # Determine if splitting is needed
            should_split = self.cluster_stable_rounds[cluster] < self.stable_rounds_for_convergence
            metric_tracker.log_cluster_similarity_matrix(server_round, matrix, update_vector_ids, cluster)
            metric_tracker.save_metrics()
            new_clusters = self.get_clusters(matrix, update_vector_ids) if should_split else []
            
            if len(new_clusters) > 1:
                # Cluster splitting logic
                cluster_client_list_mapping = {}
                cluster_client_list_mapping[cluster] = new_clusters[0]

                for i in range(len(new_clusters)):
                    if i == 0:
                        self.clusters[cluster] = self.initial_parameters
                        self.cluster_stable_rounds[cluster] = 0
                    else:
                        new_cluster_id = max(self.clusters.keys()) + 1
                        self.clusters[new_cluster_id] = self.initial_parameters
                        cluster_client_list_mapping[new_cluster_id] = new_clusters[i]
                        self.cluster_stable_rounds[new_cluster_id] = 0

                # Reassign clients to their new clusters
                for cluster_id, client_list in cluster_client_list_mapping.items():
                    for client in client_list:
                        self.clients_clusters[client] = cluster_id
            else:
                # No splitting needed, just aggregate normally
                fit_res_to_agg = []
                for client, fit_res in client_fit_results:
                    fit_res_to_agg.append((client, fit_res))
                
                if fit_res_to_agg:
                    cluster_params, _ = super().aggregate_fit(server_round, fit_res_to_agg, failures)
                    self.clusters[cluster] = cluster_params
                    self.cluster_stable_rounds[cluster] += 1

        # Log current cluster state
        metric_tracker.log_cluster_state(server_round, self.clients_clusters)
        return None, {}


    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation with cluster-specific parameters."""
        if self.early_termination: 
            return []

        # Get clients for this round of evaluation
        clients = client_manager.all().values()
        
        client_instructions = []

        for client in clients:
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            cluster_id = self.clients_clusters[client_id]
            config = {}

            # Create evaluation instruction with the correct cluster parameters
            evaluate_ins = fl.common.EvaluateIns(self.clusters[cluster_id], config)
            client_instructions.append((client, evaluate_ins))
        
        return client_instructions
            
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and update metrics for each cluster."""
        self.client_trained_metrics = {}  # Reset client metrics for this round

        for client, eval_res in results: 
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            metric_tracker.log_training_round(server_round, client_id, eval_res.loss)

        self.early_termination = True
        for cluster, num_stable_rounds in self.cluster_stable_rounds.items(): 
            if num_stable_rounds < self.max_rounds_per_cluster: 
                self.early_termination = False
                break

        return None, {}
    