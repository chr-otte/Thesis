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

    def __init__(self, model, model_type, num_layers, total_size, epochs, experiment_scenario, clustering_strategy, similarity_threshold, layer_indicies, key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # clusterid to models 
        self.clusters = { }
        self.clients_clusters = { }
        self.model = model
        self.model_type = model_type
        self.num_layers = num_layers
        self.total_size = total_size
        self.local_epochs = epochs
        self.experiment_scenario = experiment_scenario
        self.layer_indicies = layer_indicies    
        self.key = key

        self.initEpochs = 20
        self.stableEpochs = 1

        # clusterid to clientids
        self.cluster_clients = { }

        self.acceptable_similarity_threshold = similarity_threshold 
        #self.base_similarity_threshold = similarity_threshold

        self.early_termination = False 

        # Track training progress for each cluster
        self.cluster_stable_rounds = {}   # Maps cluster_id to consecutive stable rounds

        self.max_rounds_per_cluster = 6  # Minimum rounds to train each cluster

        self.stable_rounds_for_convergence = 3  # How many stable rounds to consider converged
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
                        "evaluating_cluster": cluster_id,
                        "epochs" : self.initEpochs if self.cluster_stable_rounds[cluster_id] == 0 else self.stableEpochs,
                    }
                metric_tracker.log_pretraining_round(server_round, client_id, cluster_id)
                client_instructions.append((client, fl.common.FitIns( self.clusters[cluster_id], config)))
            else:
                config = {
                        "server_round": server_round,
                        "evaluating_cluster": -1,
                        "epochs" : self.initEpochs,
                    }
                client_instructions.append((client, fl.common.FitIns(parameters, config)))
                metric_tracker.log_pretraining_round(server_round, client_id, -1)
        return client_instructions

    def cluster_base_model(self,
        client_fit_results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        cluster_id: int
    ) -> Tuple[np.ndarray, List[str]]:
        num_clients = len(client_fit_results)
        matrix = np.zeros((num_clients, num_clients))
        update_vectors_flat = []
        update_vector_ids = []

        for client_idx, (client, fit_res) in enumerate(client_fit_results):
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            update_vector_ids.append(client_id)
            client_params = fit_res.parameters.tensors
            initial_params = self.clusters[cluster_id].tensors if self.clusters[cluster_id] else None

            if initial_params:
                all_diffs = []
                for i, tensor in enumerate(client_params):
                    client_np = np.load(io.BytesIO(tensor)).astype(np.float64)
                    initial_np = np.load(io.BytesIO(initial_params[i])).astype(np.float64)
                    diff = client_np - initial_np
                    all_diffs.append(diff.flatten())
                update_vector = np.concatenate(all_diffs)
                norm = np.linalg.norm(update_vector)
                if norm > 1e-10:
                    update_vector = update_vector / norm
            else:
                deserialized = [np.frombuffer(t, dtype=np.float32).flatten() for t in client_params]
                update_vector = np.concatenate(deserialized)

            update_vectors_flat.append(update_vector)

        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                vec_i = update_vectors_flat[i].astype(np.float64)
                vec_j = update_vectors_flat[j].astype(np.float64)
                similarity = np.dot(vec_i, vec_j)
                similarity = np.clip(similarity, -1.0, 1.0)
                matrix[i][j] = similarity

        return matrix, update_vector_ids

    def cluster_layered_model(self,
        client_fit_results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        cluster_id: int,
        model_type: str = "LSTM",  # "RNN", "LSTM", or "GRU"
        num_layers: int = 5,       # 3, 5, or 10
        layer_indices: List[int] = None, 
        compare_layers: bool = True,  # Whether to generate all combinations
        server_round: int = 0
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute client similarity matrix using specified layers from different model architectures.
        
        Args:
            client_fit_results: List of client results from federated learning
            cluster_id: ID of the current cluster
            model_type: Type of recurrent model ("RNN", "LSTM", "GRU")
            num_layers: Number of layers in the model (3, 5, or 10)
            layer_indices: Which specific layers to use (None = use default combinations)
            compare_layers: Whether to generate matrices for all layer combinations
                        
        Returns:
            Similarity matrix and client IDs
        """
        import json
        import os
        from datetime import datetime

        # Generate layer combinations based on the number of layers
        if num_layers == 1: 
            layer_combinations = [
                [0]  # Only one option with a single layer
            ]
        elif num_layers == 2:
            layer_combinations = [
                # Individual layers
                [0], [1],
                # All layers
                [0, 1]
            ]
        elif num_layers == 3:
            layer_combinations = [
                # Individual layers
                [0], [1], [2],
                # Sequential pairs
                [0, 1], [1, 2],
                # Beginning/end combinations
                [0, 2],
                # All layers
                [0, 1, 2]
            ]
        elif num_layers == 4:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3],
                # Sequential pairs
                [0, 1], [1, 2], [2, 3],
                # Beginning/end combinations
                [0, 3], [0, 1, 2, 3],
                # Middle layers
                [1, 2],
                # Region-based
                [0, 1], [2, 3],
                # All layers
                [0, 1, 2, 3]
            ]
        elif num_layers == 5:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4],
                # Sequential pairs
                [0, 1], [1, 2], [2, 3], [3, 4],
                # Beginning/end combinations
                [0, 4], [0, 1, 3, 4],
                # Middle layers
                [1, 2, 3],
                # All layers
                [0, 1, 2, 3, 4]
            ]
        elif num_layers == 6:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5],
                # Sequential pairs
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                # All layers
                [0, 1, 2, 3, 4, 5]
            ]
        elif num_layers == 7:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5], [6],
                # Sequential pairs from different regions
                [0, 1], [2, 3], [4, 5], [5, 6],
                # Beginning/middle/end combinations
                [0, 3, 6], [0, 6],
                # Region-based groups
                [0, 1, 2], [2, 3, 4], [4, 5, 6],
                # Quartets spanning different regions
                [0, 2, 4, 6], [1, 2, 4, 5],
                # Middle layers
                [2, 3, 4],
                # All layers
                [0, 1, 2, 3, 4, 5, 6]
            ]
        elif num_layers == 8:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5],[6], [7],
                # Sequential pairs
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],[5,6], [6,7],
                # All layers
                [0, 1, 2, 3, 4, 5, 6, 7]
            ]
        elif num_layers == 9:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5], [6], [7], [8],
                # Sequential pairs from different regions
                [0, 1], [2, 3], [4, 5], [5, 6], [6, 8],
                # Beginning/middle/end combinations
                [0, 4, 8], [0, 8],
                # Region-based groups
                [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8],
                # Quartets spanning different regions
                [0, 2, 4, 6, 8], [1, 2, 4, 5, 7],
                # Middle layers
                [2, 3, 4, 5],
                # All layers
                [0, 1, 2, 3, 4, 5, 6, 7, 8]
            ]
        elif num_layers == 10:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                # Sequential pairs in different regions
                [0, 1], [4, 5], [8, 9],
                # Early, middle, and late layers
                [1, 2], [4, 5, 6], [8, 9],
                # Beginning/end combinations
                [0, 9], [0, 1, 8, 9],
                # Middle layers
                [3, 4, 5, 6],
                # Different regions
                [0, 4, 9],
                # All layers
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            ]
        elif num_layers == 11:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                # Sequential pairs in different regions
                [0, 1], [5, 6], [9, 10],
                # Early, middle, and late layers
                [1, 2], [5, 6], [9, 10],
                # Beginning/end combinations
                [0, 10], [0, 1, 9, 10],
                # Middle layers
                [4, 5, 6, 7],
                # Different regions
                [0, 5, 10],
                # All layers
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            ]
        elif num_layers == 13:
            layer_combinations = [
                # Individual layers
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12],
                # Sequential pairs in different regions
                [0, 1], [6, 7], [11, 12],
                # Early, middle, and late layers
                [1, 2], [6, 7], [11, 12],
                # Beginning/end combinations
                [0, 12], [0, 1, 11, 12],
                # Middle layers
                [5, 6, 7, 8],
                # Different regions
                [0, 6, 12],
                # Region-based groups
                [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12],
                # All layers
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            ]
        else:
            raise ValueError(f"Unsupported number of layers: {num_layers}")

        # Override layer combinations if not comparing layers
        if not compare_layers:
            layer_combinations = [layer_indices] if layer_indices else [[i for i in range(num_layers)]]

        # Set default layers if none provided
        default = layer_indices.copy() if layer_indices else [i for i in range(num_layers)]
        results = {}

        for layer_combination in layer_combinations:
            current_layer_indices = layer_combination        
            num_clients = len(client_fit_results)
            matrix = np.zeros((num_clients, num_clients))
            update_vector_ids = []
            update_vectors_flat = []
            
            # Map neural network layer indices to actual tensor indices in the parameter list
            # This mapping might differ between RNN, LSTM, GRU
            if model_type == "LSTM":
                param_mapping = self._map_lstm_layers_to_params(num_layers)
            elif model_type == "RNN":
                param_mapping = self._map_rnn_layers_to_params(num_layers)
            elif model_type == "GRU":
                param_mapping = self._map_gru_layers_to_params(num_layers)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # If no specific layers requested, use all layers
            if current_layer_indices is None:
                # Use all tensors
                selected_param_indices = list(range(len(client_fit_results[0][1].parameters.tensors)))
            else:
                # Map layer indices to parameter indices
                selected_param_indices = []
                for layer_idx in current_layer_indices:
                    if layer_idx in param_mapping:
                        # Add all parameter tensors for this layer
                        selected_param_indices.extend(param_mapping[layer_idx])
            
            for client_idx, (client, fit_res) in enumerate(client_fit_results):
                client_id = self.experiment_scenario.get_client_id(client.partition_id)
                update_vector_ids.append(client_id)
                client_params = fit_res.parameters.tensors
                initial_params = self.clusters[cluster_id].tensors if self.clusters[cluster_id] else None
                
                if initial_params:
                    # Calculate differences for selected parameters only
                    selected_diffs = []
                    for param_idx in selected_param_indices:
                        if param_idx < len(client_params):
                            client_np = np.load(io.BytesIO(client_params[param_idx])).astype(np.float64)
                            initial_np = np.load(io.BytesIO(initial_params[param_idx])).astype(np.float64)
                            diff = client_np - initial_np
                            selected_diffs.append(diff.flatten())
                    
                    # Concatenate selected differences
                    if selected_diffs:
                        update_vector = np.concatenate(selected_diffs)
                        # Normalize the update vector
                        norm = np.linalg.norm(update_vector)
                        if norm > 1e-10:
                            update_vector = update_vector / norm
                        update_vectors_flat.append(update_vector)
                    else:
                        # Fallback if no valid parameters were selected
                        update_vectors_flat.append(np.array([]))
                else:
                    # For initial round, just use the parameters directly
                    selected_params = [np.frombuffer(client_params[i], dtype=np.float32).flatten() 
                                    for i in selected_param_indices if i < len(client_params)]
                    if selected_params:
                        update_vector = np.concatenate(selected_params)
                        update_vectors_flat.append(update_vector)
                    else:
                        update_vectors_flat.append(np.array([]))
            
            # Compute similarity matrix
            for i in range(num_clients):
                for j in range(num_clients):
                    if i == j:
                        matrix[i][j] = 1.0  # Self-similarity is always 1
                        continue
                        
                    vec_i = update_vectors_flat[i]
                    vec_j = update_vectors_flat[j]
                    
                    # Handle empty vectors (shouldn't happen with proper layer selection)
                    if vec_i.size == 0 or vec_j.size == 0:
                        matrix[i][j] = 0.0
                        continue
                        
                    similarity = np.dot(vec_i, vec_j)
                    similarity = np.clip(similarity, -1.0, 1.0)
                    matrix[i][j] = similarity
                    
            # Store results for this layer combination
            layer_key = "_".join(map(str, layer_combination))
            results[layer_key] = {
                "matrix": matrix.tolist(),  # Convert numpy array to list
                "client_ids": update_vector_ids
            }

        # If we're comparing layers, save all results to a JSON file
        if compare_layers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "results"+"/"
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"key_{self.key}_layers_similarity_ServerRound_{server_round}_ClusterId_{cluster_id}_{timestamp}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f)
            
            print(f"Saved {model_type} {num_layers}-layer similarity matrices to {output_file}")
        
        # Return the requested matrix for the default layer_indices
        default_key = "_".join(map(str, default))
        return np.array(results[default_key]["matrix"]), results[default_key]["client_ids"]

    def _map_lstm_layers_to_params(self, num_layers):
        """
        Maps LSTM layer indices to their corresponding parameter indices for models with varying depths.
        
        Args:
            num_layers: Number of LSTM layers in the model (3, 5, or 10)
            
        Returns:
            A dictionary mapping layer indices to lists of parameter indices
        
        Notes:
            - Each LSTM layer has 4 parameter tensors: weight_ih, weight_hh, bias_ih, bias_hh
            - The final FC layer has 2 parameter tensors: weight, bias (not included in layer mappings)
        """
        layer_to_params = {}
        
        # Each LSTM layer has 4 parameter tensors
        params_per_layer = 4
        
        for i in range(num_layers):
            start_idx = i * params_per_layer
            layer_to_params[i] = list(range(start_idx, start_idx + params_per_layer))
        
        return layer_to_params

    def _map_rnn_layers_to_params(self, num_layers):
        """
        Maps RNN layer indices to their corresponding parameter indices for models with varying depths.
        
        Args:
            num_layers: Number of RNN layers in the model (3, 5, or 10)
            
        Returns:
            A dictionary mapping layer indices to lists of parameter indices
            
        Notes:
            - Each RNN layer has 2 parameter tensors: weight_ih, weight_hh (plus optional biases)
            - The final FC layer has 2 parameter tensors: weight, bias (not included in layer mappings)
        """
        layer_to_params = {}
        
        # Each RNN layer typically has 2 parameter tensors (4 if separate biases)
        # Adjust this based on your RNN implementation (with/without bias)
        params_per_layer = 4  # weight_ih, weight_hh, bias_ih, bias_hh
        
        for i in range(num_layers):
            start_idx = i * params_per_layer
            layer_to_params[i] = list(range(start_idx, start_idx + params_per_layer))
        
        return layer_to_params

    def _map_gru_layers_to_params(self, num_layers):
        """
        Maps GRU layer indices to their corresponding parameter indices for models with varying depths.
        
        Args:
            num_layers: Number of GRU layers in the model (3, 5, or 10)
            
        Returns:
            A dictionary mapping layer indices to lists of parameter indices
            
        Notes:
            - Each GRU layer has 3 weight tensors for update, reset, and hidden gates
            (6 total with biases)
            - The final FC layer has 2 parameter tensors: weight, bias (not included in layer mappings)
        """
        layer_to_params = {}
        
        # Each GRU layer typically has 6 parameter tensors:
        # weight_ih, weight_hh, bias_ih, bias_hh (each containing gates)
        params_per_layer = 4  # weight_ih, weight_hh, bias_ih, bias_hh
        
        for i in range(num_layers):
            start_idx = i * params_per_layer
            layer_to_params[i] = list(range(start_idx, start_idx + params_per_layer))
        
        return layer_to_params

    def compute_similarity_matrix(
        self, 
        client_fit_results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        cluster_id: int,
        server_round: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Dispatch similarity matrix calculation based on current model setting."""
        if self.model == "LSTM Base":
            return self.cluster_base_model(client_fit_results, cluster_id)
        elif self.model == "LSTM Layered" or self.model == "GRU Layered" or self.model == "RNN Layered":
            return self.cluster_layered_model(client_fit_results, cluster_id, num_layers=self.num_layers, model_type=self.model_type, layer_indices=self.layer_indicies, compare_layers=True, server_round=server_round)
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

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

        # Group clients by cluster
        cluster_FitRes_dic = {}
        cluster_clientId_dic = {} 
        for client, fit_res in results: 
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            if client_id in self.clients_clusters:
                cluster_id = self.clients_clusters[client_id]
                cluster_FitRes_dic.setdefault(cluster_id, []).append((client, fit_res))
                cluster_clientId_dic.setdefault(cluster_id, []).append(client_id)

        pre_split_cluster_keys = list(self.clusters.keys()).copy()        
        for cluster in pre_split_cluster_keys:
            if cluster not in cluster_FitRes_dic:
                continue
                    
            client_fit_results = cluster_FitRes_dic[cluster]
            
            if len(client_fit_results) <= 1:
                fit_res_to_agg = client_fit_results
                if fit_res_to_agg:
                    cluster_params, _ = super().aggregate_fit(server_round, fit_res_to_agg, failures)
                    self.clusters[cluster] = cluster_params
                    self.cluster_stable_rounds[cluster] += 1
                continue

            should_split = self.cluster_stable_rounds[cluster] < self.stable_rounds_for_convergence

            matrix, update_vector_ids = self.compute_similarity_matrix(client_fit_results, cluster, server_round)
            metric_tracker.log_cluster_similarity_matrix(server_round, matrix, update_vector_ids, cluster)
            new_clusters = self.get_clusters(matrix, update_vector_ids) if should_split else []

            if len(new_clusters) > 1:
                # Create mapping for all clusters including the original one
                cluster_client_list_mapping = {cluster: new_clusters[0]}
                
                # Reset the original cluster's parameters and stable rounds
                self.clusters[cluster] = self.initial_parameters  # Reset original cluster
                self.cluster_stable_rounds[cluster] = 0  # Reset stability counter for original cluster
                
                # Create new clusters as before
                for i in range(1, len(new_clusters)):
                    new_cluster_id = max(self.clusters.keys()) + 1
                    self.clusters[new_cluster_id] = self.initial_parameters
                    self.cluster_stable_rounds[new_cluster_id] = 0
                    cluster_client_list_mapping[new_cluster_id] = new_clusters[i]

                # Update client-cluster mappings
                for cluster_id, client_list in cluster_client_list_mapping.items():
                    for client in client_list:
                        self.clients_clusters[client] = cluster_id
            else:
                fit_res_to_agg = client_fit_results
                if fit_res_to_agg:
                    cluster_params, _ = super().aggregate_fit(server_round, fit_res_to_agg, failures)
                    self.clusters[cluster] = cluster_params
                    self.cluster_stable_rounds[cluster] += 1

        metric_tracker.log_cluster_state(server_round, self.clients_clusters)
        metric_tracker.save_metrics()
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
    