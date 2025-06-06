import logging
import flwr as fl
from typing import Dict, List, Tuple, Union, Optional, Any
from flwr.common import Scalar, FitRes, Parameters, EvaluateRes, EvaluateIns, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
import numpy as np 
import json 
import torch 
from strategies.SRFCA_utility.one_shot import one_shot_clustering
from strategies.SRFCA_utility.distance_metrics import l2_distance, cosine_similarity
from standardized_metric_tracking import metric_tracker
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

class SrfcaClustering(fl.server.strategy.FedAvg):
    """
    Implementation of the Iterative Federated Clustering Algorithm (IFCA).
    
    This strategy maintains multiple concurrent models, each representing a cluster.
    Clients evaluate all models and select the best one based on their local loss.
    """
    def __init__(self, experiment_scenario, config, *args, **kwargs ):
        """
        Initialize the IFCA clustering strategy.
        
        Args:
            *args, **kwargs: Arguments for FedAvg base class
        """
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.experiment_scenario = experiment_scenario
        self.config = config
        
        # State tracking
        self.client_models = {}  # Maps client_id to cluster
        self.clusters = {}       # Maps cluster_id to list of client_ids
        self.cluster_models = {} # Maps cluster_id to model parameters
        
        # Parameters from config
        self.trimming_parameter = config.get("TRIMMING_PARAMETER", 0.2)  # β in paper
        self.threshold = config.get("SIMILARITY_THRESHOLD", 0.75)                    # λ in paper
        self.min_cluster_size = config.get("MIN_CLUSTER_SIZE", 2)        # t in paper
        
        # Choose distance function based on config
        self.distance_fn = cosine_similarity
            
        # Phase tracking
        self.current_phase = "ONE_SHOT"  # ONE_SHOT, REFINE, TRACK_MODELS, DONE
        self.track_model_step = 0
        self.track_models_performance = {}
        self.refine_step = 0
        self.max_refine_steps = config.get("MAX_REFINE_STEPS", 6)
        self.current_refine_step = 0 
        self.saved_initial_parameters = None
        self.merge_threshold = config.get("MERGE_THRESHOLD", 0.9)
        self.secondLayerComparison = config.get("FULL_MODEL_COMPARISON", False)
        
        if self.secondLayerComparison == True or self.secondLayerComparison == False:
            self.secondLayerComparisonInit = self.secondLayerComparison
            self.secondLayerComparisonMerge = self.secondLayerComparison
        elif self.secondLayerComparison == "FLEX":
            self.secondLayerComparisonInit = True
            self.secondLayerComparisonMerge = False


        # Client manager will be set in configure_fit
        self.client_manager = None



    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure clients for fitting based on the current phase."""
        self.current_round = server_round
        
        match self.current_phase:
            case "ONE_SHOT": 
                self.saved_initial_parameters = parameters_to_ndarrays(parameters)
                return self._configure_one_shot(server_round, parameters, client_manager)
            case "REFINE": 
                return self._configure_refine(server_round, client_manager)
            case _: 
                return []

    def _configure_one_shot(self, server_round : int,  parameters : Parameters, client_manager: ClientManager):
        """Creating FitIns with global model for all clients"""
        clients = client_manager.sample(
            num_clients=max(int(client_manager.num_available() * self.fraction_fit), 1),
            min_num_clients=self.min_fit_clients,
        )
        if not clients:
            return []
            
        client_instructions = []
        
        # Initialize cluster models in the first round
        config = {
                "server_round": server_round,
                "local_epochs": 20,
            }
        for client in clients: 
            client_instructions.append((client, fl.common.FitIns(parameters, config)))
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            metric_tracker.log_pretraining_round(server_round, client_id, -1)

        self.track_model_step = 0
        return client_instructions

    def _configure_refine(self, server_round : int, client_manager: ClientManager):
        """
            Creating FitIns with clients prefered model
        """
            
        clients = client_manager.sample(
            num_clients=max(int(client_manager.num_available() * self.fraction_fit), 1),
            min_num_clients=self.min_fit_clients,
        )
        if not clients:
            return []
        
        client_instructions = []
        config = {
                "server_round": server_round,
                "local_epochs": 1,
            }
        for i, client in enumerate(clients):
            client_id = client.partition_id
            client_id = self.experiment_scenario.get_client_id(client_id)
            cluster_id = self.client_models[client_id]
            model_parameters = self.cluster_models[cluster_id]
            client_instructions.append((client, fl.common.FitIns(model_parameters, config)))

        return client_instructions

    def initialize_model(self, initial_parameters : Parameters, results: List[Tuple[ClientProxy, FitRes]]):
        """
            Initializes a set of parameters, where the parameters have been affected by the resulting gradients. 
            However due to the extra high learning rate in the first server round at the clients, the scale of the gradients needs to be lowered. 
        """
        if initial_parameters is None and results:
            client, fit_res = results[0]
            initial_parameters = fit_res.parameters
        
        # If we still don't have parameters or no results, return None
        if initial_parameters is None or not results:
            return None
        
        # Convert initial parameters to numpy arrays
        initial_numpy = parameters_to_ndarrays(initial_parameters)
        
        # Collect all client parameter updates
        client_updates = []
        for client, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            # Calculate updates (difference from initial parameters)
            updates = []
            for client_param, initial_param in zip(client_params, initial_numpy):
                updates.append(client_param - initial_param)
            client_updates.append(updates)
        
        # Average the updates across all clients
        avg_updates = []
        for i in range(len(initial_numpy)):
            layer_updates = [update[i] for update in client_updates]
            avg_update = np.mean(layer_updates, axis=0)
            avg_updates.append(avg_update)
        
        # Apply a dampening factor to the averaged updates
        dampening_factor = 1
        
        # Create new parameters by adding dampened updates to initial parameters
        new_params = []
        for initial_param, avg_update in zip(initial_numpy, avg_updates):
            new_params.append(initial_param + dampening_factor * avg_update)
        
        # Convert back to Parameters object
        return ndarrays_to_parameters(new_params)
    
    def _aggregate_refine_models(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
        # Group results by cluster
        cluster_specific_results = {}
        for client, fit_res in results:
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            cluster = self.client_models[client_id]
            if cluster in cluster_specific_results:
                cluster_specific_results[cluster].append((client, fit_res))
            else:
                cluster_specific_results[cluster] = [(client, fit_res)]
        
        # Process each cluster separately
        for cluster_id, cluster_results in cluster_specific_results.items():
            if len(cluster_results) < 2:
                # If only one client in cluster, use its parameters directly
                _, fit_res = cluster_results[0]
                self.cluster_models[cluster_id] = fit_res.parameters
                continue
                
            # Extract parameters from all clients in this cluster
            all_params = []
            for _, fit_res in cluster_results:
                params = parameters_to_ndarrays(fit_res.parameters)
                all_params.append(params)
                    # Simple averaging instead of trimmed mean
            averaged_params = []
            # For each parameter tensor position
            for i in range(len(all_params[0])):
                # Get this parameter from all clients
                param_i = [params[i] for params in all_params]
                
                # Simple average across all clients
                avg_param = np.mean(param_i, axis=0)
                averaged_params.append(avg_param)
            
            # Convert aggregated parameters back to Parameters format
            self.cluster_models[cluster_id] = ndarrays_to_parameters(averaged_params)    
            # Calculate the trimmed mean for each parameter tensor
            #trimmed_params = []
            #
            ## For each parameter tensor position
            #for i in range(len(all_params[0])):
            #    # Get this parameter from all clients
            #    param_i = [params[i] for params in all_params]
            #    
            #    # Convert to numpy array with shape [num_clients, *param_shape]
            #    param_stack = np.stack(param_i, axis=0)
            #    original_shape = param_stack.shape[1:]
            #    
            #    # Reshape to [num_clients, num_elements] for parameter-wise operations
            #    num_clients = param_stack.shape[0]
            #    param_stack = param_stack.reshape(num_clients, -1)
            #    
            #    # Apply trimmed mean along client dimension (axis 0)
            #    # β is the trimming parameter, we trim β fraction from both ends
            #    β = self.trimming_parameter  # e.g., 0.2
            #    k = int(β * num_clients)
            #    
            #    # For each parameter position, sort values across clients
            #    trimmed_param = np.zeros(param_stack.shape[1])
            #    for j in range(param_stack.shape[1]):
            #        sorted_values = np.sort(param_stack[:, j])
            #        # Trim k smallest and k largest values
            #        trimmed_values = sorted_values[k:num_clients-k]
            #        # Compute mean of remaining values
            #        if len(trimmed_values) > 0:
            #            trimmed_param[j] = np.mean(trimmed_values)
            #    
            #    # Reshape back to original tensor shape
            #    trimmed_param = trimmed_param.reshape(original_shape)
            #    trimmed_params.append(trimmed_param)
            #
            ## Convert aggregated parameters back to Parameters format
            #self.cluster_models[cluster_id] = ndarrays_to_parameters(all_params)

    def aggregate_fit(self, 
                    server_round: int, 
                    results: List[Tuple[ClientProxy, FitRes]], 
                    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results from clients for each cluster separately."""

        match self.current_phase: 
            case "ONE_SHOT":
                client_to_update_directions = {} 
                client_to_update_directions2 = {} 
                
                if self.secondLayerComparisonInit and results:
                    # Get number of layers in the model
                    client, fit_res = results[0]
                    client_params = parameters_to_ndarrays(fit_res.parameters)
                    # Determine number of layers based on parameter structure
                    # Assuming LSTM model where each layer has 4 parameter tensors
                    num_params = len(client_params)
                    params_per_layer = 4  # Each LSTM layer has 4 parameter tensors
                    num_layers = num_params // params_per_layer
                    
                    # Select second-to-last layer
                    second_last_layer_idx = num_layers - 2
                    
                    # Map second-to-last layer to its parameter indices
                    # Each LSTM layer has 4 parameters (weight_ih, weight_hh, bias_ih, bias_hh)
                    start_idx = second_last_layer_idx * params_per_layer
                    selected_param_indices = list(range(start_idx, start_idx + params_per_layer))
                    
                    # Extract second-to-last layer parameters for each client
                    for client, fit_res in results:
                        client_id = self.experiment_scenario.get_client_id(client.partition_id)
                        client_params = parameters_to_ndarrays(fit_res.parameters)
                        initial_params = self.saved_initial_parameters
                        
                        # Extract only second-to-last layer parameter differences
                        selected_diffs = []
                        for param_idx in selected_param_indices:
                            if param_idx < len(client_params):
                                diff = client_params[param_idx] - initial_params[param_idx]
                                selected_diffs.append(diff.flatten())
                        
                        # Concatenate into a single vector
                        if selected_diffs:
                            update_vector = np.concatenate(selected_diffs)
                            # Normalize the update vector
                            norm = np.linalg.norm(update_vector)
                            if norm > 1e-10:
                                update_vector = update_vector / norm
                            client_to_update_directions[client_id] = update_vector
                        else:
                            # Fallback - use the update direction from metrics
                            gradient_str = fit_res.metrics["update_direction"]
                            gradient_vector = np.array(json.loads(gradient_str))
                            client_to_update_directions[client_id] = gradient_vector
                else:
                    # Use entire model (original approach)
                    for client, fit_res in results:
                        client_id = self.experiment_scenario.get_client_id(client.partition_id)
                        gradient_str = fit_res.metrics["update_direction"]
                        gradient_vector = np.array(json.loads(gradient_str))
                        client_to_update_directions[client_id] = gradient_vector
                    
                # creating clusters
                clusters = one_shot_clustering(client_to_update_directions, self.distance_fn, self.threshold, self.min_cluster_size)

                # initialization of series of models 
                for cluster_id, clients in clusters.items(): 
                    for client in clients: 
                        self.client_models[client] = cluster_id
                    self.clusters[cluster_id] = clients
                    cluster_results = [res for res in results if self.experiment_scenario.get_client_id(res[0].partition_id) in clients]
                    self.cluster_models[cluster_id] = self.initialize_model(ndarrays_to_parameters(self.saved_initial_parameters), cluster_results)
                
                metric_tracker.log_cluster_state(server_round, self.client_models)
                metric_tracker.save_metrics()
                return None, {}
            case "REFINE": 
                self._aggregate_refine_models(results)

                return None, {}
            case _: # TRACK_MODELS -> only evaluate, DONE -> Nothing happens
                return None, {}


    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """Configure evaluation with clients' preferred clusters."""
        match self.current_phase: 
            case "ONE_SHOT": 
                # No model evaluation -> Not interesting in this phase. 
                self.current_phase = "TRACK_MODELS"
                return []
            case "TRACK_MODELS": 
                clients = client_manager.sample(
                    num_clients=max(int(client_manager.num_available() * self.fraction_fit), 1),
                    min_num_clients=self.min_fit_clients,
                )
                for client in clients: 
                    client_id = self.experiment_scenario.get_client_id(client.partition_id)
                    if client_id not in self.track_models_performance:
                        self.track_models_performance[client_id] = {}
                    if self.current_refine_step not in self.track_models_performance[client_id]:
                        self.track_models_performance[client_id][self.current_refine_step] = {}

                # Traking model performance 1 by 1 for each client
                return self._configure_track_model_performance(server_round, client_manager)
            case "REFINE": 
                clients = client_manager.sample(
                    num_clients=max(int(client_manager.num_available() * self.fraction_fit), 1),
                    min_num_clients=self.min_fit_clients,
                )
                if not clients:
                    return []
                client_instructions = []
                config = {
                        "server_round": server_round,
                    }
                for client in clients: 
                    client_id = self.experiment_scenario.get_client_id(client.partition_id)
                    model_parameters = self.cluster_models[self.client_models[self.experiment_scenario.get_client_id(client.partition_id)]]
                    client_instructions.append((client, fl.common.EvaluateIns(model_parameters, config)))
                    metric_tracker.log_pretraining_round(server_round, client_id, self.client_models[self.experiment_scenario.get_client_id(client.partition_id)])

                return client_instructions
            case "DONE": 
                return []
            case _: 
                return []
            
    def _configure_track_model_performance(self, server_round : int, client_manager: ClientManager): 
        if self.track_model_step == len(self.cluster_models):
            return []
            
        clients = client_manager.sample(
            num_clients=max(int(client_manager.num_available() * self.fraction_fit), 1),
            min_num_clients=self.min_fit_clients,
        )
        if not clients:
            return []

        client_instructions = []

        model_parameters = self.cluster_models[self.track_model_step]

        config = {
                "server_round": server_round,
            }
        for client in clients: 
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            metric_tracker.log_pretraining_round(server_round, client_id, self.track_model_step)
            client_instructions.append((client, fl.common.EvaluateIns(model_parameters, config)))

        return client_instructions
    

    def _get_model_similarities(self, round_num, use_second_last_layer=False):
        import numpy as np
        from scipy.spatial.distance import cosine
        
        # Get list of all cluster IDs
        cluster_ids = list(self.cluster_models.keys())
        num_clusters = len(cluster_ids)
        
        if num_clusters <= 1:
            print("Not enough models to calculate similarities.")
            return None, None
        
        # Create a similarity matrix
        similarity_matrix = np.zeros((num_clusters, num_clusters))
        
        # Convert model parameters to flat vectors for comparison
        model_vectors = {}
        for cluster_id in cluster_ids:
            # Convert Parameters to numpy arrays
            params = parameters_to_ndarrays(self.cluster_models[cluster_id])
            
            if use_second_last_layer:
                # Extract second-to-last layer parameters
                num_params = len(params)
                params_per_layer = 4  # Each LSTM layer has 4 parameter tensors
                num_layers = num_params // params_per_layer
                
                # Select second-to-last layer
                second_last_layer_idx = num_layers - 2
                
                # Map second-to-last layer to its parameter indices
                start_idx = second_last_layer_idx * params_per_layer
                selected_param_indices = list(range(start_idx, start_idx + params_per_layer))
                
                # Extract only second-to-last layer parameters
                selected_params = []
                for param_idx in selected_param_indices:
                    if param_idx < len(params):
                        selected_params.append(params[param_idx].flatten())
                
                # Concatenate into a single vector
                if selected_params:
                    flattened = np.concatenate(selected_params)
                else:
                    # Fallback to using all parameters if 2nd-last layer extraction fails
                    flattened = np.concatenate([p.flatten() for p in params])
            else:
                # Use full model - flatten and concatenate all parameters
                flattened = np.concatenate([p.flatten() for p in params])
                
            model_vectors[cluster_id] = flattened
        
        # Calculate pairwise cosine similarities
        for i, id1 in enumerate(cluster_ids):
            for j, id2 in enumerate(cluster_ids):
                if i == j:
                    # Same model, cosine similarity is 1
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate cosine similarity (1 - cosine distance)
                    similarity = 1.0 - cosine(model_vectors[id1], model_vectors[id2])
                    similarity_matrix[i, j] = similarity
        
        similarity_type = "2nd-last layer" if use_second_last_layer else "full model"
        print(f"\nModel Similarity Matrix ({similarity_type}, Cosine Similarity):")
        for i, id1 in enumerate(cluster_ids):
            row = [f"{similarity_matrix[i, j]:.4f}" for j in range(num_clusters)]
            print(f"Cluster {id1}: {' '.join(row)}")
        
        return similarity_matrix, cluster_ids

    def _merge_clusters(self, similarity_matrix, cluster_ids):
        # Create a list of all potential merges with their similarity scores
        potential_merges = []
        
        # Identify all potential merges and their similarity scores
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):  # Only check upper triangle
                    cluster_i = cluster_ids[i]
                    cluster_j = cluster_ids[j]
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > self.merge_threshold:
                        # Store as (similarity, source_cluster, target_cluster)
                        # Higher similarity will appear first after sorting
                        potential_merges.append((similarity, cluster_j, cluster_i))
        
        # Sort potential merges by similarity score (highest first)
        potential_merges.sort(reverse=True)
        
        # Track which clusters have been merged to avoid merging a cluster twice
        merged_clusters = set()
        
        # Process merges in order of decreasing similarity
        for similarity, source_cluster, target_cluster in potential_merges:
            # Skip if either cluster has already been merged
            if source_cluster in merged_clusters or target_cluster in merged_clusters:
                continue
            
            # Skip if either cluster no longer exists
            if source_cluster not in self.clusters or target_cluster not in self.clusters:
                continue
            
            # Merge client lists
            self.clusters[target_cluster].extend(self.clusters[source_cluster])
            
            # Update client assignments
            for client_id in self.clusters[source_cluster]:
                self.client_models[client_id] = target_cluster
            
            # Average the model parameters
            params_target = parameters_to_ndarrays(self.cluster_models[target_cluster])
            params_source = parameters_to_ndarrays(self.cluster_models[source_cluster])
            
            merged_params = []
            for p_target, p_source in zip(params_target, params_source):
                merged_params.append((p_target + p_source) / 2.0)
            
            self.cluster_models[target_cluster] = ndarrays_to_parameters(merged_params)
            
            # Remove the source cluster
            del self.clusters[source_cluster]
            del self.cluster_models[source_cluster]
            
            # Mark these clusters as merged
            merged_clusters.add(source_cluster)
            merged_clusters.add(target_cluster)
        
        # After all merges, renumber the clusters consecutively
        if len(merged_clusters) > 0:  # Only renumber if merges occurred
            self._renumber_clusters()

    def _renumber_clusters(self):
        """Renumber clusters to have consecutive integer IDs (0, 1, 2, ...) with no gaps."""
        # Get the current cluster IDs
        old_cluster_ids = list(self.clusters.keys())
        
        # Create a mapping from old IDs to new consecutive IDs
        new_id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_cluster_ids)}
        
        # Create new dictionaries with the updated IDs
        new_clusters = {}
        new_cluster_models = {}
        
        # Update cluster dictionaries
        for old_id, client_list in self.clusters.items():
            new_id = new_id_mapping[old_id]
            new_clusters[new_id] = client_list
            new_cluster_models[new_id] = self.cluster_models[old_id]
        
        # Update client model assignments
        for client_id, old_cluster_id in self.client_models.items():
            if old_cluster_id in new_id_mapping:  # Check if the old ID exists in mapping
                self.client_models[client_id] = new_id_mapping[old_cluster_id]
        
        # Replace the old dictionaries with the new ones
        self.clusters = new_clusters
        self.cluster_models = new_cluster_models



    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        match self.current_phase: 
            case "TRACK_MODELS": 
                # Save all model performances for each clients on all models. 
                self._aggregate_track_model_performance(server_round, results)
                if self.track_model_step == len(self.cluster_models)-1:
                    self.track_model_step = 0
                    self.current_phase = "REFINE"  # Transition to REFINE phase
                    self._recluster_clients(results) 

                    if len(list(self.cluster_models.keys())) > 1:
                        similarities, cluster_ids = self._get_model_similarities(server_round, use_second_last_layer=self.secondLayerComparisonMerge)
                        self._merge_clusters(similarities, cluster_ids)             
                    metric_tracker.log_cluster_state(server_round, self.client_models)
                    for client, evaRes in results: 
                        client_id = client.partition_id
                        client_id = self.experiment_scenario.get_client_id(client_id)
                        metric_tracker.log_training_round(server_round, client_id, evaRes.loss)

                else:
                    self.track_model_step = self.track_model_step + 1
                return None, {}
            case "REFINE":
                for client, evaRes in results: 
                    client_id = client.partition_id
                    client_id = self.experiment_scenario.get_client_id(client_id)
                    metric_tracker.log_training_round(server_round, client_id, evaRes.loss)
                if self.current_refine_step < self.max_refine_steps:
                    self.current_refine_step = self.current_refine_step + 1
                    if self.current_refine_step % 2 == 0: 
                        self.current_phase = "TRACK_MODELS"
                else: 
                    self.current_phase = "DONE" 
                return None, {}
            case "DONE": 
                return None, {}
            case _: 
                return None, {}
        metric_tracker.save_metrics()

    def _aggregate_track_model_performance(self, server_round, results):
        for client, eval_res in results: 
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            self.track_models_performance[client_id][self.current_refine_step][self.track_model_step] = eval_res

    def _recluster_clients(self, results: List[Tuple[ClientProxy, EvaluateRes]]):
        """Reassign clients to their best-performing clusters based on evaluation results."""
        # Clear existing clusters while preserving the models
        self.clusters = {cluster_id: [] for cluster_id in self.cluster_models.keys()}
        
        # Process each client's evaluation results
        for client, eval_res in results:
            client_id = self.experiment_scenario.get_client_id(client.partition_id)
            try:             
                eval_dict = self.track_models_performance[client_id][self.current_refine_step]
                
                # Find the best performing cluster for this client
                best_cluster = min(eval_dict, key=lambda cluster_id: eval_dict[cluster_id].loss)
                
                # Update client's cluster assignment
                self.client_models[client_id] = best_cluster
                
                # Add client to the appropriate cluster
                self.clusters[best_cluster].append(client_id)
            except Exception as e:
                print(f"Error reclustering client {client_id}: {str(e)}")
    

    def fit_config_fn(self, server_round: int) -> Dict[str, Scalar]:
        """
        Return training configuration for clients.
        
        Args:
            server_round: Current round of federated learning
            
        Returns:
            Configuration dictionary
        """
        config = {"local_epochs": 1, "server_round" : server_round}

        return config
    