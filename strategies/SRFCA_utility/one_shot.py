import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import json
import os
import numpy as np
import pickle

logger = logging.getLogger("clustering.one_shot")

def build_similarity_graph(client_models: Dict[str, np.array], 
                          distance_fn, 
                          threshold: float) -> Dict[str, List[str]]:
    """Build a similarity graph between clients based on model parameters."""
    graph = defaultdict(list)
    client_ids = list(client_models.keys())
    
    for i, client_i in enumerate(client_ids):
        for j, client_j in enumerate(client_ids[i+1:], i+1):
            # Calculate similarity or distance
            dist = distance_fn(client_models[client_i], client_models[client_j])
            
            # Add edge if distance is below threshold
            if dist >= threshold:
                graph[client_i].append(client_j)
                graph[client_j].append(client_i)
    
    return dict(graph)

def correlation_clustering_approx(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Implementation of the correlation clustering approximation algorithm.
    
    Based on Bansal et al. (2002) but simplified for testing.
    """
    # Convert to set for O(1) lookups
    graph_sets = {node: set(neighbors) for node, neighbors in graph.items()}
    nodes = set(graph.keys())
    clusters = []
    
    while nodes:
        # Pick a random node
        node = next(iter(nodes))
        
        # Create a new cluster with this node
        cluster = [node]
        neighbors = graph_sets.get(node, set()).copy()
        nodes.remove(node)
        
        # Add all its neighbors to the cluster
        to_process = list(neighbors)
        while to_process:
            neighbor = to_process.pop(0)
            if neighbor in nodes:
                # Check if this neighbor is connected to at least half of the cluster
                connections = sum(1 for n in cluster if n in graph_sets.get(neighbor, set()))
                if connections >= len(cluster) / 2:
                    cluster.append(neighbor)
                    nodes.remove(neighbor)
                    # Add new neighbors to process
                    for n in graph_sets.get(neighbor, set()):
                        if n in nodes and n not in to_process:
                            to_process.append(n)
        
        if cluster:
            clusters.append(cluster)
    
    return clusters

def filter_clusters_by_size(clusters: List[List[str]], min_size: int) -> Dict[int, List[str]]:
    """Filter clusters to ensure they have at least min_size nodes."""
    valid_clusters = {}
    cluster_id = 0
    
    for cluster in clusters:
        if len(cluster) >= min_size:
            valid_clusters[cluster_id] = cluster
            cluster_id += 1
    
    return valid_clusters

def one_shot_clustering(client_models: Dict[str, np.array], 
                       distance_fn, 
                       threshold: float,
                       min_cluster_size: int, 
                       save_res = False) -> Dict[int, List[str]]:
    """Perform ONE_SHOT clustering as described in SR-FCA."""
    # Build similarity graph
    if save_res:
        models_dir = "Oneshot_cluster_models"
        os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

        models_path = os.path.join(models_dir, f"{threshold}_models.pkl")
        with open(models_path, 'wb') as f:
            pickle.dump(client_models, f)
        exit(1)
    graph = build_similarity_graph(client_models, distance_fn, threshold)
    
    # Perform correlation clustering
    clusters = correlation_clustering_approx(graph)
    
    # Filter clusters by size
    valid_clusters = filter_clusters_by_size(clusters, min_cluster_size)
    # Find clients that weren't assigned to any cluster
    assigned_clients = set()
    for cluster in valid_clusters.values():
        assigned_clients.update(cluster)
    
    # Add singleton clusters for unassigned clients
    next_cluster_id = max(valid_clusters.keys(), default=-1) + 1
    for client_id in client_models.keys():
        if client_id not in assigned_clients:
            valid_clusters[next_cluster_id] = [client_id]
            next_cluster_id += 1
    logger.info(f"ONE_SHOT found {len(valid_clusters)} clusters")
    for cluster_id, clients in valid_clusters.items():
        logger.info(f"Cluster {cluster_id}: {len(clients)} clients")
    
    return valid_clusters


def test(): 
    from distance_metrics import l2_distance, cosine_similarity

    models_dir = "Oneshot_cluster_models"

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # Iterate over all .pkl files in the directory
    for filename in os.listdir(models_dir):
        if filename.endswith("model_2nd_last.pkl"):#""):
            # Full path to the file
            file_path = os.path.join(models_dir, filename)

            # Load the pickled client_models
            with open(file_path, 'rb') as f:
                client_models = pickle.load(f)

            for t in thresholds:
                clusters = one_shot_clustering(
                    client_models=client_models,
                    distance_fn=cosine_similarity,  
                    threshold=t,
                    min_cluster_size=0,
                    save_res=False  
                )

                print(f"Clusters for threshold {t}: {len(clusters)}")
