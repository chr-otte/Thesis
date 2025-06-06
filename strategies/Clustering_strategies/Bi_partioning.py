import numpy as np 

def bi_partion(similarity_matrix, clients, similarity_threshold=0.3):
    return bi_partion3(similarity_matrix, clients, similarity_threshold=similarity_threshold)
    if any(np.mean(similarity_matrix[i]) < similarity_threshold for i in range(len(clients))):
        # Find the client with lowest average similarity
        min_similarity_idx = np.argmin([np.mean(similarity_matrix[i]) for i in range(len(clients))])
        similarities_to_seed = similarity_matrix[min_similarity_idx]
        median_similarity = np.median(similarities_to_seed)

        low_similarity_cluster = []
        high_similarity_cluster = []

        for i, client_id in enumerate(clients):
            if similarities_to_seed[i] < median_similarity:
                low_similarity_cluster.append(client_id)
            else: 
                high_similarity_cluster.append(client_id)

        return [high_similarity_cluster, low_similarity_cluster]
    else:
        return [clients]  # Ensure the return type is consistent

# Not enforcing same size clusters 
def bi_partion2(similarity_matrix, clients, similarity_threshold=0.3):
    if any(np.mean(similarity_matrix[i]) < similarity_threshold for i in range(len(clients))):
        # Find the client with lowest average similarity
        min_similarity_idx = np.argmin([np.mean(similarity_matrix[i]) for i in range(len(clients))])
        similarities_to_seed = similarity_matrix[min_similarity_idx]
        
        # Instead of median, use an absolute threshold
        # This allows for clusters of varying sizes
        low_similarity_cluster = []
        high_similarity_cluster = []

        for i, client_id in enumerate(clients):
            # Use the similarity_threshold directly instead of median
            if similarities_to_seed[i] < similarity_threshold:
                low_similarity_cluster.append(client_id)
            else: 
                high_similarity_cluster.append(client_id)
                
        # Handle edge cases - if one cluster is empty, don't split
        if len(low_similarity_cluster) == 0 or len(high_similarity_cluster) == 0:
            return [clients]
            
        return [high_similarity_cluster, low_similarity_cluster]
    else:
        return [clients]

    
def bi_partion3(similarity_matrix, clients, similarity_threshold=0.5):
    # If we only have 1
    if len(clients) < 2:
        return [clients]
        
    # Create a graph representation where edges exist if similarity > threshold
    n = len(clients)
    adjacency = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i, j] >= similarity_threshold:
                adjacency[i, j] = 1
    
    # Find connected components in the graph
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(adjacency, directed=False)
    
    # If we only have one connected component or all clients are disconnected,
    # the threshold isn't creating a meaningful partition
    if n_components == 1 or n_components == n:
        return [clients]
    
    # Create clusters based on connected components
    clusters = []
    for component in range(n_components):
        component_clients = [clients[i] for i in range(n) if labels[i] == component]
        clusters.append(component_clients)
    
    if len(clusters) > 2:
        # Find the largest cluster
        largest_idx = np.argmax([len(cluster) for cluster in clusters])
        largest_cluster = clusters[largest_idx]
        
        # Merge all other clusters
        other_clients = []
        for i, cluster in enumerate(clusters):
            if i != largest_idx:
                other_clients.extend(cluster)
        
        return [largest_cluster, other_clients]
    
    return clusters



m = np.array([            [
                1.0,
                0.999820338440276,
                0.360276742008454,
                0.269919619576579,
                0.407417453732876,
                0.364016462838855,
                0.404153595675894,
                0.403007413709431,
                0.974101319932511,
                0.299543536899893,
                0.14888283924342,
                0.196193637197267
            ],
            [
                0.999820338440276,
                1.0,
                0.363427572948602,
                0.271497138501344,
                0.406767492284435,
                0.367414866172929,
                0.403373290036412,
                0.402170370343934,
                0.973741569487856,
                0.299948674674627,
                0.152137421587441,
                0.199881346629607
            ],
            [
                0.360276742008454,
                0.363427572948602,
                1.0,
                0.216675899151228,
                0.304628761113186,
                0.992896163381078,
                0.25760960932918,
                0.246575818623195,
                0.322131538611058,
                0.237135635177489,
                0.198216523987672,
                0.240093097699312
            ],
            [
                0.269919619576579,
                0.271497138501344,
                0.216675899151228,
                1.0,
                0.450645518476177,
                0.228847743046101,
                0.200543723341532,
                0.193576963728171,
                0.254196011069264,
                0.813852733799048,
                0.223373399945276,
                0.243469215347857
            ],
            [
                0.407417453732876,
                0.406767492284435,
                0.304628761113186,
                0.450645518476177,
                1.0,
                0.316822601562918,
                0.440168184664078,
                0.434385513876953,
                0.372908509951523,
                0.476549796878837,
                0.126111502653461,
                0.17424519766396
            ],
            [
                0.364016462838855,
                0.367414866172929,
                0.992896163381078,
                0.228847743046101,
                0.316822601562918,
                1.0,
                0.268497546644032,
                0.256458618823085,
                0.323396990225599,
                0.247543885857873,
                0.211842864905029,
                0.25428443316551
            ],
            [
                0.404153595675894,
                0.403373290036412,
                0.25760960932918,
                0.200543723341532,
                0.440168184664078,
                0.268497546644032,
                1.0,
                0.988134031835272,
                0.400317817337115,
                0.224957393517875,
                0.0967898281537451,
                0.127494943774026
            ],
            [
                0.403007413709431,
                0.402170370343934,
                0.246575818623195,
                0.193576963728171,
                0.434385513876953,
                0.256458618823085,
                0.988134031835272,
                1.0,
                0.399779473377318,
                0.218853058174044,
                0.0877234344304575,
                0.11416505347868
            ],
            [
                0.974101319932511,
                0.973741569487856,
                0.322131538611058,
                0.254196011069264,
                0.372908509951523,
                0.323396990225599,
                0.400317817337115,
                0.399779473377318,
                1.0,
                0.281151956791392,
                0.129289847490924,
                0.174714413477992
            ],
            [
                0.299543536899893,
                0.299948674674627,
                0.237135635177489,
                0.813852733799048,
                0.476549796878837,
                0.247543885857873,
                0.224957393517875,
                0.218853058174044,
                0.281151956791392,
                1.0,
                0.180582997884838,
                0.229437638285317
            ],
            [
                0.14888283924342,
                0.152137421587441,
                0.198216523987672,
                0.223373399945276,
                0.126111502653461,
                0.211842864905029,
                0.0967898281537451,
                0.0877234344304575,
                0.129289847490924,
                0.180582997884838,
                1.0,
                0.577450185408557
            ],
            [
                0.196193637197267,
                0.199881346629607,
                0.240093097699312,
                0.243469215347857,
                0.17424519766396,
                0.25428443316551,
                0.127494943774026,
                0.11416505347868,
                0.174714413477992,
                0.229437638285317,
                0.577450185408557,
                1.0
            ]
])

m2 = np.array([])




client_ids = [
            "45",
            "156",
            "46",
            "147",
            "73",
            "26",
            "98",
            "112",
            "152",
            "18",
            "14",
            "13"
        ]


ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for r in ranges:
    print(f"Similarity threshold: {r}")
    res = bi_partion(m, client_ids, similarity_threshold=r) 
    print(res)
    print(bi_partion2(m, client_ids, similarity_threshold=r))


def test_bi_partioning(): 
    similarity_matrix = np.array([
        [1.00, -0.12, 0.46, -0.04, -0.07, -0.05, -0.13, -0.11],  # MT_016
        [-0.12, 1.00, -0.10, -0.12, 0.14, -0.20, 0.19, 0.98],    # MT_148
        [0.46, -0.10, 1.00, -0.04, -0.07, -0.10, -0.13, -0.09],  # MT_058
        [-0.04, -0.12, -0.04, 1.00, -0.17, 0.39, -0.09, -0.13],  # MT_125
        [-0.07, 0.14, -0.07, -0.17, 1.00, -0.16, 0.41, 0.14],    # MT_202
        [-0.05, -0.20, -0.10, 0.39, -0.16, 1.00, -0.11, -0.20],  # MT_124
        [-0.13, 0.19, -0.13, -0.09, 0.41, -0.11, 1.00, 0.20],    # MT_200
        [-0.11, 0.98, -0.09, -0.13, 0.14, -0.20, 0.20, 1.00],    # MT_145
    ])

    client_ids = ["MT_016", "MT_148", "MT_058", "MT_125", "MT_202", "MT_124", "MT_200", "MT_145"]

    res = bi_partion(similarity_matrix, client_ids) 
    
    # Convert lists to sets of frozensets for unordered comparison
    expected = [
        ["MT_016", "MT_124", "MT_125", "MT_058"],
        ["MT_202", "MT_200", "MT_145", "MT_148"],
    ]
    
    assert set(frozenset(cluster) for cluster in res) == set(frozenset(cluster) for cluster in expected), "BI-partioning error"

def test_bi_partioning_2():
    from collections import defaultdict
    import json
    import os
    import numpy as np
    import pickle

    file_path = r"C:\Github\Thesis\ClusteringExperiments\Oneshot_cluster_models\0.6_models.pkl"

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    with open(file_path, 'rb') as f:
        client_models = pickle.load(f)
    similarity_matrix = []

    client_ids = []
    for c, m in client_models.items():
        similarity_matrix.append(m)
        client_ids.append(c)

    similarity_matrix = np.array(similarity_matrix)
    matrix = np.zeros((len(client_ids), len(client_ids)))
    for i in range(len(client_ids)):
        for j in range(len(client_ids)):
            if i != j:
                matrix[i][j] = np.dot(similarity_matrix[i], similarity_matrix[j]) / (np.linalg.norm(similarity_matrix[i]) * np.linalg.norm(similarity_matrix[j]))
    
    for t in thresholds:
        bi_partioned = bi_partion(matrix, client_ids, similarity_threshold=t)
        print(f"Bi-partioned clusters for threshold {t}: {bi_partioned}")

#test_bi_partioning_2()