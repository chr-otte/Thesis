import numpy as np

def split_cluster_by_graph(similarity_matrix, clients, similarity_threshold=0.3):
    # Step 1: Convert similarity matrix to an adjacency matrix
    adjacency_matrix = similarity_matrix > similarity_threshold
    
    # Step 2: Find connected components (basic communities)
    visited = [False] * len(clients)
    clusters = []
    
    for i in range(len(clients)):
        if not visited[i]:
            # Start a new cluster
            cluster = []
            # Use DFS to find all connected nodes
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    cluster.append(clients[node])
                    # Add all unvisited neighbors above threshold
                    for j in range(len(clients)):
                        if adjacency_matrix[node][j] and not visited[j]:
                            stack.append(j)
            clusters.append(cluster)
    
    return clusters


m = np.array([  [
                1.0,
                0.936563375579961,
                0.040631645238877016,
                0.02705267612422632,
                0.07851771897401308,
                0.016981951669836147,
                -0.03207979090534726,
                -0.03770277420367803
            ],
            [
                0.936563375579961,
                1.0,
                0.04443403720565341,
                0.017053263873638376,
                0.06882530883465211,
                0.011678221948340731,
                -0.03901732606685554,
                -0.04237585635101087
            ],
            [
                0.040631645238877016,
                0.04443403720565341,
                1.0,
                0.05599346306700256,
                0.02282351380344859,
                0.10265010198209186,
                0.06945255839622175,
                0.050438947807758004
            ],
            [
                0.02705267612422632,
                0.017053263873638376,
                0.05599346306700256,
                1.0,
                0.1445946276768396,
                0.07346328296475617,
                0.08864481975934577,
                0.07411074119128745
            ],
            [
                0.07851771897401308,
                0.06882530883465211,
                0.02282351380344859,
                0.1445946276768396,
                1.0,
                0.05657714905097904,
                0.027117351104565434,
                0.03985359844280109
            ],
            [
                0.016981951669836147,
                0.011678221948340731,
                0.10265010198209186,
                0.07346328296475617,
                0.05657714905097904,
                1.0,
                0.05890051219674017,
                0.07044934724449839
            ],
            [
                -0.03207979090534726,
                -0.03901732606685554,
                0.06945255839622175,
                0.08864481975934577,
                0.027117351104565434,
                0.05890051219674017,
                1.0,
                0.2837209255262009
            ],
            [
                -0.03770277420367803,
                -0.04237585635101087,
                0.050438947807758004,
                0.07411074119128745,
                0.03985359844280109,
                0.07044934724449839,
                0.2837209255262009,
                1.0
            ]
])

m2 = np.array([ [
                1.0,
                0.5951727027756418,
                0.07698049992612498,
                0.21257824176828358,
                0.16744004101909887,
                0.10987003359739991,
                0.05992424309766563,
                0.10157858971787172
            ],
            [
                0.5951727027756418,
                1.0,
                0.0567265277149345,
                0.19124747887542684,
                0.1756336445683518,
                0.07443614276617888,
                0.049924315006699374,
                0.07349045293301068
            ],
            [
                0.07698049992612498,
                0.0567265277149345,
                1.0,
                0.057679419956708354,
                0.012996132196111829,
                0.1438534116106373,
                0.10227931050941136,
                0.09811754980588863
            ],
            [
                0.21257824176828358,
                0.19124747887542684,
                0.057679419956708354,
                1.0,
                0.2873727448384746,
                0.14237875196398936,
                0.08834911773464377,
                0.11139332488650829
            ],
            [
                0.16744004101909887,
                0.1756336445683518,
                0.012996132196111829,
                0.2873727448384746,
                1.0,
                0.10973909892889877,
                0.05156243895827499,
                0.06455191747354151
            ],
            [
                0.10987003359739991,
                0.07443614276617888,
                0.1438534116106373,
                0.14237875196398936,
                0.10973909892889877,
                1.0,
                0.13392372595705052,
                0.1521555419002808
            ],
            [
                0.05992424309766563,
                0.049924315006699374,
                0.10227931050941136,
                0.08834911773464377,
                0.05156243895827499,
                0.13392372595705052,
                1.0,
                0.3127910059613677
            ],
            [
                0.10157858971787172,
                0.07349045293301068,
                0.09811754980588863,
                0.11139332488650829,
                0.06455191747354151,
                0.1521555419002808,
                0.3127910059613677,
                1.0
            ]])

m = m2 *0.5+ m *0.5 


client_ids = [
            "MT_148",
            "MT_145",
            "MT_058",
            "MT_200",
            "MT_202",
            "MT_016",
            "MT_125",
            "MT_124"
        ]

ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for r in ranges:
    print(f"Similarity threshold: {r}")
    res = split_cluster_by_graph(m2, client_ids, similarity_threshold=r) 
    print(res)



def test_split_cluster_by_graph():
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

    res = split_cluster_by_graph(similarity_matrix, client_ids) 

    # Convert lists to sets of frozensets for unordered comparison
    expected = [
        ["MT_124", "MT_125"],
        ["MT_016", "MT_058"],
        ["MT_202", "MT_200"],
        ["MT_145", "MT_148"],
    ]

    assert set(frozenset(cluster) for cluster in res) == set(frozenset(cluster) for cluster in expected), "Graph_Based_clustering error"



def test_graph_partioning_2():
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
    #bi_partioned = split_cluster_by_graph(matrix, client_ids, similarity_threshold=0.3)

    for t in thresholds:
        bi_partioned = split_cluster_by_graph(matrix, client_ids, similarity_threshold=t)
        print(f"Bi-partioned clusters for threshold {t}: {bi_partioned}")

    print()
#test_graph_partioning_2()