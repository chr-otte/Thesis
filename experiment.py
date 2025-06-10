from abc import ABC, abstractmethod

class Experiment(ABC): 
    @abstractmethod
    def get_num_clients(self) -> int: 
        pass 

    @abstractmethod 
    def get_clients(self) -> list[str]: 
        pass 
    
    @abstractmethod 
    def get_cluster_clients(self) -> dict[int, list[str]]:
        pass

    @abstractmethod
    def get_dataset_fraction(self) -> float:
        pass

    @abstractmethod 
    def get_client_id(self, idx:int) -> str: 
        pass 

def get_experiment_settings(experiment:str) -> Experiment: 
    match experiment: 
        case "base experiment": 
            return Electricity_Base()
        case "experiment1": 
            return Electricity_experiment1()
        case "experiment2": 
            return Traffic_experiment()
        case "experiment3": 
            return HAR_experiment()
        case _ : 
            raise Exception("Unknown experiment")




class Electricity_Base(Experiment): 
    def get_clients(self) -> list[str]:
        clients = [ 
           'MT_016', 'MT_058',  # cluster 0 
           'MT_145', 'MT_148',  # cluster 1
           'MT_124', 'MT_125',  # cluster 2
           'MT_200', 'MT_202',  # cluster 3 
        ]
        return clients

    def get_num_clients(self) -> int:
        return len(self.get_clients())
    
    def get_cluster_clients(self) -> dict[int, list[str]]:
        d = {
            0: ['MT_016', 'MT_058'],
            1: ['MT_145', 'MT_148'],
            2: ['MT_124', 'MT_125'],
            3: ['MT_200', 'MT_202'],
        }
        return d
    
    def get_dataset_fraction(self) -> float:
        return 0.4

    def get_client_id(self, idx) -> str:
        return self.get_clients()[idx]
    

class Electricity_experiment1(Experiment):
    def get_clients(self) -> list[str]:
        clients = [ 
            'MT_124', 'MT_125', # cluster 0
            'MT_128', 'MT_162', # cluster 1
            'MT_349', 'MT_363', # cluster 2
            'MT_340', 'MT_342', # cluster 3 
        ]
        return clients

    def get_num_clients(self) -> int:
        return len(self.get_clients())
    
    def get_cluster_clients(self) -> dict[int, list[str]]:
        d = {
            0: ['MT_124', 'MT_125'],
            1: ['MT_128', 'MT_162'],
            2: ['MT_349', 'MT_363'],
            3: ['MT_340', 'MT_342'],
        }
        return d
    
    def get_dataset_fraction(self) -> float:
        return 0.4

    def get_client_id(self, idx) -> str:
        return self.get_clients()[idx]

class Traffic_experiment(Experiment):

    def get_clients(self) -> list[str]:
        clients = [ 
            "98", "112", "46", "26", "152", "45", "156", 
            "147", "14", "13", "18", 
            "73"
        ]
        return clients

    def get_num_clients(self) -> int:
        return len(self.get_clients())
    
    def get_cluster_clients(self) -> dict[int, list[str]]:
        d = {
            0: ["98", "112", "46", "26", "152", "45", "156"],
            1: ["147", "14", "13", "18"],
            2: ["73"],
        }
        return d
    
    def use_synthetic_clustering(self) -> bool:
        return True  # or conditionally enable based on scenario

    def get_synthetic_group(self, client_id: str) -> str:
        """
        Assign each client to a synthetic transformation group.
        Options: 'seasonal', 'scaled', 'noisy', 'original'
        """
        synthetic_map = {
            "98": "seasonal",
            "112": "seasonal",
            "46": "scaled",
            "26": "scaled",
            "152": "noisy",
            "45": "noisy",
            "156": "noisy",
            "147": "original",
            "14": "original",
            "13": "original",
            "18": "original",
            "73": "original"
        }



        return synthetic_map.get(client_id, "original")

    
    def get_dataset_fraction(self) -> float:
        return 1

    def get_client_id(self, idx) -> str:
        return self.get_clients()[idx]


class HAR_experiment(Experiment):
    def get_clients(self) -> list[str]:
        clients = ["1", "3", "5", "6", "7", "8", "11", "14", "15", "16", "17", "19", "21", "22", "23", "25", "26", "27", "28", "29", "30" ]
        return clients

    def get_num_clients(self) -> int:
        return len(self.get_clients())
    
    def get_cluster_clients(self) -> dict[int, list[str]]:
        d = {
            0: ["14","23","7","21","16","17"],
            1: ["28"],
            2: ["25"],
            3: ["5", "6", "11","15"],
            4: ["19", "26", "22", "27"],
            5: ["8", "29", "30", "1", "3"],
        }
        return d
    
    def use_synthetic_clustering(self) -> bool:
        return True  # Enable synthetic clustering
    
    def get_synthetic_group(self, client_id: str) -> str:
        """
        Map each cluster to a specific type of synthetic transformation.
        This can be used for more detailed control of synthetic elements.
        """
        # Map client to its cluster first
        client_cluster = None
        for cluster_id, client_list in self.get_cluster_clients().items():
            if client_id in client_list:
                client_cluster = cluster_id
                break
        
        # Map clusters to transformation types
        cluster_to_transformation = {
            0: "bouncy",     # More bouncy walking style
            1: "lateral",    # More lateral movement
            2: "forceful",   # More forceful movements 
            3: "vibration",  # High frequency vibration
            4: "drift",      # Slight drift in stationary positions
            5: "phase_shift" # Phase shifts and amplitude scaling
        }
        
        return cluster_to_transformation.get(client_cluster, "original")
    
    def get_dataset_fraction(self) -> float:
        return 1

    def get_client_id(self, idx) -> str:
        return self.get_clients()[idx]
