import flwr as fl
from server import start_server
from client import create_client
from flwr.server.client_manager import SimpleClientManager
from strategies.strategy import get_strategy
import ray
from standardized_metric_tracking import metric_tracker
from logger import logger 
import flwr as fl
from experiment_scenarios.experiment import get_experiment_settings
import os 
import torch 
from flwr.common import parameters_to_ndarrays
from experiment_configurations.Baseline_experiments import get_baseline_experiments
from experiment_configurations.Ifca_experiments import get_ifca_experiments
from experiment_configurations.Csc_experiments import get_csc_experiments
from experiment_configurations.Srfca_experiments import get_srfca_experiments

def run_experiment(experiment_config):
    """Run FL experiment using Ray for client simulation."""
    logger.info(f"Starting experiment with config: {experiment_config}")
    
    experiment_settings = get_experiment_settings(experiment_config["EXPERIMENT_SETTING"])

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Start clients asynchronously
    def client_fn(cid):
        logger.info(f"Creating client with ID: {cid}")
        try:
            client = create_client(int(cid), experiment_settings, experiment_config)
            logger.info(f"Client {cid} created successfully")
            return client.to_client()
        except Exception as e:
            logger.error(f"Error creating client {cid}: {e}")
            raise

    strategy = get_strategy(experiment_config, experiment_settings)
    logger.info(f"Using strategy: {experiment_config['STRATEGY_NAME']}")

    # Configure server
    server_config = fl.server.ServerConfig(num_rounds=experiment_config["NUM_ROUNDS"])
    
    # Log simulation parameters
    logger.info(f"Starting simulation with {experiment_settings.get_num_clients()} clients")
    logger.info(f"Planning for {experiment_config['NUM_ROUNDS']} rounds")
    logger.info(f"Fraction fit: {experiment_config['FRACTION_FIT']}")


    try:
        history = None
        custom_config = fl.server.ServerConfig(
            num_rounds=experiment_config["NUM_ROUNDS"]
        )
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=experiment_settings.get_num_clients(),
            client_resources={"num_cpus": experiment_config["NUM_CPU"]},
            config=custom_config,
            strategy=strategy,
        )

        logger.info("Simulation completed successfully")
        metric_tracker.save_metrics()

        if experiment_config["STRATEGY_NAME"] == "FedAvg":
            # Create saved_models directory if it doesn't exist
            os.makedirs('saved_models', exist_ok=True)
            
            # Create a filename based on the experiment config
            model_filename = f"saved_models/{experiment_config['EXPERIMENT_NAME']}.pt"
            
            # Get the final model parameters from the strategy
            if hasattr(strategy, 'parameters'):
                final_parameters = strategy.parameters
                
                # Save the model
                torch.save({
                    'model_state_ndarrays': parameters_to_ndarrays(final_parameters),
                    'experiment_config': experiment_config,
                }, model_filename)
                
                logger.info(f"Model saved to {model_filename}")
            else:
                logger.warning("Could not save model: strategy doesn't have parameters attribute")


        return history
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        raise

# default configuration
DEFAULT_CONFIG = {
    "FRACTION_FIT": 1,
    "SEQ_LENGTH": 24,
    "PRED_LENGTH": 24,
    "NUM_CPU":1,
    "EFFECTIVE_ROUNDS":3
}
def id_generator(config):
    return config["EXPERIMENT_NAME"]

def main(): 

    def get_similarity_matrices():
        exp = get_csc_experiments("experiment3")
        default = exp[0].copy()
        exp = []
        layers = [3,4,5,7] # [3, 5, 7, 10]
        hidden_sizes = [32]#, 1024, 2048]
        epochs = [1,2,3,4,5,6,7,8,9,10]
        models = ["LSTM Layered"] #, "GRU Layered", "RNN Layered"]
        model_types = ["LSTM"] #, "GRU", "RNN"]
        
        for epoch in epochs: 
            for i in models:
                for j in layers:
                    for k in hidden_sizes: 
                        setting = default.copy()
                        setting["MODEL_NAME"] = i
                        setting["NUM_LAYERS"] = j
                        setting["MODEL_TYPE"] = model_types[models.index(i)]
                        setting["NUM_ROUNDS"] = 1
                        setting["HIDDEN_SIZE"] = k
                        setting["LOCAL_EPOCHS"] = epoch
                        exp.append(setting)
        return exp
    
    def create_csc_experiments(experiment_name):
        exp = get_csc_experiments(experiment_name)
        return exp

    import glob

    results = []

    exp = get_srfca_experiments("Experiment3") #create_csc_experiments("Base Experiment")
    #for e in create_csc_experiments("Experiment2"):
    #    exp.append(e)
    generating_similarity_matrices = False
    

    for idx, exp_config in enumerate(exp):
        print(f"Running experiment {idx+1}/{len(exp)}")
        DEFAULT_CONFIG.update(exp_config) 
        DEFAULT_CONFIG["KEY"] = id_generator(DEFAULT_CONFIG)
        DEFAULT_CONFIG["LOCAL_EPOCHS"] = 1
        if generating_similarity_matrices:
            output_dir = "similarity_matrices" + "/" + str(DEFAULT_CONFIG["LOCAL_EPOCHS"]) + "/"
            pattern = f"{DEFAULT_CONFIG['MODEL_TYPE']}_{DEFAULT_CONFIG['NUM_LAYERS']}_{DEFAULT_CONFIG['HIDDEN_SIZE']}layers_similarity*.json"
            output_pattern = os.path.join(output_dir, pattern)

            matching_files = glob.glob(output_pattern)

            if matching_files:  # If list is not empty
                continue
        
        #if os.path.exists("results/" + DEFAULT_CONFIG["EXPERIMENT_NAME"] + ".json"):
        #    continue 

        result = run_experiment(DEFAULT_CONFIG)
        results.append(result)

    # Analyze and compare results
    print(results)

if __name__ == "__main__":
    try :
        main()
    except Exception as e:
        main()
