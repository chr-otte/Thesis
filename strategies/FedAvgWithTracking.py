import logging
from typing import Optional, Dict, List, Tuple, Union
import flwr as fl
from flwr.common import Scalar, EvaluateRes, FitRes, Parameters
from standardized_metric_tracking import metric_tracker

logger = logging.getLogger("FL-Strategy")

class FedAvgWithTracking(fl.server.strategy.FedAvg):
    """Strategy that extends FedAvg to track client metrics for overfitting detection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.client_metrics = {}  # Store metrics across rounds
        
        # Initialize structures to track metrics over time
        self.training_history = {}  # Client training metrics by round
        self.evaluation_history = {}  # Client evaluation metrics by round
        logger.info("FedAvg with tracking initialized")
        self.latest_parameters = None

    def fit_config_fn(self, server_round: int):
        """Return training configuration for clients."""
        return {
            "server_round": server_round,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and track client training metrics."""
        self.current_round = server_round
                
        # Call the parent's aggregate_fit
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Store the aggregated parameters
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and track client performance metrics."""
        # Track client evaluation metrics for this round
        client_ids = []
        for client, eval_res in results: 
            client_id = eval_res.metrics["Id"]
            client_ids.append(client_id)
            metric_tracker.log_training_round(server_round, client_id, eval_res.loss)
            metric_tracker.log_pretraining_round(server_round, client_id, 0) # single global model
        metric_tracker.log_cluster_state(server_round, {0:client_ids})

        # Call the parent's aggregate_evaluate
        return super().aggregate_evaluate(server_round, results, failures)
    
    @property
    def parameters(self):
        """Access to the latest model parameters."""
        return self.latest_parameters
