import flwr as fl
from typing import Dict, Optional, List, Tuple
from strategies.strategy import get_strategy

def start_server(num_rounds: int, fraction_fit: float, strategy_name: str, config):
    """Start Flower server with the specified strategy."""
    # Create strategy based on strategy_name
            # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(config),
    )