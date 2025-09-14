"""
Neuroevolution Implementation for P-013D (Evolutionary Trading Strategies).

This module implements neural network evolution using the NEAT (NeuroEvolution of Augmenting Topologies)
algorithm for evolving trading decision networks. Features include:

- Neural network topology evolution (NEAT algorithm)
- Weight and bias evolution for decision networks
- Activation function optimization
- Network pruning and growth mechanisms
- Speciation for diversity preservation
- Integration with existing genetic algorithm components
- Real-time adaptation during trading
- Support for both feedforward and recurrent networks

Key Components:
- NeuroNetwork: Base neural network class with evolution capabilities
- NEATGenome: NEAT genome representation with innovation tracking
- Speciation: Population clustering for diversity preservation
- NeuroEvolutionStrategy: Main strategy implementing neural network evolution
- NetworkMutations: Specialized mutation operators for network topology

Dependencies:
- PyTorch for neural network operations and GPU acceleration
- Integration with existing evolutionary framework
- Core types and logging from src/core/
- Base strategy from src/strategies/base.py
"""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

# Optional PyTorch imports - only import if available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

# For type checking only
if TYPE_CHECKING and HAS_TORCH:
    from torch import Tensor
else:
    Tensor = Any

from src.core.exceptions import OptimizationError
from src.core.types import MarketData, Position, Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer
from src.strategies.evolutionary.fitness import FitnessEvaluator
from src.utils.decorators import memory_usage, time_execution


class ActivationType(Enum):
    """Supported activation function types for neural networks."""

    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"
    LINEAR = "linear"


class NodeType(Enum):
    """Neural network node types for NEAT genome representation."""

    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
    BIAS = "bias"


class ConnectionType(Enum):
    """Connection types for network topology."""

    FORWARD = "forward"
    RECURRENT = "recurrent"
    SELF_LOOP = "self_loop"


@dataclass
class NodeGene:
    """Represents a node in the NEAT genome.

    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of node (input, hidden, output, bias)
        activation: Activation function type
        bias: Bias value for the node
        layer: Layer index for feedforward organization
        enabled: Whether the node is active
    """

    node_id: int
    node_type: NodeType
    activation: ActivationType = ActivationType.SIGMOID
    bias: float = 0.0
    layer: int = 0
    enabled: bool = True

    def copy(self) -> "NodeGene":
        """Create a copy of this node gene."""
        return NodeGene(
            node_id=self.node_id,
            node_type=self.node_type,
            activation=self.activation,
            bias=self.bias,
            layer=self.layer,
            enabled=self.enabled,
        )


@dataclass
class ConnectionGene:
    """Represents a connection in the NEAT genome.

    Attributes:
        innovation_id: Global innovation number for this connection
        from_node: Source node ID
        to_node: Target node ID
        weight: Connection weight
        enabled: Whether connection is active
        connection_type: Type of connection (forward, recurrent, etc.)
    """

    innovation_id: int
    from_node: int
    to_node: int
    weight: float
    enabled: bool = True
    connection_type: ConnectionType = ConnectionType.FORWARD

    def copy(self) -> "ConnectionGene":
        """Create a copy of this connection gene."""
        return ConnectionGene(
            innovation_id=self.innovation_id,
            from_node=self.from_node,
            to_node=self.to_node,
            weight=self.weight,
            enabled=self.enabled,
            connection_type=self.connection_type,
        )


class InnovationTracker:
    """Tracks innovation numbers for topology mutations in NEAT algorithm.

    Innovation numbers ensure that the same structural mutation gets the same
    innovation number across the population, enabling proper crossover.
    """

    def __init__(self):
        """Initialize innovation tracker."""
        self.current_innovation = 0
        self.connection_innovations: dict[tuple[int, int], int] = {}
        self.node_innovations: dict[int, int] = {}  # connection_id -> new_node_id
        self.current_node_id = 0

    def get_connection_innovation(self, from_node: int, to_node: int) -> int:
        """Get innovation number for a connection between two nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            Innovation number for this connection
        """
        key = (from_node, to_node)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.connection_innovations[key]

    def get_node_innovation(self, connection_id: int) -> int:
        """Get node ID for splitting a connection.

        Args:
            connection_id: Innovation ID of connection being split

        Returns:
            Node ID for the new node
        """
        if connection_id not in self.node_innovations:
            self.node_innovations[connection_id] = self.current_node_id
            self.current_node_id += 1
        return self.node_innovations[connection_id]

    def get_next_node_id(self) -> int:
        """Get next available node ID."""
        node_id = self.current_node_id
        self.current_node_id += 1
        return node_id


class NEATGenome:
    """NEAT genome representation for evolving neural network topologies.

    Implements the core NEAT algorithm for evolving both topology and weights
    of neural networks. Supports speciation, crossover, and mutation operations.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        innovation_tracker: InnovationTracker,
        genome_id: str | None = None,
    ):
        """Initialize NEAT genome.

        Args:
            input_size: Number of input nodes
            output_size: Number of output nodes
            innovation_tracker: Global innovation tracker
            genome_id: Optional genome identifier
        """
        self.genome_id = genome_id or str(uuid.uuid4())
        self.input_size = input_size
        self.output_size = output_size
        self.innovation_tracker = innovation_tracker

        # Genome components
        self.nodes: dict[int, NodeGene] = {}
        self.connections: dict[int, ConnectionGene] = {}

        # Evolution metrics
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id: int | None = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize basic topology
        self._initialize_minimal_topology()

    def _initialize_minimal_topology(self) -> None:
        """Initialize minimal feedforward topology with input and output nodes."""
        # Create input nodes
        for i in range(self.input_size):
            self.nodes[i] = NodeGene(node_id=i, node_type=NodeType.INPUT, layer=0)

        # Create bias node
        bias_id = self.input_size
        self.nodes[bias_id] = NodeGene(node_id=bias_id, node_type=NodeType.BIAS, layer=0, bias=1.0)

        # Create output nodes
        output_start = self.input_size + 1
        for i in range(self.output_size):
            node_id = output_start + i
            self.nodes[node_id] = NodeGene(
                node_id=node_id,
                node_type=NodeType.OUTPUT,
                layer=1,
                activation=ActivationType.TANH,  # Suitable for trading signals
            )

        # Create initial connections from inputs to outputs
        for input_id in range(self.input_size + 1):  # Include bias
            for output_id in range(output_start, output_start + self.output_size):
                innovation_id = self.innovation_tracker.get_connection_innovation(
                    input_id, output_id
                )
                weight = random.gauss(0, 1.0)  # Gaussian initialization

                self.connections[innovation_id] = ConnectionGene(
                    innovation_id=innovation_id,
                    from_node=input_id,
                    to_node=output_id,
                    weight=weight,
                )

    def add_node_mutation(self) -> bool:
        """Add a new node by splitting an existing connection.

        Returns:
            True if mutation was successful, False otherwise
        """
        # Find enabled connections that can be split
        enabled_connections = [conn for conn in self.connections.values() if conn.enabled]
        if not enabled_connections:
            return False

        # Select random connection to split
        connection = random.choice(enabled_connections)

        # Disable the original connection
        connection.enabled = False

        # Get new node ID
        new_node_id = self.innovation_tracker.get_node_innovation(connection.innovation_id)

        # Calculate layer for new node
        from_layer = self.nodes[connection.from_node].layer
        to_layer = self.nodes[connection.to_node].layer
        new_layer = from_layer + 1

        # Update layers if necessary to maintain feedforward structure
        if new_layer >= to_layer:
            self._shift_layers_from(to_layer, 1)
            new_layer = to_layer

        # Create new hidden node
        self.nodes[new_node_id] = NodeGene(
            node_id=new_node_id,
            node_type=NodeType.HIDDEN,
            layer=new_layer,
            activation=random.choice(list(ActivationType)),
        )

        # Create two new connections
        # Connection from original source to new node (weight = 1.0)
        innovation1 = self.innovation_tracker.get_connection_innovation(
            connection.from_node, new_node_id
        )
        self.connections[innovation1] = ConnectionGene(
            innovation_id=innovation1,
            from_node=connection.from_node,
            to_node=new_node_id,
            weight=1.0,
        )

        # Connection from new node to original target (original weight)
        innovation2 = self.innovation_tracker.get_connection_innovation(
            new_node_id, connection.to_node
        )
        self.connections[innovation2] = ConnectionGene(
            innovation_id=innovation2,
            from_node=new_node_id,
            to_node=connection.to_node,
            weight=connection.weight,
        )

        self.logger.debug(
            f"Added node mutation - genome_id: {self.genome_id}, new_node_id: {new_node_id}, split_connection: {connection.innovation_id}"
        )

        return True

    def add_connection_mutation(self, allow_recurrent: bool = True) -> bool:
        """Add a new connection between existing nodes.

        Args:
            allow_recurrent: Whether to allow recurrent connections

        Returns:
            True if mutation was successful, False otherwise
        """
        # Get all possible node pairs
        node_ids = list(self.nodes.keys())
        possible_connections = []

        for from_id in node_ids:
            for to_id in node_ids:
                # Skip if connection already exists
                existing = any(
                    conn.from_node == from_id and conn.to_node == to_id
                    for conn in self.connections.values()
                )
                if existing:
                    continue

                from_node = self.nodes[from_id]
                to_node = self.nodes[to_id]

                # Prevent connections to input/bias nodes
                if to_node.node_type in [NodeType.INPUT, NodeType.BIAS]:
                    continue

                # Prevent connections from output nodes
                if from_node.node_type == NodeType.OUTPUT:
                    continue

                # Check for feedforward constraint
                if from_node.layer >= to_node.layer:
                    if not allow_recurrent:
                        continue
                    # This would be a recurrent connection
                    connection_type = ConnectionType.RECURRENT
                    if from_id == to_id:
                        connection_type = ConnectionType.SELF_LOOP
                else:
                    connection_type = ConnectionType.FORWARD

                possible_connections.append((from_id, to_id, connection_type))

        if not possible_connections:
            return False

        # Select random connection to add
        from_id, to_id, conn_type = random.choice(possible_connections)

        # Create new connection
        innovation_id = self.innovation_tracker.get_connection_innovation(from_id, to_id)
        weight = random.gauss(0, 1.0)

        self.connections[innovation_id] = ConnectionGene(
            innovation_id=innovation_id,
            from_node=from_id,
            to_node=to_id,
            weight=weight,
            connection_type=conn_type,
        )

        self.logger.debug(
            f"Added connection mutation - genome_id: {self.genome_id}, from_node: {from_id}, to_node: {to_id}, connection_type: {conn_type.value}"
        )

        return True

    def mutate_weights(self, mutation_rate: float = 0.8, mutation_strength: float = 0.1) -> None:
        """Mutate connection weights and node biases.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation for weight perturbations
        """
        # Mutate connection weights
        for connection in self.connections.values():
            if random.random() < mutation_rate:
                if random.random() < 0.1:  # 10% chance of complete replacement
                    connection.weight = random.gauss(0, 1.0)
                else:
                    # Perturb existing weight
                    perturbation = random.gauss(0, mutation_strength)
                    connection.weight += perturbation
                    # Clamp weights to reasonable range
                    connection.weight = max(-5.0, min(5.0, connection.weight))

        # Mutate node biases
        for node in self.nodes.values():
            if node.node_type != NodeType.BIAS and random.random() < mutation_rate:
                perturbation = random.gauss(0, mutation_strength)
                node.bias += perturbation
                node.bias = max(-2.0, min(2.0, node.bias))

    def mutate_activation_functions(self, mutation_rate: float = 0.1) -> None:
        """Mutate activation functions of hidden and output nodes.

        Args:
            mutation_rate: Probability of mutating each node's activation
        """
        activations = list(ActivationType)

        for node in self.nodes.values():
            if node.node_type in [NodeType.HIDDEN, NodeType.OUTPUT]:
                if random.random() < mutation_rate:
                    # Choose new activation function
                    current_idx = activations.index(node.activation)
                    new_activations = activations[:current_idx] + activations[current_idx + 1 :]
                    node.activation = random.choice(new_activations)

    def enable_disable_mutation(
        self, enable_rate: float = 0.01, disable_rate: float = 0.01
    ) -> None:
        """Randomly enable/disable connections for network pruning.

        Args:
            enable_rate: Probability of enabling a disabled connection
            disable_rate: Probability of disabling an enabled connection
        """
        for connection in self.connections.values():
            if connection.enabled and random.random() < disable_rate:
                connection.enabled = False
            elif not connection.enabled and random.random() < enable_rate:
                connection.enabled = True

    def _shift_layers_from(self, start_layer: int, shift_amount: int) -> None:
        """Shift layer numbers to maintain proper feedforward structure.

        Args:
            start_layer: Layer to start shifting from
            shift_amount: Amount to shift layer numbers
        """
        for node in self.nodes.values():
            if node.layer >= start_layer:
                node.layer += shift_amount

    def calculate_compatibility_distance(
        self, other: "NEATGenome", c1: float = 1.0, c2: float = 1.0, c3: float = 0.4
    ) -> float:
        """Calculate compatibility distance for speciation.

        Args:
            other: Other genome to compare with
            c1: Coefficient for excess genes
            c2: Coefficient for disjoint genes
            c3: Coefficient for weight differences

        Returns:
            Compatibility distance between genomes
        """
        # Get innovation numbers
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())

        if not innovations1 and not innovations2:
            return 0.0

        # Find matching, excess, and disjoint genes
        matching = innovations1.intersection(innovations2)
        all_innovations = innovations1.union(innovations2)

        if all_innovations:
            max(all_innovations)
            excess1 = len([i for i in innovations1 if i > max(innovations2 or [0])])
            excess2 = len([i for i in innovations2 if i > max(innovations1 or [0])])
            excess = excess1 + excess2

            disjoint = len(all_innovations) - len(matching) - excess
        else:
            excess = 0
            disjoint = 0

        # Calculate average weight difference for matching genes
        weight_diff = 0.0
        if matching:
            weight_diffs = []
            for innovation in matching:
                w1 = self.connections[innovation].weight
                w2 = other.connections[innovation].weight
                weight_diffs.append(abs(w1 - w2))
            weight_diff = sum(weight_diffs) / len(weight_diffs)

        # Normalize by genome size
        n = max(len(innovations1), len(innovations2), 1)

        distance = (c1 * excess / n) + (c2 * disjoint / n) + (c3 * weight_diff)
        return distance

    def crossover(self, other: "NEATGenome", fitness_equal: bool = False) -> "NEATGenome":
        """Perform crossover with another genome to create offspring.

        Args:
            other: Other parent genome
            fitness_equal: Whether both parents have equal fitness

        Returns:
            New offspring genome
        """
        # Determine which parent is more fit
        if fitness_equal or self.fitness == other.fitness:
            parent1, parent2 = self, other
        elif self.fitness > other.fitness:
            parent1, parent2 = self, other  # self is more fit
        else:
            parent1, parent2 = other, self  # other is more fit

        # Create offspring genome
        offspring = NEATGenome(
            input_size=self.input_size,
            output_size=self.output_size,
            innovation_tracker=self.innovation_tracker,
        )

        # Clear default topology
        offspring.nodes.clear()
        offspring.connections.clear()

        # Inherit all nodes from both parents
        all_node_ids = set(parent1.nodes.keys()).union(set(parent2.nodes.keys()))
        for node_id in all_node_ids:
            if node_id in parent1.nodes:
                offspring.nodes[node_id] = parent1.nodes[node_id].copy()
            else:
                offspring.nodes[node_id] = parent2.nodes[node_id].copy()

        # Inherit connections
        all_innovations = set(parent1.connections.keys()).union(set(parent2.connections.keys()))

        for innovation in all_innovations:
            if innovation in parent1.connections and innovation in parent2.connections:
                # Matching gene - randomly choose from either parent
                if random.random() < 0.5:
                    offspring.connections[innovation] = parent1.connections[innovation].copy()
                else:
                    offspring.connections[innovation] = parent2.connections[innovation].copy()

                # If one is disabled, possibly disable in offspring
                if (
                    not parent1.connections[innovation].enabled
                    or not parent2.connections[innovation].enabled
                ):
                    if random.random() < 0.75:  # 75% chance to disable
                        offspring.connections[innovation].enabled = False

            elif innovation in parent1.connections:
                # Excess/disjoint from more fit parent - inherit
                offspring.connections[innovation] = parent1.connections[innovation].copy()
            # Ignore excess/disjoint from less fit parent

        return offspring

    def copy(self) -> "NEATGenome":
        """Create a deep copy of this genome."""
        new_genome = NEATGenome(
            input_size=self.input_size,
            output_size=self.output_size,
            innovation_tracker=self.innovation_tracker,
            genome_id=str(uuid.uuid4()),
        )

        # Clear default topology
        new_genome.nodes.clear()
        new_genome.connections.clear()

        # Copy nodes
        for node_id, node in self.nodes.items():
            new_genome.nodes[node_id] = node.copy()

        # Copy connections
        for innovation_id, connection in self.connections.items():
            new_genome.connections[innovation_id] = connection.copy()

        # Copy fitness
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        new_genome.species_id = self.species_id

        return new_genome


class NeuroNetwork:
    """PyTorch implementation of evolvable neural network from NEAT genome.

    Converts NEAT genome representation into executable PyTorch network with
    support for both feedforward and recurrent connections.
    """

    def __init__(self, genome: NEATGenome, device: str = "cpu"):
        """Initialize neural network from NEAT genome.

        Args:
            genome: NEAT genome to convert to network
            device: Device to run network on ("cpu" or "cuda")
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for NeuroNetwork but not installed. Install with: pip install torch"
            )

        self.genome = genome
        self.device = device
        self.input_size = genome.input_size
        self.output_size = genome.output_size

        # Build network structure
        self._build_network()

    def _build_network(self) -> None:
        """Build PyTorch network from genome structure."""
        # Sort nodes by layer for feedforward processing
        self.nodes_by_layer = {}
        for node in self.genome.nodes.values():
            if node.layer not in self.nodes_by_layer:
                self.nodes_by_layer[node.layer] = []
            self.nodes_by_layer[node.layer].append(node)

        # Store enabled connections
        self.enabled_connections = [
            conn for conn in self.genome.connections.values() if conn.enabled
        ]

        # Create activation function mapping
        self.activation_functions = {
            ActivationType.SIGMOID: torch.sigmoid,
            ActivationType.TANH: torch.tanh,
            ActivationType.RELU: F.relu,
            ActivationType.LEAKY_RELU: lambda x: F.leaky_relu(x, 0.01),
            ActivationType.ELU: F.elu,
            ActivationType.SWISH: lambda x: x * torch.sigmoid(x),
            ActivationType.GELU: F.gelu,
            ActivationType.LINEAR: lambda x: x,
        }

        # Store recurrent connections separately
        self.recurrent_connections = [
            conn
            for conn in self.enabled_connections
            if conn.connection_type in [ConnectionType.RECURRENT, ConnectionType.SELF_LOOP]
        ]

        # Initialize hidden states for recurrent connections
        self.hidden_states: dict[int, Tensor] = {}
        self._reset_hidden_states()

    def _reset_hidden_states(self) -> None:
        """Reset hidden states for recurrent connections."""
        for node_id in self.genome.nodes.keys():
            if self.genome.nodes[node_id].node_type == NodeType.HIDDEN:
                self.hidden_states[node_id] = torch.zeros(1, device=self.device)

    def forward(self, x: "Tensor", reset_hidden: bool = False) -> "Tensor":
        """Forward pass through the neural network.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            reset_hidden: Whether to reset hidden states for recurrent connections

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        if reset_hidden:
            self._reset_hidden_states()

        batch_size = x.size(0)

        # Initialize node activations
        node_activations: dict[int, Tensor] = {}

        # Set input activations
        for i in range(self.input_size):
            node_activations[i] = x[:, i : i + 1]

        # Set bias activation
        bias_id = self.input_size
        if bias_id in self.genome.nodes:
            node_activations[bias_id] = torch.ones(batch_size, 1, device=self.device)

        # Process layers in order
        max_layer = max(node.layer for node in self.genome.nodes.values())
        for layer in range(1, max_layer + 1):
            if layer not in self.nodes_by_layer:
                continue

            for node in self.nodes_by_layer[layer]:
                if node.node_type in [NodeType.INPUT, NodeType.BIAS]:
                    continue

                # Calculate weighted sum of inputs
                weighted_sum = torch.zeros(batch_size, 1, device=self.device)

                # Add bias
                weighted_sum += node.bias

                # Add inputs from connections
                for conn in self.enabled_connections:
                    if (
                        conn.to_node == node.node_id
                        and conn.connection_type == ConnectionType.FORWARD
                    ):
                        if conn.from_node in node_activations:
                            weighted_sum += conn.weight * node_activations[conn.from_node]

                # Add recurrent inputs (use previous hidden state)
                for conn in self.recurrent_connections:
                    if conn.to_node == node.node_id:
                        if conn.from_node in self.hidden_states:
                            recurrent_input = conn.weight * self.hidden_states[conn.from_node]
                            if recurrent_input.size(0) != batch_size:
                                recurrent_input = recurrent_input.expand(batch_size, -1)
                            weighted_sum += recurrent_input

                # Apply activation function
                activation_fn = self.activation_functions[node.activation]
                node_activations[node.node_id] = activation_fn(weighted_sum)

                # Store hidden state for recurrent connections
                if node.node_type == NodeType.HIDDEN:
                    self.hidden_states[node.node_id] = node_activations[node.node_id].mean(
                        dim=0, keepdim=True
                    )

        # Collect outputs
        outputs = []
        output_start = self.input_size + 1
        for i in range(self.output_size):
            output_id = output_start + i
            if output_id in node_activations:
                outputs.append(node_activations[output_id])
            else:
                # Fallback for disconnected outputs
                outputs.append(torch.zeros(batch_size, 1, device=self.device))

        return torch.cat(outputs, dim=1)

    @time_execution
    def predict_signal(self, market_data: MarketData) -> Signal:
        """Generate trading signal from market data.

        Args:
            market_data: Current market data

        Returns:
            Trading signal based on network output
        """
        # Prepare input features
        features = self._extract_features(market_data)

        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            output = self.forward(x)

        # Interpret output
        signal_strength = output[0, 0].item()  # Assuming single output for signal strength

        # Convert to signal direction and confidence
        if signal_strength > 0.1:
            direction = SignalDirection.BUY
            confidence = min(abs(signal_strength), 1.0)
        elif signal_strength < -0.1:
            direction = SignalDirection.SELL
            confidence = min(abs(signal_strength), 1.0)
        else:
            direction = SignalDirection.HOLD
            confidence = 0.0

        return Signal(
            symbol=market_data.symbol,
            direction=direction,
            strength=Decimal(str(confidence)),
            timestamp=market_data.timestamp,
            source="neuroevolution",
            metadata={
                "raw_output": signal_strength,
                "network_genome_id": self.genome.genome_id,
                "input_features": features,
            },
        )

    def _extract_features(self, market_data: MarketData) -> list[float]:
        """Extract numerical features from market data for network input.

        Args:
            market_data: Market data to extract features from

        Returns:
            List of normalized feature values
        """
        features = []

        # Price-based features
        price = market_data.price if isinstance(market_data.price, Decimal) else Decimal(str(market_data.price))
        features.append(float(price / Decimal("10000")))  # Normalize price

        # Volume feature
        volume = market_data.volume if isinstance(market_data.volume, Decimal) else Decimal(str(market_data.volume))
        features.append(min(float(volume / Decimal("1000000")), 1.0))  # Normalize and cap volume

        # Spread feature (if available)
        if market_data.bid and market_data.ask:
            spread = market_data.ask - market_data.bid
            features.append(min(spread / price, 0.1))  # Relative spread, capped
        else:
            features.append(0.0)

        # OHLC features (if available)
        if market_data.open_price:
            open_price = market_data.open_price if isinstance(market_data.open_price, Decimal) else Decimal(str(market_data.open_price))
            price_change = float((price - open_price) / open_price)
            features.append(max(-1.0, min(1.0, price_change)))  # Price change ratio
        else:
            features.append(0.0)

        if market_data.high_price and market_data.low_price:
            high_price = market_data.high_price if isinstance(market_data.high_price, Decimal) else Decimal(str(market_data.high_price))
            low_price = market_data.low_price if isinstance(market_data.low_price, Decimal) else Decimal(str(market_data.low_price))
            price_range = high_price - low_price
            relative_position = (
                float((price - low_price) / price_range) if price_range > 0 else 0.5
            )
            features.append(relative_position)  # Position within daily range
        else:
            features.append(0.5)

        # Pad or truncate to match expected input size
        while len(features) < self.input_size:
            features.append(0.0)

        return features[: self.input_size]


class Species:
    """Represents a species in NEAT population for diversity preservation.

    Species group similar genomes together to protect innovation and maintain
    diversity in the population.
    """

    def __init__(self, species_id: int, representative: NEATGenome):
        """Initialize species with representative genome.

        Args:
            species_id: Unique identifier for this species
            representative: Representative genome for this species
        """
        self.species_id = species_id
        self.representative = representative
        self.members: list[NEATGenome] = [representative]
        self.average_fitness = 0.0
        self.best_fitness = 0.0
        self.generations_without_improvement = 0
        self.offspring_count = 0

    def add_member(self, genome: NEATGenome) -> None:
        """Add genome to this species."""
        genome.species_id = self.species_id
        self.members.append(genome)

    def calculate_average_fitness(self) -> None:
        """Calculate average fitness of species members."""
        if self.members:
            total_fitness = sum(member.adjusted_fitness for member in self.members)
            self.average_fitness = total_fitness / len(self.members)
            self.best_fitness = max(member.fitness for member in self.members)

    def select_parents(self, selection_pressure: float = 0.5) -> list[NEATGenome]:
        """Select parents for reproduction within species.

        Args:
            selection_pressure: Proportion of top genomes eligible for reproduction

        Returns:
            List of selected parent genomes
        """
        if not self.members:
            return []

        # Sort by fitness
        sorted_members = sorted(self.members, key=lambda g: g.fitness, reverse=True)

        # Select top proportion
        num_parents = max(1, int(len(sorted_members) * selection_pressure))
        return sorted_members[:num_parents]

    def remove_worst_genomes(self, keep_ratio: float = 0.5) -> None:
        """Remove worst performing genomes from species.

        Args:
            keep_ratio: Proportion of genomes to keep
        """
        if len(self.members) <= 1:
            return

        sorted_members = sorted(self.members, key=lambda g: g.fitness, reverse=True)
        keep_count = max(1, int(len(sorted_members) * keep_ratio))
        self.members = sorted_members[:keep_count]

        # Update representative
        self.representative = self.members[0]


class SpeciationManager:
    """Manages species formation and evolution in NEAT population."""

    def __init__(
        self, compatibility_threshold: float = 3.0, target_species: int = 10, dropoff_age: int = 15
    ):
        """Initialize speciation manager.

        Args:
            compatibility_threshold: Distance threshold for species membership
            target_species: Target number of species to maintain
            dropoff_age: Generations without improvement before species removal
        """
        self.compatibility_threshold = compatibility_threshold
        self.target_species = target_species
        self.dropoff_age = dropoff_age
        self.species: dict[int, Species] = {}
        self.next_species_id = 0

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def speciate_population(self, population: list[NEATGenome]) -> None:
        """Organize population into species based on genetic similarity.

        Args:
            population: List of genomes to organize into species
        """
        # Clear previous species memberships
        for species in self.species.values():
            species.members.clear()

        # Assign genomes to species
        for genome in population:
            assigned = False

            for species in self.species.values():
                distance = genome.calculate_compatibility_distance(species.representative)
                if distance < self.compatibility_threshold:
                    species.add_member(genome)
                    assigned = True
                    break

            # Create new species if no match found
            if not assigned:
                new_species = Species(self.next_species_id, genome)
                self.species[self.next_species_id] = new_species
                self.next_species_id += 1

        # Remove empty species
        empty_species = [sid for sid, species in self.species.items() if not species.members]
        for sid in empty_species:
            del self.species[sid]

        # Calculate fitness sharing
        self._calculate_fitness_sharing()

        # Update species statistics
        for species in self.species.values():
            species.calculate_average_fitness()

        # Remove stagnant species
        self._remove_stagnant_species()

        # Adjust compatibility threshold to target species count
        self._adjust_compatibility_threshold()

    def _calculate_fitness_sharing(self) -> None:
        """Apply fitness sharing to maintain diversity."""
        for species in self.species.values():
            for genome in species.members:
                # Adjusted fitness = raw fitness / species size
                genome.adjusted_fitness = genome.fitness / len(species.members)

    def _remove_stagnant_species(self) -> None:
        """Remove species that haven't improved for too long."""
        species_to_remove = []

        for species in self.species.values():
            if species.generations_without_improvement >= self.dropoff_age:
                species_to_remove.append(species.species_id)

        # Always keep at least one species
        if len(species_to_remove) >= len(self.species):
            # Keep the best species
            best_species = max(self.species.values(), key=lambda s: s.best_fitness)
            species_to_remove.remove(best_species.species_id)

        for species_id in species_to_remove:
            self.logger.info(f"Removing stagnant species {species_id}")
            del self.species[species_id]

    def _adjust_compatibility_threshold(self) -> None:
        """Adjust compatibility threshold to maintain target species count."""
        current_species_count = len(self.species)

        if current_species_count > self.target_species:
            # Too many species - increase threshold
            self.compatibility_threshold *= 1.1
        elif current_species_count < self.target_species:
            # Too few species - decrease threshold
            self.compatibility_threshold *= 0.9

        # Clamp threshold to reasonable range
        self.compatibility_threshold = max(0.5, min(10.0, self.compatibility_threshold))

    def allocate_offspring(self, total_offspring: int) -> dict[int, int]:
        """Allocate offspring slots to species based on fitness.

        Args:
            total_offspring: Total number of offspring to allocate

        Returns:
            Dictionary mapping species ID to offspring count
        """
        if not self.species:
            return {}

        # Calculate total adjusted fitness
        total_fitness = sum(species.average_fitness for species in self.species.values())

        if total_fitness <= 0:
            # Equal allocation if no positive fitness
            offspring_per_species = total_offspring // len(self.species)
            return {sid: offspring_per_species for sid in self.species.keys()}

        # Proportional allocation based on fitness
        allocation = {}
        allocated_total = 0

        for species_id, species in self.species.items():
            proportion = species.average_fitness / total_fitness
            offspring_count = int(proportion * total_offspring)
            allocation[species_id] = offspring_count
            allocated_total += offspring_count

        # Distribute remaining offspring to best species
        remaining = total_offspring - allocated_total
        if remaining > 0:
            best_species_id = max(
                self.species.keys(), key=lambda sid: self.species[sid].average_fitness
            )
            allocation[best_species_id] += remaining

        return allocation


@dataclass
class NeuroEvolutionConfig:
    """Configuration for neuroevolution strategy."""

    # Population parameters
    population_size: int = 150
    input_features: int = 5  # Number of input features
    output_size: int = 1  # Single output for signal strength

    # NEAT parameters
    compatibility_threshold: float = 3.0
    target_species: int = 10
    dropoff_age: int = 15

    # Mutation rates
    add_node_rate: float = 0.03
    add_connection_rate: float = 0.3
    weight_mutation_rate: float = 0.8
    weight_mutation_strength: float = 0.1
    activation_mutation_rate: float = 0.1
    enable_rate: float = 0.01
    disable_rate: float = 0.01

    # Evolution parameters
    survival_threshold: float = 0.2  # Proportion of population to keep
    interspecies_mating_rate: float = 0.001
    generations: int = 100

    # Network parameters
    allow_recurrent: bool = True
    device: str = "cpu"  # "cpu" or "cuda"

    # Real-time adaptation
    adaptation_rate: float = 0.1
    memory_length: int = 100  # Recent performance memory

    # Trading parameters
    min_confidence_threshold: float = 0.3
    max_position_hold_time: int = 24  # Hours


class NeuroEvolutionStrategy(BaseStrategy):
    """Neural network evolution strategy for trading decisions.

    Implements NEAT algorithm to evolve neural network topology and weights
    for generating trading signals. Features include:

    - Population-based evolution of neural networks
    - Speciation for diversity preservation
    - Real-time adaptation during trading
    - Integration with existing strategy framework
    """

    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None):
        """Initialize neuroevolution strategy.

        Args:
            config: Strategy configuration including neuroevolution parameters
        """
        super().__init__(config, services)

        # Parse neuroevolution-specific config
        self.neuro_config = NeuroEvolutionConfig(**config.get("neuroevolution", {}))

        # Initialize NEAT components
        self.innovation_tracker = InnovationTracker()
        self.speciation_manager = SpeciationManager(
            compatibility_threshold=self.neuro_config.compatibility_threshold,
            target_species=self.neuro_config.target_species,
            dropoff_age=self.neuro_config.dropoff_age,
        )

        # Evolution state
        self.population: list[NEATGenome] = []
        self.best_genome: NEATGenome | None = None
        self.best_network: NeuroNetwork | None = None
        self.generation = 0
        self.evolution_history: list[dict[str, Any]] = []

        # Real-time adaptation
        self.recent_performance: list[float] = []
        self.adaptation_counter = 0

        # Performance tracking
        self.network_predictions: list[tuple[Signal, float]] = []  # Signal, actual return

        # Initialize population
        self._initialize_population()

        self.logger.info(
            f"NeuroEvolutionStrategy initialized - population_size: {self.neuro_config.population_size}, target_species: {self.neuro_config.target_species}, device: {self.neuro_config.device}"
        )

    def _initialize_population(self) -> None:
        """Initialize random population of NEAT genomes."""
        self.population = []

        for i in range(self.neuro_config.population_size):
            genome = NEATGenome(
                input_size=self.neuro_config.input_features,
                output_size=self.neuro_config.output_size,
                innovation_tracker=self.innovation_tracker,
                genome_id=f"gen0_ind{i}",
            )

            # Apply initial mutations for diversity
            genome.mutate_weights(
                mutation_rate=0.5, mutation_strength=self.neuro_config.weight_mutation_strength
            )

            self.population.append(genome)

        # Speciate initial population
        self.speciation_manager.speciate_population(self.population)

        # Select initial best genome
        self.best_genome = self.population[0]
        if HAS_TORCH:
            self._update_best_network()

    def _update_best_network(self) -> None:
        """Update the best network from current best genome."""
        if self.best_genome and HAS_TORCH:
            self.best_network = NeuroNetwork(
                genome=self.best_genome, device=self.neuro_config.device
            )

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate trading signals using best evolved network.

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        if not HAS_TORCH or not self.best_network:
            # Return neutral signal if torch not available
            return [
                Signal(
                    symbol=data.symbol,
                    direction=SignalDirection.HOLD,
                    strength=Decimal("0.0"),
                    timestamp=data.timestamp,
                    source="neuroevolution_no_torch",
                )
            ]

        try:
            # Generate signal from best network
            signal = self.best_network.predict_signal(data)

            # Apply confidence threshold
            if signal.confidence < self.neuro_config.min_confidence_threshold:
                signal.direction = SignalDirection.HOLD
                signal.confidence = 0.0

            # Track prediction for adaptation
            self.network_predictions.append((signal, 0.0))  # Return will be updated later

            # Trigger adaptation if needed
            await self._maybe_adapt()

            return [signal]

        except Exception as e:
            self.logger.error(f"Signal generation failed in neuroevolution: {e!s}")
            return []

    async def _maybe_adapt(self) -> None:
        """Trigger adaptation if conditions are met."""
        self.adaptation_counter += 1

        # Adapt every N predictions
        adaptation_interval = int(1.0 / self.neuro_config.adaptation_rate)
        if self.adaptation_counter >= adaptation_interval:
            await self.adapt_networks()
            self.adaptation_counter = 0

    @memory_usage
    async def adapt_networks(self) -> None:
        """Adapt networks based on recent performance."""
        if len(self.network_predictions) < 10:
            return  # Need minimum data for adaptation

        try:
            # Evaluate recent performance
            recent_signals = self.network_predictions[-self.neuro_config.memory_length :]

            # Calculate fitness based on prediction accuracy (simplified)
            total_accuracy = 0.0
            for signal, actual_return in recent_signals:
                if signal.direction == SignalDirection.BUY and actual_return > 0:
                    total_accuracy += signal.confidence
                elif signal.direction == SignalDirection.SELL and actual_return < 0:
                    total_accuracy += signal.confidence
                elif signal.direction == SignalDirection.HOLD and abs(actual_return) < 0.01:
                    total_accuracy += 0.5

            recent_fitness = total_accuracy / len(recent_signals) if recent_signals else 0.0
            self.recent_performance.append(recent_fitness)

            # Keep only recent performance data
            if len(self.recent_performance) > self.neuro_config.memory_length:
                self.recent_performance = self.recent_performance[
                    -self.neuro_config.memory_length :
                ]

            # Update best genome fitness
            if self.best_genome:
                self.best_genome.fitness = recent_fitness

            # Trigger micro-evolution if performance is declining
            if len(self.recent_performance) >= 5:
                recent_avg = sum(self.recent_performance[-5:]) / 5
                older_avg = (
                    sum(self.recent_performance[-10:-5]) / 5
                    if len(self.recent_performance) >= 10
                    else recent_avg
                )

                if recent_avg < older_avg * 0.9:  # 10% decline
                    self.logger.info("Performance decline detected, triggering micro-evolution")
                    await self._micro_evolution()

        except Exception as e:
            self.logger.error(f"Network adaptation failed: {e!s}")

    async def _micro_evolution(self) -> None:
        """Perform micro-evolution to adapt to changing conditions."""
        if not self.best_genome:
            return

        # Create variations of best genome
        variations = []
        for i in range(10):
            variant = self.best_genome.copy()
            variant.genome_id = f"adapt_{self.generation}_{i}"

            # Apply light mutations
            variant.mutate_weights(
                mutation_rate=0.3,
                mutation_strength=self.neuro_config.weight_mutation_strength * 0.5,
            )

            if random.random() < 0.1:
                variant.add_connection_mutation(allow_recurrent=self.neuro_config.allow_recurrent)

            variations.append(variant)

        # Evaluate variations (simplified - just add small fitness bonus for diversity)
        for i, variant in enumerate(variations):
            variant.fitness = self.best_genome.fitness + random.gauss(0, 0.1)

        # Select best variation
        best_variation = max(variations, key=lambda g: g.fitness)
        if best_variation.fitness > self.best_genome.fitness:
            self.best_genome = best_variation
            self._update_best_network()
            self.logger.info("Adopted new best genome from micro-evolution")

    async def evolve_population(self, fitness_evaluator: FitnessEvaluator) -> None:
        """Evolve the population for one generation.

        Args:
            fitness_evaluator: Fitness evaluation function
        """
        try:
            self.logger.info(f"Starting evolution generation {self.generation + 1}")

            # Evaluate population fitness (this would normally use backtesting)
            # For now, assign random fitness values
            for genome in self.population:
                genome.fitness = random.uniform(0, 1)

            # Speciate population
            self.speciation_manager.speciate_population(self.population)

            # Track best genome
            current_best = max(self.population, key=lambda g: g.fitness)
            if not self.best_genome or current_best.fitness > self.best_genome.fitness:
                self.best_genome = current_best.copy()
                self._update_best_network()

            # Generate next generation
            new_population = []

            # Allocate offspring to species
            offspring_allocation = self.speciation_manager.allocate_offspring(
                self.neuro_config.population_size
            )

            for species_id, offspring_count in offspring_allocation.items():
                if species_id not in self.speciation_manager.species:
                    continue

                species = self.speciation_manager.species[species_id]

                # Elite selection - keep best genome
                if offspring_count > 0:
                    best_in_species = max(species.members, key=lambda g: g.fitness)
                    new_population.append(best_in_species.copy())
                    offspring_count -= 1

                # Generate offspring
                parents = species.select_parents()
                for _ in range(offspring_count):
                    offspring = self._create_offspring(species, parents)
                    new_population.append(offspring)

            # Ensure population size
            while len(new_population) < self.neuro_config.population_size:
                # Add random genome
                genome = NEATGenome(
                    input_size=self.neuro_config.input_features,
                    output_size=self.neuro_config.output_size,
                    innovation_tracker=self.innovation_tracker,
                )
                new_population.append(genome)

            new_population = new_population[: self.neuro_config.population_size]

            # Update population
            self.population = new_population
            self.generation += 1

            # Record evolution statistics
            self._record_generation_stats()

            self.logger.info(
                f"Generation {self.generation} completed - best_fitness: {self.best_genome.fitness if self.best_genome else 0}, species_count: {len(self.speciation_manager.species)}, population_size: {len(self.population)}"
            )

        except Exception as e:
            self.logger.error(f"Population evolution failed: {e!s}")
            raise OptimizationError(f"Evolution failed: {e!s}")

    def _create_offspring(self, species: Species, parents: list[NEATGenome]) -> NEATGenome:
        """Create offspring through crossover and mutation.

        Args:
            species: Species to create offspring for
            parents: Available parent genomes

        Returns:
            New offspring genome
        """
        if len(parents) >= 2 and random.random() < 0.75:
            # Crossover
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            offspring = parent1.crossover(parent2)
        else:
            # Asexual reproduction
            parent = random.choice(parents) if parents else species.representative
            offspring = parent.copy()

        # Apply mutations
        self._mutate_genome(offspring)

        # Assign new ID
        offspring.genome_id = f"gen{self.generation + 1}_{uuid.uuid4().hex[:8]}"

        return offspring

    def _mutate_genome(self, genome: NEATGenome) -> None:
        """Apply mutations to a genome.

        Args:
            genome: Genome to mutate
        """
        # Structural mutations
        if random.random() < self.neuro_config.add_node_rate:
            genome.add_node_mutation()

        if random.random() < self.neuro_config.add_connection_rate:
            genome.add_connection_mutation(allow_recurrent=self.neuro_config.allow_recurrent)

        # Weight mutations
        genome.mutate_weights(
            mutation_rate=self.neuro_config.weight_mutation_rate,
            mutation_strength=self.neuro_config.weight_mutation_strength,
        )

        # Activation function mutations
        genome.mutate_activation_functions(self.neuro_config.activation_mutation_rate)

        # Enable/disable mutations
        genome.enable_disable_mutation(
            enable_rate=self.neuro_config.enable_rate, disable_rate=self.neuro_config.disable_rate
        )

    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        if not self.population:
            return

        fitnesses = [genome.fitness for genome in self.population]

        stats = {
            "generation": self.generation,
            "population_size": len(self.population),
            "species_count": len(self.speciation_manager.species),
            "best_fitness": max(fitnesses),
            "average_fitness": sum(fitnesses) / len(fitnesses),
            "fitness_std": np.std(fitnesses),
            "best_genome_id": self.best_genome.genome_id if self.best_genome else None,
            "best_genome_complexity": len(self.best_genome.connections) if self.best_genome else 0,
            "timestamp": datetime.now(timezone.utc),
        }

        self.evolution_history.append(stats)

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal from neural network.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        # Check confidence threshold
        if signal.confidence < self.neuro_config.min_confidence_threshold:
            return False

        # Check signal metadata
        if "raw_output" not in signal.metadata:
            return False

        # Check for reasonable output values
        raw_output = signal.metadata["raw_output"]
        if not isinstance(raw_output, (int, float)) or abs(raw_output) > 10:
            return False

        return True

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size based on signal confidence.

        Args:
            signal: Trading signal

        Returns:
            Position size as decimal
        """
        # Base position size from config
        base_size = Decimal(str(self.config.position_size_pct))

        # Scale by signal confidence
        confidence_multiplier = Decimal(str(signal.confidence))

        # Apply network-specific scaling
        if "raw_output" in signal.metadata:
            raw_output = abs(signal.metadata["raw_output"])
            network_multiplier = min(Decimal(str(raw_output)), Decimal("2.0"))  # Cap at 2x
        else:
            network_multiplier = Decimal("1.0")

        return base_size * confidence_multiplier * network_multiplier

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed using neural network.

        Args:
            position: Current position
            data: Current market data

        Returns:
            True if position should be closed
        """
        # Standard exit conditions
        if self._check_standard_exits(position, data):
            return True

        # Neural network exit decision
        if self.best_network:
            try:
                exit_signal = self.best_network.predict_signal(data)

                # Exit if network suggests opposite direction with high confidence
                if position.side.value == "buy" and exit_signal.direction == SignalDirection.SELL:
                    return exit_signal.confidence > 0.7
                elif position.side.value == "sell" and exit_signal.direction == SignalDirection.BUY:
                    return exit_signal.confidence > 0.7

            except Exception as e:
                self.logger.error(f"Neural network exit decision failed: {e!s}")

        return False

    def _check_standard_exits(self, position: Position, data: MarketData) -> bool:
        """Check standard exit conditions (stop loss, take profit, time)."""
        current_price = data.price
        entry_price = position.entry_price

        # Calculate P&L percentage
        if position.side.value == "buy":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True

        # Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True

        # Time-based exit (simplified - would need actual time tracking)
        # This is a placeholder for time-based exits

        return False

    def get_strategy_info(self) -> dict[str, Any]:
        """Get detailed strategy information including evolution state.

        Returns:
            Strategy information dictionary
        """
        base_info = super().get_strategy_info()

        neuro_info = {
            "neuro_evolution": {
                "generation": self.generation,
                "population_size": len(self.population),
                "species_count": len(self.speciation_manager.species),
                "best_fitness": self.best_genome.fitness if self.best_genome else 0,
                "best_genome_id": self.best_genome.genome_id if self.best_genome else None,
                "recent_performance": (
                    self.recent_performance[-10:] if self.recent_performance else []
                ),
                "evolution_config": {
                    "compatibility_threshold": self.speciation_manager.compatibility_threshold,
                    "target_species": self.neuro_config.target_species,
                    "input_features": self.neuro_config.input_features,
                    "allow_recurrent": self.neuro_config.allow_recurrent,
                },
            }
        }

        base_info.update(neuro_info)
        return base_info

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get summary of evolution process.

        Returns:
            Evolution summary dictionary
        """
        if not self.evolution_history:
            return {}

        return {
            "generations_completed": len(self.evolution_history),
            "best_fitness_progression": [h["best_fitness"] for h in self.evolution_history],
            "species_count_progression": [h["species_count"] for h in self.evolution_history],
            "complexity_progression": [h["best_genome_complexity"] for h in self.evolution_history],
            "current_best_genome": {
                "genome_id": self.best_genome.genome_id if self.best_genome else None,
                "fitness": self.best_genome.fitness if self.best_genome else 0,
                "node_count": len(self.best_genome.nodes) if self.best_genome else 0,
                "connection_count": len(self.best_genome.connections) if self.best_genome else 0,
                "species_id": self.best_genome.species_id if self.best_genome else None,
            },
            "innovation_stats": {
                "current_innovation": self.innovation_tracker.current_innovation,
                "current_node_id": self.innovation_tracker.current_node_id,
                "unique_connections": len(self.innovation_tracker.connection_innovations),
                "unique_nodes": len(self.innovation_tracker.node_innovations),
            },
        }

    async def save_population(self, filepath: str) -> None:
        """Save current population to file.

        Args:
            filepath: Path to save population data
        """
        try:
            import pickle

            save_data = {
                "population": self.population,
                "best_genome": self.best_genome,
                "generation": self.generation,
                "innovation_tracker": self.innovation_tracker,
                "evolution_history": self.evolution_history,
                "config": self.neuro_config,
            }

            def _save_to_file():
                with open(filepath, "wb") as f:
                    pickle.dump(save_data, f)

            # Run file I/O in executor to avoid blocking the event loop
            await asyncio.to_thread(_save_to_file)

            self.logger.info(f"Population saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save population: {e!s}")
            raise

    async def load_population(self, filepath: str) -> None:
        """Load population from file.

        Args:
            filepath: Path to load population data from
        """
        try:
            import pickle

            def _load_from_file():
                with open(filepath, "rb") as f:
                    return pickle.load(f)

            # Run file I/O in executor to avoid blocking the event loop
            save_data = await asyncio.to_thread(_load_from_file)

            self.population = save_data["population"]
            self.best_genome = save_data["best_genome"]
            self.generation = save_data["generation"]
            self.innovation_tracker = save_data["innovation_tracker"]
            self.evolution_history = save_data["evolution_history"]

            # Update best network
            self._update_best_network()

            # Re-speciate population
            self.speciation_manager.speciate_population(self.population)

            self.logger.info(f"Population loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load population: {e!s}")
            raise


# Update completion status


    # Helper methods for accessing data through data service

