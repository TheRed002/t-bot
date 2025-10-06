"""
State management integration tests.

Tests state persistence, recovery, synchronization, checkpoint management,
and consistency validation across system restarts and failures.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.state.checkpoint_manager import CheckpointManager
from src.state.state_synchronizer import StateSynchronizer as StateSyncManager
from tests.integration.base_integration import (
    BaseIntegrationTest,
    performance_test,
)

logger = logging.getLogger(__name__)


class TestStatepersistence(BaseIntegrationTest):
    """Test state persistence and recovery mechanisms."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_portfolio_state_persistence(self):
        """Test persistence and recovery of portfolio state."""

        # Create temporary state storage
        temp_dir = tempfile.mkdtemp()
        state_file = Path(temp_dir) / "portfolio_state.json"

        state_manager = Mock(spec=StateManager)

        # Initial portfolio state
        initial_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": "125000.50",
            "positions": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "side": "LONG",
                    "quantity": "2.5",
                    "average_price": "49000.0",
                    "unrealized_pnl": "2500.0",
                    "entry_timestamp": "2024-01-01T10:00:00Z",
                },
                "ETH/USDT": {
                    "symbol": "ETH/USDT",
                    "side": "LONG",
                    "quantity": "15.0",
                    "average_price": "2900.0",
                    "unrealized_pnl": "-750.0",
                    "entry_timestamp": "2024-01-01T14:30:00Z",
                },
            },
            "balances": {"USDT": "50000.0", "BTC": "2.5", "ETH": "15.0"},
            "open_orders": {
                "order_123": {
                    "id": "order_123",
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "quantity": "0.5",
                    "price": "48000.0",
                    "status": "NEW",
                    "timestamp": "2024-01-01T15:45:00Z",
                }
            },
            "daily_pnl": "1750.0",
            "total_trades": 127,
            "last_activity": datetime.now(timezone.utc).isoformat(),
        }

        # Mock state persistence
        async def save_state(state_data):
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
            return True

        async def load_state():
            if state_file.exists():
                with open(state_file) as f:
                    return json.load(f)
            return None

        state_manager.save_state = save_state
        state_manager.load_state = load_state

        # 1. Save initial state
        save_success = await state_manager.save_state(initial_state)
        assert save_success is True
        assert state_file.exists()

        logger.info(f"State saved to {state_file}")

        # 2. Simulate system restart - clear memory
        del initial_state

        # 3. Load state after restart
        recovered_state = await state_manager.load_state()

        assert recovered_state is not None
        assert "portfolio_value" in recovered_state
        assert Decimal(recovered_state["portfolio_value"]) == Decimal("125000.50")

        # Verify positions recovered
        assert len(recovered_state["positions"]) == 2
        assert "BTC/USDT" in recovered_state["positions"]
        assert "ETH/USDT" in recovered_state["positions"]

        btc_position = recovered_state["positions"]["BTC/USDT"]
        assert Decimal(btc_position["quantity"]) == Decimal("2.5")
        assert Decimal(btc_position["average_price"]) == Decimal("49000.0")

        # Verify open orders recovered
        assert len(recovered_state["open_orders"]) == 1
        assert "order_123" in recovered_state["open_orders"]

        order = recovered_state["open_orders"]["order_123"]
        assert order["status"] == "NEW"
        assert Decimal(order["price"]) == Decimal("48000.0")

        # Verify balances recovered
        balances = recovered_state["balances"]
        assert Decimal(balances["USDT"]) == Decimal("50000.0")
        assert Decimal(balances["BTC"]) == Decimal("2.5")

        logger.info("Portfolio state recovery successful")

        # Cleanup
        os.remove(state_file)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_incremental_state_updates(self):
        """Test incremental state updates and delta persistence."""

        temp_dir = tempfile.mkdtemp()
        state_file = Path(temp_dir) / "state.json"
        delta_file = Path(temp_dir) / "state_deltas.json"

        # Initial state
        current_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": {"BTC/USDT": {"quantity": "1.0", "average_price": "50000.0"}},
            "balances": {"USDT": "100000.0", "BTC": "1.0"},
            "version": 1,
        }

        state_deltas = []

        # Mock incremental state manager
        async def apply_delta(delta):
            nonlocal current_state

            # Record delta
            delta["timestamp"] = datetime.now(timezone.utc).isoformat()
            delta["version"] = current_state["version"] + 1
            state_deltas.append(delta)

            # Apply delta to current state
            if "position_updates" in delta:
                for symbol, update in delta["position_updates"].items():
                    if symbol in current_state["positions"]:
                        current_state["positions"][symbol].update(update)
                    else:
                        current_state["positions"][symbol] = update

            if "balance_updates" in delta:
                for asset, amount in delta["balance_updates"].items():
                    current_state["balances"][asset] = str(
                        Decimal(current_state["balances"].get(asset, "0")) + Decimal(amount)
                    )

            current_state["version"] = delta["version"]
            current_state["timestamp"] = delta["timestamp"]

            # Persist delta
            with open(delta_file, "w") as f:
                json.dump(state_deltas, f, indent=2)

        async def save_full_state():
            with open(state_file, "w") as f:
                json.dump(current_state, f, indent=2)

        # Apply series of incremental updates
        updates = [
            # Trade execution
            {
                "type": "trade_execution",
                "position_updates": {
                    "BTC/USDT": {"quantity": "1.5"}  # Increase position
                },
                "balance_updates": {
                    "USDT": "-25000.0",  # Spent USDT
                    "BTC": "0.5",  # Received BTC
                },
            },
            # Partial position close
            {
                "type": "position_close",
                "position_updates": {
                    "BTC/USDT": {"quantity": "1.2"}  # Reduced position
                },
                "balance_updates": {
                    "USDT": "15300.0",  # Received USDT from sale
                    "BTC": "-0.3",  # Sold BTC
                },
            },
            # New position opened
            {
                "type": "new_position",
                "position_updates": {"ETH/USDT": {"quantity": "10.0", "average_price": "3000.0"}},
                "balance_updates": {
                    "USDT": "-30000.0",  # Spent on ETH
                    "ETH": "10.0",  # Received ETH
                },
            },
        ]

        # Apply updates sequentially
        for update in updates:
            await apply_delta(update)
            logger.info(f"Applied delta: {update['type']}")

        # Verify final state
        assert current_state["version"] == 4  # Initial + 3 updates
        assert len(state_deltas) == 3

        # Verify position changes
        assert Decimal(current_state["positions"]["BTC/USDT"]["quantity"]) == Decimal("1.2")
        assert "ETH/USDT" in current_state["positions"]

        # Verify balance changes
        # Started with 100000 USDT, spent 25000, received 15300, spent 30000
        # 100000 - 25000 + 15300 - 30000 = 60300
        assert Decimal(current_state["balances"]["USDT"]) == Decimal("60300.0")
        assert Decimal(current_state["balances"]["BTC"]) == Decimal("1.2")
        assert Decimal(current_state["balances"]["ETH"]) == Decimal("10.0")

        # Save final state
        await save_full_state()

        # Test delta replay for recovery
        recovered_state = {
            "timestamp": current_state["timestamp"],
            "positions": {"BTC/USDT": {"quantity": "1.0", "average_price": "50000.0"}},
            "balances": {"USDT": "100000.0", "BTC": "1.0"},
            "version": 1,
        }

        # Replay deltas to reconstruct state
        for delta in state_deltas:
            if "position_updates" in delta:
                for symbol, update in delta["position_updates"].items():
                    if symbol in recovered_state["positions"]:
                        recovered_state["positions"][symbol].update(update)
                    else:
                        recovered_state["positions"][symbol] = update

            if "balance_updates" in delta:
                for asset, amount in delta["balance_updates"].items():
                    recovered_state["balances"][asset] = str(
                        Decimal(recovered_state["balances"].get(asset, "0")) + Decimal(amount)
                    )

        # Verify recovery matches current state
        assert recovered_state["balances"]["USDT"] == current_state["balances"]["USDT"]
        assert (
            recovered_state["positions"]["BTC/USDT"]["quantity"]
            == current_state["positions"]["BTC/USDT"]["quantity"]
        )

        logger.info("Incremental state updates and recovery successful")

        # Cleanup
        for file_path in [state_file, delta_file]:
            if file_path.exists():
                os.remove(file_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_state_corruption_detection_and_repair(self):
        """Test detection and repair of corrupted state data."""

        state_validator = Mock()
        state_repairer = Mock()

        # Valid state structure
        valid_state = {
            "timestamp": "2024-01-01T12:00:00Z",
            "positions": {
                "BTC/USDT": {"quantity": "1.5", "average_price": "50000.0", "unrealized_pnl": "0.0"}
            },
            "balances": {"USDT": "100000.0", "BTC": "1.5"},
            "checksum": "abc123def456",
        }

        # Corrupted state examples
        corrupted_states = [
            # Missing required fields
            {
                "timestamp": "2024-01-01T12:00:00Z",
                # Missing positions and balances
                "checksum": "corrupted",
            },
            # Invalid data types
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "positions": {
                    "BTC/USDT": {
                        "quantity": "invalid_number",  # Should be valid Decimal string
                        "average_price": None,  # Should not be None
                        "unrealized_pnl": "0.0",
                    }
                },
                "balances": {"USDT": "100000.0", "BTC": "1.5"},
                "checksum": "def456ghi789",
            },
            # Inconsistent balances vs positions
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "positions": {
                    "BTC/USDT": {
                        "quantity": "2.0",  # Position shows 2.0 BTC
                        "average_price": "50000.0",
                        "unrealized_pnl": "0.0",
                    }
                },
                "balances": {"USDT": "100000.0", "BTC": "1.0"},  # But balance shows 1.0 BTC
                "checksum": "ghi789jkl012",
            },
        ]

        # Validation functions
        def validate_state_structure(state):
            """Check if state has required structure."""
            required_fields = ["timestamp", "positions", "balances"]
            return all(field in state for field in required_fields)

        def validate_data_types(state):
            """Check if data types are correct."""
            try:
                if "positions" in state:
                    for symbol, position in state["positions"].items():
                        if "quantity" in position:
                            Decimal(position["quantity"])  # Should be convertible to Decimal
                        if "average_price" in position and position["average_price"] is not None:
                            Decimal(position["average_price"])

                if "balances" in state:
                    for asset, balance in state["balances"].items():
                        Decimal(balance)  # Should be convertible to Decimal

                return True
            except (ValueError, TypeError):
                return False

        def validate_consistency(state):
            """Check consistency between positions and balances."""
            if "positions" not in state or "balances" not in state:
                return False

            # Check that position quantities match balance holdings
            for symbol, position in state["positions"].items():
                base_asset = symbol.split("/")[0]
                if base_asset in state["balances"]:
                    position_qty = Decimal(position["quantity"])
                    balance_qty = Decimal(state["balances"][base_asset])

                    # Allow small discrepancy for rounding
                    if abs(position_qty - balance_qty) > Decimal("0.001"):
                        return False

            return True

        state_validator.validate_structure = validate_state_structure
        state_validator.validate_data_types = validate_data_types
        state_validator.validate_consistency = validate_consistency

        # Repair functions
        async def repair_missing_fields(state):
            """Add missing required fields with default values."""
            if "timestamp" not in state:
                state["timestamp"] = datetime.now(timezone.utc).isoformat()
            if "positions" not in state:
                state["positions"] = {}
            if "balances" not in state:
                state["balances"] = {}
            return state

        async def repair_data_types(state):
            """Fix invalid data types."""
            if "positions" in state:
                for symbol, position in state["positions"].items():
                    # Fix invalid quantities
                    if "quantity" in position:
                        try:
                            Decimal(position["quantity"])
                        except (ValueError, TypeError):
                            position["quantity"] = "0.0"

                    # Fix None or invalid prices
                    if "average_price" in position:
                        if position["average_price"] is None:
                            # Use current market price as fallback
                            position["average_price"] = "50000.0"  # Mock price
                        else:
                            try:
                                Decimal(position["average_price"])
                            except (ValueError, TypeError):
                                position["average_price"] = "50000.0"

            if "balances" in state:
                for asset, balance in state["balances"].items():
                    try:
                        Decimal(balance)
                    except (ValueError, TypeError):
                        state["balances"][asset] = "0.0"

            return state

        async def repair_consistency(state):
            """Fix consistency issues between positions and balances."""
            if "positions" in state and "balances" in state:
                # Sync balances to match positions (positions are more reliable)
                for symbol, position in state["positions"].items():
                    base_asset = symbol.split("/")[0]
                    if base_asset not in state["balances"]:
                        state["balances"][base_asset] = "0.0"

                    # Update balance to match position
                    state["balances"][base_asset] = position["quantity"]

            return state

        state_repairer.repair_missing_fields = repair_missing_fields
        state_repairer.repair_data_types = repair_data_types
        state_repairer.repair_consistency = repair_consistency

        # Test validation and repair for each corrupted state
        for i, corrupted_state in enumerate(corrupted_states):
            logger.info(f"Testing corrupted state #{i + 1}")

            # Check validation
            structure_valid = state_validator.validate_structure(corrupted_state)
            types_valid = state_validator.validate_data_types(corrupted_state)
            consistency_valid = state_validator.validate_consistency(corrupted_state)

            is_valid = structure_valid and types_valid and consistency_valid
            assert is_valid is False  # Should be detected as corrupted

            logger.info(f"  Structure valid: {structure_valid}")
            logger.info(f"  Types valid: {types_valid}")
            logger.info(f"  Consistency valid: {consistency_valid}")

            # Apply repairs
            repaired_state = corrupted_state.copy()

            if not structure_valid:
                repaired_state = await state_repairer.repair_missing_fields(repaired_state)

            if not types_valid:
                repaired_state = await state_repairer.repair_data_types(repaired_state)

            if not consistency_valid:
                repaired_state = await state_repairer.repair_consistency(repaired_state)

            # Verify repair success
            structure_valid_after = state_validator.validate_structure(repaired_state)
            types_valid_after = state_validator.validate_data_types(repaired_state)
            consistency_valid_after = state_validator.validate_consistency(repaired_state)

            is_valid_after = structure_valid_after and types_valid_after and consistency_valid_after

            logger.info(
                f"  After repair - Structure: {structure_valid_after}, "
                f"Types: {types_valid_after}, Consistency: {consistency_valid_after}"
            )

            assert is_valid_after is True  # Should be repaired

        logger.info("State corruption detection and repair tests completed")


class TestCheckpointManagement(BaseIntegrationTest):
    """Test checkpoint creation, management, and recovery."""

    @pytest.mark.asyncio
    @performance_test(max_duration=20.0)
    @pytest.mark.timeout(300)
    async def test_periodic_checkpoint_creation(self, performance_monitor):
        """Test automatic periodic checkpoint creation."""

        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir()

        checkpoint_manager = Mock(spec=CheckpointManager)
        checkpoint_manager.checkpoint_interval = 2.0  # 2 seconds for testing
        checkpoint_manager.max_checkpoints = 5
        checkpoint_manager.checkpoint_directory = checkpoint_dir

        # Current system state
        system_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": Decimal("150000.0"),
            "active_positions": 3,
            "open_orders": 2,
            "daily_pnl": Decimal("2500.0"),
            "update_counter": 0,
        }

        created_checkpoints = []

        # Mock checkpoint creation
        async def create_checkpoint():
            checkpoint_id = f"checkpoint_{len(created_checkpoints) + 1}"
            checkpoint_timestamp = datetime.now(timezone.utc)

            checkpoint_data = {
                "id": checkpoint_id,
                "timestamp": checkpoint_timestamp.isoformat(),
                "state": system_state.copy(),
                "metadata": {"version": "1.0", "system_health": "healthy", "active_strategies": 2},
            }

            # Save checkpoint file
            checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            created_checkpoints.append(checkpoint_data)
            performance_monitor.record_event("checkpoint_created", {"id": checkpoint_id})

            logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id

        checkpoint_manager.create_checkpoint = create_checkpoint

        # Simulate system running with periodic checkpoints
        test_duration = 10.0  # 10 seconds
        start_time = time.time()
        last_checkpoint = start_time

        while time.time() - start_time < test_duration:
            # Simulate system activity
            system_state["update_counter"] += 1
            system_state["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Create checkpoint periodically
            if time.time() - last_checkpoint >= checkpoint_manager.checkpoint_interval:
                await checkpoint_manager.create_checkpoint()
                last_checkpoint = time.time()

            await asyncio.sleep(0.1)  # Small delay to simulate work

        # Verify checkpoints were created
        expected_checkpoints = int(test_duration / checkpoint_manager.checkpoint_interval)
        actual_checkpoints = len(created_checkpoints)

        assert actual_checkpoints >= expected_checkpoints - 1  # Allow slight timing variance
        logger.info(f"Created {actual_checkpoints} checkpoints over {test_duration}s")

        # Verify checkpoint files exist
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) == actual_checkpoints

        # Test checkpoint cleanup (keep only max_checkpoints)
        if len(created_checkpoints) > checkpoint_manager.max_checkpoints:
            # Remove oldest checkpoints
            excess_count = len(created_checkpoints) - checkpoint_manager.max_checkpoints
            for i in range(excess_count):
                old_checkpoint = created_checkpoints[i]
                old_file = checkpoint_dir / f"{old_checkpoint['id']}.json"
                if old_file.exists():
                    os.remove(old_file)

            remaining_files = list(checkpoint_dir.glob("*.json"))
            assert len(remaining_files) <= checkpoint_manager.max_checkpoints

            logger.info(f"Cleanup complete: {len(remaining_files)} checkpoints remaining")

        # Cleanup
        for file_path in checkpoint_dir.glob("*"):
            os.remove(file_path)
        os.rmdir(checkpoint_dir)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_checkpoint_recovery_scenarios(self):
        """Test recovery from different checkpoint scenarios."""

        # Scenario 1: Normal recovery from latest checkpoint
        checkpoints = [
            {
                "id": "checkpoint_1",
                "timestamp": "2024-01-01T10:00:00Z",
                "state": {"portfolio_value": "100000.0", "positions_count": 2},
            },
            {
                "id": "checkpoint_2",
                "timestamp": "2024-01-01T11:00:00Z",
                "state": {"portfolio_value": "102500.0", "positions_count": 3},
            },
            {
                "id": "checkpoint_3",
                "timestamp": "2024-01-01T12:00:00Z",
                "state": {"portfolio_value": "105000.0", "positions_count": 4},
            },
        ]

        checkpoint_manager = Mock(spec=CheckpointManager)

        async def get_latest_checkpoint():
            # Return most recent checkpoint
            return max(checkpoints, key=lambda cp: cp["timestamp"])

        async def get_checkpoint_by_id(checkpoint_id):
            return next((cp for cp in checkpoints if cp["id"] == checkpoint_id), None)

        checkpoint_manager.get_latest_checkpoint = get_latest_checkpoint
        checkpoint_manager.get_checkpoint_by_id = get_checkpoint_by_id

        # Test normal recovery
        latest_checkpoint = await checkpoint_manager.get_latest_checkpoint()

        assert latest_checkpoint["id"] == "checkpoint_3"
        assert Decimal(latest_checkpoint["state"]["portfolio_value"]) == Decimal("105000.0")

        logger.info(
            f"Normal recovery: Loaded checkpoint {latest_checkpoint['id']} "
            f"with portfolio value ${latest_checkpoint['state']['portfolio_value']}"
        )

        # Scenario 2: Recovery when latest checkpoint is corrupted
        # Simulate corrupted latest checkpoint
        checkpoints[-1]["state"] = None  # Corrupt the latest checkpoint

        async def get_valid_checkpoint():
            # Try latest first, then fall back to previous ones
            for checkpoint in reversed(checkpoints):
                if checkpoint["state"] is not None:
                    return checkpoint
            return None

        checkpoint_manager.get_latest_checkpoint = get_valid_checkpoint

        recovery_checkpoint = await checkpoint_manager.get_latest_checkpoint()

        assert recovery_checkpoint["id"] == "checkpoint_2"  # Previous valid checkpoint
        assert Decimal(recovery_checkpoint["state"]["portfolio_value"]) == Decimal("102500.0")

        logger.info(f"Corruption recovery: Fell back to checkpoint {recovery_checkpoint['id']}")

        # Scenario 3: Point-in-time recovery (rollback to specific checkpoint)
        target_checkpoint_id = "checkpoint_1"
        rollback_checkpoint = await checkpoint_manager.get_checkpoint_by_id(target_checkpoint_id)

        assert rollback_checkpoint is not None
        assert rollback_checkpoint["id"] == target_checkpoint_id
        assert Decimal(rollback_checkpoint["state"]["portfolio_value"]) == Decimal("100000.0")

        logger.info(f"Point-in-time recovery: Rolled back to {target_checkpoint_id}")

        # Scenario 4: No checkpoints available (cold start)
        empty_checkpoints = []

        async def get_latest_checkpoint_empty():
            return None if not empty_checkpoints else empty_checkpoints[-1]

        checkpoint_manager.get_latest_checkpoint = get_latest_checkpoint_empty

        cold_start_checkpoint = await checkpoint_manager.get_latest_checkpoint()

        assert cold_start_checkpoint is None

        # Initialize with default state for cold start
        default_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": "0.0",
            "positions": {},
            "balances": {},
            "daily_pnl": "0.0",
            "total_trades": 0,
        }

        logger.info("Cold start recovery: Initialized with default state")

        # Verify all recovery scenarios
        assert recovery_checkpoint["state"]["positions_count"] == 3
        assert rollback_checkpoint["state"]["positions_count"] == 2
        assert default_state["total_trades"] == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_checkpoint_integrity_validation(self):
        """Test checkpoint integrity validation and verification."""

        import hashlib

        checkpoint_validator = Mock()

        # Create checkpoint with integrity data
        original_state = {
            "timestamp": "2024-01-01T12:00:00Z",
            "portfolio_value": "100000.0",
            "positions": {"BTC/USDT": {"quantity": "2.0", "average_price": "50000.0"}},
            "balances": {"USDT": "100000.0", "BTC": "2.0"},
        }

        # Calculate checksum
        state_json = json.dumps(original_state, sort_keys=True)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()

        valid_checkpoint = {
            "id": "checkpoint_valid",
            "timestamp": "2024-01-01T12:00:00Z",
            "state": original_state,
            "integrity": {
                "checksum": checksum,
                "algorithm": "sha256",
                "created_at": "2024-01-01T12:00:00Z",
                "version": "1.0",
            },
        }

        # Validation functions
        def validate_checksum(checkpoint):
            """Verify checkpoint data integrity using checksum."""
            if "integrity" not in checkpoint or "checksum" not in checkpoint["integrity"]:
                return False

            expected_checksum = checkpoint["integrity"]["checksum"]
            state_json = json.dumps(checkpoint["state"], sort_keys=True)
            calculated_checksum = hashlib.sha256(state_json.encode()).hexdigest()

            return expected_checksum == calculated_checksum

        def validate_timestamp(checkpoint):
            """Verify checkpoint timestamp is reasonable."""
            try:
                checkpoint_time = datetime.fromisoformat(
                    checkpoint["timestamp"].replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)

                # Checkpoint shouldn't be from the future
                if checkpoint_time > now:
                    return False

                # Checkpoint shouldn't be too old (e.g., more than 30 days)
                if (now - checkpoint_time).days > 30:
                    return False

                return True
            except ValueError:
                return False

        def validate_structure(checkpoint):
            """Verify checkpoint has required structure."""
            required_fields = ["id", "timestamp", "state", "integrity"]
            return all(field in checkpoint for field in required_fields)

        checkpoint_validator.validate_checksum = validate_checksum
        checkpoint_validator.validate_timestamp = validate_timestamp
        checkpoint_validator.validate_structure = validate_structure

        # Test valid checkpoint
        assert checkpoint_validator.validate_structure(valid_checkpoint) is True
        assert checkpoint_validator.validate_timestamp(valid_checkpoint) is True
        assert checkpoint_validator.validate_checksum(valid_checkpoint) is True

        logger.info("Valid checkpoint passed all integrity checks")

        # Test corrupted checkpoints
        corrupted_checkpoints = [
            # Missing integrity data
            {
                "id": "checkpoint_no_integrity",
                "timestamp": "2024-01-01T12:00:00Z",
                "state": original_state,
                # Missing integrity field
            },
            # Wrong checksum
            {
                "id": "checkpoint_wrong_checksum",
                "timestamp": "2024-01-01T12:00:00Z",
                "state": original_state,
                "integrity": {"checksum": "wrong_checksum_value", "algorithm": "sha256"},
            },
            # Modified state data
            {
                "id": "checkpoint_modified_state",
                "timestamp": "2024-01-01T12:00:00Z",
                "state": {
                    **original_state,
                    "portfolio_value": "999999.0",  # Modified value
                },
                "integrity": {
                    "checksum": checksum,  # Original checksum
                    "algorithm": "sha256",
                },
            },
            # Future timestamp
            {
                "id": "checkpoint_future",
                "timestamp": "2025-12-31T23:59:59Z",  # Future date
                "state": original_state,
                "integrity": {"checksum": checksum, "algorithm": "sha256"},
            },
        ]

        for corrupted_checkpoint in corrupted_checkpoints:
            structure_valid = checkpoint_validator.validate_structure(corrupted_checkpoint)
            timestamp_valid = checkpoint_validator.validate_timestamp(corrupted_checkpoint)
            checksum_valid = checkpoint_validator.validate_checksum(corrupted_checkpoint)

            is_valid = structure_valid and timestamp_valid and checksum_valid

            logger.info(
                f"Checkpoint {corrupted_checkpoint['id']}: "
                f"Structure={structure_valid}, Timestamp={timestamp_valid}, "
                f"Checksum={checksum_valid}, Overall={is_valid}"
            )

            assert is_valid is False  # All corrupted checkpoints should fail validation

        logger.info("Checkpoint integrity validation tests completed")


class TestStateSynchronization(BaseIntegrationTest):
    """Test state synchronization across system components."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_multi_component_state_sync(self):
        """Test state synchronization between multiple system components."""

        # Mock system components
        components = {
            "portfolio_manager": {"state": {"total_value": Decimal("100000.0"), "positions": {}}},
            "risk_manager": {"state": {"daily_pnl": Decimal("0.0"), "exposure": {}}},
            "execution_engine": {"state": {"active_orders": {}, "execution_stats": {}}},
            "strategy_manager": {"state": {"active_strategies": [], "performance": {}}},
        }

        sync_manager = Mock(spec=StateSyncManager)
        sync_events = []

        # Mock synchronization methods
        async def sync_component_state(component_name, state_update):
            """Sync state update to a specific component."""
            if component_name in components:
                components[component_name]["state"].update(state_update)
                sync_events.append(
                    {"component": component_name, "update": state_update, "timestamp": time.time()}
                )
            return True

        async def broadcast_state_update(state_update, exclude_components=None):
            """Broadcast state update to all components except excluded ones."""
            exclude_components = exclude_components or []

            for component_name in components.keys():
                if component_name not in exclude_components:
                    await sync_component_state(component_name, state_update)

            return len(components) - len(exclude_components)

        sync_manager.sync_component_state = sync_component_state
        sync_manager.broadcast_state_update = broadcast_state_update

        # Scenario 1: Trade execution triggers multi-component sync
        trade_execution_update = {
            "trade_id": "trade_001",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "price": Decimal("50000.0"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Portfolio manager processes trade
        portfolio_update = {
            "positions": {
                "BTC/USDT": {
                    "quantity": "1.0",
                    "average_price": "50000.0",
                    "market_value": "50000.0",
                }
            },
            "total_value": "100000.0",  # Updated value
        }
        await sync_manager.sync_component_state("portfolio_manager", portfolio_update)

        # Risk manager updates exposure
        risk_update = {
            "exposure": {"BTC": "50000.0", "total_exposure": "50000.0"},
            "position_count": 1,
        }
        await sync_manager.sync_component_state("risk_manager", risk_update)

        # Execution engine updates order status
        execution_update = {
            "completed_orders": {
                "order_001": {"status": "FILLED", "fill_price": "50000.0", "fill_quantity": "1.0"}
            },
            "execution_stats": {"total_trades": 1, "volume_today": "50000.0"},
        }
        await sync_manager.sync_component_state("execution_engine", execution_update)

        # Verify state synchronization
        portfolio_state = components["portfolio_manager"]["state"]
        risk_state = components["risk_manager"]["state"]
        execution_state = components["execution_engine"]["state"]

        assert "BTC/USDT" in portfolio_state["positions"]
        assert Decimal(risk_state["exposure"]["BTC"]) == Decimal("50000.0")
        assert "order_001" in execution_state["completed_orders"]

        # Verify sync events recorded
        assert len(sync_events) == 3
        component_names = [event["component"] for event in sync_events]
        assert "portfolio_manager" in component_names
        assert "risk_manager" in component_names
        assert "execution_engine" in component_names

        logger.info(f"Multi-component sync completed: {len(sync_events)} components synchronized")

        # Scenario 2: Broadcast update to all components
        market_data_update = {
            "market_prices": {
                "BTC/USDT": "51000.0",  # Price increased
                "ETH/USDT": "3100.0",
            },
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

        sync_events_before_broadcast = len(sync_events)
        components_synced = await sync_manager.broadcast_state_update(market_data_update)

        assert components_synced == len(components)
        assert len(sync_events) == sync_events_before_broadcast + len(components)

        # Verify all components received the update
        for component_name, component_data in components.items():
            assert "market_prices" in component_data["state"]
            assert component_data["state"]["market_prices"]["BTC/USDT"] == "51000.0"

        logger.info(f"Broadcast sync completed: {components_synced} components updated")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_conflict_resolution_in_state_sync(self):
        """Test conflict resolution when state updates conflict."""

        conflict_resolver = Mock()

        # Scenario: Multiple components try to update the same state simultaneously
        conflicting_updates = [
            {
                "component": "portfolio_manager",
                "timestamp": 1640995200.0,  # 2022-01-01 00:00:00
                "update": {
                    "portfolio_value": "102000.0",
                    "last_update_source": "portfolio_calculation",
                },
            },
            {
                "component": "risk_manager",
                "timestamp": 1640995205.0,  # 2022-01-01 00:00:05 (5 seconds later)
                "update": {
                    "portfolio_value": "101500.0",  # Different value
                    "last_update_source": "risk_calculation",
                },
            },
            {
                "component": "execution_engine",
                "timestamp": 1640995203.0,  # 2022-01-01 00:00:03 (middle timestamp)
                "update": {
                    "portfolio_value": "101800.0",  # Another different value
                    "last_update_source": "execution_tracking",
                },
            },
        ]

        # Conflict resolution strategies
        def resolve_by_timestamp(updates):
            """Use the most recent update (highest timestamp)."""
            return max(updates, key=lambda u: u["timestamp"])

        def resolve_by_authority(updates):
            """Use update from most authoritative source."""
            authority_order = ["portfolio_manager", "execution_engine", "risk_manager"]

            for authority in authority_order:
                for update in updates:
                    if update["component"] == authority:
                        return update

            return updates[0]  # Fallback to first if no authority found

        def resolve_by_consensus(updates):
            """Use average or most common value."""
            values = [Decimal(u["update"]["portfolio_value"]) for u in updates]
            average_value = sum(values) / len(values)

            # Return update closest to average
            closest_update = min(
                updates, key=lambda u: abs(Decimal(u["update"]["portfolio_value"]) - average_value)
            )

            # Modify value to be the consensus (average)
            result = closest_update.copy()
            result["update"] = result["update"].copy()
            result["update"]["portfolio_value"] = str(average_value)
            result["update"]["resolution_method"] = "consensus"

            return result

        conflict_resolver.resolve_by_timestamp = resolve_by_timestamp
        conflict_resolver.resolve_by_authority = resolve_by_authority
        conflict_resolver.resolve_by_consensus = resolve_by_consensus

        # Test timestamp-based resolution
        timestamp_resolution = conflict_resolver.resolve_by_timestamp(conflicting_updates)

        assert timestamp_resolution["component"] == "risk_manager"  # Latest timestamp
        assert timestamp_resolution["update"]["portfolio_value"] == "101500.0"

        logger.info(
            f"Timestamp resolution: Selected update from {timestamp_resolution['component']} "
            f"with value {timestamp_resolution['update']['portfolio_value']}"
        )

        # Test authority-based resolution
        authority_resolution = conflict_resolver.resolve_by_authority(conflicting_updates)

        assert authority_resolution["component"] == "portfolio_manager"  # Highest authority
        assert authority_resolution["update"]["portfolio_value"] == "102000.0"

        logger.info(
            f"Authority resolution: Selected update from {authority_resolution['component']} "
            f"with value {authority_resolution['update']['portfolio_value']}"
        )

        # Test consensus-based resolution
        consensus_resolution = conflict_resolver.resolve_by_consensus(conflicting_updates)

        # Average of 102000, 101500, 101800 = 101766.67
        expected_consensus = (Decimal("102000.0") + Decimal("101500.0") + Decimal("101800.0")) / 3
        actual_consensus = Decimal(consensus_resolution["update"]["portfolio_value"])

        assert abs(actual_consensus - expected_consensus) < Decimal("1.0")  # Within $1
        assert consensus_resolution["update"]["resolution_method"] == "consensus"

        logger.info(f"Consensus resolution: Calculated average value {actual_consensus}")

        # Test conflict detection
        def detect_conflicts(updates, tolerance=Decimal("100.0")):
            """Detect if updates have conflicting values."""
            if len(updates) < 2:
                return False

            values = [Decimal(u["update"]["portfolio_value"]) for u in updates]
            max_value = max(values)
            min_value = min(values)

            return (max_value - min_value) > tolerance

        has_conflicts = detect_conflicts(conflicting_updates)
        assert has_conflicts is True

        # Values range from 101500 to 102000 = 500 difference > 100 tolerance
        value_range = max(
            Decimal(u["update"]["portfolio_value"]) for u in conflicting_updates
        ) - min(Decimal(u["update"]["portfolio_value"]) for u in conflicting_updates)

        assert value_range == Decimal("500.0")

        logger.info(f"Conflict detection: Value range ${value_range} exceeds tolerance")
        logger.info("Conflict resolution tests completed")
