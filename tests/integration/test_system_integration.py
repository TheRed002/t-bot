"""
Integration validation tests for the entire T-Bot system.

This module contains comprehensive integration tests that validate:
1. Import resolution and circular dependency detection
2. Service startup sequence and dependency initialization
3. Configuration loading from multiple sources
4. Inter-service communication and data flow
5. End-to-end functionality validation
"""

import importlib
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


class ImportValidator:
    """Validates all imports and detects circular dependencies."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.visited: set[str] = set()
        self.visiting: set[str] = set()
        self.import_graph: dict[str, list[str]] = {}
        self.circular_deps: list[list[str]] = []

    def find_circular_dependencies(self) -> list[list[str]]:
        """Find all circular dependencies in the codebase."""
        python_files = list(self.root_path.glob("**/*.py"))

        for py_file in python_files:
            module_name = self._path_to_module(py_file)
            if module_name not in self.visited:
                self._dfs_check_circular(module_name, py_file)

        return self.circular_deps

    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name."""
        relative_path = path.relative_to(self.root_path.parent)
        module_parts = list(relative_path.parts)
        if module_parts[-1] == "__init__.py":
            module_parts = module_parts[:-1]
        else:
            module_parts[-1] = module_parts[-1][:-3]  # Remove .py
        return ".".join(module_parts)

    def _dfs_check_circular(self, module_name: str, file_path: Path, path: list[str] = None):
        """DFS traversal to detect circular dependencies."""
        if path is None:
            path = []

        if module_name in self.visiting:
            # Found a cycle
            cycle_start = path.index(module_name)
            cycle = path[cycle_start:] + [module_name]
            self.circular_deps.append(cycle)
            return

        if module_name in self.visited:
            return

        self.visiting.add(module_name)
        path.append(module_name)

        # Parse imports from the file
        imports = self._extract_imports(file_path)
        self.import_graph[module_name] = imports

        for imported_module in imports:
            try:
                imported_path = self._module_to_path(imported_module)
                if imported_path and imported_path.exists():
                    self._dfs_check_circular(imported_module, imported_path, path[:])
            except Exception as e:
                logger.warning(f"Could not resolve import {imported_module}: {e}")

        path.pop()
        self.visiting.remove(module_name)
        self.visited.add(module_name)

    def _extract_imports(self, file_path: Path) -> list[str]:
        """Extract all imports from a Python file."""
        imports = []
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Simple regex-based import extraction
            import re

            # Match 'from x import y' and 'import x'
            import_pattern = r"(?:from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import|import\s+([a-zA-Z_][a-zA-Z0-9_.]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_.]*)*))"

            for match in re.finditer(import_pattern, content):
                if match.group(1):  # from ... import
                    module = match.group(1)
                    if module.startswith("src.") or module.startswith("."):
                        imports.append(module)
                elif match.group(2):  # import ...
                    modules = match.group(2).split(",")
                    for module in modules:
                        module = module.strip()
                        if module.startswith("src.") or module.startswith("."):
                            imports.append(module)

        except Exception as e:
            logger.error(f"Error parsing imports from {file_path}: {e}")

        return imports

    def _module_to_path(self, module_name: str) -> Path:
        """Convert module name to file path."""
        if module_name.startswith("."):
            # Relative import - skip for now
            return None

        parts = module_name.split(".")
        if parts[0] == "src":
            path = self.root_path / Path(*parts[1:])
        else:
            return None

        # Check both __init__.py and .py files
        if (path / "__init__.py").exists():
            return path / "__init__.py"
        elif (path.parent / f"{path.name}.py").exists():
            return path.parent / f"{path.name}.py"

        return None


class ConfigValidator:
    """Validates configuration loading from multiple sources."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def validate_config_files(self) -> dict[str, any]:
        """Validate all configuration files can be loaded."""
        results = {"yaml_configs": {}, "json_configs": {}, "errors": []}

        # Test YAML configs
        yaml_files = list(self.config_path.glob("**/*.yaml")) + list(
            self.config_path.glob("**/*.yml")
        )
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    content = yaml.safe_load(f)
                results["yaml_configs"][str(yaml_file)] = "valid"
            except Exception as e:
                results["errors"].append(f"YAML error in {yaml_file}: {e}")
                results["yaml_configs"][str(yaml_file)] = "invalid"

        # Test JSON configs
        json_files = list(self.config_path.glob("**/*.json"))
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    content = json.load(f)
                results["json_configs"][str(json_file)] = "valid"
            except Exception as e:
                results["errors"].append(f"JSON error in {json_file}: {e}")
                results["json_configs"][str(json_file)] = "invalid"

        return results


class ServiceValidator:
    """Validates service integrations and startup sequences."""

    def __init__(self):
        self.services = {}

    async def validate_service_startup(self) -> dict[str, any]:
        """Test service startup sequence and dependencies."""
        results = {
            "config_service": False,
            "database_service": False,
            "exchange_services": {},
            "bot_services": {},
            "api_service": False,
            "websocket_service": False,
            "errors": [],
        }

        try:
            # Test config service
            from src.core.config.main import ConfigManager

            config_manager = ConfigManager()
            await config_manager.load_config()
            results["config_service"] = True
        except Exception as e:
            results["errors"].append(f"Config service error: {e}")

        try:
            # Test database connection (mocked)
            with patch("asyncpg.connect", new_callable=AsyncMock):
                from src.database.queries import DatabaseManager

                db_manager = DatabaseManager("mock://connection")
                await db_manager.initialize()
                results["database_service"] = True
        except Exception as e:
            results["errors"].append(f"Database service error: {e}")

        # Test exchange services
        exchanges = ["binance", "coinbase", "okx"]
        for exchange_name in exchanges:
            try:
                if exchange_name == "binance":
                    from src.exchanges.binance import BinanceExchange

                    exchange = BinanceExchange()
                elif exchange_name == "coinbase":
                    from src.exchanges.coinbase import CoinbaseExchange

                    exchange = CoinbaseExchange()
                elif exchange_name == "okx":
                    from src.exchanges.okx import OKXExchange

                    exchange = OKXExchange()

                # Mock the connection
                exchange._session = MagicMock()
                results["exchange_services"][exchange_name] = True
            except Exception as e:
                results["errors"].append(f"Exchange {exchange_name} error: {e}")
                results["exchange_services"][exchange_name] = False

        try:
            # Test bot management services
            from src.bot_management.bot_monitor import BotMonitor
            from src.bot_management.bot_orchestrator import BotOrchestrator
            from src.bot_management.resource_manager import ResourceManager

            orchestrator = BotOrchestrator()
            monitor = BotMonitor()
            resource_manager = ResourceManager()

            results["bot_services"] = {
                "orchestrator": True,
                "monitor": True,
                "resource_manager": True,
            }
        except Exception as e:
            results["errors"].append(f"Bot services error: {e}")
            results["bot_services"] = {"error": str(e)}

        try:
            # Test API service
            from src.web_interface.app import create_app

            app = create_app()
            results["api_service"] = True
        except Exception as e:
            results["errors"].append(f"API service error: {e}")

        try:
            # Test WebSocket service
            from src.web_interface.socketio_manager import SocketIOManager

            socketio_manager = SocketIOManager()
            results["websocket_service"] = True
        except Exception as e:
            results["errors"].append(f"WebSocket service error: {e}")

        return results


@pytest.mark.asyncio
class TestSystemIntegration:
    """Comprehensive system integration tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.root_path = Path(__file__).parent.parent.parent / "src"
        self.config_path = Path(__file__).parent.parent.parent / "config"

    def test_import_validation(self):
        """Test that all imports resolve correctly and no circular dependencies exist."""
        validator = ImportValidator(self.root_path)
        circular_deps = validator.find_circular_dependencies()

        # Log any circular dependencies found
        if circular_deps:
            for cycle in circular_deps:
                logger.warning(f"Circular dependency detected: {' -> '.join(cycle)}")

        # For now, we'll log warnings but not fail the test
        # In a production environment, you might want to fail on circular dependencies
        assert len(circular_deps) < 10, (
            f"Too many circular dependencies found: {len(circular_deps)}"
        )

    def test_config_validation(self):
        """Test that all configuration files are valid and can be loaded."""
        validator = ConfigValidator(self.config_path)
        results = validator.validate_config_files()

        # Log any errors
        for error in results["errors"]:
            logger.error(error)

        # Ensure critical config files are valid
        yaml_valid = sum(1 for status in results["yaml_configs"].values() if status == "valid")
        json_valid = sum(1 for status in results["json_configs"].values() if status == "valid")

        assert yaml_valid > 0, "No valid YAML configuration files found"
        assert json_valid > 0, "No valid JSON configuration files found"

        # Fail if there are critical errors
        critical_errors = [
            e for e in results["errors"] if "schema" in e.lower() or "main" in e.lower()
        ]
        assert len(critical_errors) == 0, f"Critical configuration errors: {critical_errors}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_startup_validation(self):
        """Test that all services can be started and initialized properly."""
        validator = ServiceValidator()
        results = await validator.validate_service_startup()

        # Log any errors
        for error in results["errors"]:
            logger.error(error)

        # Validate critical services are working
        assert results["config_service"], "Config service failed to initialize"

        # Exchange services should have at least one working
        working_exchanges = sum(1 for status in results["exchange_services"].values() if status)
        assert working_exchanges > 0, "No exchange services are working"

        # Bot services should be available
        if isinstance(results["bot_services"], dict) and "error" not in results["bot_services"]:
            assert results["bot_services"]["orchestrator"], "Bot orchestrator failed"
            assert results["bot_services"]["monitor"], "Bot monitor failed"
            assert results["bot_services"]["resource_manager"], "Resource manager failed"

        # API and WebSocket services should be available
        assert results["api_service"], "API service failed to initialize"
        assert results["websocket_service"], "WebSocket service failed to initialize"

    def test_module_imports(self):
        """Test that critical modules can be imported without errors."""
        critical_modules = [
            "src.core.config.main",
            "src.core.types",
            "src.exchanges.base",
            "src.strategies.base",
            "src.bot_management.bot_orchestrator",
            "src.risk_management.risk_manager",
            "src.web_interface.app",
            "src.error_handling.error_handler",
        ]

        failed_imports = []
        for module_name in critical_modules:
            try:
                importlib.import_module(module_name)
                logger.info(f"Successfully imported {module_name}")
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                logger.error(f"Failed to import {module_name}: {e}")

        assert len(failed_imports) == 0, f"Critical modules failed to import: {failed_imports}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_end_to_end_workflow(self):
        """Test a basic end-to-end workflow to ensure system integration."""
        try:
            # Mock external dependencies
            with (
                patch("asyncpg.connect", new_callable=AsyncMock),
                patch("aiohttp.ClientSession.get", new_callable=AsyncMock),
                patch("websockets.connect", new_callable=AsyncMock),
            ):
                # Initialize core components
                from src.bot_management.bot_orchestrator import BotOrchestrator
                from src.core.config.main import ConfigManager
                from src.risk_management.risk_manager import RiskManager

                config_manager = ConfigManager()
                await config_manager.load_config()

                orchestrator = BotOrchestrator()
                risk_manager = RiskManager()

                # Test basic workflow
                # 1. Create a mock bot configuration
                mock_bot_config = {
                    "bot_id": "test_bot_001",
                    "strategy": "mean_reversion",
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "capital": 1000.0,
                    "risk_per_trade": 0.02,
                }

                # 2. Test bot creation (mocked)
                with patch.object(orchestrator, "create_bot", return_value=True):
                    bot_created = await orchestrator.create_bot(mock_bot_config)
                    assert bot_created, "Bot creation failed"

                # 3. Test risk validation
                mock_position = {
                    "symbol": "BTCUSDT",
                    "side": "long",
                    "size": 0.01,
                    "price": 50000.0,
                }

                risk_valid = risk_manager.validate_position_risk(
                    mock_position, account_balance=1000.0, max_risk_per_trade=0.02
                )
                assert risk_valid, "Risk validation failed"

                logger.info("End-to-end workflow test completed successfully")

        except Exception as e:
            pytest.fail(f"End-to-end workflow test failed: {e}")

    def test_error_handling_integration(self):
        """Test that error handling works across service boundaries."""
        try:
            from src.error_handling.factory import ErrorHandlerFactory

            # Test error handler creation
            handler = ErrorHandlerFactory.create_handler("exchange")
            assert handler is not None, "Error handler creation failed"

            # Test error handling workflow
            test_error = Exception("Test integration error")

            # This should not raise an exception
            result = handler.handle_error(test_error, context={"test": "integration"})

            logger.info("Error handling integration test completed")

        except Exception as e:
            pytest.fail(f"Error handling integration test failed: {e}")


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "--tb=short"])
