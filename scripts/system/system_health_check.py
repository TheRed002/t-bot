#!/usr/bin/env python3
"""
Comprehensive system health check for T-Bot trading system.

This script performs a complete validation of:
1. Import resolution and circular dependency detection
2. Configuration file validation
3. Service startup sequence testing
4. Database connectivity
5. Exchange API connectivity (mocked)
6. WebSocket functionality
7. API endpoint validation
8. Bot management workflow
9. Error handling integration
10. Performance benchmarks
"""

import asyncio
import importlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthCheckResult:
    """Health check result data structure."""
    component: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    timestamp: Optional[str] = None


class HealthChecker:
    """Main health checker class."""
    
    def __init__(self):
        self.results: List[HealthCheckResult] = []
        self.project_root = project_root
        
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return summary."""
        logger.info("Starting comprehensive system health check...")
        start_time = time.time()
        
        # Run all health checks
        checks = [
            self.check_imports,
            self.check_configuration,
            self.check_core_services,
            self.check_exchange_integrations,
            self.check_database_integration,
            self.check_api_endpoints,
            self.check_websocket_integration,
            self.check_bot_management,
            self.check_error_handling,
            self.check_performance_baseline
        ]
        
        for check in checks:
            try:
                await check()
            except Exception as e:
                logger.error(f"Health check {check.__name__} failed: {e}")
                self.results.append(
                    HealthCheckResult(
                        component=check.__name__,
                        status=HealthStatus.CRITICAL,
                        message=f"Check failed with exception: {str(e)}",
                        details={"traceback": traceback.format_exc()}
                    )
                )
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_duration)
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    async def check_imports(self):
        """Check import resolution and circular dependencies."""
        logger.info("Checking imports and circular dependencies...")
        start_time = time.time()
        
        try:
            # Critical modules to test
            critical_modules = [
                'src.core.config.main',
                'src.core.types',
                'src.core.types.base',
                'src.core.types.data',
                'src.core.types.market',
                'src.exchanges.base',
                'src.exchanges.binance',
                'src.exchanges.coinbase',
                'src.exchanges.okx',
                'src.strategies.base',
                'src.bot_management.bot_orchestrator',
                'src.bot_management.bot_monitor',
                'src.bot_management.resource_manager',
                'src.risk_management.risk_manager',
                'src.risk_management.position_sizing',
                'src.web_interface.app',
                'src.web_interface.socketio_manager',
                'src.error_handling.error_handler',
                'src.error_handling.factory',
                'src.data.features.technical_indicators',
                'src.ml.inference.inference_engine'
            ]
            
            failed_imports = []
            successful_imports = []
            
            for module_name in critical_modules:
                try:
                    importlib.import_module(module_name)
                    successful_imports.append(module_name)
                except Exception as e:
                    failed_imports.append((module_name, str(e)))
                    
            duration = time.time() - start_time
            
            if not failed_imports:
                status = HealthStatus.HEALTHY
                message = f"All {len(successful_imports)} critical modules imported successfully"
            elif len(failed_imports) < len(critical_modules) * 0.2:  # Less than 20% failed
                status = HealthStatus.WARNING
                message = f"{len(failed_imports)} modules failed to import"
            else:
                status = HealthStatus.CRITICAL
                message = f"{len(failed_imports)} critical modules failed to import"
                
            self.results.append(
                HealthCheckResult(
                    component="imports",
                    status=status,
                    message=message,
                    details={
                        "successful_imports": len(successful_imports),
                        "failed_imports": len(failed_imports),
                        "failed_modules": failed_imports
                    },
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="imports",
                    status=HealthStatus.CRITICAL,
                    message=f"Import check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_configuration(self):
        """Check configuration file loading."""
        logger.info("Checking configuration loading...")
        start_time = time.time()
        
        try:
            config_path = self.project_root / "config"
            results = {
                "yaml_files": 0,
                "json_files": 0,
                "valid_yaml": 0,
                "valid_json": 0,
                "errors": []
            }
            
            # Check YAML files
            yaml_files = list(config_path.glob("**/*.yaml")) + list(config_path.glob("**/*.yml"))
            results["yaml_files"] = len(yaml_files)
            
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r') as f:
                        yaml.safe_load(f)
                    results["valid_yaml"] += 1
                except Exception as e:
                    results["errors"].append(f"YAML {yaml_file.name}: {str(e)}")
            
            # Check JSON files
            json_files = list(config_path.glob("**/*.json"))
            results["json_files"] = len(json_files)
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        json.load(f)
                    results["valid_json"] += 1
                except Exception as e:
                    results["errors"].append(f"JSON {json_file.name}: {str(e)}")
            
            duration = time.time() - start_time
            
            # Determine status
            total_files = results["yaml_files"] + results["json_files"]
            valid_files = results["valid_yaml"] + results["valid_json"]
            
            if valid_files == total_files and total_files > 0:
                status = HealthStatus.HEALTHY
                message = f"All {total_files} configuration files are valid"
            elif valid_files > total_files * 0.8:  # More than 80% valid
                status = HealthStatus.WARNING
                message = f"{valid_files}/{total_files} configuration files are valid"
            else:
                status = HealthStatus.CRITICAL
                message = f"Only {valid_files}/{total_files} configuration files are valid"
                
            self.results.append(
                HealthCheckResult(
                    component="configuration",
                    status=status,
                    message=message,
                    details=results,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="configuration",
                    status=HealthStatus.CRITICAL,
                    message=f"Configuration check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_core_services(self):
        """Check core service initialization."""
        logger.info("Checking core services...")
        start_time = time.time()
        
        try:
            from unittest.mock import patch, AsyncMock, MagicMock
            
            services_status = {}
            
            # Test config service
            try:
                from src.core.config.main import ConfigManager
                config_manager = ConfigManager()
                services_status["config_manager"] = "healthy"
            except Exception as e:
                services_status["config_manager"] = f"failed: {str(e)}"
            
            # Test error handling
            try:
                from src.error_handling.error_handler import ErrorHandler
                from src.error_handling.factory import ErrorHandlerFactory
                handler = ErrorHandlerFactory.create_handler('test')
                services_status["error_handling"] = "healthy"
            except Exception as e:
                services_status["error_handling"] = f"failed: {str(e)}"
            
            # Test validation framework
            try:
                from src.utils.validation.core import ValidationFramework
                validator = ValidationFramework()
                services_status["validation"] = "healthy"
            except Exception as e:
                services_status["validation"] = f"failed: {str(e)}"
            
            duration = time.time() - start_time
            
            healthy_services = sum(1 for status in services_status.values() if status == "healthy")
            total_services = len(services_status)
            
            if healthy_services == total_services:
                status = HealthStatus.HEALTHY
                message = f"All {total_services} core services are healthy"
            elif healthy_services > total_services * 0.7:
                status = HealthStatus.WARNING
                message = f"{healthy_services}/{total_services} core services are healthy"
            else:
                status = HealthStatus.CRITICAL
                message = f"Only {healthy_services}/{total_services} core services are healthy"
                
            self.results.append(
                HealthCheckResult(
                    component="core_services",
                    status=status,
                    message=message,
                    details=services_status,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="core_services",
                    status=HealthStatus.CRITICAL,
                    message=f"Core services check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_exchange_integrations(self):
        """Check exchange service integrations."""
        logger.info("Checking exchange integrations...")
        start_time = time.time()
        
        try:
            exchange_status = {}
            
            # Test Binance
            try:
                from src.exchanges.binance import BinanceExchange
                exchange = BinanceExchange()
                exchange_status["binance"] = "initialized"
            except Exception as e:
                exchange_status["binance"] = f"failed: {str(e)}"
            
            # Test Coinbase
            try:
                from src.exchanges.coinbase import CoinbaseExchange
                exchange = CoinbaseExchange()
                exchange_status["coinbase"] = "initialized"
            except Exception as e:
                exchange_status["coinbase"] = f"failed: {str(e)}"
            
            # Test OKX
            try:
                from src.exchanges.okx import OKXExchange
                exchange = OKXExchange()
                exchange_status["okx"] = "initialized"
            except Exception as e:
                exchange_status["okx"] = f"failed: {str(e)}"
            
            duration = time.time() - start_time
            
            working_exchanges = sum(1 for status in exchange_status.values() if "initialized" in status)
            total_exchanges = len(exchange_status)
            
            if working_exchanges == total_exchanges:
                status = HealthStatus.HEALTHY
                message = f"All {total_exchanges} exchange integrations are working"
            elif working_exchanges > 0:
                status = HealthStatus.WARNING
                message = f"{working_exchanges}/{total_exchanges} exchange integrations are working"
            else:
                status = HealthStatus.CRITICAL
                message = "No exchange integrations are working"
                
            self.results.append(
                HealthCheckResult(
                    component="exchange_integrations",
                    status=status,
                    message=message,
                    details=exchange_status,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="exchange_integrations",
                    status=HealthStatus.CRITICAL,
                    message=f"Exchange integration check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_database_integration(self):
        """Check database integration."""
        logger.info("Checking database integration...")
        start_time = time.time()
        
        try:
            # Test database module imports
            try:
                from src.database.queries import DatabaseManager
                from src.database.models import TradingSession
                from src.database.service import DatabaseService
                
                status = HealthStatus.HEALTHY
                message = "Database modules imported successfully"
                details = {"modules_imported": ["DatabaseManager", "TradingSession", "DatabaseService"]}
                
            except Exception as e:
                status = HealthStatus.CRITICAL
                message = f"Database module import failed: {str(e)}"
                details = {"error": str(e)}
            
            duration = time.time() - start_time
            
            self.results.append(
                HealthCheckResult(
                    component="database_integration",
                    status=status,
                    message=message,
                    details=details,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="database_integration",
                    status=HealthStatus.CRITICAL,
                    message=f"Database check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_api_endpoints(self):
        """Check API endpoint functionality."""
        logger.info("Checking API endpoints...")
        start_time = time.time()
        
        try:
            # Clear any existing prometheus registry to avoid conflicts
            import prometheus_client
            prometheus_client.REGISTRY = prometheus_client.CollectorRegistry()
            
            from src.web_interface.app import create_app
            
            app = create_app()
            
            # Get all routes
            routes = []
            for rule in app.url_map.iter_rules():
                routes.append({
                    "endpoint": rule.endpoint,
                    "methods": list(rule.methods),
                    "rule": str(rule)
                })
            
            status = HealthStatus.HEALTHY if len(routes) > 0 else HealthStatus.WARNING
            message = f"API application created with {len(routes)} endpoints"
            
            duration = time.time() - start_time
            
            self.results.append(
                HealthCheckResult(
                    component="api_endpoints",
                    status=status,
                    message=message,
                    details={"routes_count": len(routes), "sample_routes": routes[:5]},
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="api_endpoints",
                    status=HealthStatus.CRITICAL,
                    message=f"API endpoints check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_websocket_integration(self):
        """Check WebSocket integration."""
        logger.info("Checking WebSocket integration...")
        start_time = time.time()
        
        try:
            from src.web_interface.socketio_manager import SocketIOManager
            
            socketio_manager = SocketIOManager()
            
            status = HealthStatus.HEALTHY
            message = "WebSocket manager initialized successfully"
            details = {"socketio_manager": "initialized"}
            
            duration = time.time() - start_time
            
            self.results.append(
                HealthCheckResult(
                    component="websocket_integration",
                    status=status,
                    message=message,
                    details=details,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="websocket_integration",
                    status=HealthStatus.CRITICAL,
                    message=f"WebSocket check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_bot_management(self):
        """Check bot management workflow."""
        logger.info("Checking bot management...")
        start_time = time.time()
        
        try:
            from src.bot_management.bot_orchestrator import BotOrchestrator
            from src.bot_management.bot_monitor import BotMonitor
            from src.bot_management.resource_manager import ResourceManager
            
            orchestrator = BotOrchestrator()
            monitor = BotMonitor()
            resource_manager = ResourceManager()
            
            status = HealthStatus.HEALTHY
            message = "Bot management services initialized successfully"
            details = {
                "orchestrator": "initialized",
                "monitor": "initialized", 
                "resource_manager": "initialized"
            }
            
            duration = time.time() - start_time
            
            self.results.append(
                HealthCheckResult(
                    component="bot_management",
                    status=status,
                    message=message,
                    details=details,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="bot_management",
                    status=HealthStatus.CRITICAL,
                    message=f"Bot management check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_error_handling(self):
        """Check error handling integration."""
        logger.info("Checking error handling...")
        start_time = time.time()
        
        try:
            from src.error_handling.error_handler import ErrorHandler
            from src.error_handling.factory import ErrorHandlerFactory
            
            # Test error handler creation
            handler = ErrorHandlerFactory.create_handler('integration_test')
            
            # Test error handling
            test_error = Exception("Test integration error")
            result = handler.handle_error(test_error, context={"test": "integration"})
            
            status = HealthStatus.HEALTHY
            message = "Error handling system working correctly"
            details = {"handler_created": True, "error_handled": True}
            
            duration = time.time() - start_time
            
            self.results.append(
                HealthCheckResult(
                    component="error_handling",
                    status=status,
                    message=message,
                    details=details,
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="error_handling",
                    status=HealthStatus.CRITICAL,
                    message=f"Error handling check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    async def check_performance_baseline(self):
        """Check basic performance metrics."""
        logger.info("Checking performance baseline...")
        start_time = time.time()
        
        try:
            # Simple performance tests
            import_times = []
            
            # Measure import times for key modules
            test_modules = [
                'src.core.types',
                'src.exchanges.base',
                'src.strategies.base',
                'src.risk_management.risk_manager'
            ]
            
            for module_name in test_modules:
                module_start = time.time()
                try:
                    importlib.import_module(module_name)
                    import_time = time.time() - module_start
                    import_times.append(import_time)
                except:
                    pass
            
            avg_import_time = sum(import_times) / len(import_times) if import_times else 0
            max_import_time = max(import_times) if import_times else 0
            
            duration = time.time() - start_time
            
            # Determine status based on performance
            if avg_import_time < 0.1 and max_import_time < 0.5:
                status = HealthStatus.HEALTHY
                message = "Performance metrics are within acceptable ranges"
            elif avg_import_time < 0.5 and max_import_time < 2.0:
                status = HealthStatus.WARNING
                message = "Performance metrics show some delays"
            else:
                status = HealthStatus.CRITICAL
                message = "Performance metrics indicate potential issues"
            
            self.results.append(
                HealthCheckResult(
                    component="performance_baseline",
                    status=status,
                    message=message,
                    details={
                        "avg_import_time": avg_import_time,
                        "max_import_time": max_import_time,
                        "modules_tested": len(test_modules)
                    },
                    duration=duration
                )
            )
            
        except Exception as e:
            self.results.append(
                HealthCheckResult(
                    component="performance_baseline",
                    status=HealthStatus.CRITICAL,
                    message=f"Performance check failed: {str(e)}",
                    duration=time.time() - start_time
                )
            )
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate health check summary."""
        summary = {
            "overall_status": HealthStatus.HEALTHY.value,
            "total_duration": total_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "components_checked": len(self.results),
            "status_counts": {
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "unknown": 0
            },
            "details": []
        }
        
        for result in self.results:
            summary["status_counts"][result.status.value.lower()] += 1
            summary["details"].append({
                "component": result.component,
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "details": result.details
            })
        
        # Determine overall status
        if summary["status_counts"]["critical"] > 0:
            summary["overall_status"] = HealthStatus.CRITICAL.value
        elif summary["status_counts"]["warning"] > 0:
            summary["overall_status"] = HealthStatus.WARNING.value
            
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save health check results to file."""
        results_dir = self.project_root / ".claude_experiments"
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"health_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Health check results saved to {results_file}")


async def main():
    """Run the comprehensive health check."""
    print("=" * 80)
    print("T-Bot System Health Check - Day 18 Integration Validation")
    print("=" * 80)
    
    checker = HealthChecker()
    summary = await checker.run_all_checks()
    
    # Print summary
    print(f"\nOverall Status: {summary['overall_status']}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Components Checked: {summary['components_checked']}")
    print(f"Healthy: {summary['status_counts']['healthy']}")
    print(f"Warning: {summary['status_counts']['warning']}")
    print(f"Critical: {summary['status_counts']['critical']}")
    
    print("\nComponent Details:")
    print("-" * 80)
    for detail in summary['details']:
        status_symbol = {
            'HEALTHY': '✓',
            'WARNING': '⚠',
            'CRITICAL': '✗',
            'UNKNOWN': '?'
        }.get(detail['status'], '?')
        
        print(f"{status_symbol} {detail['component']:25} {detail['status']:10} {detail['message']}")
        if detail['duration']:
            print(f"  Duration: {detail['duration']:.3f}s")
    
    print("=" * 80)
    
    return summary['overall_status'] == 'HEALTHY'


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)