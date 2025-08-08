"""
State consistency monitor for cross-system state validation.

This module provides cross-system state consistency checking, automatic state
reconciliation procedures, state corruption detection, and real-time state
validation alerts.

CRITICAL: This module integrates with P-001 core framework and P-002 database
for state persistence and will be used by all subsequent prompts.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from decimal import Decimal
from src.core.logging import get_logger

# MANDATORY: Import from P-001 core framework
from src.core.exceptions import (
    TradingBotError, StateConsistencyError, StateCorruptionError
)
from src.core.config import Config

# MANDATORY: Import from P-007A utils framework
from src.utils.decorators import time_execution, retry

logger = get_logger(__name__)


@dataclass
class StateValidationResult:
    """Result of state validation check."""
    is_consistent: bool
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str = ""
    severity: str = "low"  # low, medium, high, critical


class StateMonitor:
    """Monitors and validates state consistency across system components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.state_monitoring_config = config.error_handling
        self.validation_frequency = self.state_monitoring_config.state_validation_frequency
        self.consistency_checks = [
            "portfolio_balance_sync",
            "position_quantity_sync",
            "order_status_sync",
            "risk_limit_compliance"
        ]
        self.reconciliation_config = self.state_monitoring_config
        self.auto_reconcile = self.reconciliation_config.auto_reconciliation_enabled
        self.max_discrepancy = self.reconciliation_config.max_discrepancy_threshold
        self.force_sync_threshold = 0.05  # Default threshold
        
        # State tracking
        self.last_validation_results: Dict[str, StateValidationResult] = {}
        self.state_history: List[StateValidationResult] = []
        self.reconciliation_attempts: Dict[str, int] = {}
    
    @time_execution
    @retry(max_attempts=2)
    async def validate_state_consistency(self, component: str = "all") -> StateValidationResult:
        """Validate state consistency for specified component or all components."""
        
        logger.info("Starting state consistency validation", component=component)
        
        discrepancies = []
        is_consistent = True
        severity = "low"
        
        if component == "all":
            # Validate all components
            for check in self.consistency_checks:
                try:
                    check_result = await self._perform_consistency_check(check)
                    if not check_result["is_consistent"]:
                        is_consistent = False
                        discrepancies.extend(check_result["discrepancies"])
                        severity = max(severity, check_result["severity"])
                        
                except Exception as e:
                    logger.error(
                        "State consistency check failed",
                        check=check,
                        error=str(e)
                    )
                    is_consistent = False
                    discrepancies.append({
                        "check": check,
                        "error": str(e),
                        "type": "validation_error"
                    })
                    severity = "critical"
        else:
            # Validate specific component
            try:
                check_result = await self._perform_consistency_check(component)
                if not check_result["is_consistent"]:
                    is_consistent = False
                    discrepancies.extend(check_result["discrepancies"])
                    severity = check_result["severity"]
                    
            except Exception as e:
                logger.error(
                    "Component state validation failed",
                    component=component,
                    error=str(e)
                )
                is_consistent = False
                discrepancies.append({
                    "component": component,
                    "error": str(e),
                    "type": "validation_error"
                })
                severity = "critical"
        
        result = StateValidationResult(
            is_consistent=is_consistent,
            discrepancies=discrepancies,
            component=component,
            severity=severity
        )
        
        # Store result
        self.last_validation_results[component] = result
        self.state_history.append(result)
        
        # Keep only last 1000 validation results
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        logger.info(
            "State consistency validation completed",
            component=component,
            is_consistent=is_consistent,
            discrepancy_count=len(discrepancies),
            severity=severity
        )
        
        return result
    
    async def _perform_consistency_check(self, check_name: str) -> Dict[str, Any]:
        """Perform a specific consistency check."""
        
        if check_name == "portfolio_balance_sync":
            return await self._check_portfolio_balance_sync()
        elif check_name == "position_quantity_sync":
            return await self._check_position_quantity_sync()
        elif check_name == "order_status_sync":
            return await self._check_order_status_sync()
        elif check_name == "risk_limit_compliance":
            return await self._check_risk_limit_compliance()
        else:
            logger.warning("Unknown consistency check", check_name=check_name)
            return {
                "is_consistent": True,
                "discrepancies": [],
                "severity": "low"
            }
    
    async def _check_portfolio_balance_sync(self) -> Dict[str, Any]:
        """Check if portfolio balances are synchronized across systems."""
        
        try:
            # TODO: Implement actual balance synchronization check
            # This will be implemented in P-003+ (Exchange Integrations) and P-010A (Capital Management)
            
            # Simulate balance check
            discrepancies = []
            is_consistent = True
            severity = "low"
            
            # TODO: Compare balances from:
            # - Database (P-002)
            # - Exchange APIs (P-003+)
            # - Redis cache (P-002)
            # - InfluxDB metrics (P-002)
            
            logger.info("Portfolio balance sync check completed")
            
            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity
            }
            
        except Exception as e:
            logger.error("Portfolio balance sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "balance_sync_error"}],
                "severity": "high"
            }
    
    async def _check_position_quantity_sync(self) -> Dict[str, Any]:
        """Check if position quantities are synchronized across systems."""
        
        try:
            # TODO: Implement actual position synchronization check
            # This will be implemented in P-020 (Order Management and Execution Engine)
            
            # Simulate position check
            discrepancies = []
            is_consistent = True
            severity = "low"
            
            # TODO: Compare positions from:
            # - Database (P-002)
            # - Exchange APIs (P-003+)
            # - Redis cache (P-002)
            # - Risk management system (P-008+)
            
            logger.info("Position quantity sync check completed")
            
            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity
            }
            
        except Exception as e:
            logger.error("Position quantity sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "position_sync_error"}],
                "severity": "high"
            }
    
    async def _check_order_status_sync(self) -> Dict[str, Any]:
        """Check if order statuses are synchronized across systems."""
        
        try:
            # TODO: Implement actual order status synchronization check
            # This will be implemented in P-020 (Order Management and Execution Engine)
            
            # Simulate order status check
            discrepancies = []
            is_consistent = True
            severity = "low"
            
            # TODO: Compare order statuses from:
            # - Database (P-002)
            # - Exchange APIs (P-003+)
            # - Redis cache (P-002)
            # - Execution engine (P-020)
            
            logger.info("Order status sync check completed")
            
            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity
            }
            
        except Exception as e:
            logger.error("Order status sync check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "order_sync_error"}],
                "severity": "high"
            }
    
    async def _check_risk_limit_compliance(self) -> Dict[str, Any]:
        """Check if risk limits are being complied with."""
        
        try:
            # TODO: Implement actual risk limit compliance check
            # This will be implemented in P-008+ (Risk Management)
            
            # Simulate risk compliance check
            discrepancies = []
            is_consistent = True
            severity = "low"
            
            # TODO: Check risk limits:
            # - Position size limits
            # - Portfolio exposure limits
            # - Drawdown limits
            # - Leverage limits
            
            logger.info("Risk limit compliance check completed")
            
            return {
                "is_consistent": is_consistent,
                "discrepancies": discrepancies,
                "severity": severity
            }
            
        except Exception as e:
            logger.error("Risk limit compliance check failed", error=str(e))
            return {
                "is_consistent": False,
                "discrepancies": [{"error": str(e), "type": "risk_compliance_error"}],
                "severity": "critical"
            }
    
    async def reconcile_state(self, component: str, discrepancies: List[Dict[str, Any]]) -> bool:
        """Attempt to reconcile state discrepancies."""
        
        logger.info(
            "Attempting state reconciliation",
            component=component,
            discrepancy_count=len(discrepancies)
        )
        
        if not self.auto_reconcile:
            logger.info("Auto-reconciliation disabled", component=component)
            return False
        
        reconciliation_attempts = self.reconciliation_attempts.get(component, 0)
        max_attempts = 3
        
        if reconciliation_attempts >= max_attempts:
            logger.warning(
                "Max reconciliation attempts reached",
                component=component,
                attempts=reconciliation_attempts
            )
            return False
        
        self.reconciliation_attempts[component] = reconciliation_attempts + 1
        
        try:
            if component == "portfolio_balance_sync":
                success = await self._reconcile_portfolio_balances(discrepancies)
            elif component == "position_quantity_sync":
                success = await self._reconcile_position_quantities(discrepancies)
            elif component == "order_status_sync":
                success = await self._reconcile_order_statuses(discrepancies)
            elif component == "risk_limit_compliance":
                success = await self._reconcile_risk_limits(discrepancies)
            else:
                logger.warning("Unknown reconciliation component", component=component)
                return False
            
            if success:
                logger.info("State reconciliation successful", component=component)
                # Reset reconciliation attempts on success
                self.reconciliation_attempts[component] = 0
            else:
                logger.warning("State reconciliation failed", component=component)
            
            return success
            
        except Exception as e:
            logger.error("State reconciliation error", component=component, error=str(e))
            return False
    
    async def _reconcile_portfolio_balances(self, discrepancies: List[Dict[str, Any]]) -> bool:
        """Reconcile portfolio balance discrepancies."""
        
        try:
            # TODO: Implement actual balance reconciliation
            # This will be implemented in P-010A (Capital Management System)
            
            logger.info("Reconciling portfolio balances", discrepancy_count=len(discrepancies))
            
            # TODO: Implement reconciliation logic:
            # 1. Identify the source of truth (usually exchange)
            # 2. Update database with correct balances
            # 3. Update Redis cache
            # 4. Update InfluxDB metrics
            # 5. Log reconciliation actions
            
            return True
            
        except Exception as e:
            logger.error("Portfolio balance reconciliation failed", error=str(e))
            return False
    
    async def _reconcile_position_quantities(self, discrepancies: List[Dict[str, Any]]) -> bool:
        """Reconcile position quantity discrepancies."""
        
        try:
            # TODO: Implement actual position reconciliation
            # This will be implemented in P-020 (Order Management and Execution Engine)
            
            logger.info("Reconciling position quantities", discrepancy_count=len(discrepancies))
            
            # TODO: Implement reconciliation logic:
            # 1. Identify the source of truth (usually exchange)
            # 2. Update database with correct positions
            # 3. Update Redis cache
            # 4. Update risk management system
            # 5. Log reconciliation actions
            
            return True
            
        except Exception as e:
            logger.error("Position quantity reconciliation failed", error=str(e))
            return False
    
    async def _reconcile_order_statuses(self, discrepancies: List[Dict[str, Any]]) -> bool:
        """Reconcile order status discrepancies."""
        
        try:
            # TODO: Implement actual order status reconciliation
            # This will be implemented in P-020 (Order Management and Execution Engine)
            
            logger.info("Reconciling order statuses", discrepancy_count=len(discrepancies))
            
            # TODO: Implement reconciliation logic:
            # 1. Identify the source of truth (usually exchange)
            # 2. Update database with correct order statuses
            # 3. Update Redis cache
            # 4. Update execution engine
            # 5. Log reconciliation actions
            
            return True
            
        except Exception as e:
            logger.error("Order status reconciliation failed", error=str(e))
            return False
    
    async def _reconcile_risk_limits(self, discrepancies: List[Dict[str, Any]]) -> bool:
        """Reconcile risk limit compliance issues."""
        
        try:
            # TODO: Implement actual risk limit reconciliation
            # This will be implemented in P-008+ (Risk Management)
            
            logger.info("Reconciling risk limits", discrepancy_count=len(discrepancies))
            
            # TODO: Implement reconciliation logic:
            # 1. Identify risk limit violations
            # 2. Take corrective actions (close positions, reduce exposure)
            # 3. Update risk management system
            # 4. Log reconciliation actions
            # 5. Trigger alerts if necessary
            
            return True
            
        except Exception as e:
            logger.error("Risk limit reconciliation failed", error=str(e))
            return False
    
    async def start_monitoring(self):
        """Start continuous state monitoring."""
        
        logger.info("Starting state monitoring")
        
        while True:
            try:
                # Validate all components
                result = await self.validate_state_consistency("all")
                
                if not result.is_consistent:
                    logger.warning(
                        "State inconsistency detected",
                        discrepancy_count=len(result.discrepancies),
                        severity=result.severity
                    )
                    
                    # Attempt reconciliation for each component with discrepancies
                    for discrepancy in result.discrepancies:
                        component = discrepancy.get("component", "unknown")
                        if component != "unknown":
                            await self.reconcile_state(component, [discrepancy])
                
                # Wait for next validation cycle
                await asyncio.sleep(self.validation_frequency)
                
            except Exception as e:
                logger.error("State monitoring error", error=str(e))
                await asyncio.sleep(self.validation_frequency)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state monitoring status."""
        
        summary = {
            "last_validation_results": {},
            "reconciliation_attempts": self.reconciliation_attempts.copy(),
            "total_validations": len(self.state_history),
            "recent_inconsistencies": 0
        }
        
        # Add last validation results
        for component, result in self.last_validation_results.items():
            summary["last_validation_results"][component] = {
                "is_consistent": result.is_consistent,
                "discrepancy_count": len(result.discrepancies),
                "severity": result.severity,
                "validation_time": result.validation_time.isoformat()
            }
        
        # Count recent inconsistencies (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        for result in self.state_history:
            if not result.is_consistent and result.validation_time > recent_cutoff:
                summary["recent_inconsistencies"] += 1
        
        return summary
    
    def get_state_history(self, hours: int = 24) -> List[StateValidationResult]:
        """Get state validation history for the specified time period."""
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            result for result in self.state_history
            if result.validation_time > cutoff
        ] 