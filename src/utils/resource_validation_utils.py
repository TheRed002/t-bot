"""
Shared resource management validation utilities.

Extracted from duplicated code in ResourceManager, BotResourceService, and ResourceManagementService.
"""

from decimal import Decimal
from typing import Any, Dict
from enum import Enum

from src.core.logging import get_logger
from src.core.types import BotPriority, ResourceType
from src.core.exceptions import ValidationError

logger = get_logger(__name__)


class ResourceValidationError(ValidationError):
    """Specific validation error for resource operations."""
    pass


class ResourceValidator:
    """Consolidated resource validation logic."""

    @staticmethod
    def validate_resource_request(
        bot_id: str,
        capital_amount: Decimal,
        priority: BotPriority = BotPriority.NORMAL,
        additional_resources: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Validate a resource request with standard rules.

        Args:
            bot_id: Bot identifier
            capital_amount: Requested capital amount
            priority: Bot priority level
            additional_resources: Additional resource requirements

        Returns:
            Validation result with requirements and issues

        Raises:
            ResourceValidationError: If validation fails
        """
        if not bot_id or not isinstance(bot_id, str):
            raise ResourceValidationError("Invalid bot_id: must be non-empty string")

        if not isinstance(capital_amount, Decimal) or capital_amount <= 0:
            raise ResourceValidationError("Invalid capital_amount: must be positive Decimal")

        if not isinstance(priority, BotPriority):
            raise ResourceValidationError("Invalid priority: must be BotPriority enum")

        try:
            # Calculate standard resource requirements
            requirements = ResourceValidator.calculate_resource_requirements(
                bot_id, capital_amount, priority, additional_resources
            )

            # Validate individual resource types
            validation_issues = []
            for resource_type, amount in requirements.items():
                issues = ResourceValidator._validate_resource_type(resource_type, amount)
                validation_issues.extend(issues)

            return {
                'valid': len(validation_issues) == 0,
                'requirements': requirements,
                'issues': validation_issues,
                'bot_id': bot_id,
                'priority': priority
            }

        except Exception as e:
            logger.error(f"Resource validation failed for bot {bot_id}: {e}")
            raise ResourceValidationError(f"Validation failed: {e}") from e

    @staticmethod
    def calculate_resource_requirements(
        bot_id: str,
        capital_amount: Decimal,
        priority: BotPriority = BotPriority.NORMAL,
        additional_resources: dict[str, Any] | None = None
    ) -> dict[ResourceType, Decimal]:
        """
        Calculate standardized resource requirements for a bot.

        This consolidates the resource calculation logic used across multiple services.
        """
        requirements: dict[ResourceType, Decimal] = {}

        try:
            # Capital allocation (always required)
            requirements[ResourceType.CAPITAL] = capital_amount

            # API rate limits based on priority and capital
            base_api_requests = Decimal('100')  # Base requests per minute
            priority_multiplier = {
                BotPriority.LOW: Decimal('0.5'),
                BotPriority.NORMAL: Decimal('1.0'),
                BotPriority.HIGH: Decimal('1.5'),
                BotPriority.CRITICAL: Decimal('2.0')
            }

            # Scale API requests with capital amount (more capital = more trading = more API calls)
            capital_scale = min(Decimal('5.0'), capital_amount / Decimal('1000'))
            api_requests = base_api_requests * priority_multiplier[priority] * capital_scale
            requirements[ResourceType.API_CALLS] = api_requests

            # Database connections
            base_db_connections = Decimal('2')  # Base connections
            db_connections = base_db_connections * priority_multiplier[priority]
            requirements[ResourceType.DATABASE_CONNECTIONS] = db_connections

            # Memory allocation (estimated based on complexity)
            base_memory = Decimal('100')  # Base MB
            memory_mb = base_memory * priority_multiplier[priority]
            requirements[ResourceType.MEMORY] = memory_mb

            # CPU allocation
            base_cpu = Decimal('10')  # Base CPU percentage
            cpu_percent = base_cpu * priority_multiplier[priority]
            requirements[ResourceType.CPU] = cpu_percent

            # Add any additional resource requirements
            if additional_resources:
                for resource_name, amount in additional_resources.items():
                    try:
                        resource_type = ResourceType(resource_name)
                        requirements[resource_type] = Decimal(str(amount))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid additional resource {resource_name}: {e}")

            return requirements

        except Exception as e:
            logger.error(f"Error calculating resource requirements for bot {bot_id}: {e}")
            # Return minimal requirements as fallback
            return {
                ResourceType.CAPITAL: capital_amount,
                ResourceType.API_CALLS: Decimal('50'),
                ResourceType.DATABASE_CONNECTIONS: Decimal('1'),
                ResourceType.MEMORY: Decimal('50'),
                ResourceType.CPU: Decimal('5')
            }

    @staticmethod
    def _validate_resource_type(resource_type: ResourceType, amount: Decimal) -> list[str]:
        """Validate individual resource type and amount."""
        issues = []

        try:
            # Resource-specific validation rules
            if resource_type == ResourceType.CAPITAL:
                if amount < Decimal('10'):
                    issues.append(f"Capital amount too low: {amount} (minimum: 10)")
                elif amount > Decimal('1000000'):
                    issues.append(f"Capital amount too high: {amount} (maximum: 1,000,000)")

            elif resource_type == ResourceType.API_CALLS:
                if amount < Decimal('1'):
                    issues.append(f"API requests too low: {amount} (minimum: 1)")
                elif amount > Decimal('10000'):
                    issues.append(f"API requests too high: {amount} (maximum: 10,000)")

            elif resource_type == ResourceType.DATABASE_CONNECTIONS:
                if amount < Decimal('1'):
                    issues.append(f"Database connections too low: {amount} (minimum: 1)")
                elif amount > Decimal('50'):
                    issues.append(f"Database connections too high: {amount} (maximum: 50)")

            elif resource_type == ResourceType.MEMORY:
                if amount < Decimal('10'):
                    issues.append(f"Memory allocation too low: {amount}MB (minimum: 10MB)")
                elif amount > Decimal('8192'):
                    issues.append(f"Memory allocation too high: {amount}MB (maximum: 8GB)")

            elif resource_type == ResourceType.CPU:
                if amount < Decimal('1'):
                    issues.append(f"CPU allocation too low: {amount}% (minimum: 1%)")
                elif amount > Decimal('100'):
                    issues.append(f"CPU allocation too high: {amount}% (maximum: 100%)")

        except Exception as e:
            issues.append(f"Validation error for {resource_type}: {e}")

        return issues

    @staticmethod
    def check_resource_availability(
        requirements: dict[ResourceType, Decimal],
        current_allocations: dict[ResourceType, Decimal],
        limits: dict[ResourceType, Decimal]
    ) -> dict[str, Any]:
        """
        Check if resources are available for allocation.

        Args:
            requirements: Required resources
            current_allocations: Currently allocated resources
            limits: Resource limits

        Returns:
            Availability check result
        """
        result = {
            'available': True,
            'conflicts': [],
            'utilization': {},
            'remaining': {}
        }

        try:
            for resource_type, required_amount in requirements.items():
                current = current_allocations.get(resource_type, Decimal('0'))
                limit = limits.get(resource_type, Decimal('0'))

                if limit <= 0:
                    result['conflicts'].append(f"No limit set for {resource_type}")
                    result['available'] = False
                    continue

                total_needed = current + required_amount
                utilization = float(total_needed / limit)
                remaining = limit - total_needed

                result['utilization'][resource_type] = utilization
                result['remaining'][resource_type] = float(remaining)

                if total_needed > limit:
                    result['conflicts'].append(
                        f"{resource_type} would exceed limit: "
                        f"need {total_needed}, limit {limit}"
                    )
                    result['available'] = False

                # Warn if utilization is very high
                elif utilization > 0.9:
                    result['conflicts'].append(
                        f"{resource_type} utilization would be high: {utilization:.1%}"
                    )

            return result

        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            return {
                'available': False,
                'conflicts': [f"Availability check failed: {e}"],
                'utilization': {},
                'remaining': {}
            }


class ResourceAllocationTracker:
    """Track and validate resource allocations."""

    @staticmethod
    def validate_allocation_state(
        bot_id: str,
        allocation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate the state of a resource allocation.

        Args:
            bot_id: Bot identifier
            allocation_data: Allocation data to validate

        Returns:
            Validation result with status and issues
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'allocation_status': 'unknown'
        }

        try:
            # Check required fields
            required_fields = ['resource_type', 'allocated_amount', 'allocation_time']
            for field in required_fields:
                if field not in allocation_data:
                    validation_result['issues'].append(f"Missing required field: {field}")
                    validation_result['valid'] = False

            if not validation_result['valid']:
                return validation_result

            # Validate resource type
            try:
                resource_type = ResourceType(allocation_data['resource_type'])
            except ValueError:
                validation_result['issues'].append(
                    f"Invalid resource type: {allocation_data['resource_type']}"
                )
                validation_result['valid'] = False
                return validation_result

            # Validate allocated amount
            try:
                amount = Decimal(str(allocation_data['allocated_amount']))
                if amount <= 0:
                    validation_result['issues'].append("Allocated amount must be positive")
                    validation_result['valid'] = False
            except (ValueError, TypeError):
                validation_result['issues'].append("Invalid allocated amount format")
                validation_result['valid'] = False

            # Check allocation age
            try:
                from datetime import datetime, timezone
                allocation_time = datetime.fromisoformat(
                    allocation_data['allocation_time'].replace('Z', '+00:00')
                )
                age = datetime.now(timezone.utc) - allocation_time

                if age.total_seconds() > 86400:  # 24 hours
                    validation_result['warnings'].append("Allocation is over 24 hours old")

                validation_result['allocation_status'] = 'active'

            except (ValueError, TypeError):
                validation_result['warnings'].append("Invalid allocation timestamp")

            # Validate connection data if present
            if 'connections' in allocation_data:
                connections = allocation_data['connections']
                if isinstance(connections, dict):
                    if 'database' in connections and connections['database'] is None:
                        validation_result['warnings'].append("Database connection is None")
                    if 'websockets' in connections and not connections['websockets']:
                        validation_result['warnings'].append("No websocket connections")

            return validation_result

        except Exception as e:
            logger.error(f"Error validating allocation for bot {bot_id}: {e}")
            return {
                'valid': False,
                'issues': [f"Validation error: {e}"],
                'warnings': [],
                'allocation_status': 'error'
            }


class ResourceConflictDetector:
    """Detect and analyze resource conflicts."""

    @staticmethod
    def detect_allocation_conflicts(
        all_allocations: dict[str, dict[str, Any]],
        resource_limits: dict[ResourceType, Decimal]
    ) -> list[dict[str, Any]]:
        """
        Detect conflicts in resource allocations across all bots.

        Args:
            all_allocations: All current resource allocations by bot_id
            resource_limits: Global resource limits

        Returns:
            List of detected conflicts
        """
        conflicts = []

        try:
            # Calculate total resource usage by type
            total_usage: dict[ResourceType, Decimal] = {}

            for bot_id, bot_allocations in all_allocations.items():
                for resource_type_str, allocation_data in bot_allocations.items():
                    try:
                        resource_type = ResourceType(resource_type_str)
                        amount = Decimal(str(allocation_data.get('allocated_amount', 0)))

                        if resource_type not in total_usage:
                            total_usage[resource_type] = Decimal('0')

                        total_usage[resource_type] += amount

                    except (ValueError, TypeError) as e:
                        conflicts.append({
                            'type': 'invalid_allocation',
                            'bot_id': bot_id,
                            'resource_type': resource_type_str,
                            'issue': f"Invalid allocation data: {e}",
                            'severity': 'medium'
                        })

            # Check for limit violations
            for resource_type, total_used in total_usage.items():
                limit = resource_limits.get(resource_type, Decimal('0'))

                if limit > 0 and total_used > limit:
                    conflicts.append({
                        'type': 'limit_violation',
                        'resource_type': resource_type.value,
                        'total_used': float(total_used),
                        'limit': float(limit),
                        'excess': float(total_used - limit),
                        'severity': 'high',
                        'issue': f"{resource_type.value} usage exceeds limit"
                    })

                elif limit > 0 and total_used > limit * Decimal('0.9'):
                    conflicts.append({
                        'type': 'approaching_limit',
                        'resource_type': resource_type.value,
                        'total_used': float(total_used),
                        'limit': float(limit),
                        'utilization': float(total_used / limit),
                        'severity': 'medium',
                        'issue': f"{resource_type.value} usage is high"
                    })

            return conflicts

        except Exception as e:
            logger.error(f"Error detecting resource conflicts: {e}")
            return [{
                'type': 'detection_error',
                'issue': f"Conflict detection failed: {e}",
                'severity': 'high'
            }]