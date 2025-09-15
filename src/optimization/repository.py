"""
Optimization repository for result storage and retrieval.

This module provides repository implementations for persisting
optimization results and metadata with proper database models,
foreign key relationships, and financial data constraints.
"""

import asyncio
import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.core.base import BaseComponent
from src.core.event_constants import OptimizationEvents
from src.core.exceptions import RepositoryError
from src.database.models.optimization import (
    OptimizationResult as OptimizationResultDB,
    OptimizationRun,
    ParameterSet,
)
from src.optimization.core import OptimizationResult
from src.optimization.interfaces import OptimizationRepositoryProtocol


class OptimizationRepository(BaseComponent, OptimizationRepositoryProtocol):
    """
    Repository for optimization result persistence using database models.

    Provides database-backed storage for optimization runs, results,
    parameter sets, and objectives with proper relationships and constraints.
    """

    def __init__(
        self,
        session: AsyncSession | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize optimization repository.

        Args:
            session: Database session for database operations (optional for testing)
            name: Component name for identification
            config: Component configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "OptimizationRepository", config, correlation_id)
        self._session = session

        # Add database dependency if session is provided
        if session:
            self.add_dependency("AsyncSession")

        self._logger.info("OptimizationRepository initialized", has_session=session is not None)

    async def save_optimization_result(
        self,
        result: OptimizationResult,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save optimization result with proper database relationships.

        Args:
            result: Optimization result to save
            metadata: Additional metadata to store

        Returns:
            Optimization result ID
        """
        if not self._session:
            raise RepositoryError(
                "Database session not available - cannot save optimization result",
                error_code="REPO_001",
                context={"result_id": result.optimization_id},
            )

        try:
            # Create optimization run if it doesn't exist
            run_query = select(OptimizationRun).where(OptimizationRun.id == result.optimization_id)
            db_run = await self._session.scalar(run_query)

            if not db_run:
                # Create optimization run record
                db_run = OptimizationRun(
                    id=result.optimization_id,
                    algorithm_name=result.algorithm_name,
                    strategy_name=metadata.get("strategy_name") if metadata else None,
                    parameter_space={},  # Will be populated with actual parameter space
                    objectives_config={},  # Will be populated with objectives
                    status="completed",
                    current_iteration=result.iterations_completed,
                    total_iterations=result.iterations_completed,
                    completion_percentage=Decimal("100.00"),
                    evaluations_completed=result.evaluations_completed,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    total_duration_seconds=result.total_duration_seconds,
                    best_objective_value=result.optimal_objective_value,
                    convergence_achieved=result.convergence_achieved,
                    trading_mode=metadata.get("trading_mode") if metadata else None,
                    data_start_date=metadata.get("data_start_date") if metadata else None,
                    data_end_date=metadata.get("data_end_date") if metadata else None,
                    initial_capital=metadata.get("initial_capital") if metadata else None,
                    warnings=result.warnings or [],
                )
                self._session.add(db_run)
                await self._session.flush()  # Ensure run ID is available

            # Create optimization result record
            db_result = OptimizationResultDB(
                id=str(uuid.uuid4()),
                optimization_run_id=result.optimization_id,
                optimal_parameters=result.optimal_parameters,
                optimal_objective_value=result.optimal_objective_value,
                objective_values=result.objective_values,
                validation_score=result.validation_score,
                overfitting_score=result.overfitting_score,
                robustness_score=result.robustness_score,
                parameter_stability=result.parameter_stability or {},
                sensitivity_analysis=result.sensitivity_analysis or {},
                statistical_significance=result.statistical_significance,
                confidence_interval_lower=result.confidence_interval[0]
                if result.confidence_interval
                else None,
                confidence_interval_upper=result.confidence_interval[1]
                if result.confidence_interval
                else None,
                is_statistically_significant=result.is_statistically_significant(),
            )

            # Extract financial metrics from objective values
            if result.objective_values:
                db_result.sharpe_ratio = result.objective_values.get("sharpe_ratio")
                db_result.max_drawdown = result.objective_values.get("max_drawdown")
                db_result.total_return = result.objective_values.get("total_return")
                db_result.win_rate = result.objective_values.get("win_rate")
                db_result.profit_factor = result.objective_values.get("profit_factor")
                db_result.volatility = result.objective_values.get("volatility")
                db_result.value_at_risk = result.objective_values.get("value_at_risk")
                db_result.conditional_var = result.objective_values.get("conditional_var")
                db_result.beta = result.objective_values.get("beta")
                db_result.alpha = result.objective_values.get("alpha")

            self._session.add(db_result)
            await self._session.commit()

            self._logger.info(
                "Optimization result saved to database",
                optimization_id=result.optimization_id,
                algorithm=result.algorithm_name,
                result_id=db_result.id,
            )

            # Emit result saved event with proper async context and timeout
            try:
                if hasattr(self, "emit_event") and callable(self.emit_event):
                    event_data = {
                        "optimization_id": result.optimization_id,
                        "algorithm_name": result.algorithm_name,
                        "result_id": db_result.id,
                        "optimal_objective_value": str(result.optimal_objective_value),
                        "timestamp": datetime.now().isoformat(),
                    }
                    await asyncio.wait_for(
                        self.emit_event(OptimizationEvents.RESULT_SAVED, event_data),
                        timeout=5.0,  # 5 second timeout for WebSocket operations
                    )
            except (AttributeError, asyncio.TimeoutError) as e:
                # Graceful fallback if event emission not available or times out
                self._logger.debug(f"Event emission not available or timed out: {e}")
            except Exception as e:
                # Log but don't fail the save operation for WebSocket issues
                self._logger.warning(f"Failed to emit result saved event: {e}")

            return result.optimization_id

        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"Failed to save optimization result: {e}")
            raise RepositoryError(f"Failed to save optimization result: {e}") from e
        finally:
            # Ensure session resources are properly handled
            try:
                # Expire all cached objects to prevent stale data
                self._session.expunge_all()
            except Exception as cleanup_error:
                self._logger.debug(f"Error during session cleanup: {cleanup_error}")

    async def get_optimization_result(self, optimization_id: str) -> OptimizationResult | None:
        """
        Retrieve optimization result by ID from database.

        Args:
            optimization_id: Optimization result ID

        Returns:
            Optimization result or None if not found
        """
        if not self._session:
            self._logger.warning(
                "Database session not available - cannot retrieve optimization result"
            )
            return None

        try:
            # Query database for optimization result
            query = (
                select(OptimizationResultDB)
                .options(joinedload(OptimizationResultDB.optimization_run))
                .where(OptimizationResultDB.optimization_run_id == optimization_id)
            )
            result = await self._session.execute(query)
            row = result.first()

            if not row:
                return None

            db_result = row[0]
            db_run = db_result.optimization_run

            # Convert database model to domain model
            optimization_result = OptimizationResult(
                optimization_id=db_result.optimization_run_id,
                algorithm_name=db_run.algorithm_name,
                optimal_parameters=db_result.optimal_parameters,
                optimal_objective_value=db_result.optimal_objective_value,
                objective_values=db_result.objective_values,
                validation_score=db_result.validation_score,
                overfitting_score=db_result.overfitting_score,
                robustness_score=db_result.robustness_score,
                iterations_completed=db_run.current_iteration,
                evaluations_completed=db_run.evaluations_completed,
                convergence_achieved=db_run.convergence_achieved,
                start_time=db_run.start_time,
                end_time=db_run.end_time,
                total_duration_seconds=db_run.total_duration_seconds,
                parameter_stability=db_result.parameter_stability or {},
                sensitivity_analysis=db_result.sensitivity_analysis or {},
                warnings=db_run.warnings or [],
                config_used=db_run.algorithm_config or {},
                statistical_significance=db_result.statistical_significance,
                confidence_interval=(
                    (db_result.confidence_interval_lower, db_result.confidence_interval_upper)
                    if db_result.confidence_interval_lower is not None
                    and db_result.confidence_interval_upper is not None
                    else None
                ),
            )

            return optimization_result

        except Exception as e:
            self._logger.error(f"Failed to retrieve optimization result: {e}")
            raise RepositoryError(f"Failed to retrieve optimization result: {e}") from e

    async def list_optimization_results(
        self,
        strategy_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[OptimizationResult]:
        """
        List optimization results with optional filtering from database.

        Args:
            strategy_name: Optional strategy name filter
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            List of optimization results
        """
        if not self._session:
            self._logger.warning("Database session not available - returning empty list")
            return []

        try:
            # Build query with optional filtering
            query = (
                select(OptimizationResultDB)
                .options(joinedload(OptimizationResultDB.optimization_run))
                .order_by(desc(OptimizationResultDB.created_at))
            )

            # Apply strategy name filter if provided
            if strategy_name:
                query = query.join(OptimizationRun).where(
                    OptimizationRun.strategy_name == strategy_name
                )

            # Apply pagination
            query = query.offset(offset).limit(limit)

            result = await self._session.execute(query)
            rows = result.fetchall()

            # Convert database models to domain models
            optimization_results = []
            for row in rows:
                db_result = row[0]
                db_run = db_result.optimization_run
                optimization_result = OptimizationResult(
                    optimization_id=db_result.optimization_run_id,
                    algorithm_name=db_run.algorithm_name,
                    optimal_parameters=db_result.optimal_parameters,
                    optimal_objective_value=db_result.optimal_objective_value,
                    objective_values=db_result.objective_values,
                    validation_score=db_result.validation_score,
                    overfitting_score=db_result.overfitting_score,
                    robustness_score=db_result.robustness_score,
                    iterations_completed=db_run.current_iteration,
                    evaluations_completed=db_run.evaluations_completed,
                    convergence_achieved=db_run.convergence_achieved,
                    start_time=db_run.start_time,
                    end_time=db_run.end_time,
                    total_duration_seconds=db_run.total_duration_seconds,
                    parameter_stability=db_result.parameter_stability or {},
                    sensitivity_analysis=db_result.sensitivity_analysis or {},
                    warnings=db_run.warnings or [],
                    config_used=db_run.algorithm_config or {},
                    statistical_significance=db_result.statistical_significance,
                    confidence_interval=(
                        (db_result.confidence_interval_lower, db_result.confidence_interval_upper)
                        if db_result.confidence_interval_lower is not None
                        and db_result.confidence_interval_upper is not None
                        else None
                    ),
                )
                optimization_results.append(optimization_result)

            return optimization_results

        except Exception as e:
            self._logger.error(f"Failed to list optimization results: {e}")
            raise RepositoryError(f"Failed to list optimization results: {e}") from e

    async def delete_optimization_result(self, optimization_id: str) -> bool:
        """
        Delete optimization result by ID from database.

        Args:
            optimization_id: Optimization result ID

        Returns:
            True if deleted successfully
        """
        try:
            # Find and delete optimization run (cascade will handle related records)
            run_query = select(OptimizationRun).where(OptimizationRun.id == optimization_id)
            db_run = await self._session.scalar(run_query)

            if db_run:
                self._session.delete(db_run)
                await self._session.commit()

                self._logger.info(
                    "Optimization result deleted from database", optimization_id=optimization_id
                )
                return True
            else:
                self._logger.warning(
                    "Optimization result not found for deletion", optimization_id=optimization_id
                )
                return False

        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"Failed to delete optimization result: {e}")
            raise RepositoryError(f"Failed to delete optimization result: {e}") from e
        finally:
            # Ensure session resources are properly handled
            try:
                # Clear any cached objects after deletion
                self._session.expunge_all()
            except Exception as cleanup_error:
                self._logger.debug(f"Error during deletion session cleanup: {cleanup_error}")

    async def save_parameter_set(
        self,
        optimization_id: str,
        parameters: dict[str, Any],
        objective_value: Decimal | None,
        iteration_number: int,
        **kwargs: Any,
    ) -> str:
        """
        Save parameter set evaluation result.

        Args:
            optimization_id: Optimization run ID
            parameters: Parameter values
            objective_value: Objective value achieved
            iteration_number: Iteration number
            **kwargs: Additional parameter set data

        Returns:
            Parameter set ID
        """
        try:
            # Create parameter hash for deduplication
            import hashlib

            param_str = json.dumps(parameters, sort_keys=True, default=str)
            parameter_hash = hashlib.sha256(param_str.encode()).hexdigest()

            # Create parameter set record
            parameter_set = ParameterSet(
                id=str(uuid.uuid4()),
                optimization_run_id=optimization_id,
                parameters=parameters,
                parameter_hash=parameter_hash,
                objective_value=objective_value,
                objective_values=kwargs.get("objective_values"),
                constraint_violations=kwargs.get("constraint_violations"),
                is_feasible=kwargs.get("is_feasible", True),
                iteration_number=iteration_number,
                evaluation_time_seconds=kwargs.get("evaluation_time_seconds"),
                evaluation_status=kwargs.get("evaluation_status", "completed"),
                evaluation_error=kwargs.get("evaluation_error"),
                sharpe_ratio=kwargs.get("sharpe_ratio"),
                total_return=kwargs.get("total_return"),
                max_drawdown=kwargs.get("max_drawdown"),
                rank_by_objective=kwargs.get("rank_by_objective"),
                percentile_rank=kwargs.get("percentile_rank"),
            )

            self._session.add(parameter_set)
            await self._session.commit()

            self._logger.debug(
                "Parameter set saved",
                optimization_id=optimization_id,
                parameter_set_id=parameter_set.id,
                iteration=iteration_number,
            )

            return parameter_set.id

        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"Failed to save parameter set: {e}")
            raise RepositoryError(f"Failed to save parameter set: {e}") from e
        finally:
            # Ensure session resources are properly handled
            try:
                # Expire all cached objects to prevent memory leaks
                self._session.expunge_all()
            except Exception as cleanup_error:
                self._logger.debug(f"Error during parameter set session cleanup: {cleanup_error}")

    async def get_parameter_sets(
        self,
        optimization_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get parameter sets for optimization run.

        Args:
            optimization_id: Optimization run ID
            limit: Maximum number of parameter sets to return

        Returns:
            List of parameter set data
        """
        try:
            query = (
                select(ParameterSet)
                .where(ParameterSet.optimization_run_id == optimization_id)
                .order_by(ParameterSet.iteration_number)
            )

            if limit:
                query = query.limit(limit)

            result = await self._session.execute(query)
            parameter_sets = result.scalars().all()

            # Convert to dictionaries
            parameter_set_data = []
            for ps in parameter_sets:
                parameter_set_data.append(
                    {
                        "id": ps.id,
                        "parameters": ps.parameters,
                        "objective_value": ps.objective_value,
                        "objective_values": ps.objective_values,
                        "is_feasible": ps.is_feasible,
                        "iteration_number": ps.iteration_number,
                        "evaluation_time_seconds": ps.evaluation_time_seconds,
                        "evaluation_status": ps.evaluation_status,
                        "rank_by_objective": ps.rank_by_objective,
                        "percentile_rank": ps.percentile_rank,
                        "created_at": ps.created_at,
                    }
                )

            return parameter_set_data

        except Exception as e:
            self._logger.error(f"Failed to get parameter sets: {e}")
            raise RepositoryError(f"Failed to get parameter sets: {e}") from e
