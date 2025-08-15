"""
Grafana dashboard configurations for T-Bot Trading System.

This module provides comprehensive dashboard configurations for visualizing
trading system metrics, performance data, and operational health.

Key Features:
- Trading operations dashboard
- System performance monitoring
- Exchange health monitoring
- Risk management visualization
- Alert dashboard
- Custom panels and graphs
"""

import json
from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Panel:
    """Grafana panel configuration."""

    id: int
    title: str
    type: str  # graph, stat, table, heatmap, etc.
    targets: list[dict[str, Any]]
    gridPos: dict[str, int]  # {"x": 0, "y": 0, "w": 12, "h": 8}
    options: dict[str, Any] = field(default_factory=dict)
    fieldConfig: dict[str, Any] = field(default_factory=dict)
    datasource: str = "prometheus"

    def to_dict(self) -> dict[str, Any]:
        """Convert panel to Grafana JSON format."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "targets": self.targets,
            "gridPos": self.gridPos,
            "options": self.options,
            "fieldConfig": self.fieldConfig,
            "datasource": {"type": "prometheus", "uid": self.datasource},
        }


@dataclass
class Dashboard:
    """Grafana dashboard configuration."""

    title: str
    description: str
    tags: list[str]
    panels: list[Panel]
    uid: str = ""
    refresh: str = "30s"
    time_range: dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    variables: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert dashboard to Grafana JSON format."""
        return {
            "dashboard": {
                "id": None,
                "uid": self.uid,
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in self.panels],
                "time": self.time_range,
                "timepicker": {},
                "templating": {"list": self.variables},
                "annotations": {"list": []},
                "refresh": self.refresh,
                "schemaVersion": 30,
                "version": 1,
                "links": [],
            },
            "overwrite": True,
        }


class DashboardBuilder:
    """Builder for creating Grafana dashboards for T-Bot."""

    def __init__(self):
        """Initialize dashboard builder."""
        self.panel_id_counter = 1

    def get_next_panel_id(self) -> int:
        """Get next panel ID."""
        panel_id = self.panel_id_counter
        self.panel_id_counter += 1
        return panel_id

    def create_trading_overview_dashboard(self) -> Dashboard:
        """
        Create trading overview dashboard.

        Returns:
            Trading overview dashboard configuration
        """
        panels = []

        # Row 1: Key Metrics
        panels.extend(
            [
                # Total Portfolio Value
                Panel(
                    id=self.get_next_panel_id(),
                    title="Total Portfolio Value (USD)",
                    type="stat",
                    targets=[{"expr": "sum(tbot_portfolio_value_usd)", "refId": "A"}],
                    gridPos={"x": 0, "y": 0, "w": 6, "h": 4},
                    fieldConfig={"defaults": {"unit": "currencyUSD", "decimals": 2}},
                ),
                # Active Orders
                Panel(
                    id=self.get_next_panel_id(),
                    title="Active Orders",
                    type="stat",
                    targets=[
                        {"expr": 'sum(rate(tbot_orders_total{status="open"}[5m]))', "refId": "A"}
                    ],
                    gridPos={"x": 6, "y": 0, "w": 6, "h": 4},
                ),
                # Daily P&L
                Panel(
                    id=self.get_next_panel_id(),
                    title="Daily P&L (USD)",
                    type="stat",
                    targets=[{"expr": 'sum(tbot_portfolio_pnl_usd{timeframe="1d"})', "refId": "A"}],
                    gridPos={"x": 12, "y": 0, "w": 6, "h": 4},
                    fieldConfig={
                        "defaults": {
                            "unit": "currencyUSD",
                            "decimals": 2,
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "green", "value": 0},
                                ]
                            },
                        }
                    },
                ),
                # Active Alerts
                Panel(
                    id=self.get_next_panel_id(),
                    title="Active Alerts",
                    type="stat",
                    targets=[{"expr": "sum(tbot_alerts_active)", "refId": "A"}],
                    gridPos={"x": 18, "y": 0, "w": 6, "h": 4},
                    fieldConfig={
                        "defaults": {
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 1},
                                    {"color": "red", "value": 5},
                                ]
                            }
                        }
                    },
                ),
            ]
        )

        # Row 2: Trading Volume and Performance
        panels.extend(
            [
                # Trading Volume Over Time
                Panel(
                    id=self.get_next_panel_id(),
                    title="Trading Volume (USD)",
                    type="graph",
                    targets=[
                        {
                            "expr": "sum(rate(tbot_trades_volume_usd_sum[5m])) by (exchange)",
                            "refId": "A",
                            "legendFormat": "{{exchange}}",
                        }
                    ],
                    gridPos={"x": 0, "y": 4, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "currencyUSD"}},
                ),
                # P&L Distribution
                Panel(
                    id=self.get_next_panel_id(),
                    title="P&L Distribution",
                    type="histogram",
                    targets=[{"expr": "tbot_trades_pnl_usd", "refId": "A"}],
                    gridPos={"x": 12, "y": 4, "w": 12, "h": 8},
                ),
            ]
        )

        # Row 3: Exchange Health
        panels.extend(
            [
                # Exchange API Response Times
                Panel(
                    id=self.get_next_panel_id(),
                    title="Exchange API Response Times",
                    type="graph",
                    targets=[
                        {
                            "expr": "histogram_quantile(0.95, sum(rate(tbot_exchange_api_response_time_seconds_bucket[5m])) by (le, exchange))",
                            "refId": "A",
                            "legendFormat": "{{exchange}} 95th percentile",
                        }
                    ],
                    gridPos={"x": 0, "y": 12, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "s"}},
                ),
                # Exchange Health Scores
                Panel(
                    id=self.get_next_panel_id(),
                    title="Exchange Health Scores",
                    type="gauge",
                    targets=[
                        {
                            "expr": "tbot_exchange_health_score",
                            "refId": "A",
                            "legendFormat": "{{exchange}}",
                        }
                    ],
                    gridPos={"x": 12, "y": 12, "w": 12, "h": 8},
                    fieldConfig={
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "yellow", "value": 0.7},
                                    {"color": "green", "value": 0.9},
                                ]
                            },
                        }
                    },
                ),
            ]
        )

        # Variables for filtering
        variables = [
            {
                "name": "exchange",
                "type": "query",
                "query": "label_values(tbot_orders_total, exchange)",
                "multi": True,
                "includeAll": True,
            },
            {
                "name": "strategy",
                "type": "query",
                "query": "label_values(tbot_strategy_signals_total, strategy)",
                "multi": True,
                "includeAll": True,
            },
        ]

        return Dashboard(
            title="T-Bot Trading Overview",
            description="Comprehensive overview of T-Bot trading system performance",
            tags=["tbot", "trading", "overview"],
            panels=panels,
            uid="tbot-trading-overview",
            variables=variables,
        )

    def create_system_performance_dashboard(self) -> Dashboard:
        """
        Create system performance dashboard.

        Returns:
            System performance dashboard configuration
        """
        panels = []

        # Row 1: System Health
        panels.extend(
            [
                # CPU Usage
                Panel(
                    id=self.get_next_panel_id(),
                    title="CPU Usage (%)",
                    type="graph",
                    targets=[{"expr": "tbot_system_cpu_usage_percent", "refId": "A"}],
                    gridPos={"x": 0, "y": 0, "w": 8, "h": 8},
                    fieldConfig={"defaults": {"unit": "percent", "min": 0, "max": 100}},
                ),
                # Memory Usage
                Panel(
                    id=self.get_next_panel_id(),
                    title="Memory Usage",
                    type="graph",
                    targets=[
                        {
                            "expr": "tbot_system_memory_usage_percent",
                            "refId": "A",
                            "legendFormat": "Memory Usage %",
                        }
                    ],
                    gridPos={"x": 8, "y": 0, "w": 8, "h": 8},
                    fieldConfig={"defaults": {"unit": "percent", "min": 0, "max": 100}},
                ),
                # Disk Usage
                Panel(
                    id=self.get_next_panel_id(),
                    title="Disk Usage",
                    type="stat",
                    targets=[{"expr": "tbot_system_disk_usage_percent", "refId": "A"}],
                    gridPos={"x": 16, "y": 0, "w": 8, "h": 8},
                    fieldConfig={
                        "defaults": {
                            "unit": "percent",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 70},
                                    {"color": "red", "value": 90},
                                ]
                            },
                        }
                    },
                ),
            ]
        )

        # Row 2: Application Metrics
        panels.extend(
            [
                # Database Connections
                Panel(
                    id=self.get_next_panel_id(),
                    title="Database Connections",
                    type="graph",
                    targets=[
                        {
                            "expr": "tbot_database_connections_active",
                            "refId": "A",
                            "legendFormat": "{{database}}",
                        }
                    ],
                    gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                ),
                # Cache Hit Rate
                Panel(
                    id=self.get_next_panel_id(),
                    title="Cache Hit Rate",
                    type="graph",
                    targets=[
                        {
                            "expr": "tbot_cache_hit_rate_percent",
                            "refId": "A",
                            "legendFormat": "{{cache_type}}",
                        }
                    ],
                    gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "percent", "min": 0, "max": 100}},
                ),
            ]
        )

        # Row 3: Network and Latency
        panels.extend(
            [
                # Network I/O
                Panel(
                    id=self.get_next_panel_id(),
                    title="Network I/O",
                    type="graph",
                    targets=[
                        {
                            "expr": "rate(tbot_system_network_bytes_sent_total[5m])",
                            "refId": "A",
                            "legendFormat": "Bytes Sent/sec",
                        },
                        {
                            "expr": "rate(tbot_system_network_bytes_recv_total[5m])",
                            "refId": "B",
                            "legendFormat": "Bytes Received/sec",
                        },
                    ],
                    gridPos={"x": 0, "y": 16, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "binBps"}},
                ),
                # Application Uptime
                Panel(
                    id=self.get_next_panel_id(),
                    title="Application Uptime",
                    type="stat",
                    targets=[{"expr": "tbot_application_uptime_seconds", "refId": "A"}],
                    gridPos={"x": 12, "y": 16, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "s"}},
                ),
            ]
        )

        return Dashboard(
            title="T-Bot System Performance",
            description="System performance metrics and resource utilization",
            tags=["tbot", "system", "performance"],
            panels=panels,
            uid="tbot-system-performance",
        )

    def create_risk_management_dashboard(self) -> Dashboard:
        """
        Create risk management dashboard.

        Returns:
            Risk management dashboard configuration
        """
        panels = []

        # Row 1: Risk Metrics
        panels.extend(
            [
                # Value at Risk
                Panel(
                    id=self.get_next_panel_id(),
                    title="Value at Risk (VaR)",
                    type="graph",
                    targets=[
                        {
                            "expr": "tbot_risk_var_usd",
                            "refId": "A",
                            "legendFormat": "{{confidence_level}} confidence - {{timeframe}}",
                        }
                    ],
                    gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "currencyUSD"}},
                ),
                # Maximum Drawdown
                Panel(
                    id=self.get_next_panel_id(),
                    title="Maximum Drawdown",
                    type="stat",
                    targets=[{"expr": "max(tbot_risk_max_drawdown_percent)", "refId": "A"}],
                    gridPos={"x": 12, "y": 0, "w": 6, "h": 8},
                    fieldConfig={
                        "defaults": {
                            "unit": "percent",
                            "decimals": 2,
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": -10},
                                    {"color": "red", "value": -25},
                                ]
                            },
                        }
                    },
                ),
                # Sharpe Ratio
                Panel(
                    id=self.get_next_panel_id(),
                    title="Sharpe Ratio",
                    type="stat",
                    targets=[{"expr": 'tbot_risk_sharpe_ratio{timeframe="1d"}', "refId": "A"}],
                    gridPos={"x": 18, "y": 0, "w": 6, "h": 8},
                    fieldConfig={
                        "defaults": {
                            "decimals": 2,
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "yellow", "value": 1},
                                    {"color": "green", "value": 2},
                                ]
                            },
                        }
                    },
                ),
            ]
        )

        # Row 2: Position and Exposure
        panels.extend(
            [
                # Position Sizes
                Panel(
                    id=self.get_next_panel_id(),
                    title="Position Sizes (USD)",
                    type="graph",
                    targets=[
                        {
                            "expr": "tbot_risk_position_size_usd",
                            "refId": "A",
                            "legendFormat": "{{exchange}} - {{symbol}}",
                        }
                    ],
                    gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                    fieldConfig={"defaults": {"unit": "currencyUSD"}},
                ),
                # Portfolio Exposure
                Panel(
                    id=self.get_next_panel_id(),
                    title="Portfolio Exposure by Asset",
                    type="piechart",
                    targets=[
                        {
                            "expr": "tbot_portfolio_exposure_percent",
                            "refId": "A",
                            "legendFormat": "{{asset}}",
                        }
                    ],
                    gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                ),
            ]
        )

        # Row 3: Risk Violations and Circuit Breakers
        panels.extend(
            [
                # Risk Limit Violations
                Panel(
                    id=self.get_next_panel_id(),
                    title="Risk Limit Violations",
                    type="graph",
                    targets=[
                        {
                            "expr": "rate(tbot_risk_limit_violations_total[5m])",
                            "refId": "A",
                            "legendFormat": "{{limit_type}} - {{severity}}",
                        }
                    ],
                    gridPos={"x": 0, "y": 16, "w": 12, "h": 8},
                ),
                # Circuit Breaker Triggers
                Panel(
                    id=self.get_next_panel_id(),
                    title="Circuit Breaker Triggers",
                    type="table",
                    targets=[
                        {
                            "expr": "increase(tbot_risk_circuit_breaker_triggers_total[1h])",
                            "refId": "A",
                            "format": "table",
                        }
                    ],
                    gridPos={"x": 12, "y": 16, "w": 12, "h": 8},
                ),
            ]
        )

        return Dashboard(
            title="T-Bot Risk Management",
            description="Risk metrics and compliance monitoring",
            tags=["tbot", "risk", "compliance"],
            panels=panels,
            uid="tbot-risk-management",
        )

    def create_alerts_dashboard(self) -> Dashboard:
        """
        Create alerts monitoring dashboard.

        Returns:
            Alerts dashboard configuration
        """
        panels = []

        # Row 1: Alert Overview
        panels.extend(
            [
                # Active Alerts by Severity
                Panel(
                    id=self.get_next_panel_id(),
                    title="Active Alerts by Severity",
                    type="piechart",
                    targets=[
                        {
                            "expr": "sum(tbot_alerts_active) by (severity)",
                            "refId": "A",
                            "legendFormat": "{{severity}}",
                        }
                    ],
                    gridPos={"x": 0, "y": 0, "w": 8, "h": 8},
                ),
                # Alert Rate
                Panel(
                    id=self.get_next_panel_id(),
                    title="Alert Fire Rate (per minute)",
                    type="graph",
                    targets=[
                        {
                            "expr": "rate(tbot_alerts_fired_total[5m]) * 60",
                            "refId": "A",
                            "legendFormat": "Alerts/min",
                        }
                    ],
                    gridPos={"x": 8, "y": 0, "w": 8, "h": 8},
                ),
                # Mean Time to Resolution
                Panel(
                    id=self.get_next_panel_id(),
                    title="Mean Time to Resolution",
                    type="stat",
                    targets=[{"expr": "avg(tbot_alert_resolution_duration_seconds)", "refId": "A"}],
                    gridPos={"x": 16, "y": 0, "w": 8, "h": 8},
                    fieldConfig={"defaults": {"unit": "s"}},
                ),
            ]
        )

        # Row 2: Alert History and Trends
        panels.extend(
            [
                # Alert History
                Panel(
                    id=self.get_next_panel_id(),
                    title="Alert History",
                    type="table",
                    targets=[{"expr": "tbot_alerts_history", "refId": "A", "format": "table"}],
                    gridPos={"x": 0, "y": 8, "w": 24, "h": 8},
                )
            ]
        )

        return Dashboard(
            title="T-Bot Alerts",
            description="Alert monitoring and management",
            tags=["tbot", "alerts", "monitoring"],
            panels=panels,
            uid="tbot-alerts",
        )


class GrafanaDashboardManager:
    """Manager for Grafana dashboard operations."""

    def __init__(self, grafana_url: str, api_key: str):
        """
        Initialize Grafana dashboard manager.

        Args:
            grafana_url: Grafana server URL
            api_key: Grafana API key
        """
        self.grafana_url = grafana_url.rstrip("/")
        self.api_key = api_key
        self.builder = DashboardBuilder()

    async def deploy_all_dashboards(self) -> dict[str, bool]:
        """
        Deploy all T-Bot dashboards to Grafana.

        Returns:
            Dictionary with deployment results
        """
        dashboards = [
            self.builder.create_trading_overview_dashboard(),
            self.builder.create_system_performance_dashboard(),
            self.builder.create_risk_management_dashboard(),
            self.builder.create_alerts_dashboard(),
        ]

        results = {}
        for dashboard in dashboards:
            try:
                success = await self.deploy_dashboard(dashboard)
                results[dashboard.title] = success
                if success:
                    logger.info(f"Successfully deployed dashboard: {dashboard.title}")
                else:
                    logger.error(f"Failed to deploy dashboard: {dashboard.title}")
            except Exception as e:
                logger.error(f"Error deploying dashboard {dashboard.title}: {e}")
                results[dashboard.title] = False

        return results

    async def deploy_dashboard(self, dashboard: Dashboard) -> bool:
        """
        Deploy a single dashboard to Grafana.

        Args:
            dashboard: Dashboard to deploy

        Returns:
            True if deployment was successful
        """
        import aiohttp

        url = f"{self.grafana_url}/api/dashboards/db"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=dashboard.to_dict(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status in [200, 201]:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Grafana API error {response.status}: {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Error deploying dashboard to Grafana: {e}")
            return False

    def export_dashboards_to_files(self, output_dir: str) -> None:
        """
        Export dashboard configurations to JSON files.

        Args:
            output_dir: Directory to save dashboard files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        dashboards = [
            self.builder.create_trading_overview_dashboard(),
            self.builder.create_system_performance_dashboard(),
            self.builder.create_risk_management_dashboard(),
            self.builder.create_alerts_dashboard(),
        ]

        for dashboard in dashboards:
            filename = f"{dashboard.uid}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(dashboard.to_dict(), f, indent=2)

            logger.info(f"Exported dashboard to {filepath}")


def create_default_dashboards() -> list[Dashboard]:
    """
    Create all default T-Bot dashboards.

    Returns:
        List of default dashboard configurations
    """
    builder = DashboardBuilder()
    return [
        builder.create_trading_overview_dashboard(),
        builder.create_system_performance_dashboard(),
        builder.create_risk_management_dashboard(),
        builder.create_alerts_dashboard(),
    ]
