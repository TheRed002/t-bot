"""Tests for backtesting analysis module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from src.backtesting.analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
from src.core.config import Config


class TestMonteCarloAnalyzer:
    """Test MonteCarloAnalyzer."""

    def test_monte_carlo_analyzer_creation(self):
        """Test creating MonteCarloAnalyzer."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        analyzer = MonteCarloAnalyzer(config=config, engine_factory=engine_factory)
        
        assert analyzer.config == config
        assert analyzer.engine_factory == engine_factory

    def test_monte_carlo_analyzer_attributes(self):
        """Test MonteCarloAnalyzer has expected attributes."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        analyzer = MonteCarloAnalyzer(config=config, engine_factory=engine_factory)
        
        # Should have basic attributes
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'engine_factory')

    def test_monte_carlo_run_analysis_method_exists(self):
        """Test that run_analysis method exists."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        analyzer = MonteCarloAnalyzer(config=config, engine_factory=engine_factory)
        
        assert hasattr(analyzer, 'run_analysis')
        assert callable(analyzer.run_analysis)


class TestWalkForwardAnalyzer:
    """Test WalkForwardAnalyzer."""

    def test_walk_forward_analyzer_creation(self):
        """Test creating WalkForwardAnalyzer."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        analyzer = WalkForwardAnalyzer(config=config, engine_factory=engine_factory)
        
        assert analyzer.config == config
        assert analyzer.engine_factory == engine_factory

    def test_walk_forward_analyzer_attributes(self):
        """Test WalkForwardAnalyzer has expected attributes."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        analyzer = WalkForwardAnalyzer(config=config, engine_factory=engine_factory)
        
        # Should have basic attributes
        assert hasattr(analyzer, 'config')
        assert hasattr(analyzer, 'engine_factory')

    def test_walk_forward_run_analysis_method_exists(self):
        """Test that run_analysis method exists."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        analyzer = WalkForwardAnalyzer(config=config, engine_factory=engine_factory)
        
        assert hasattr(analyzer, 'run_analysis')
        assert callable(analyzer.run_analysis)


class TestAnalysisModuleImports:
    """Test that analysis module imports work correctly."""

    def test_monte_carlo_analyzer_import(self):
        """Test MonteCarloAnalyzer can be imported."""
        from src.backtesting.analysis import MonteCarloAnalyzer
        
        assert MonteCarloAnalyzer is not None
        assert hasattr(MonteCarloAnalyzer, '__init__')

    def test_walk_forward_analyzer_import(self):
        """Test WalkForwardAnalyzer can be imported."""
        from src.backtesting.analysis import WalkForwardAnalyzer
        
        assert WalkForwardAnalyzer is not None
        assert hasattr(WalkForwardAnalyzer, '__init__')

    def test_analyzer_classes_are_different(self):
        """Test that the analyzer classes are different."""
        from src.backtesting.analysis import MonteCarloAnalyzer, WalkForwardAnalyzer
        
        assert MonteCarloAnalyzer != WalkForwardAnalyzer
        assert MonteCarloAnalyzer.__name__ == "MonteCarloAnalyzer"
        assert WalkForwardAnalyzer.__name__ == "WalkForwardAnalyzer"


class TestAnalysisIntegration:
    """Test analysis module integration."""

    def test_analyzers_work_with_mocked_dependencies(self):
        """Test that analyzers can be created with mocked dependencies."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        # Should be able to create both analyzers
        mc_analyzer = MonteCarloAnalyzer(config=config, engine_factory=engine_factory)
        wf_analyzer = WalkForwardAnalyzer(config=config, engine_factory=engine_factory)
        
        assert mc_analyzer is not None
        assert wf_analyzer is not None
        assert mc_analyzer != wf_analyzer

    def test_analyzers_inherit_from_base_interface(self):
        """Test that analyzers have interface-like behavior."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        mc_analyzer = MonteCarloAnalyzer(config=config, engine_factory=engine_factory)
        wf_analyzer = WalkForwardAnalyzer(config=config, engine_factory=engine_factory)
        
        # Check they have the interface method
        assert hasattr(mc_analyzer, 'run_analysis')
        assert hasattr(wf_analyzer, 'run_analysis')
        assert callable(mc_analyzer.run_analysis)
        assert callable(wf_analyzer.run_analysis)

    def test_analyzers_have_run_analysis_method(self):
        """Test that analyzers have run_analysis method."""
        config = MagicMock(spec=Config)
        engine_factory = MagicMock()
        
        mc_analyzer = MonteCarloAnalyzer(config=config, engine_factory=engine_factory)
        wf_analyzer = WalkForwardAnalyzer(config=config, engine_factory=engine_factory)
        
        assert hasattr(mc_analyzer, 'run_analysis')
        assert hasattr(wf_analyzer, 'run_analysis')
        assert callable(mc_analyzer.run_analysis)
        assert callable(wf_analyzer.run_analysis)