"""
spec_analyze — Stage 1: Spec Understanding & Feature Extraction

Analyzes Verilog source files or natural-language specifications and
produces a unified feature set for downstream path selection (Stage 2)
and RAG retrieval (Stage 3).

Usage:
    from spec_analyze import Analyzer

    analyzer = Analyzer()                        # loads .env automatically
    result = analyzer.analyze_file("uart_tx.v")  # from Verilog
    result = analyzer.analyze_spec("A 32-bit UART transmitter with parity")
    result = analyzer.analyze_dir("benchmark/")  # batch
"""

from spec_analyze.analyzer import Analyzer
from spec_analyze.schema import FeatureResult, validate_features
from spec_analyze.spec_loader import load_spec_entries

__all__ = ["Analyzer", "FeatureResult", "load_spec_entries", "validate_features"]
