"""
RTL_DIRECT_compare — direct spec-to-RTL comparison experiment.

This package generates Verilog directly from Stage 1 specs for the same
benchmarks that Module 2 routed to the c_first path, so the resulting RTL
can be compared against the Module 3→4→5 flow.
"""

from RTL_DIRECT_compare.generator import RTLDirectGenerator
from RTL_DIRECT_compare.schema import RTLDirectResult

__all__ = ["RTLDirectGenerator", "RTLDirectResult"]
