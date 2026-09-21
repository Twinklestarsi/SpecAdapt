"""
c_gen — Module 3: C code generation for HLS-based optimization path.

Generates HLS-compatible C code from Verilog or natural-language specifications.
"""

from c_gen.schema import CGenResult
from c_gen.generator import CGenerator

__all__ = ["CGenResult", "CGenerator"]
