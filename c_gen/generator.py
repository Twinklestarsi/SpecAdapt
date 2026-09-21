"""
generator.py — CGenerator orchestrator for Module 3.

Coordinates the full C generation pipeline:
  1. Dispatch to llm_gen based on input_type
  2. Write raw C to temp file
  3. Validate with clang-16, retry with errors if needed
  4. On total LLM failure + Verilog input → try v2c fallback
  5. Run postprocess (state writeback patch; optional legacy HLS pragmas)
  6. Final validation
  7. Return CGenResult

On total failure (both LLM and v2c fail), returns success=False.
Downstream modules can then re-route to RTL-direct path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from c_gen.schema import CGenResult
from c_gen import llm_gen, validator, postprocess, v2c_fallback
from token_counter import TokenUsage


class CGenerator:
    """Module 3 orchestrator: C code generation with retry and fallback."""

    def __init__(
        self,
        output_dir: str | Path = "/tmp/c_gen_output",
        env_path: str | Path = ".env",
        max_llm_retries: int = 3,
        memory_session: Optional[Any] = None,
        inject_interface_pragmas: bool = True,
    ):
        """
        Args:
            output_dir: Directory to write generated .c files
            env_path: Path to .env file for LLM API config
            max_llm_retries: Max retry attempts for LLM generation with clang errors
            inject_interface_pragmas: Keep legacy HLS pragma injection when True.
                The active C-first AI-to-RTL route sets this to False.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_path = Path(env_path)
        self.max_llm_retries = max_llm_retries
        self.memory_session = memory_session
        self.inject_interface_pragmas = inject_interface_pragmas

    def _postprocess(self, c_path: Path) -> None:
        postprocess.patch_state_writebacks(c_path)
        if self.inject_interface_pragmas:
            postprocess.inject_hls_pragmas(c_path)

    def generate(
        self,
        benchmark: str,
        input_type: str,
        verilog_path: Optional[str] = None,
        spec_text: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> CGenResult:
        """
        Generate C code for one benchmark.

        Args:
            benchmark: Benchmark name
            input_type: "verilog" | "spec" | "mixed"
            verilog_path: Path to .v file (for verilog/mixed)
            spec_text: Natural-language spec (for spec/mixed)
            features: Feature dict from Module 1 (for spec mode)

        Returns:
            CGenResult with success status and output path
        """
        c_path = self.output_dir / f"{benchmark}.c"

        guidance_text = ""
        guidance = []
        if self.memory_session is not None:
            guidance = self.memory_session.failure_guidance(
                "module3",
                "c_generation",
                {
                    "input_type": input_type,
                    "benchmark": benchmark,
                    "failure_type": "c_generation_failed",
                },
            )
            successful_fixes = [
                str(item.get("fix_applied") or "")
                for item in guidance
                if item.get("fix_succeeded") is True
                and item.get("fix_applied")
                and float(item.get("score", 0.0)) > 0.0
            ]
            if successful_fixes:
                guidance_text = (
                    "\n\nHistorical fixes that succeeded in similar cases:\n- "
                    + "\n- ".join(successful_fixes[:3])
                )

        input_payload = {
            "input_type": input_type,
            "verilog_path": verilog_path,
            "has_spec": bool(spec_text),
            "failure_guidance_count": len(guidance),
        }
        if input_type == "spec":
            result = self._generate_from_spec(
                benchmark,
                (spec_text or "") + guidance_text,
                features,
                c_path,
            )
        elif input_type == "verilog":
            result = self._generate_from_verilog(
                benchmark, verilog_path, c_path, guidance_text
            )
        elif input_type == "mixed":
            result = self._generate_mixed(
                benchmark,
                verilog_path,
                (spec_text or "") + guidance_text,
                c_path,
            )
        else:
            result = CGenResult(
                benchmark=benchmark,
                c_path=str(c_path),
                method="unknown",
                success=False,
                error=f"Unknown input_type: {input_type}",
            )
        if self.memory_session is not None:
            self.memory_session.record_c_generation(result, input_payload)
        return result

    def _generate_from_spec(
        self,
        benchmark: str,
        spec_text: str,
        features: Dict[str, Any],
        c_path: Path,
    ) -> CGenResult:
        """Generate C from natural-language spec (primary path)."""
        if not spec_text:
            return CGenResult(
                benchmark=benchmark,
                c_path=str(c_path),
                method="llm_spec",
                success=False,
                error="No spec_text provided",
            )

        funcname = benchmark.replace(" ", "_").replace("-", "_")
        usage = TokenUsage()
        pre_review_path = c_path.with_suffix(".pre_semantic_review.c")

        # LLM generation with retry loop
        for attempt in range(self.max_llm_retries):
            c_code, attempt_usage = llm_gen.generate_from_spec(
                spec_text=spec_text,
                funcname=funcname,
                features=features or {},
                max_retries=0,  # retries handled here
                env_path=self.env_path,
            )
            usage += attempt_usage

            if c_code is None:
                if attempt < self.max_llm_retries - 1:
                    continue
                return CGenResult(
                    benchmark=benchmark,
                    c_path=str(c_path),
                    method="llm_spec",
                    success=False,
                    error="LLM generation failed",
                    token_usage=usage,
                )

            # Write and validate
            c_path.write_text(c_code)
            ok, stderr = validator.validate(c_path)

            if ok:
                # A syntax-valid C program can still violate cycle-level RTL
                # semantics.  Run one serial semantic audit before allowing the
                # C-first route to enter CDFG/MCTS planning.
                pre_review_path.write_text(c_code, encoding="utf-8")
                reviewed_code, review_usage = llm_gen.review_from_spec(
                    spec_text=spec_text,
                    funcname=funcname,
                    features=features or {},
                    c_code=c_code,
                    max_retries=1,
                    env_path=self.env_path,
                )
                usage += review_usage
                if reviewed_code is None:
                    stderr = "Spec-to-C semantic review LLM failed"
                    if attempt < self.max_llm_retries - 1:
                        spec_text += (
                            "\n\nThe previous Spec-to-C candidate could not complete "
                            "the mandatory cycle-semantics review. Regenerate it using "
                            "strict old_state/next_state semantics."
                        )
                        continue
                    return CGenResult(
                        benchmark=benchmark,
                        c_path=str(c_path),
                        method="llm_spec",
                        success=False,
                        error=stderr,
                        token_usage=usage,
                        semantic_review_status="failed",
                        pre_review_c_path=str(pre_review_path),
                    )

                c_path.write_text(reviewed_code, encoding="utf-8")
                ok_review, review_stderr = validator.validate(c_path)
                if not ok_review:
                    stderr = (
                        "Semantic review returned invalid C:\n" + review_stderr
                    )
                    if attempt < self.max_llm_retries - 1:
                        spec_text += (
                            "\n\nThe previous semantic-review output failed clang "
                            f"validation:\n{review_stderr}\nReturn complete corrected C."
                        )
                        continue
                    return CGenResult(
                        benchmark=benchmark,
                        c_path=str(c_path),
                        method="llm_spec",
                        success=False,
                        error=stderr,
                        token_usage=usage,
                        semantic_review_status="failed",
                        pre_review_c_path=str(pre_review_path),
                    )

                # Postprocess
                self._postprocess(c_path)

                # Final validation
                ok_final, _ = validator.validate(c_path)
                if ok_final:
                    return CGenResult(
                        benchmark=benchmark,
                        c_path=str(c_path),
                        method="llm_spec",
                        success=True,
                        token_usage=usage,
                        semantic_review_status="passed",
                        pre_review_c_path=str(pre_review_path),
                    )

            # Retry with error feedback
            if attempt < self.max_llm_retries - 1:
                spec_text += f"\n\nPrevious attempt failed clang validation:\n{stderr}\nFix these errors."

        return CGenResult(
            benchmark=benchmark,
            c_path=str(c_path),
            method="llm_spec",
            success=False,
            error=f"Failed after {self.max_llm_retries} attempts",
            token_usage=usage,
            semantic_review_status="failed",
            pre_review_c_path=str(pre_review_path) if pre_review_path.exists() else "",
        )

    def _generate_from_verilog(
        self,
        benchmark: str,
        verilog_path: str,
        c_path: Path,
        guidance_text: str = "",
    ) -> CGenResult:
        """Generate C from Verilog (secondary path with v2c fallback)."""
        if not verilog_path or not Path(verilog_path).exists():
            return CGenResult(
                benchmark=benchmark,
                c_path=str(c_path),
                method="llm_verilog",
                success=False,
                error=f"Verilog file not found: {verilog_path}",
            )

        verilog_source = Path(verilog_path).read_text()
        if guidance_text:
            verilog_source += "\n// " + guidance_text.replace("\n", "\n// ")
        usage = TokenUsage()

        # LLM generation with retry loop
        for attempt in range(self.max_llm_retries):
            c_code, attempt_usage = llm_gen.generate_from_verilog(
                verilog_source=verilog_source,
                max_retries=0,
                env_path=self.env_path,
            )
            usage += attempt_usage

            if c_code is None:
                break  # Fall through to v2c fallback

            c_path.write_text(c_code)
            ok, stderr = validator.validate(c_path)

            if ok:
                self._postprocess(c_path)

                ok_final, _ = validator.validate(c_path)
                if ok_final:
                    return CGenResult(
                        benchmark=benchmark,
                        c_path=str(c_path),
                        method="llm_verilog",
                        success=True,
                        token_usage=usage,
                    )

            if attempt < self.max_llm_retries - 1:
                verilog_source += f"\n// Previous attempt failed:\n// {stderr}\n"

        # v2c fallback
        ok, error = v2c_fallback.v2c_convert(verilog_path, c_path)
        if ok:
            self._postprocess(c_path)

            ok_final, _ = validator.validate(c_path)
            if ok_final:
                return CGenResult(
                    benchmark=benchmark,
                    c_path=str(c_path),
                    method="v2c_fallback",
                    success=True,
                    token_usage=usage,
                )

        return CGenResult(
            benchmark=benchmark,
            c_path=str(c_path),
            method="llm_verilog",
            success=False,
            error=f"LLM and v2c both failed. Last error: {error}",
            token_usage=usage,
        )

    def _generate_mixed(
        self,
        benchmark: str,
        verilog_path: str,
        spec_text: str,
        c_path: Path,
    ) -> CGenResult:
        """Generate C from both Verilog and spec."""
        if not verilog_path or not Path(verilog_path).exists():
            return CGenResult(
                benchmark=benchmark,
                c_path=str(c_path),
                method="llm_mixed",
                success=False,
                error=f"Verilog file not found: {verilog_path}",
            )

        verilog_source = Path(verilog_path).read_text()
        usage = TokenUsage()

        for attempt in range(self.max_llm_retries):
            c_code, attempt_usage = llm_gen.generate_mixed(
                verilog_source=verilog_source,
                spec_text=spec_text or "",
                max_retries=0,
                env_path=self.env_path,
            )
            usage += attempt_usage

            if c_code is None:
                if attempt < self.max_llm_retries - 1:
                    continue
                return CGenResult(
                    benchmark=benchmark,
                    c_path=str(c_path),
                    method="llm_mixed",
                    success=False,
                    error="LLM generation failed",
                    token_usage=usage,
                )

            c_path.write_text(c_code)
            ok, stderr = validator.validate(c_path)

            if ok:
                self._postprocess(c_path)

                ok_final, _ = validator.validate(c_path)
                if ok_final:
                    return CGenResult(
                        benchmark=benchmark,
                        c_path=str(c_path),
                        method="llm_mixed",
                        success=True,
                        token_usage=usage,
                    )

            if attempt < self.max_llm_retries - 1:
                spec_text += f"\n\nPrevious attempt failed:\n{stderr}\n"

        return CGenResult(
            benchmark=benchmark,
            c_path=str(c_path),
            method="llm_mixed",
            success=False,
            error=f"Failed after {self.max_llm_retries} attempts",
            token_usage=usage,
        )

    def generate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> List[CGenResult]:
        """
        Generate C for multiple benchmarks from Module 1 output.

        Args:
            results: List of FeatureResult dicts from spec_analyze

        Returns:
            List of CGenResult
        """
        outputs = []
        for r in results:
            benchmark = r.get("benchmark", "unknown")
            input_type = r.get("input_type", "spec")
            verilog_path = r.get("verilog_path")
            spec_text = r.get("spec_text")
            features = r.get("llm_features", {})

            result = self.generate(
                benchmark=benchmark,
                input_type=input_type,
                verilog_path=verilog_path,
                spec_text=spec_text,
                features=features,
            )
            outputs.append(result)

        return outputs

    def generate_from_json(
        self,
        json_path: str | Path,
    ) -> List[CGenResult]:
        """
        Generate C from spec_analysis.json output.

        Args:
            json_path: Path to spec_analysis.json

        Returns:
            List of CGenResult
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        return self.generate_batch(data)

    def generate_from_pipeline_json(
        self,
        features_json: str | Path,
        decisions_json: str | Path,
    ) -> List[CGenResult]:
        """
        Generate C only for benchmarks routed to c_first by Module 2.

        Args:
            features_json: Path to Module 1 output (spec_analysis.json)
            decisions_json: Path to Module 2 output (PathDecision list JSON)

        Returns:
            List of CGenResult for c_first benchmarks only
        """
        with open(features_json, encoding="utf-8") as f:
            features_data = json.load(f)
        with open(decisions_json, encoding="utf-8") as f:
            decisions_data = json.load(f)

        if isinstance(features_data, dict):
            features_data = [features_data]
        if isinstance(decisions_data, dict):
            if "decisions" in decisions_data:
                decisions_data = decisions_data["decisions"]
            else:
                decisions_data = [decisions_data]

        feature_map = {
            item.get("benchmark", "unknown"): item
            for item in features_data
            if isinstance(item, dict)
        }

        selected_features: List[Dict[str, Any]] = []
        for decision in decisions_data:
            if not isinstance(decision, dict):
                continue
            if decision.get("path") != "c_first":
                continue

            benchmark = decision.get("benchmark")
            if not benchmark:
                continue

            feature = feature_map.get(benchmark)
            if feature is not None:
                selected_features.append(feature)

        return self.generate_batch(selected_features)
