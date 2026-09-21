"""
generator.py — Orchestrator for direct spec-to-RTL comparison runs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import dotenv_values

from RTL_DIRECT_compare import llm_gen, validator
from RTL_DIRECT_compare.schema import RTLDirectResult
from module5.rtl_direct_runner import _build_syntax_retry_message
from token_counter import TokenUsage


def _sanitize_identifier(name: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]", "_", name or "")
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "generated_module"
    if text[0].isdigit():
        text = f"m_{text}"
    return text


class RTLDirectGenerator:
    """Generate RTL directly from a specification and validate its syntax."""

    def __init__(
        self,
        output_dir: str | Path,
        env_path: str | Path = ".env",
        max_llm_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_path = Path(env_path)
        self.max_llm_retries = max_llm_retries

    def _resolved_model(self, model_id: str | None) -> str:
        if model_id:
            return model_id
        configured = dotenv_values(self.env_path)
        return str(
            os.environ.get("OPENAI_MODEL")
            or configured.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )

    def generate(
        self,
        benchmark: str,
        spec_text: str,
        features: Dict[str, Any] | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        output_dir: Path | None = None,
        interface_contract: str = "",
    ) -> RTLDirectResult:
        features = features or {}
        resolved_model = self._resolved_model(model_id)
        module_name = _sanitize_identifier(
            features.get("module_name") or benchmark
        )
        out_dir = output_dir or self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        rtl_path = out_dir / f"{benchmark}.v"
        attempts_dir = out_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        usage = TokenUsage()
        attempt_artifacts: List[Dict[str, Any]] = []
        failure_history: List[str] = []
        last_error = ""

        if not spec_text:
            return RTLDirectResult(
                benchmark=benchmark,
                module_name=module_name,
                rtl_path=str(rtl_path),
                method="llm_spec_to_rtl",
                success=False,
                model=resolved_model,
                error="No spec_text provided",
                token_usage=usage,
            )

        current_spec = spec_text
        for attempt in range(self.max_llm_retries):
            rtl_code, attempt_usage = llm_gen.generate_from_spec(
                benchmark=benchmark,
                spec_text=current_spec,
                module_name=module_name,
                features=features,
                max_retries=0,
                env_path=self.env_path,
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                # Repeated on every attempt: a syntax retry must not silently
                # drop the interface the equivalence check will enforce.
                interface_contract=interface_contract,
            )
            usage += attempt_usage

            if rtl_code is None:
                last_error = "LLM generation failed"
                attempt_artifacts.append(
                    {
                        "attempt": attempt,
                        "rtl_path": "",
                        "stderr_path": "",
                        "syntax_ok": False,
                        "error": last_error,
                    }
                )
                if attempt < self.max_llm_retries - 1:
                    continue
                break

            rtl_path.write_text(rtl_code, encoding="utf-8")
            attempt_rtl_path = attempts_dir / f"attempt_{attempt}.v"
            attempt_rtl_path.write_text(rtl_code, encoding="utf-8")
            ok, stderr = validator.validate(attempt_rtl_path)
            stderr_path = attempts_dir / f"attempt_{attempt}.stderr.txt"
            stderr_path.write_text(stderr or "", encoding="utf-8")
            last_error = stderr or ""
            attempt_artifacts.append(
                {
                    "attempt": attempt,
                    "rtl_path": str(attempt_rtl_path),
                    "stderr_path": str(stderr_path),
                    "syntax_ok": bool(ok),
                    "error": stderr or "",
                }
            )
            if ok:
                return RTLDirectResult(
                    benchmark=benchmark,
                    module_name=module_name,
                    rtl_path=str(rtl_path),
                    method="llm_spec_to_rtl",
                    success=True,
                    model=resolved_model,
                    syntax_ok=True,
                    attempt_count=attempt + 1,
                    attempt_artifacts=attempt_artifacts,
                    token_usage=usage,
                )

            if attempt < self.max_llm_retries - 1:
                retry_message = _build_syntax_retry_message(
                    stderr,
                    rtl_code,
                    attempt=attempt,
                    module_name=module_name,
                    failure_history=tuple(failure_history),
                )
                current_spec = (
                    f"{spec_text}\n\n"
                    f"{retry_message}"
                )
                headline = (
                    (stderr or "").strip().splitlines() or ["(no validator output)"]
                )[0]
                failure_history.append(f"attempt {attempt}: {headline}")

        return RTLDirectResult(
            benchmark=benchmark,
            module_name=module_name,
            rtl_path=str(rtl_path),
            method="llm_spec_to_rtl",
            success=False,
            model=resolved_model,
            syntax_ok=False,
            error=(
                f"Failed after {self.max_llm_retries} attempts. "
                f"Last error: {last_error or 'unknown generation failure'}"
            ),
            attempt_count=len(attempt_artifacts),
            attempt_artifacts=attempt_artifacts,
            token_usage=usage,
        )

    def generate_batch(self, results: List[Dict[str, Any]]) -> List[RTLDirectResult]:
        outputs: List[RTLDirectResult] = []
        for item in results:
            outputs.append(
                self.generate(
                    benchmark=item.get("benchmark", "unknown"),
                    spec_text=item.get("spec_text", ""),
                    features=item.get("llm_features", {}) or {},
                )
            )
        return outputs

    def load_selected_features(
        self,
        features_json: str | Path,
        decisions_json: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Join Module 1 and Module 2 outputs, keeping c_first only.
        """
        with open(features_json, encoding="utf-8") as f:
            features_data = json.load(f)
        with open(decisions_json, encoding="utf-8") as f:
            decisions_data = json.load(f)

        if isinstance(features_data, dict):
            features_data = [features_data]
        if isinstance(decisions_data, dict):
            decisions_data = decisions_data.get("decisions", [decisions_data])

        feature_map = {
            item.get("benchmark", "unknown"): item
            for item in features_data
            if isinstance(item, dict)
        }

        selected: List[Dict[str, Any]] = []
        for decision in decisions_data:
            if not isinstance(decision, dict):
                continue
            if decision.get("path") != "c_first":
                continue
            benchmark = decision.get("benchmark")
            if benchmark in feature_map:
                selected.append(feature_map[benchmark])

        return selected

    def generate_from_pipeline_json(
        self,
        features_json: str | Path,
        decisions_json: str | Path,
    ) -> List[RTLDirectResult]:
        selected = self.load_selected_features(features_json, decisions_json)
        return self.generate_batch(selected)

    def generate_multi_model(
        self,
        selected: List[Dict[str, Any]],
        model_configs: List[Dict[str, Any]],
    ) -> List[RTLDirectResult]:
        """Run generation across multiple models sequentially."""
        all_results: List[RTLDirectResult] = []
        for cfg in model_configs:
            mid = cfg["model_id"]
            model_dir = self.output_dir / mid
            print(f"\n{'='*60}\nModel: {mid}\n{'='*60}")
            for item in selected:
                benchmark = item.get("benchmark", "unknown")
                result = self.generate(
                    benchmark=benchmark,
                    spec_text=item.get("spec_text", ""),
                    features=item.get("llm_features", {}) or {},
                    model_id=mid,
                    api_key=cfg.get("api_key"),
                    base_url=cfg.get("base_url"),
                    output_dir=model_dir,
                )
                all_results.append(result)
                status = "OK" if result.success else "FAIL"
                print(f"  {status} {benchmark}")
        return all_results

    @staticmethod
    def load_model_config(config_path: str | Path) -> List[Dict[str, Any]]:
        """Load model configuration from JSON file."""
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
