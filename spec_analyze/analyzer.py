"""
analyzer.py — Orchestrator for Stage 1 feature extraction.

Detects input type (Verilog file, spec text, spec file, or mixed),
routes to the appropriate extraction path, merges results, and
returns a unified FeatureResult.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spec_analyze.schema import FeatureResult
from spec_analyze.confidence_rules import Confidence
from spec_analyze import regex_features
from spec_analyze import llm_features


class Analyzer:
    """
    Main entry point for Stage 1 analysis.

    Usage:
        analyzer = Analyzer()                         # auto-loads .env
        analyzer = Analyzer(env_path="/path/to/.env")
        analyzer = Analyzer(model="gpt-4o")           # override model

        result = analyzer.analyze_file("uart_tx.v")
        result = analyzer.analyze_spec("A 32-bit UART TX with parity")
        result = analyzer.analyze_file("uart_tx.v", spec_text="...")
        results = analyzer.analyze_dir("benchmark/")
    """

    def __init__(
        self,
        env_path: Optional[str] = None,
        model: Optional[str] = None,
        memory_session: Optional[Any] = None,
    ):
        # Resolve .env path
        if env_path:
            self._env_path = Path(env_path)
        else:
            self._env_path = Path(__file__).resolve().parent.parent / ".env"

        # Initialize LLM client
        llm_features.init_client(self._env_path)
        self._model = model or llm_features.get_model()
        self._memory_session = memory_session

    def _finish(self, result: FeatureResult) -> FeatureResult:
        if self._memory_session is not None:
            self._memory_session.record_feature_result(result)
        return result

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", "")

    @staticmethod
    def resolve_optimization_target(
        benchmark: str,
        spec_text: Optional[str] = None,
        explicit_target: Optional[str] = None,
    ) -> str:
        """
        Normalize an explicit optimization target, or assign one deterministically.

        This keeps experiments reproducible while still "randomly" distributing
        AREA vs TIMING when the input did not specify either.
        """
        if explicit_target:
            normalized = str(explicit_target).strip().upper()
            if normalized in {"AREA", "TIMING"}:
                return normalized

        seed_text = f"{benchmark}::{spec_text or ''}"
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return "AREA" if int(digest[:8], 16) % 2 == 0 else "TIMING"

    # ── Single-file analysis ──────────────────────────────────────────

    def analyze_file(
        self,
        verilog_path: str,
        spec_text: Optional[str] = None,
        subcategory: str = "",
        benchmark_name: str = "",
        optimization_target: Optional[str] = None,
    ) -> FeatureResult:
        """
        Analyze a Verilog file, optionally with an accompanying spec.

        If spec_text is provided, uses the mixed prompt for richer analysis.
        Always runs regex extraction on the Verilog source.
        """
        vpath = Path(verilog_path)
        bname = benchmark_name or vpath.stem
        verilog_src = vpath.read_text(errors="replace")
        target = self.resolve_optimization_target(
            bname, spec_text=spec_text, explicit_target=optimization_target
        )

        # Regex features (always available for Verilog)
        rfeat = regex_features.extract(vpath, subcategory, bname)

        # LLM features
        if spec_text:
            input_type = "mixed"
            lfeat, conf, raw, _ev, usage = llm_features.extract_from_mixed(
                verilog_src, spec_text, model=self._model
            )
        else:
            input_type = "verilog"
            lfeat, conf, raw, usage = llm_features.extract_from_verilog(
                verilog_src, model=self._model
            )

        # Verilog grounds the features — overall confidence is high
        if not lfeat:
            overall = Confidence.NONE.value
        else:
            overall = Confidence.HIGH.value

        return self._finish(FeatureResult(
            benchmark=bname,
            input_type=input_type,
            optimization_target=target,
            verilog_path=str(vpath),
            spec_text=spec_text,
            llm_features=lfeat,
            confidence=conf,
            overall_confidence=overall,
            regex_features=rfeat,
            token_usage=usage,
            llm_raw=raw,
        ))

    # ── Spec-only analysis ────────────────────────────────────────────

    def analyze_spec(
        self,
        spec_text: str,
        benchmark_name: str = "",
        optimization_target: Optional[str] = None,
    ) -> FeatureResult:
        """
        Analyze a natural-language specification (no Verilog source).
        Confidence is scored by cross-checking spec text against LLM output.
        """
        lfeat, conf, raw, evidence, usage = llm_features.extract_from_spec(
            spec_text, model=self._model
        )

        bname = benchmark_name or (lfeat.get("module_name", "spec_input") if lfeat else "spec_input")
        target = self.resolve_optimization_target(
            bname, spec_text=spec_text, explicit_target=optimization_target
        )

        if not lfeat:
            overall = Confidence.NONE.value
        else:
            from spec_analyze.confidence_rules import score_confidence
            conf, overall, evidence = score_confidence(spec_text, lfeat)

        return self._finish(FeatureResult(
            benchmark=bname,
            input_type="spec",
            optimization_target=target,
            spec_text=spec_text,
            llm_features=lfeat,
            confidence=conf,
            overall_confidence=overall,
            regex_features=None,
            token_usage=usage,
            llm_raw=raw,
        ))

    # ── Spec file analysis ────────────────────────────────────────────

    def analyze_spec_file(
        self,
        spec_path: str,
        benchmark_name: str = "",
        optimization_target: Optional[str] = None,
    ) -> FeatureResult:
        """Read a .txt spec file and analyze it."""
        text = Path(spec_path).read_text(errors="replace")
        result = self.analyze_spec(
            text,
            benchmark_name=benchmark_name,
            optimization_target=optimization_target,
        )
        return result

    # ── Directory batch analysis ──────────────────────────────────────

    def analyze_dir(
        self,
        bench_dir: str,
        delay: float = 1.0,
    ) -> List[FeatureResult]:
        """
        Analyze all benchmarks in a directory.

        Supports:
          - Flat layout:   dir/*.v  (and optional dir/*.txt specs)
          - Subcategorized: dir/<subcat>/<name>/<name>.v
        """
        results: List[FeatureResult] = []
        benchmarks = _discover_benchmarks(bench_dir)

        if not benchmarks:
            return results

        total = len(benchmarks)
        for i, (vpath, spec_path, subcat, bname) in enumerate(benchmarks):
            print(f"[{i + 1}/{total}] {bname}")

            spec_text = None
            if spec_path and spec_path.exists():
                spec_text = spec_path.read_text(errors="replace")

            if vpath:
                result = self.analyze_file(
                    str(vpath), spec_text=spec_text,
                    subcategory=subcat, benchmark_name=bname,
                    optimization_target=self.resolve_optimization_target(
                        bname, spec_text=spec_text
                    ),
                )
            elif spec_text:
                result = self.analyze_spec(
                    spec_text,
                    benchmark_name=bname,
                    optimization_target=self.resolve_optimization_target(
                        bname, spec_text=spec_text
                    ),
                )
            else:
                continue

            feat = result.llm_features
            if feat:
                print(f"  -> {feat.get('architecture_pattern', '?')} | "
                      f"{feat.get('complexity', '?')} | "
                      f"{feat.get('purpose', '?')[:60]}")
            else:
                print(f"  -> LLM ERROR")

            results.append(result)

            if i < total - 1 and delay > 0:
                import time
                time.sleep(delay)

        return results


# ── Benchmark discovery ───────────────────────────────────────────────

def _discover_benchmarks(
    bench_dir: str,
) -> List[Tuple[Optional[Path], Optional[Path], str, str]]:
    """
    Walk the benchmark directory and return a list of
    (verilog_path, spec_path, subcategory, benchmark_name) tuples.

    Handles:
      - Flat:           dir/<name>.v  +  optional dir/<name>.txt
      - Subcategorized: dir/<subcat>/<name>/<name>.v  +  optional *_spec.txt
      - Spec-only:      dir/<name>.txt  (no .v file)
    """
    results: List[Tuple[Optional[Path], Optional[Path], str, str]] = []
    root = Path(bench_dir)

    # Collect .v files at the top level (flat layout)
    flat_v = sorted(
        f for f in root.iterdir()
        if f.is_file() and f.suffix == ".v" and ".bak" not in f.name
    )
    flat_txt = sorted(
        f for f in root.iterdir()
        if f.is_file() and f.suffix == ".txt"
    )
    flat_txt_map = {f.stem: f for f in flat_txt}

    if flat_v:
        # Flat layout
        seen_stems = set()
        for vf in flat_v:
            bname = vf.stem
            seen_stems.add(bname)
            spec_file = flat_txt_map.get(bname)
            results.append((vf, spec_file, "", bname))

        # Spec-only .txt files (no matching .v)
        for txt_f in flat_txt:
            if txt_f.stem not in seen_stems:
                results.append((None, txt_f, "", txt_f.stem))
    else:
        # Subcategorized layout
        for subcat_dir in sorted(root.iterdir()):
            if not subcat_dir.is_dir():
                continue
            subcat = subcat_dir.name

            for entry in sorted(subcat_dir.iterdir()):
                if entry.is_file() and entry.suffix == ".v" and ".bak" not in entry.name:
                    spec_file = entry.with_suffix(".txt")
                    if not spec_file.exists():
                        spec_file = None
                    results.append((entry, spec_file, subcat, entry.stem))
                elif entry.is_dir():
                    # Look for .v and _spec.txt inside
                    v_files = list(entry.glob("*_original.v"))
                    if not v_files:
                        v_files = [
                            f for f in entry.glob("*.v") if ".bak" not in f.name
                        ]
                    spec_files = list(entry.glob("*_spec.txt")) + list(entry.glob("*.txt"))

                    vf = v_files[0] if v_files else None
                    sf = spec_files[0] if spec_files else None
                    if vf or sf:
                        results.append((vf, sf, subcat, entry.name))

    return results
