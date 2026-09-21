#!/usr/bin/env python3
"""Run the upstream MAGE agent three times on the NVDLA-22 C-first set.

This adapter intentionally calls MAGE's public ``TopAgent`` entry point rather
than reimplementing its generation prompts.  The NVDLA pilot cases do not have
the Verilog-Eval testbench layout expected by MAGE's example script, so the
runner feeds each case's verified ``spec.txt`` directly to ``TopAgent.run``.
The golden closure is retained as provenance only; it is not passed to MAGE's
generation or simulation prompts.  This keeps the run specification-only and
avoids compiling two modules with the same NVDLA top-level name.

The run is deliberately resumable.  A case result is written after each
case/round, and ``summary.csv``, ``summary.md`` and ``summary.json`` are
rewritten from all available result files.  The default settings preserve the
full MAGE pipeline while bounding retries so that 22 cases x 3 rounds remains
an auditable experiment rather than an unbounded LLM/EDA job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
# When invoked by pathname, Python puts only this script's directory on
# sys.path.  Add the repository root so the local module5 DC wrapper remains
# importable for the post-generation PPA stage.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUT_ROOT = ROOT / "runs/external_spec_agents/20260915_cfirst_nvdla22_no_jg"
PILOT_ROOT = ROOT / "runs/phase6/nvdla22_njg_20260908/pilot"
PILOT_MANIFEST = PILOT_ROOT / "pilot_manifest.json"
DEFAULT_RUN_ROOT = ROOT / "runs/external_spec_agents/20260919_mage_nvdla22_3x"
PYTHON = ROOT / ".conda-env/bin/python"
IVERILOG = ROOT / ".conda-env/bin/iverilog"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")


def extract_module_name(verilog: Path) -> str | None:
    text = verilog.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^\s*module\s+([A-Za-z_]\w*)\b", text)
    return match.group(1) if match else None


def load_cases() -> list[dict[str, Any]]:
    """Load the 22 unique cases represented by the user's input directory."""

    summary = read_json(INPUT_ROOT / "summary.json")
    results = summary.get("results") or []
    by_id: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or not item.get("case_id"):
            continue
        by_id.setdefault(str(item["case_id"]), item)

    manifest = read_json(PILOT_MANIFEST)
    manifest_by_id = {str(item["id"]): item for item in manifest.get("cases", [])}
    expected = int(summary.get("case_count") or manifest.get("case_count") or 22)
    if len(by_id) != expected:
        raise RuntimeError(
            f"Input dataset has {len(by_id)} unique cases, expected {expected}: "
            f"{INPUT_ROOT / 'summary.json'}"
        )

    cases: list[dict[str, Any]] = []
    for case_id in sorted(by_id):
        old = by_id[case_id]
        manifest_item = manifest_by_id.get(case_id, {})
        spec = Path(str(old.get("spec_path") or "")).resolve()
        golden = Path(str(old.get("golden_rtl_path") or "")).resolve()
        if not spec.is_file() or not golden.is_file():
            # Fall back to the pilot manifest when an older summary stored a
            # relative or stale path.
            spec = (PILOT_ROOT / str(manifest_item.get("spec_path", ""))).resolve()
            golden = (PILOT_ROOT / str(manifest_item.get("golden_rtl_path", ""))).resolve()
        if not spec.is_file() or not golden.is_file():
            raise FileNotFoundError(f"Missing NVDLA input for {case_id}: {spec} / {golden}")
        top = str(manifest_item.get("golden_top_module") or "") or extract_module_name(golden)
        if not top:
            raise RuntimeError(f"Could not determine top module for {case_id}: {golden}")
        cases.append(
            {
                "case_id": case_id,
                "spec_path": str(spec),
                "golden_rtl_path": str(golden),
                "top_module": top,
                "design_type": manifest_item.get("design_type"),
                "family_id": manifest_item.get("family_id"),
                "source_group": manifest_item.get("source_group"),
                "spec_sha256": sha256(spec),
                "golden_sha256": sha256(golden),
            }
        )
    return cases


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        # The project already exports the variables in normal operation.  Do
        # not make python-dotenv an additional hard dependency for this runner.
        pass


def build_llm(model: str, max_tokens: int | None, temperature: float, top_p: float):
    """Construct the OpenAI-compatible LlamaIndex LLM used by MAGE."""

    load_env_file()
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("CLOUD_API_BASE_URL")
    )
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CLOUD_API_KEY")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL/OPENAI_API_BASE_URL is not configured")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    # LlamaIndex's OpenAI wrapper reads OPENAI_API_BASE in some releases and
    # api_base in others.  Set both, while passing api_base explicitly.
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE_URL"] = base_url

    # LlamaIndex validates model names against its OpenAI catalogue.  The
    # selected C-first model is served through an OpenAI-compatible endpoint,
    # but is not an OpenAI catalogue name; make the wrapper treat it as a chat
    # model with a generous context window while preserving the actual model
    # string in the request payload.
    import llama_index.llms.openai.base as openai_base

    original_context_size = openai_base.openai_modelname_to_contextsize

    def context_size_for_compat_model(name: str) -> int:
        try:
            return original_context_size(name)
        except ValueError:
            return 128_000

    openai_base.openai_modelname_to_contextsize = context_size_for_compat_model
    openai_base.is_chat_model = lambda model=None, **_: True

    from llama_index.llms.openai import OpenAI
    try:
        from llm_request import deepseek_thinking_kwargs

        request_controls = deepseek_thinking_kwargs(model)
    except Exception:
        request_controls = {}
    # Do not force the gateway's ``response_format=json_object`` mode here.
    # On long NVDLA specifications that mode has been observed to spend the
    # entire completion budget on an empty/unfinished object.  MAGE already
    # supplies and parses its own JSON-format prompt; the compatible endpoint
    # returns a valid JSON object reliably when left in normal chat mode.

    # ``temperature`` and ``top_p`` are also supplied by MAGE's TokenCounter
    # on every request; keeping them here documents the experiment settings.
    return OpenAI(
        model=model,
        api_key=api_key,
        api_base=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_retries=1,
        timeout=300.0,
        additional_kwargs=request_controls,
    )


def patch_unknown_tokenizer() -> str:
    """Use MAGE's native counter with a deterministic fallback for DeepSeek."""

    import tiktoken

    original = tiktoken.encoding_for_model

    def safe_encoding_for_model(model: str):
        try:
            return original(model)
        except KeyError:
            # DeepSeek's model name is not registered by every tiktoken
            # release.  cl100k_base is a conservative local accounting
            # fallback; provider usage is not fabricated or mixed into it.
            return tiktoken.get_encoding("cl100k_base")

    tiktoken.encoding_for_model = safe_encoding_for_model
    return "tiktoken:cl100k_base fallback for unregistered model names"


def install_token_counter_adapter() -> None:
    """Make MAGE token accounting robust for OpenAI-compatible reasoning APIs.

    DeepSeek-compatible gateways may return ``content=None`` while a response
    is still exhausting its reasoning budget.  MAGE's upstream counter assumes
    a string and otherwise raises before it can save the case.  The adapter
    normalizes such content and prefers the provider's prompt/completion usage
    fields (which include reasoning tokens) when they are present; otherwise it
    falls back to MAGE's local tiktoken count.
    """

    from mage import token_counter as token_counter_module

    if getattr(token_counter_module.TokenCounter, "_nvdla22_adapter_installed", False):
        return

    TokenCounter = token_counter_module.TokenCounter
    TokenCount = token_counter_module.TokenCount
    from llama_index.core.base.llms.types import ChatMessage
    original_count_chat = TokenCounter.count_chat

    def content_to_text(content: Any, raw: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for block in content:
                if isinstance(block, str):
                    pieces.append(block)
                elif isinstance(block, dict):
                    value = block.get("text") or block.get("content")
                    if value is not None:
                        pieces.append(str(value))
                else:
                    value = getattr(block, "text", None) or getattr(block, "content", None)
                    if value is not None:
                        pieces.append(str(value))
            if pieces:
                return "".join(pieces)
        # OpenAI-compatible reasoning responses expose the final text and
        # hidden reasoning as separate message fields.  Use whichever is
        # available so JSON parsing can fail gracefully instead of aborting
        # token accounting with a TypeError.
        try:
            message = raw.choices[0].message
            value = getattr(message, "content", None)
            if isinstance(value, str):
                return value
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str):
                return reasoning
        except Exception:
            pass
        return "" if content is None else str(content)

    def count_chat_with_provider_usage(self, messages, llm=None):
        llm = llm or self.llm
        in_token_cnt_local = self.count(llm.messages_to_prompt(messages))
        response = llm.chat(
            messages,
            top_p=token_counter_module.settings.top_p,
            temperature=token_counter_module.settings.temperature,
        )
        try:
            original_content = response.message.content
        except Exception:
            # Newer LlamaIndex versions expose reasoning and answer as
            # multiple content blocks; the convenience ``content`` property
            # intentionally raises in that case.
            original_content = getattr(response.message, "blocks", None)
        normalized_content = content_to_text(original_content, response.raw)
        response.message = ChatMessage(
            role=response.message.role,
            content=normalized_content,
            additional_kwargs=dict(response.message.additional_kwargs),
        )
        usage = getattr(response.raw, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            in_token_cnt = prompt_tokens
            out_token_cnt = completion_tokens
            self._token_usage_source = "provider usage (prompt_tokens + completion_tokens)"
            self._provider_request_count = getattr(self, "_provider_request_count", 0) + 1
        else:
            in_token_cnt = in_token_cnt_local
            out_token_cnt = self.count(normalized_content)
            self._token_usage_source = "MAGE TokenCounter local tiktoken accounting"
        token_cnt = TokenCount(in_token_cnt=in_token_cnt, out_token_cnt=out_token_cnt)
        self.token_cnts[self.cur_tag].append(token_cnt)
        if self.enable_reformat_json:
            response.message.content = token_counter_module.reformat_json_string(response.message.content)
        return response, token_cnt

    TokenCounter.count_chat = count_chat_with_provider_usage
    TokenCounter._nvdla22_adapter_installed = True


def bounded_agent(llm):
    """Return a TopAgent using MAGE's flow with bounded retry/candidate caps."""

    from mage.agent import TopAgent

    class BoundedTopAgent(TopAgent):
        def _apply_bounds(self) -> None:
            # These are caps only; all stages remain MAGE's own TB generator,
            # RTL generator, simulation judge/reviewer and RTL editor.
            self.sim_max_retry = 1
            self.rtl_max_candidates = 1
            self.rtl_selected_candidates = 1
            if self.tb_gen is not None:
                # The queue-display prompt can make the JSON response exceed
                # a gateway's completion cap on long NVDLA interfaces.  The
                # upstream TB generator supports the shorter moment-display
                # variant as its normal fallback path.
                self.tb_gen.gen_display_queue = False
                self.tb_gen.json_decode_max_trial = 2
            if self.rtl_gen is not None:
                # Bound malformed-response retries so a long NVDLA spec
                # cannot consume an unbounded number of completion tokens.
                self.rtl_gen.max_trials = 2
            if self.rtl_edit is not None:
                self.rtl_edit.max_trials = 1

        def run_instance(self, spec: str):
            self._apply_bounds()
            return super().run_instance(spec)

        def run_instance_ablation(self, spec: str):
            self._apply_bounds()
            return super().run_instance_ablation(spec)

    return BoundedTopAgent(llm)


def token_counts(agent) -> dict[str, Any]:
    try:
        count = agent.token_counter.get_sum_count()
        in_count = int(count.in_token_cnt)
        out_count = int(count.out_token_cnt)
        requests: list[dict[str, int]] = []
        for tag_counts in getattr(agent.token_counter, "token_cnts", {}).values():
            for item in tag_counts:
                request_in = getattr(item, "in_token_cnt", None)
                request_out = getattr(item, "out_token_cnt", None)
                if isinstance(request_in, int) and isinstance(request_out, int):
                    requests.append(
                        {
                            "prompt_tokens": request_in,
                            "completion_tokens": request_out,
                            "total_tokens": request_in + request_out,
                        }
                    )
        return {
            "prompt_tokens": in_count,
            "completion_tokens": out_count,
            "total_tokens": in_count + out_count,
            "source": getattr(agent.token_counter, "_token_usage_source", "MAGE TokenCounter"),
            "request_count": getattr(agent.token_counter, "_provider_request_count", None),
            "requests": requests,
        }
    except Exception as exc:  # noqa: BLE001
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "source": f"unavailable: {type(exc).__name__}", "request_count": None, "requests": []}


def syntax_check(rtl: Path, top: str, out: Path) -> dict[str, Any]:
    log_path = out / "syntax.log"
    try:
        completed = subprocess.run(
            [str(IVERILOG), "-g2012", "-s", top, "-t", "null", str(rtl)],
            cwd=str(out), capture_output=True, text=True, timeout=120, check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "log_path": str(log_path),
        }
    except Exception as exc:  # noqa: BLE001
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"status": "error", "returncode": None, "log_path": str(log_path), "error": str(exc)}


def dc_run(rtl: Path, case: dict[str, Any], out: Path, goal: str, round_index: int) -> dict[str, Any]:
    try:
        from module5.dc_runner import run_dc_for_verilog

        dc = run_dc_for_verilog(
            rtl,
            benchmark=str(case["case_id"]),
            goal=goal,
            stem=f"mage_round_{round_index:02d}__{safe_name(str(case['case_id']))}__{goal}",
            output_root=out / "dc" / goal,
            top_module=str(case["top_module"]),
            max_cores=1,
        )
        metrics = dc.get("metrics") if isinstance(dc, dict) else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        value = metrics.get("total_cell_area") if goal == "area" else metrics.get("data_arrival_time_ps", metrics.get("delay_ps"))
        ok = bool(dc.get("success")) and value is not None
        return {
            "status": "passed" if ok else "failed",
            "metric": value,
            "metrics": metrics,
            "raw_status": dc.get("status"),
            "success": bool(dc.get("success")),
            "report_dir": dc.get("report_dir"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "metric": None, "error": f"{type(exc).__name__}: {exc}"}


def run_one(
    agent_llm,
    case: dict[str, Any],
    round_index: int,
    run_root: Path,
    model: str,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    force: bool,
    ablation: bool,
    redirect_log: bool = False,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = run_root / f"round_{round_index:02d}" / safe_name(case_id)
    result_path = case_dir / "result.json"
    if result_path.is_file() and not force:
        return read_json(result_path)

    case_dir.mkdir(parents=True, exist_ok=True)
    spec = Path(str(case["spec_path"]))
    spec_text = spec.read_text(encoding="utf-8", errors="replace")
    output_dir = case_dir / "mage_output"
    log_dir = case_dir / "mage_logs"
    result: dict[str, Any] = {
        "schema_version": "mage_nvdla22_case_result_v1",
        "agent": "stable-lab/MAGE",
        "mode": "ablation_generation_only" if ablation else "full_pipeline",
        "round": round_index,
        "case_id": case_id,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "top_module": case["top_module"],
        "spec_path": case["spec_path"],
        "golden_rtl_path": case["golden_rtl_path"],
        "golden_used_for_generation": False,
        "golden_used_for_mage_simulation": False,
        "verification_mode": "none",
        "spec_sha256": case["spec_sha256"],
        "golden_sha256": case["golden_sha256"],
        "caps": {"sim_max_retry": 1, "rtl_max_candidates": 1, "rtl_syntax_trials": 2, "editor_trials": 1, "tb_json_trials": 2, "tb_display_queue": False},
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    started = time.time()
    agent = bounded_agent(agent_llm)
    agent.set_output_path(str(output_dir))
    agent.set_log_path(str(log_dir))
    agent.set_ablation(ablation)
    agent.set_redirect_log(redirect_log)
    # No golden TB or golden RTL is passed.  In ablation mode MAGE invokes its
    # documented spec-only RTLGenerator path; in full mode it would generate
    # and review a TB from the natural-language specification.
    try:
        if ablation:
            # This is the implementation called by MAGE's documented
            # TopAgent.run_instance_ablation().  Calling the generator
            # directly avoids an OpenAI-compatible gateway quirk observed at
            # the TopAgent stdout/log redirection boundary: the same MAGE
            # prompt could otherwise return an empty JSON object at the
            # completion limit.  The prompt, parser, syntax checker, retry
            # loop and TokenCounter remain MAGE's own code.
            from mage.log_utils import set_log_dir, switch_log_to_file, switch_log_to_stdout
            from mage.rtl_generator import RTLGenerator

            agent.token_counter.reset()
            agent.rtl_gen = RTLGenerator(agent.token_counter)
            agent._apply_bounds()
            mage_log_dir = log_dir / f"VERILOG_EVAL_V2_{case_id}"
            mage_output_dir = output_dir / f"VERILOG_EVAL_V2_{case_id}"
            mage_log_dir.mkdir(parents=True, exist_ok=True)
            mage_output_dir.mkdir(parents=True, exist_ok=True)
            set_log_dir(str(mage_log_dir))
            switch_log_to_file()
            try:
                mage_pass, mage_text = agent.rtl_gen.ablation_chat(
                    input_spec=spec_text,
                    rtl_path=str(mage_output_dir / "rtl.sv"),
                )
            finally:
                switch_log_to_stdout()
        else:
            mage_pass, mage_text = agent.run(
                benchmark_type_name="VERILOG_EVAL_V2",
                task_id=case_id,
                spec=spec_text,
                golden_tb_path=None,
                golden_rtl_blackbox_path=None,
            )
        result["mage_return"] = {"is_pass": bool(mage_pass), "message": str(mage_text)[:5000]}
    except Exception as exc:  # noqa: BLE001
        result["mage_return"] = {"is_pass": False, "message": f"{type(exc).__name__}: {exc}"}
    finally:
        # TopAgent.run normally restores stdout/stderr itself.  The explicit
        # restoration also protects the outer resumable loop if an upstream
        # exception occurs before that restoration.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    result["token_usage"] = token_counts(agent)
    candidate = output_dir / f"VERILOG_EVAL_V2_{case_id}" / "rtl.sv"
    result["candidate_path"] = str(candidate) if candidate.is_file() else None
    candidate_top = extract_module_name(candidate) if candidate.is_file() else None
    result["candidate_top_module"] = candidate_top
    result["syntax"] = {}
    result["synthesis"] = {}
    if not candidate.is_file() or candidate.stat().st_size == 0:
        result["terminal_status"] = "generation_failed"
    else:
        # The natural-language NVDLA specs describe the interface but do not
        # require the golden module name.  MAGE commonly emits ``TopModule``;
        # use the generated declaration for syntax/DC while retaining the
        # golden name separately as provenance.
        synthesis_case = dict(case)
        synthesis_case["top_module"] = candidate_top or case["top_module"]
        result["syntax"] = syntax_check(
            candidate, str(synthesis_case["top_module"]), case_dir
        )
        if result["syntax"].get("status") != "passed":
            result["terminal_status"] = "candidate_syntax_failed"
        else:
            for goal in ("area", "timing"):
                result["synthesis"][goal] = dc_run(
                    candidate, synthesis_case, case_dir, goal, round_index
                )
            ppa_ok = any(block.get("status") == "passed" for block in result["synthesis"].values())
            result["terminal_status"] = "ppa_complete" if ppa_ok else "synthesis_failed"

    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(result_path, result)
    return result


def iter_results(run_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("round_*/**/result.json")):
        try:
            results.append(read_json(path))
        except Exception:
            continue
    return sorted(results, key=lambda x: (int(x.get("round", 0)), str(x.get("case_id", ""))))


def metric(result: dict[str, Any], goal: str) -> Any:
    block = (result.get("synthesis") or {}).get(goal) or {}
    return block.get("metric") if block.get("status") == "passed" else None


def token_value(result: dict[str, Any], key: str) -> Any:
    return (result.get("token_usage") or {}).get(key)


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def write_reports(run_root: Path, cases: list[dict[str, Any]], model: str, rounds: int, tokenizer_note: str) -> None:
    results = iter_results(run_root)
    write_json(
        run_root / "summary.json",
        {
            "schema_version": "mage_nvdla22_summary_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dataset": str(INPUT_ROOT),
            "case_count": len(cases),
            "rounds_requested": rounds,
            "rounds_present": sorted({int(r.get("round", 0)) for r in results}),
            "agent": "stable-lab/MAGE",
            "model": model,
            "verification_mode": "none",
            "token_accounting": tokenizer_note,
            "results": results,
        },
    )

    csv_path = run_root / "summary.csv"
    csv_fields = [
        "round", "case_id", "top_module", "candidate_top_module", "mode", "terminal_status", "mage_is_pass", "syntax_status",
        "area", "timing_ps", "prompt_tokens", "completion_tokens", "total_tokens",
        "request_count", "token_source", "elapsed_seconds", "candidate_path", "result_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "round": result.get("round"),
                "case_id": result.get("case_id"),
                "top_module": result.get("top_module"),
                "candidate_top_module": result.get("candidate_top_module"),
                "mode": result.get("mode"),
                "terminal_status": result.get("terminal_status"),
                "mage_is_pass": (result.get("mage_return") or {}).get("is_pass"),
                "syntax_status": (result.get("syntax") or {}).get("status"),
                "area": metric(result, "area"),
                "timing_ps": metric(result, "timing"),
                "prompt_tokens": token_value(result, "prompt_tokens"),
                "completion_tokens": token_value(result, "completion_tokens"),
                "total_tokens": token_value(result, "total_tokens"),
                "request_count": token_value(result, "request_count"),
                "token_source": token_value(result, "source"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "candidate_path": result.get("candidate_path"),
                "result_path": str(run_root / f"round_{int(result.get('round', 0)):02d}" / safe_name(str(result.get("case_id", ""))) / "result.json"),
            })

    lines = [
        "# MAGE × NVDLA22 三轮实验",
        "",
        f"- 输入数据集：`{INPUT_ROOT}`（22 个 case）",
        f"- Agent：`stable-lab/MAGE`；模型：`{model}`；请求轮数：{rounds}",
        "- 运行目录：本目录的 `round_01/`、`round_02/`、`round_03/`；每个 case 保留 MAGE 日志、生成 RTL、Icarus 日志和 DC 结果。",
        f"- 生成方式：调用 MAGE 官方 `TopAgent.run_instance_ablation` 所使用的 `RTLGenerator.ablation_chat`，模式为 `{('ablation_generation_only' if any(r.get('mode') == 'ablation_generation_only' for r in results) else 'full_pipeline')}`；未把 golden RTL/TB 传入生成或 MAGE 仿真，避免数据泄漏和同名模块冲突。",
        "- 验证口径：本轮不运行 JasperGold；Area/Timing 是通过语法门后执行的 provisional DC 指标，不能当成功能等价证明。",
        "- 约束：MAGE 流程保留，但为 66 次 case-run 设置 `sim_max_retry=1`、`rtl_max_candidates=1`、RTL 语法重试 2 次、编辑器重试 1 次、TB JSON 重试 2 次；TB 使用 MAGE 的较短 moment-display 变体以避免长规格输出被截断。",
        f"- Token：{tokenizer_note}；详细 prompt/completion/total 在 `summary.csv` 和每个 `result.json`。",
        "",
        "## 每轮汇总",
        "",
        "| Round | Cases | MAGE pass | Syntax pass | Area DC | Timing DC | PPA any | Tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for round_index in range(1, rounds + 1):
        rs = [r for r in results if int(r.get("round", 0)) == round_index]
        mage_ok = sum(bool((r.get("mage_return") or {}).get("is_pass")) for r in rs)
        syntax_ok = sum((r.get("syntax") or {}).get("status") == "passed" for r in rs)
        area_ok = sum(metric(r, "area") is not None for r in rs)
        timing_ok = sum(metric(r, "timing") is not None for r in rs)
        ppa_ok = sum(str(r.get("terminal_status")) == "ppa_complete" for r in rs)
        tokens = [token_value(r, "total_tokens") for r in rs if isinstance(token_value(r, "total_tokens"), int)]
        lines.append(f"| {round_index} | {len(rs)} | {mage_ok} | {syntax_ok} | {area_ok} | {timing_ok} | {ppa_ok} | {sum(tokens) if tokens else '—'} |")

    lines.extend([
        "",
        "## 逐 case 结果",
        "",
        "数值为空表示该阶段未成功，不做估算。Timing 单位为 ps；Area 是 DC `total_cell_area`。",
        "",
        "| Round | Case | MAGE | Syntax | Area | Timing (ps) | Prompt tok. | Completion tok. | Total tok. | API req. | Status |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result in results:
        lines.append(
            f"| {result.get('round')} | {result.get('case_id')} | "
            f"{'pass' if (result.get('mage_return') or {}).get('is_pass') else 'fail'} | "
            f"{(result.get('syntax') or {}).get('status', '—')} | {fmt(metric(result, 'area'))} | "
            f"{fmt(metric(result, 'timing'))} | {fmt(token_value(result, 'prompt_tokens'))} | "
            f"{fmt(token_value(result, 'completion_tokens'))} | {fmt(token_value(result, 'total_tokens'))} | "
            f"{fmt(token_value(result, 'request_count'))} | "
            f"{result.get('terminal_status', '—')} |"
        )

    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test prefix; default runs all 22 cases")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Use MAGE's documented spec-only RTL generation path")
    args = parser.parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    # MAGE's own syntax checker invokes ``iverilog`` by name.  Use the
    # project-pinned binary, just as the adapter-level gate below does.
    os.environ["PATH"] = str(ROOT / ".conda-env" / "bin") + os.pathsep + os.environ.get("PATH", "")
    cases = load_cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["case_id"] in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected")

    # Persist the exact input manifest before external calls begin.
    write_json(run_root / "input_manifest.json", {"source": str(INPUT_ROOT), "cases": cases})
    tokenizer_note = patch_unknown_tokenizer()
    install_token_counter_adapter()
    from mage.gen_config import set_exp_setting

    set_exp_setting(temperature=args.temperature, top_p=args.top_p)
    llm = build_llm(args.model, args.max_tokens, args.temperature, args.top_p)
    write_json(
        run_root / "run_config.json",
        {
            "agent": "stable-lab/MAGE",
            "mode": "ablation_generation_only" if args.ablation else "full_pipeline",
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "rounds": args.rounds,
            "case_count": len(cases),
            "input_dataset": str(INPUT_ROOT),
            "golden_passed_to_mage": False,
            "token_accounting": tokenizer_note,
        },
    )

    print(f"MAGE run root: {run_root}", flush=True)
    print(f"Cases: {len(cases)}; rounds: {args.rounds}; model: {args.model}", flush=True)
    for round_index in range(1, args.rounds + 1):
        for case in cases:
            print(f"[round {round_index}/{args.rounds}] {case['case_id']}", flush=True)
            try:
                # Build a fresh TopAgent per case.  This is important because
                # MAGE resets a TokenCounter at the start of each run and its
                # conversation histories are intentionally case-local.
                agent_llm = llm
                result = run_one(
                    agent_llm, case, round_index, run_root, args.model,
                    args.max_tokens, args.temperature, args.top_p, args.force,
                    args.ablation,
                )
                print(
                    f"  -> {result.get('terminal_status')} "
                    f"tokens={token_value(result, 'total_tokens')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                case_dir = run_root / f"round_{round_index:02d}" / safe_name(str(case["case_id"]))
                failure = {
                    "schema_version": "mage_nvdla22_case_result_v1",
                    "agent": "stable-lab/MAGE",
                    "round": round_index,
                    "case_id": case["case_id"],
                    "model": args.model,
                    "top_module": case["top_module"],
                    "spec_path": case["spec_path"],
                    "golden_rtl_path": case["golden_rtl_path"],
                    "terminal_status": "runner_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                write_json(case_dir / "result.json", failure)
                print(f"  -> runner_exception: {type(exc).__name__}: {exc}", flush=True)
            finally:
                write_reports(run_root, cases, args.model, args.rounds, tokenizer_note)

    write_reports(run_root, cases, args.model, args.rounds, tokenizer_note)
    print(f"Reports: {run_root / 'summary.md'} and {run_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
