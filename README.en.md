# VIVADO Dual-Route RTL Optimization

> [中文版 / Chinese version →](README.md)

LLM-driven optimization of hardware RTL, with **formal equivalence checking to guarantee
the optimized design still does the same thing**.

> ### ⚠️ About the dataset
>
> **This repository contains code only — no datasets, no reference RTL.**
>
> The original experiments used open-source hardware from NVIDIA NVDLA and OpenTitan,
> which is not republished here for licensing and data-distribution reasons. A fresh clone
> therefore **will not run out of the box**: you need to supply your own test cases.
> See [section 5](#test-case-format-bring-your-own) for the format.
>
> Also not included: `memory_agent.db` (experience database), `rag_retrieve/indices/`
> (RAG indices), `LLM_DC_LOG/` (knowledge base). These accumulate from running
> experiments over time and have to be built up from scratch.

---

## 1. What this project does

Chip designers write **RTL** (hardware description code, in Verilog). The same function can be
written many ways, and the wording changes the resulting chip — smaller **area**, or faster
**timing**. Tuning this by hand is tedious; this project has an LLM do it.

But letting an LLM rewrite hardware has one fatal risk: **it can silently break the function.**
So the core of this project is a strict pipeline:

```
   Specification                              fails → send back to the LLM to fix
   (what the circuit must do)                           ↑
        │                                               │
        ├──► Route A: C-first ────┐                     │
        │    spec → C → optimize  │                     │
        │    C → back to RTL      │                     │
        │                         ├──► Formal check ────┘──► Synthesis ──► measure area/speed
        ├──► Route B: RTL-direct ─┘    (JasperGold)          (DesignCompiler)
             LLM writes RTL directly    mathematically         real EDA tool
                                        *proves* the new
                                        RTL is equivalent
```

**Both routes solve the same problem and compete.** That is what "dual-route" means.

Three key players:

| Name | What it is | Role |
|------|-----------|------|
| **LLM** | e.g. DeepSeek | Rewrites the code |
| **JasperGold** (JG) | Formal verification tool | *Proves* equivalence — this is **proof, not testing** |
| **DesignCompiler** (DC) | Logic synthesis tool | Turns RTL into a real circuit, measures area and speed |

Only results that **pass formal verification AND synthesize successfully** count. Anything that
fails verification is explicitly flagged and never silently enters the dataset.

---

## 2. Quick start

> **Prerequisites**: set up the Python environment, `.env` and your test cases first —
> see [section 4](#4-requirements). If data is missing, step 3 of the script tells you exactly what.

### Step 1 — Self-check (free, strongly recommended first)

```bash
./run_all.sh --check
```

Checks the Python environment, EDA tools, data files and API configuration, and **automatically
fills in any missing data files**. All green means you are ready:

```
━━━ Step 1/6  Python environment ━━━
  ✓ Python: Python 3.11.16
  ✓ EDA tool clang-16
  ✓ EDA tool iverilog
  ...
━━━ Step 5/6  preflight ━━━
  ✓ LLM endpoint configured
  ✓ DesignCompiler remote configured
  ✓ JasperGold configured (enabled=True)
  ✓ C-first route ready
  ✓ RTL-direct route ready
```

### Step 2 — See what would run (still free)

```bash
./run_all.sh
```

With no arguments this is **safe mode**: it only writes the experiment plan. **No LLM calls,
no cost.**

### Step 3 — Actually run it

```bash
./run_all.sh --execute
```

`--execute` is the real thing. The script asks you to type `yes` to confirm, because this will:

- call the LLM API (**costs money**)
- hold remote JasperGold and DesignCompiler licenses
- take **over an hour** for the 13 test cases

---

## 3. `run_all.sh` options

```bash
./run_all.sh --help
```

| Option | Meaning |
|--------|---------|
| *(none)* | Safe mode: plan only, no cost |
| `--check` | Environment self-check only |
| `--execute` | **Real run** (costs money) |
| `--objective area` | Optimize for **area** — default |
| `--objective timing` | Optimize for **timing** (speed) |
| `--verification-mode none` | Skip formal verification, synthesize only. Faster, but results are **not guaranteed functionally correct** and are marked unusable for training |
| `--repeats N` | Repetitions per case (default 1) |
| `--output-root <dir>` | Where results go (default `runs/phase6/oneclick_<objective>`) |
| `--summarize-existing` | Re-aggregate existing results without running anything new |
| `--yes` | Skip the confirmation prompt (for unattended runs) |

Common combinations:

```bash
./run_all.sh --objective timing --execute      # real timing run
./run_all.sh --execute --yes                   # unattended
./run_all.sh --summarize-existing              # rebuild the report only
```

> **Resuming**: re-running with the same `--output-root` reuses cases that already finished.
> But **different run configurations cannot share one output directory** — the program validates
> the config hash and refuses, so old and new results never get mixed.

---

## 4. Requirements

The script checks all of this and tells you exactly what is missing.

**Python environment** (default `/path/to/Vivado/.conda-env`, Python 3.11)

This conda environment already contains every EDA tool: `clang-16`, `opt`, `iverilog`, `yosys`, `dot`.
`toolchain.py` **looks in the Python prefix's `bin/` first**, so using the right Python resolves the
whole toolchain automatically — **no PATH changes needed**.

To use a different environment:

```bash
VIVADO_ENV=/your/conda/env ./run_all.sh --check
```

**Data assets** (9 files, ~82MB — not in the public repo)

These large files are required at runtime but are **not part of this repository**. If you have
an original working tree, point the script at it and step 3 copies them in (with `cp -n`, so it
**never overwrites** anything that already exists):

```bash
VIVADO_SRC_TREE=/your/source/tree ./run_all.sh --check
```

If you don't, you will have to build them up from scratch — `memory_agent.db` and the RAG
indices are accumulated by running experiments; `rag_retrieve/build_rag.py` rebuilds the indices.

The list:

| File | Size | Purpose |
|------|------|---------|
| `memory_agent.db` | 41M | Memory Agent experience database |
| `path_decisions_log.json` | 88K | Historical routing decisions |
| `rag_retrieve/indices/module4_historical_region_index_v3.json` | 28M | RAG historical region index |
| `rag_retrieve/indices/module4_cdfg_rag_index_v3.json` | 9.3M | CDFG joined index |
| `LLM_DC_LOG/rag_knowledge_base_v3.csv` | 3.6M | RAG knowledge base |
| 4 smaller files | <100K | Manifest, stats, rebuild report |

To use a different source tree:

```bash
VIVADO_SRC_TREE=/your/source/tree ./run_all.sh --check
```

**`.env`** (currently a symlink into the source tree)

Holds the LLM endpoint (`OPENAI_*` / `CLOUD_*`), DesignCompiler remote (`DC_*`) and JasperGold
remote (`JG_*`) settings. **It contains API keys — never commit it, never share it.**

---

## 5. Code layout

### The 8 modules

| Module | Directory | Purpose |
|--------|-----------|---------|
| Module 1 | `spec_analyze/` | Parse the spec, extract circuit features |
| Module 2 | `path_select/` | Decide which route this design takes |
| Module 3 | `c_gen/` | Generate C from the circuit description |
| Module 4 | `rag_retrieve/` | RAG retrieval over past experience |
| Module 4.5 | *(inside Module 5)* | MCTS planning — what to change and how |
| Module 5 | `module5/` | Execute: edit C → synthesize → verify → measure PPA |
| Module 8 | `memory_agent/` | Memory hub: record every decision and outcome |
| Support | `token_counter/` | Token accounting |

### Experiment drivers

| Path | Purpose |
|------|---------|
| `experiments/path_oracle/run_phase6_pilot.py` | **Main entry point** — what `run_all.sh` calls |
| `experiments/path_oracle/run_dual_path.py` | Dual-route run for a single spec |
| `pipeline/__main__.py` | Full Module 1–5 pipeline for one design |
| `pipeline/preflight.py` | Environment self-check |
| `module5/jg_verifier.py` | JasperGold formal verification (124K, one of the core files) |
| `module5/rtl_direct_runner.py` | RTL-direct route executor |

### Test case format (bring your own)

**This repository ships no test cases** — you supply them. The original experiments used 13
official NVDLA circuits (NVDLA is NVIDIA's open-source deep learning accelerator); you can
source your own from [nvdla/hw](https://github.com/nvdla/hw) or any Verilog project.

Directory layout:

```
your-cases/
  pilot_manifest.json          # case list
  cases/
    <case-name>/
      spec.txt                 # the specification handed to the LLM
      golden_source.v          # original source
      golden_closure.v         # self-contained closure (reference for formal checking)
```

`pilot_manifest.json` must contain these fields:

```json
{
  "case_count": 1,
  "execution_policy": { "mode": "serial", "max_concurrency": 1 },
  "cases": [
    {
      "id": "my_case",
      "family_id": "my_family",
      "design_type": "sequential",
      "golden_top_module": "my_module",
      "spec_path": "cases/my_case/spec.txt",
      "golden_rtl_path": "cases/my_case/golden_closure.v",
      "golden_source_rtl_path": "cases/my_case/golden_source.v"
    }
  ]
}
```

Hard requirements — the program refuses to start otherwise:

- `execution_policy` must be exactly `{"mode": "serial", "max_concurrency": 1}`
- `case_count` must equal the actual length of `cases`
- the files named by `spec_path` and `golden_rtl_path` must exist

**What "closure" means**: formal verification needs a design that elaborates standalone. If your
module depends on other files (submodules, RAM models, library functions), flatten every
dependency into one file — that is `golden_closure.v`. Keep `golden_source.v` as the untouched
original, for the record.

Then point the script at it:

```bash
./run_all.sh --pilot-root /your/cases --execute
```

### Other

- `golden_datas/`, `goldenRTL/` — reference RTL and spec documents
- `logs/` — run logs (`run_all.sh` writes one per run)
- `run_dc.py`, `verilog_dc.py` — DesignCompiler invocation
- `llm_v2v.py`, `llm_request.py` — LLM call wrappers

---

## 6. Reading the results

Results land under `--output-root` (default `runs/phase6/oneclick_area/`):

```
runs/phase6/oneclick_area/
  phase6_collection.json      # ← top-level summary, start here
  pairs/
    <case>_rep1/
      phase6_pair_result.json # per-case dual-route comparison
```

Key fields in `phase6_collection.json`:

| Field | Meaning |
|-------|---------|
| `mode` | `plan_only` = plan only; `execute` = really ran |
| `objective` | `AREA` or `TIMING` |
| `planned_pair_count` | Pairs planned |
| `completed_pair_count` | Pairs finished |
| `run_config_sha256` | Config hash, validated when resuming |

Per-case status values (in `phase6_pair_result.json`):

| Status | Meaning |
|--------|---------|
| `success` | Passed end to end, valid PPA data ✓ |
| `equivalence_failed` | Formal check found a functional mismatch — **the LLM broke it** |
| `rtl_failed` | Generated RTL failed syntax or elaboration |
| `c_generation_failed` | C generation failed |
| `validate_failed` | Validation failed |
| `pipeline_exception` | Pipeline error (usually environment or missing files) |

> **How to read `equivalence_failed`**: this is not a bug — it is **formal verification doing its
> job**. It caught the LLM breaking the design and produced a counterexample. In practice, across
> a 19-case run only a handful reached `success`; most were stopped by verification. **That is
> exactly the point of this pipeline**: without formal checking, those broken "optimizations"
> would have been recorded as wins.

---

## 7. FAQ

**Q: How much does a run cost / how long does it take?**
13 cases at `--repeats 1`, a few minutes each, over an hour total. Cost depends on the model
configured in `.env`. Run safe mode first, then `--execute`.

**Q: "Project Memory baseline files are required"**
`memory_agent.db` or `path_decisions_log.json` is missing. `./run_all.sh --check` fills them in.

**Q: "Required tool not found"**
An EDA tool was not found. Make sure you are using the conda environment that ships the toolchain
(`VIVADO_ENV`). Note: **do not rely on `conda activate` plus a bare `python`** — `run_all.sh` uses
the interpreter's absolute path, which is more reliable.

**Q: Can I skip formal verification for a quick look?**
`./run_all.sh --verification-mode none --execute`. But the results are flagged
`training_eligible=false` and **must not be treated as trustworthy data**.

**Q: Can I run it in parallel to go faster?**
The main entry point is **serial by design** (`max_concurrency=1`); there is no `--workers` option.
This is deliberate: concurrent LLM calls and simultaneous DC jobs would compromise the fairness of
the route comparison.

**Q: How do I prepare test cases?**
See [the format in section 5](#test-case-format-bring-your-own). Point `--pilot-root` at your own
directory. The manifest must declare `"execution_policy": {"mode": "serial", "max_concurrency": 1}`
or the program refuses to start.

**Q: How do I resume after an interruption?**
Re-run with the same `--output-root`; finished cases are reused automatically.

---

## 8. Safety notes

- `.env` holds API keys and server credentials — **never commit it, never share it**
- `--execute` incurs real API cost and holds EDA licenses; the confirmation prompt is deliberate
- Asset backfill uses `cp -n` and **never overwrites** existing files, so re-running is safe
- The repo's `.gitignore` already excludes secrets, datasets and experiment artifacts — think
  before loosening it
