# SpecAdapt

This repository contains the SpecAdapt codebase. The main workflow starts from a specification, automatically selects either the `c_first` or `rtl_direct` path, generates candidate RTL, performs functional equivalence verification with JasperGold, and finally evaluates area or timing with Design Compiler.

## 1. The Four Agents in the Main Workflow

The four Agents are implemented jointly by the modules below. JasperGold and Design Compiler are backends for verification, synthesis, and PPA evaluation; they are not among the four Agents.

| Agent | Current implementation | Responsibilities |
| --- | --- | --- |
| **Spec Agent** | [`spec_analyze/`](spec_analyze/), [`path_select/`](path_select/) | Reads the spec and extracts structured features; in `auto` mode, uses the Adapter together with rules, the model, and Memory records to select `c_first` or `rtl_direct` |
| **C Agent** | [`c_gen/`](c_gen/), [`rag_retrieve/`](rag_retrieve/), [`module5/`](module5/) | Generates C on the `c_first` path; retrieves historical transformations and uses MCTS to form an optimization action chain; then modifies and checks the C code, using HLS to implement C-to-RTL|
| **RTL Agent** | [`pipeline/`](pipeline/), [`module5/`](module5/) | Generates, syntax-checks, and retries RTL. `rtl_direct` generates it directly from the spec|
| **Memory Agent** | [`memory_agent/`](memory_agent/) | Stores and retrieves features, path decisions, historical experience, failure feedback, and PPA results to provide context for later stages |


The verification backends are implemented in [`module5/jg_verifier.py`](module5/jg_verifier.py) (JasperGold) and [`module5/dc_runner.py`](module5/dc_runner.py) (Design Compiler). With `--verification-mode jaspergold`, a candidate must first pass syntax checking and JasperGold equivalence verification before entering DC.

## 2. Environment Setup

First clone the repository and enter its root directory:


The project does not assume a fixed Python or conda path and does not provide a unified dependency installation script. Prepare a Python 3 environment that can run this project and make sure `python` is on PATH. If you use a dedicated environment, you can explicitly specify its root directory:

```bash
export VIVADO_ENV=/path/to/your/python-env
export PATH="$VIVADO_ENV/bin:$PATH"
```

Only after confirming that `.env` does not exist in a newly cloned directory, copy the configuration template and fill in your service configuration; if `.env` already exists, do not overwrite it:

```bash
cp .env.example .env
```

`.env.example` contains configuration entries for the LLM, JasperGold, and Design Compiler. Fill in the corresponding API, remote server, and tool paths for your environment; if you do not use a backend, you can leave its placeholder configuration in place. Running HLS requires Vitis `v++`, as well as `yosys` and `iverilog` on PATH; if `v++` is not yet on PATH, specify the Vitis environment script first:

```bash
export VITIS_ENV_SCRIPT=/path/to/Vitis/settings64.sh
```

If `v++` is already on PATH, you do not need to set `VITIS_ENV_SCRIPT`.

## 3. Running an Experiment: `python -m pipeline`

Running `python -m pipeline` processes one design. The default settings let the path selection module choose the processing path automatically. Each run must specify an optimization objective with `--objective`. `area` means minimizing chip area, while `timing` means minimizing circuit delay.

The path selection module works in the following order:

1. The path selection module first checks two types of fixed rules. For multi-clock, purely structural, or extremely simple circuits, the program directly selects `rtl_direct`, which generates RTL (register-transfer-level hardware description code) directly from the design requirements.
2. If none of these rules match, the program calls the trained MLP (multilayer perceptron model) to calculate the probability of selecting `c_first`. `c_first` means generating C code first and then generating RTL from the C code.

### 3.1 Using a spec File

The example is as follows: `golden_closure.v` is the reference design for JasperGold, and `--golden-top your_top` should match the top-level module name for the case:

```bash
# If v++ is not on PATH, uncomment the next line and replace it with the path to your Vitis installation script.
# export VITIS_ENV_SCRIPT=/path/to/Vitis/settings64.sh

SPEC=/path/to/your/spec.txt
GOLDEN=/path/to/your/golden_closure.v

python -m pipeline \
  --spec-file "$SPEC" \
  --benchmark your_benchmark \
  --objective area \
  --backend hls \
  --verification-mode jaspergold \
  --golden-rtl "$GOLDEN" \
  --golden-top your_top \
  --output-root "$PWD/pipeline_runs/your_benchmark_area"
```



### 3.2 Experiment Result Locations

If no path arguments are provided, `pipeline/orchestrator.py` uses `memory_agent.db`, `path_decisions_log.json`, `pipeline_runs/`, and `c_gen_output/` in the repository root by default. After `--output-root` is provided, each run's plan and candidate results are written under that directory, for example:

```text
pipeline_runs/csrng_area/
├── summary.json
└── <run_id>/
    ├── rtl_direct/              # rtl_direct path
    ├── module4_plan.json        # C-first retrieval results
    ├── module45_plan.json       # C-first MCTS action chain
    └── module5/                  # Modified C, RTL, and JG/DC results
```

## 4. Common CLI Quick Reference

```bash
python -m pipeline --help
./run_all.sh --help
```

Common parameters for a single design: `--spec-file`, `--verilog`, `--objective area|timing`, `--backend direct_rtl|hls`, `--forced-path auto|c_first|rtl_direct`, `--verification-mode jaspergold|none`, `--golden-rtl`, `--golden-top`, `--output-root`.

Common Phase-6 parameters: `--pilot-root`, `--output-root`, `--objective area|timing`, `--repeats N`, `--execute`, `--yes`, `--check`, `--summarize-existing`. Refer to the output of these two `--help` commands and the current script implementation for the complete parameters and default values.
