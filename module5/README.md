# Module 5

Module 5 executes one `Module5Action` from Module 4.5, edits the source C code,
validates it, runs HLS to generate RTL, evaluates the generated Verilog with
Design Compiler, and writes a structured result JSON.

## Files

- `cli.py`: command-line entry point
- `executor.py`: main orchestration
- `editor.py`: constrained LLM-based C edit
- `validators.py`: C syntax and structural checks
- `hls_runner.py`: HLS wrapper around `hls_rtl_batch_enhanced.py`
- `dc_runner.py`: single-Verilog DC evaluation wrapper
- `reward.py`: objective-aware reward calculation
- `result_schema.py`: result dataclasses
- `io_utils.py`: JSON loading and workdir helpers

## Requirements

- `.env` with `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and optional `OPENAI_MODEL`
- `clang-16` on `PATH`
- Vitis Python environment for `hls_rtl_batch_enhanced.py`
- Remote DC access configured in `.env` for `run_dc.py`

## Usage

Run one best action from a Module 4.5 plan:

```bash
python3 -m module5.cli \
  --plan /path/to/project/rag_retrieve/indices/module45/counter_8bit_mcts_area_v2.json \
  --action-index 0 \
  --output-root /tmp/module5_runs
```

Run from a single action JSON:

```bash
python3 -m module5.cli \
  --action-json /path/to/action.json \
  --output-root /tmp/module5_runs
```

Skip baseline evaluation:

```bash
python3 -m module5.cli \
  --plan /path/to/module45.json \
  --action-index 0 \
  --no-baseline
```

## Output

Each run writes a work directory:

```text
/tmp/module5_runs/<benchmark>/<action_id_sanitized>/
  action.json
  source.c
  edited.c
  result.json
  candidate_hls/
  baseline_hls/
```

`result.json` includes:

- execution status
- edited C path
- generated Verilog path
- syntax/HLS/DC status
- area/timing metrics
- baseline deltas when enabled
- objective-aware reward summary
