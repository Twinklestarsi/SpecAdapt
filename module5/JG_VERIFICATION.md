# JasperGold Verification Integration for Module 5

## Overview

This integration adds formal verification capabilities to Module 5's direct RTL generation flow using JasperGold. The generated Verilog is verified against a golden reference (`csrng.sv`) through an iterative verification-and-repair loop.

## Architecture

### Workflow

```
1. Module 5 generates initial RTL via LLM
2. JasperGold verifies RTL against golden reference (csrng.sv)
3. If verification fails:
   a. Extract error feedback from JasperGold output
   b. Pass feedback to LLM for correction
   c. LLM generates corrected RTL
   d. Repeat from step 2
4. If verification passes or max retries exhausted, return result
```

### Key Components

#### 1. `module5/jg_verifier.py` (NEW)
Core JasperGold verification module with the following functions:

- **`verify_rtl_with_jg()`**: Main entry point for verification loop
- **`run_jaspergold()`**: Execute JasperGold on remote server via SSH
- **`parse_verilog_ports()`**: Extract module interface from Verilog
- **`generate_fpv_wrapper()`**: Create FPV wrapper for equivalence checking
- **`generate_fpv_tcl()`**: Generate JasperGold TCL script
- **`build_correction_prompt()`**: Build LLM prompt with JG feedback

#### 2. `module5/rtl_direct_runner.py` (MODIFIED)
Updated to support JasperGold verification:

- Added `enable_jg_verification` parameter
- Added `golden_rtl_path` parameter
- Added `jg_max_retries` parameter
- Calls `verify_rtl_with_jg()` after syntax validation

#### 3. `module5/executor.py` (MODIFIED)
Updated to pass JG parameters through the execution pipeline:

- `_evaluate_candidate()` accepts JG parameters
- `execute_action()` accepts JG parameters
- Parameters propagated to `generate_direct_rtl()`

#### 4. `module5/cli.py` (MODIFIED)
Added command-line arguments:

- `--enable-jg-verification`: Enable JG verification
- `--golden-rtl-path`: Path to golden reference RTL
- `--jg-max-retries`: Max verification retry attempts

## Usage

### Basic Usage

```bash
python3 -m module5.cli \
  --plan /path/to/module45_plan.json \
  --action-index 0 \
  --output-root /tmp/module5_runs \
  --backend direct_rtl \
  --enable-jg-verification \
  --golden-rtl-path /path/to/project/goldenRTL/csrng.sv \
  --jg-max-retries 3
```

### Test Script

A test script is provided at `/path/to/project/test_jg_verification.sh`:

```bash
./test_jg_verification.sh
```

### Batch Execution

Update `run_module5_batch.sh` to enable JG verification:

```bash
python3 -m module5.cli \
  --plan "$plan_file" \
  --action-index "$ACTION_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --backend direct_rtl \
  --env-path "$ENV_PATH" \
  --spec-db-path "$SPEC_DB_PATH" \
  --enable-jg-verification \
  --golden-rtl-path /path/to/project/goldenRTL/csrng.sv \
  --jg-max-retries 3 \
  $BASELINE_FLAG
```

## Intermediate Files

All intermediate files are stored in `/path/to/project/module5_mid/<benchmark>/`:

### Per-Attempt Files

For each verification attempt `N`:

- **`attempt_N.v`**: Generated Verilog (with `_opt` suffix on module name)
- **`attempt_N_FPV.sv`**: FPV wrapper instantiating golden and optimized modules
- **`attempt_N_FPV.tcl`**: JasperGold TCL script for equivalence checking
- **`attempt_N_jg_output.txt`**: Raw JasperGold output (stdout + stderr)
- **`attempt_N_correction_prompt.txt`**: LLM correction prompt (if retry needed)
- **`attempt_N_llm_response.txt`**: LLM correction response (if retry needed)

### Golden Reference

- **`csrng.sv`**: Copy of golden reference RTL

## Configuration

### Environment Variables (.env)

JasperGold configuration is loaded from `.env`:

```bash
# JasperGold remote server
JG_REMOTE_USER=2660001
JG_REMOTE_HOST=your.jg.server.example.com
JG_REMOTE_BASE=/home/2660001/wyf
JG_TIMEOUT=600

# LLM configuration (for correction)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=http://...
OPENAI_MODEL=gpt-4o-mini
```

### Golden Reference

The golden reference RTL is located at:
```
/path/to/project/goldenRTL/csrng.sv
```

This is the functionally correct implementation that generated RTL must match.

## Verification Process

### 1. FPV Wrapper Generation

The verifier generates a SystemVerilog wrapper that:
- Instantiates both golden (`u_ref`) and optimized (`u_opt`) modules
- Connects all inputs to both instances
- Separates outputs with `_ref` and `_opt` suffixes
- Asserts equivalence: `output_ref == output_opt`

Example wrapper structure:
```systemverilog
module csrng_FPV(
    input logic clk_i,
    input logic rst_ni,
    // ... other inputs ...
    output logic entropy_src_req_o_ref,
    output logic entropy_src_req_o_opt
);
    csrng u_ref(
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        // ...
        .entropy_src_req_o(entropy_src_req_o_ref)
    );

    csrng_opt u_opt(
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        // ...
        .entropy_src_req_o(entropy_src_req_o_opt)
    );

    property p_eq_entropy_src_req_o;
        @(posedge clk_i) disable iff (!rst_ni)
            entropy_src_req_o_ref == entropy_src_req_o_opt;
    endproperty
    assert property (p_eq_entropy_src_req_o);
    // ... assertions for other outputs ...
endmodule
```

### 2. JasperGold Execution

The verifier:
1. Copies files to remote server via SSH path mapping
2. Runs JasperGold in batch mode with generated TCL script
3. Parses SUMMARY section from output
4. Extracts: `proven`, `cex` (counterexample), `undetermined`, `unknown`

Success criteria:
- All assertions proven
- No counterexamples
- No undetermined/unknown results

### 3. Error Feedback Extraction

If verification fails, the verifier extracts:
- Error type: `counterexample found`, `undetermined`, `compile error`, etc.
- Last 1500 characters of JasperGold output
- Specific assertion failures (if available)

### 4. LLM Correction

The correction prompt includes:
- Golden reference RTL (functional specification)
- Previous (incorrect) Verilog output
- JasperGold error details
- Original C specification
- Specification description
- Critical requirements (preserve interface, fix functional bug)

The LLM generates corrected Verilog, which is fed back into the verification loop.

## Return Values

### Success Case

```json
{
  "success": true,
  "status": "verified",
  "generated_verilog_path": "/path/to/final.v",
  "syntax_ok": true,
  "verified": true,
  "jg_attempts": 2,
  "stderr": "",
  "module_name": "csrng",
  "prompt_path": "/path/to/rtl_prompt.json",
  "raw_response_path": "/path/to/rtl_raw_response.txt"
}
```

### Failure Case

```json
{
  "success": false,
  "status": "verification_failed",
  "generated_verilog_path": "/path/to/final.v",
  "syntax_ok": true,
  "verified": false,
  "jg_attempts": 4,
  "stderr": "Verification failed after 4 attempts: counterexample found",
  "module_name": "csrng",
  "prompt_path": "/path/to/rtl_prompt.json",
  "raw_response_path": "/path/to/rtl_raw_response.txt"
}
```

## Implementation Details

### Port Detection

The verifier automatically detects:
- **Clock**: Looks for `clk`, `clk_i`, `clock` input ports
- **Reset**: Looks for `rst`, `rst_ni`, `reset` input ports
- **Reset polarity**: Active-low if name contains `_n`, `_ni`, or `n` suffix
- **Sequential logic**: Detects `always @(posedge/negedge)` patterns

### Module Renaming

To avoid naming conflicts:
- Golden module keeps original name (e.g., `csrng`)
- Optimized module renamed to `<original>_opt` (e.g., `csrng_opt`)
- FPV wrapper named `<original>_FPV` (e.g., `csrng_FPV`)

### Remote Path Mapping

Local paths under `/path/to/project/` are mapped to remote paths under `/path/on/remote/` (configurable via `JG_REMOTE_BASE`).

### Cleanup

After timeout or lock errors, the verifier:
1. Kills stale JasperGold processes via `pkill -9`
2. Removes project directories: `jgproject*`, `sessionLogs*`
3. Waits briefly for processes to die before retry

## Limitations

1. **Golden reference is fixed**: Currently hardcoded to `csrng.sv`
2. **Single benchmark**: Designed for CSRNG benchmark specifically
3. **Interface must match**: Generated RTL must have same ports as golden
4. **Sequential logic assumed**: Optimized for clocked designs
5. **Remote execution required**: JasperGold must be available on remote server

## Future Enhancements

1. **Multi-benchmark support**: Parameterize golden reference per benchmark
2. **Behavioral equivalence**: Add support for interface transformations
3. **Parallel verification**: Run multiple JG instances concurrently
4. **Incremental verification**: Cache proven sub-modules
5. **Coverage-guided correction**: Use JG coverage to guide LLM fixes
6. **Assertion mining**: Automatically extract assertions from golden RTL

## Troubleshooting

### JasperGold Timeout

If JG times out (default 600s):
- Check remote server load
- Increase `JG_TIMEOUT` in `.env`
- Simplify design or reduce assertion complexity

### Project Directory Lock

If you see "cannot obtain ownership of project directory":
- The cleanup function should handle this automatically
- If persistent, manually SSH and remove `jgproject*` directories

### Compile Errors

If JG reports compile errors:
- Check `attempt_N_jg_output.txt` for details
- Verify golden reference is valid SystemVerilog
- Check for syntax errors in generated Verilog

### LLM Correction Fails

If LLM returns empty or invalid Verilog:
- Check LLM API configuration in `.env`
- Review correction prompt in `attempt_N_correction_prompt.txt`
- Try different LLM model or increase temperature

### Verification Never Passes

If all retries exhausted without success:
- Review JG outputs to understand failure mode
- Check if golden reference matches C specification
- Verify that optimization is functionally valid
- Consider increasing `--jg-max-retries`

## References

- **llm_v2v_2.py**: Reference implementation for JG verification loop
- **goldenRTL/csrng.sv**: Golden reference RTL
- **MODULE5.md**: Module 5 architecture documentation
