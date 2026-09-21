This repository contains an experimental framework for using large language models (LLMs)
to explore function-preserving optimizations of register-transfer level (RTL) designs.
The framework evaluates two complementary routes against the same specification:

- **C-first** translates the design description into C, applies guided transformations,
  and generates RTL from the optimized representation.
- **RTL-direct** asks the LLM to produce an RTL candidate directly.

Both routes are orchestrated through a common pipeline. Specification analysis and route
selection are combined with retrieval-augmented generation (RAG), Monte Carlo Tree Search
(MCTS), and a project Memory component that records decisions and outcomes. Candidate RTL is
checked for syntax and elaboration before it proceeds to optional formal verification and
logic synthesis. Design Compiler (DC) supplies area and timing measurements; JasperGold (JG)
can provide formal equivalence evidence against the reference RTL.

The framework is intended for controlled experiments, not for assuming that an LLM-generated
design is correct. A candidate has equivalence evidence only when JasperGold reports a
successful equivalence check for that candidate. Syntax success, synthesis success, or a good
PPA value alone does not establish functional equivalence. Results from runs with formal
verification disabled are explicitly unverified and are not training-eligible.

## Repository scope and exclusions

The GitHub repository is intentionally a code-and-documentation distribution. It does not
include the experimental datasets or reference designs used by prior runs. In particular,
the repository excludes:

- `datasets/`, `golden_datas/`, `goldenRTL/`, and `runs/`;
- reference RTL, test-case specifications, and generated experiment results;
- `.env` and other local credentials or deployment-specific configuration;
- `memory_agent.db`, RAG indices, RAG knowledge bases, and related runtime state.

A fresh clone therefore cannot execute the complete experiment without additional runtime
assets. Users must supply their own test cases, reference RTL, any required Memory/RAG assets,
LLM configuration, and access to the required EDA backends. The repository's ignore rules are
intended to keep these inputs and generated artifacts out of ordinary commits; review the
working tree before publishing it.

## Architecture

The high-level execution path is deliberately serial so that route comparisons remain
reproducible at the orchestration level:

```text
Specification + reference RTL
              |
              v
     Specification analysis
              |
              v
      Route selection / RAG / Memory
              |
       +------+------+
       |             |
       v             v
   C-first       RTL-direct
  spec -> C ->    LLM writes
  optimize -> RTL  candidate RTL
       |             |
       +------+------+
              v
     Syntax and elaboration checks
              |
              v
   Optional JasperGold equivalence
              |
              v
    Design Compiler synthesis
              |
              v
       Area / timing / PPA results
              |
              v
      Result summary and Memory update
```

The two route outputs are compared using the configured PPA objective. Formal verification is
an evidence gate when `--verification-mode jaspergold` is selected; it is not a replacement for
reviewing the generated RTL, synthesis reports, or experiment configuration.

## Capabilities

The current phase-6 driver supports the following experiment controls and safeguards:

| Capability | Description |
| --- | --- |
| Dual routes | Compares C-first and RTL-direct candidates for the same case. |
| Area objective | Scores lower synthesized cell area as the preferred result. |
| Timing objective | Scores lower critical-path delay / arrival time as the preferred result. |
| RAG retrieval | Reuses relevant historical transformations and synthesis experience when the required assets are available. |
| MCTS planning | Searches candidate C-first actions under configurable iteration, depth, and action limits. |
| Formal gate | Runs syntax, JasperGold equivalence, optional LLM repair, and then DC in the strict verification mode. |
| PPA measurement | Records synthesis status and area/timing fields from the DC backend. |
| Serial policy | Enforces `execution_policy.mode=serial` and `max_concurrency=1` in the pilot manifest. |
| Plan-only mode | Creates an experiment plan and configuration record without issuing LLM or EDA execution calls. |
| Resume support | Reuses completed pairs only when content and configuration fingerprints still match. |
| Config hashing | Records a run-configuration hash to prevent incompatible results from being mixed. |

## Requirements

### Operating system and Python

The supported execution environment is Linux. Python 3.11 is recommended. The launcher can
use an environment selected by `VIVADO_ENV`, a project or parent environment discovered by the
launcher, the active conda environment, or a `python3` found on `PATH`.

The Python packages checked by [`run_all.sh`](run_all.sh) are:

```text
python-dotenv
openai
numpy
pandas
PyYAML
requests
tiktoken
networkx
scikit-learn
```

For a new Python environment, the package installation command is:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  python-dotenv openai numpy pandas PyYAML requests \
  tiktoken networkx scikit-learn
```

The C-first path also requires the command-line tools `clang-16`, `opt`, `iverilog`, `yosys`,
and `dot`. They must be available in the selected environment's `bin/` directory or on
`PATH`; they are not installed by this README's Python command.

### Services and EDA backends

An OpenAI-compatible LLM endpoint is required for execution. The endpoint, model, and API key
are configured through `.env`.

Full verified execution additionally requires access to remote Design Compiler and JasperGold
installations, including their environment scripts, executable paths, and remote workspaces.
These commercial tools and licenses are not distributed with this repository. A plan-only run
can be used to inspect a configuration without issuing LLM or EDA work, subject to the normal
preflight behavior of the launcher.

## Setup

### Clone and prepare the environment

Use the repository URL for your own GitHub repository in the following example:

```bash
git clone https://github.com/<owner>/<repository>.git
cd <repository>
```

Create or select an environment that contains the Python packages and command-line tools above.
If the tools are installed in a dedicated environment, point the launcher at it:

```bash
export VIVADO_ENV=/path/to/python-or-conda-environment
```

Copy the configuration template and edit the resulting local file:

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

`.env` contains secrets and deployment-specific paths. Keep it local, do not commit it, and do
not paste its values into issue reports or experiment logs shared outside the project.

### Configuration groups

The template in [`.env.example`](.env.example) is organized into these groups:

| Group | Variables | Purpose |
| --- | --- | --- |
| Cloud LLM | `CLOUD_API_BASE_URL`, `CLOUD_API_KEY`, `CLOUD_MODEL`, token and timeout settings | OpenAI-compatible remote model access. |
| Compatibility aliases | `OPENAI_API_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | Names used by older and shared pipeline components. |
| Optional reasoning | `DEEPSEEK_ENABLE_THINKING`, `DEEPSEEK_REASONING_EFFORT` | Provider-specific reasoning controls when supported by the configured model. |
| Local LLM | `LOCAL_API_BASE_URL`, `LOCAL_API_KEY`, `LOCAL_MODEL`, token and timeout settings | Local API use by the local V2V path. |
| Design Compiler | `DC_REMOTE_USER`, `DC_REMOTE_HOST`, `DC_REMOTE_BASE`, `DC_ENV_SCRIPT`, `DC_SHELL_PATH`, `ASAP7_DB_PATH` | Remote synthesis and technology-library configuration. |
| JasperGold | `JG_ENABLED`, `JG_REMOTE_USER`, `JG_REMOTE_HOST`, `JG_REMOTE_BASE`, `JG_ENV_SCRIPT`, `JG_BIN`, `JG_TIMEOUT` | Remote formal-equivalence configuration. |
| Memory | `MEMORY_AGENT_DB_PATH` | Optional location of the project Memory database. |
| Proxy bypass | `NO_PROXY`, `no_proxy` | Keeps local and internal EDA/API traffic off configured proxies. |

Use an OpenAI-compatible base URL and model that your provider supports. The compatibility
aliases may be set to the corresponding `CLOUD_*` values as shown in the template.

## Runtime assets

The code distribution and the execution distribution are separate. Runtime assets may include
the project Memory database, the historical routing log, RAG indices, and the knowledge-base
files consumed by the retrieval components. Test cases and reference RTL are separate inputs
described in [Test-case format](#test-case-format).

If an existing source tree already contains the required runtime assets, set
`VIVADO_SRC_TREE` to that tree before invoking the launcher:

```bash
export VIVADO_SRC_TREE=/path/to/existing/source-tree
./run_all.sh --check --pilot-root /path/to/cases
```

The launcher uses `cp -n` for asset backfill, so it does not overwrite files that are already
present. `VIVADO_SRC_TREE` is only a source for an existing, compatible tree; it is not a data
generator. Without such a source tree, the required assets must be provided or regenerated by
the user through the applicable project tooling. `run_all.sh` does not construct the complete
Memory and RAG state from an empty clone automatically.

## Test-case format

The phase-6 pilot expects a case root containing `pilot_manifest.json` and the files named by
that manifest. A minimal example is:

```text
/path/to/cases/
├── pilot_manifest.json
└── cases/
    └── example_case/
        ├── spec.txt
        ├── golden_source.v
        └── golden_closure.v
```

Example `pilot_manifest.json`:

```json
{
  "case_count": 1,
  "execution_policy": {
    "mode": "serial",
    "max_concurrency": 1
  },
  "cases": [
    {
      "id": "example_case",
      "family_id": "example_family",
      "design_type": "sequential",
      "golden_top_module": "example_top",
      "spec_path": "cases/example_case/spec.txt",
      "golden_rtl_path": "cases/example_case/golden_closure.v",
      "golden_source_rtl_path": "cases/example_case/golden_source.v"
    }
  ]
}
```

The runner requires `case_count` to equal the length of `cases`. It also requires the files
referenced by `spec_path` and `golden_rtl_path` to exist. The execution policy must be exactly
serial with `max_concurrency` equal to one; the phase-6 driver intentionally refuses a
parallel pilot. `golden_source_rtl_path` identifies the untouched source RTL for provenance
when it is available.

`golden_closure.v` should be a self-contained elaboration closure for formal verification. If
the original module depends on submodules, RAM models, or other library files, include the
needed dependencies in the closure while preserving `golden_source.v` as the original source.
The `golden_top_module` and `design_type` fields must agree with the design supplied to the
formal and synthesis backends.

## Quick start

The commands below assume that `/path/to/cases` contains the manifest and case files described
above. Supplying `--pilot-root` explicitly is required for a fresh clone because the repository
does not ship the original pilot cases.

1. Check the environment and case inputs:

   ```bash
   ./run_all.sh --check --pilot-root /path/to/cases
   ```

2. Generate a plan without executing LLM or EDA work:

   ```bash
   ./run_all.sh \
     --pilot-root /path/to/cases \
     --output-root ./runs/phase6/plan_area \
     --objective area
   ```

3. Execute the verified area-oriented experiment after reviewing the plan:

   ```bash
   ./run_all.sh \
     --pilot-root /path/to/cases \
     --output-root ./runs/phase6/area \
     --objective area \
     --verification-mode jaspergold \
     --execute
   ```

4. Execute a timing-oriented experiment in a separate output directory:

   ```bash
   ./run_all.sh \
     --pilot-root /path/to/cases \
     --output-root ./runs/phase6/timing \
     --objective timing \
     --verification-mode jaspergold \
     --execute
   ```

The `--execute` commands issue LLM requests, use remote JasperGold and Design Compiler
resources, and can incur provider charges or consume shared licenses. Any cost, license
availability, and runtime depend on the selected model, case set, configuration, and remote
infrastructure; this README does not promise a fixed amount.

To rebuild a summary from completed results without new LLM or EDA calls, use the same pilot
root, output root, repeat count, and objective:

```bash
./run_all.sh \
  --pilot-root /path/to/cases \
  --output-root ./runs/phase6/area \
  --objective area \
  --summarize-existing
```

## Command-line options

Run `./run_all.sh --help` for the launcher-generated help text. The main options are:

| Option | Meaning |
| --- | --- |
| *(no execution flag)* | Check the environment and write a plan; does not issue experiment calls. |
| `--check` | Run environment and input checks only. |
| `--execute` | Run the serial experiment using the configured LLM and EDA backends. |
| `--summarize-existing` | Rebuild summaries from completed pair results without new AI/DC execution. |
| `--pilot-root <dir>` | Test-case root containing `pilot_manifest.json`. |
| `--output-root <dir>` | Result directory; otherwise the launcher derives one under `runs/phase6/`. |
| `--objective area|timing` | Select the area or timing PPA objective; default is `area`. |
| `--repeats N` | Number of repeats per case; default is `1` in `run_all.sh`. |
| `--verification-mode jaspergold|none` | Use formal equivalence, or skip equivalence and mark results unverified. |
| `--max-actions N` | Maximum C-first planning actions; default is `5`. |
| `--mcts-iterations N` | MCTS iterations; default is `240`. |
| `--mcts-max-depth N` | MCTS search depth limit. |
| `--mcts-candidate-limit N` | Maximum candidates considered by MCTS. |
| `--rtl-max-retries N` | RTL-direct retry limit after generation failures. |
| `--dc-max-retries N` | Design Compiler retry limit. |
| `--jg-max-retries N` | Maximum JasperGold-driven repair attempts; default is `1`. |
| `--verification-timeout N` | JasperGold timeout in seconds; default is `1000`. |
| `--ppa-tie-tolerance-pct P` | Percentage tolerance used when comparing route PPA medians. |
| `--yes` | Skip the confirmation prompt before execution. |
| `--skip-assets` | Do not attempt runtime-asset backfill; use only when assets are already available. |
| `-h`, `--help` | Display launcher help. |

The phase-6 implementation also exposes an internal `--require-jaspergold` compatibility alias
when invoked directly. Use `run_all.sh --verification-mode jaspergold` for the documented
launcher interface.

## Outputs and status interpretation

Results are written below `--output-root`. A typical layout is:

```text
<output-root>/
├── phase6_collection.json
├── phase6_summary.json
└── pairs/
    └── <case>_repN/
        └── phase6_pair_result.json
```

`phase6_collection.json` records the run mode (`plan_only` or `execute`), objective, pilot
manifest, execution policy, preflight data, planned/completed pair counts, and the
`run_config_sha256` fingerprint used for safe resumption. `phase6_summary.json` contains
provisional route labels and aggregate evidence when execution or summary reconstruction is
complete.

Each `phase6_pair_result.json` contains the pair metadata, the C-first and RTL-direct run
records, verification records, synthesis status, and PPA fields. Common route-level statuses
include `success`, `rtl_failed`, `equivalence_failed`, `c_generation_failed`, `validate_failed`,
`dc_failed`, and `pipeline_exception`; the exact failure stage and reason should be read from
the JSON rather than inferred from a label. Verification records use statuses such as
`passed`, `failed`, `timeout`, `error`, or `not_run`.

For `--verification-mode jaspergold`, a route contributes verified PPA evidence only when its
equivalence record passes and its synthesis succeeds with a usable metric. A `success` status
is still a record of this pipeline's outcome, not a universal proof of every integration-level
property of a design.

For `--verification-mode none`, equivalence is not run. The summary may calculate an
unverified syntax-and-synthesis label, but it must be treated as unverified and is not
training-eligible. `training_eligible` is therefore a conservative summary field: it does not
mean that every generated artifact is correct, and it is false for the no-verification mode.

## Repository layout

| Path | Role |
| --- | --- |
| `run_all.sh` | Environment checks, asset handling, planning, execution, and summary entry point. |
| `experiments/path_oracle/run_phase6_pilot.py` | Serial phase-6 pilot driver and result aggregation. |
| `experiments/path_oracle/run_dual_path.py` | Builds and executes paired route plans. |
| `spec_analyze/` | Specification parsing and feature analysis. |
| `path_select/` | Route-selection logic. |
| `c_gen/` | C-generation and C-first support. |
| `rag_retrieve/` | Retrieval utilities and RAG integration. |
| `module5/` | Candidate editing, verification, synthesis orchestration, and route execution. |
| `pipeline/` | Shared pipeline orchestration and preflight checks. |
| `memory_agent/` | Memory persistence and experience integration. |
| `RTL_DIRECT_compare/` | RTL-direct generation and comparison support. |
| `token_counter/` | Token accounting utilities. |
| `.env.example` | Non-secret configuration template. |
| `README.md` / `README.cn.md` | English and Chinese project documentation. |

## Reproducibility and resume behavior

The runner processes one pair at a time and uses an exclusive serial lock. A plan records the
objective, repeat count, execution policy, MCTS settings, backend configuration, source hashes,
prompt hashes, and Memory-baseline hashes. These records make it possible to identify when two
runs were configured differently.

To resume an interrupted execution, rerun the same command with the same `--output-root` and
compatible inputs. Completed pairs are reused only when the pair metadata, case content,
verification mode, code and prompt fingerprints, and run-configuration hash still match. A
stale or partial pair is refused for automatic reuse; review it manually or select a new output
directory. Do not combine outputs from different objectives or incompatible configurations.

Reproducibility is bounded by the configured LLM, remote service behavior, tool versions,
technology libraries, and backend state. The framework records configuration and content
fingerprints, but it cannot make an external model or proprietary EDA service deterministic.

## Security and operational notes

- Treat `.env` as a secret file. It can contain LLM API keys, SSH-related remote settings, and
  internal filesystem paths.
- Inspect generated logs and JSON before sharing them; prompts, RTL, error text, and remote
  paths may be copied into experiment artifacts.
- Keep test cases, reference RTL, Memory databases, RAG indices, and generated results in
  controlled storage. Do not loosen ignore rules merely to make a local run convenient.
- Confirm the LLM endpoint and model before `--execute`; requests may be billable or subject to
  provider retention policies.
- Confirm remote Design Compiler and JasperGold targets, license availability, and workspace
  permissions before execution.
- Use `--verification-mode none` only for exploratory measurements. Its outputs do not provide
  functional-equivalence evidence and are not training-eligible.
- Use a separate output directory for materially different configurations and preserve the
  recorded configuration hash with any results that are retained.

## License

No project-level license has been provided in this repository. No permission to use, copy,
modify, or redistribute the project should be inferred from its publication or from this
README. Review and add the appropriate licensing terms before distributing the project.
