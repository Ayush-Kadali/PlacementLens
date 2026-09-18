# PlacementLens

An explainable compiler pipeline for placing AI graph operators on a CPU and an accelerator. SegFault 2026, open theme *Explainable compilers*, team Boundary Conditions.

An accelerator usually supports only part of a model. Sending it every operator it supports can split the graph, and the copies and synchronizations at each boundary can cost more than the compute saved. PlacementLens compiles small matrix graphs into explicit CPU/device command plans, executes them on a functional software device, checks the result, and records why each compiler step did what it did.

```sh
bash scripts/run_demo.sh
build/placementlens explain --graph examples/fragmented.plg --report artifacts/segfault/run-report.md
```

The report shows validation, fusion decisions, capability checks, placement alternatives, transfer insertion, buffer allocations, cost components, and correctness results. `verify` executes every feasible placement. Tests include a 45-case matrix with 591 placement executions across chain, shared-intermediate, and residual-style graphs.

Accelerator costs in this version are illustrative; the virtual device executes in software on the host.

More commands to try:

```sh
build/placementlens --help
build/placementlens compare --graph examples/fragmented.plg
build/placementlens compare --graph examples/fragmented.plg --relu-supported
build/placementlens compare --graph examples/fragmented.plg --boundary-ms 5
build/placementlens explain --graph examples/fragmented.plg --device-bytes 80
build/placementlens explain --graph examples/branched.plg --relu-supported --fuse
build/placementlens verify  --graph examples/branched.plg --relu-supported
ctest --test-dir build --output-on-failure
```

[EXPERIMENTS.md](EXPERIMENTS.md) has the things we tried on this code and what happened, including the ones that did not work out.

## Status

| | |
|---|---|
| Built and tested | Graph language and validation, guarded MatMul/ReLU fusion, capability analysis, three placement policies, transfer insertion, lowering to commands, checked runtime, reference evaluator, per-pass explanation records, CLI, 45-case verification matrix |
| First version only | Functional software device (correctness and protocol rules, no timing behaviour); analytical cost model with hand-set constants; per-command host timing; memory accounting without frees |
| Planned, not built | Calibrated profiles, a measured oracle, placement regret, an uncertainty-aware policy, asynchronous queues, a real edge backend, ONNX import |

Costs are illustrative. No accelerator speedup is claimed.

## The larger project

PlacementLens is the first working slice of a larger project: an explainable, hardware-aware runtime for heterogeneous edge-AI execution. That project asks how often "offload every supported operator" is the wrong decision on edge devices, and by how much, measured against an exhaustively measured best plan. The roadmap below is not a claim that these items exist in this repository:

- a C++ host runtime with explicit buffers, commands, queues, events and failure handling;
- a configurable functional accelerator simulator with explicit device memory;
- calibrated cost profiles, a measured oracle for small graphs, and placement regret per policy;
- deterministic traces and profiling, and a view that links placement, memory, commands and timelines;
- a bounded comparison on a real accelerator backend.

Development is laptop-first so that building and testing never depend on proprietary hardware. Real hardware is a validation backend, not a prerequisite.

## Build

Requires a C++17 compiler, CMake 3.16+, and Python 3 for the full checks.
