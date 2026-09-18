# Things we tried

Small experiments on the current code, with the actual result of each. They were run on 18 September 2026 on the submitted revision. All costs are the illustrative model, not hardware. Each one can be repeated with the commands given.

## 1. Can a simple "skip small device islands" rule replace the exhaustive search?

**Why we tried it.** Real runtimes avoid tiny accelerator chunks with a rule of thumb. Torch-TensorRT, for example, has `min_block_size`. Our cost policy tries every placement, which cannot scale. A rule like this would.

**What we did.** Start from maximal placement. Move every run of consecutive device nodes shorter than `k` back to the CPU. Compare the modeled cost with the exhaustive optimum on all 45 verification cases.

```sh
python3 scripts/check_matrix.py
python3 scripts/experiment_min_island.py
```

| k | Matches the optimum | Plan infeasible | Mean excess, ms | Worst excess, ms |
|---:|---:|---:|---:|---:|
| 1 (plain maximal) | 18 / 45 | 9 | 6.6510 | 27.5582 |
| 2 | 21 / 45 | 9 | 3.1752 | 17.0541 |
| 3 | 24 / 45 | 9 | 0.6953 | 4.5999 |

**What happened.** The rule helps on average. It does not replace the search.

- It is wrong in both directions. With `k=1` the worst case keeps bouncing to the device when copies are expensive (`residual-16-expensive-copies`: chose `ACAACA`, best was `CCCCCC`). With `k=3` the worst case throws away useful work (`residual-2-fragmented`: chose `CCCCCC`, best was `ACAACC`).
- That second case is the interesting one. The best plan keeps device islands of length one, because each is a MatMul, the expensive operator. Island *length* is the wrong thing to measure. What the island contains matters more.
- The 9 infeasible cases are the constrained-memory configurations. Dropping small islands does nothing about a plan that does not fit. The rule has no idea about memory.

**Decision.** Keep exhaustive enumeration as the reference for small graphs. A scalable policy should weigh what an island saves against what its boundaries cost, not count nodes. Not implemented.

## 2. Does the correctness check need a tolerance?

**Why we tried it.** The executor accumulates MatMul in `float` and the reference in `double`. We compare with `1e-5 + 1e-5 * |expected|`. We wanted to know if that tolerance is doing anything, or if exact equality would pass.

**What we did.** Changed the comparison in `execute` to `error > 0.0`, rebuilt, ran everything, then reverted.

**What happened.**

- The 45-case matrix still passed: 591 plans, maximum absolute error exactly 0. We did not expect that. The generated inputs are bounded binary fractions, so every intermediate value is exactly representable and float and double agree bit for bit.
- `core_contracts` failed at "random original graph/all placements". That test uses arbitrary random values, and there the two accumulations differ in the last bits.

**Decision.** Keep the tolerance. Also noted: the matrix alone cannot detect float/double divergence, because of how its inputs are generated. The random-value test is what covers that. A version of the matrix with non-dyadic inputs would be a better check. Not done yet.

## 3. What if compute cost grows with tensor size?

**Why we tried it.** Our compute costs are constants per operator. A 16x16 MatMul costs the same as a 2x2 one in the model, which is clearly wrong. We wanted to see what the results look like with even a crude size term.

**What we did.** In `insert_transfers`, multiplied each compute cost by `output elements / 4`, so the 2x2 demo keeps its numbers. Rebuilt, reran the tests, the matrix, and experiment 1. The line is left commented out in `src/core.cpp`.

**What happened.**

- The demo and all three test suites were unchanged, as intended.
- Plain maximal placement matched the optimum in 30 cases instead of 18.
- At widths 8 and 16 the best plan used the device in all 30 cases, including the expensive-copy configurations. Only three width-2 cases still preferred all-CPU.
- The min-island rule from experiment 1 became much worse. With `k=2` the mean excess went from 3.18 ms to 77.61 ms, worst 416.70 ms. Dropping a device island now throws away a large compute saving.

**What this tells us.** Under a size-aware cost, "offloading everything loses" is an effect of small tensors or expensive links. It shrinks quickly as tensors grow. Our headline example is a small-tensor case, and we should say so.

**Decision.** Not kept. The factor is as made up as the constants: MatMul work goes with rows x cols x inner dimension, not output size, and the device and CPU would not scale the same way. This needs measured profiles. Until then the constants stay, with this caveat written down.

## Not tried yet

- A proper FLOP-based compute term (rows x cols x inner for MatMul), separately per backend.
- Freeing device buffers after their last use, and how many more plans become feasible under a tight budget.
- Anything on real hardware.
