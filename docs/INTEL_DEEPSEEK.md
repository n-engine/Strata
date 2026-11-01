# DeepSeek V4.1-Flash on an Intel Arc card (SYCL)

This branch (`dsv41-sycl`) runs Strata's DeepSeek engine, `strata-dsv41`, on an Intel Arc card through SYCL. It is a
community port on top of Strata 0.2.8. Upstream Strata does not ship it: [INTEL.md](INTEL.md) still says DeepSeek does not
run on Intel. Everything here was measured on one machine, listed below. Treat it as a working preview, not a release.

## What it is

- `sycl/src/dsv41/` is the DeepSeek engine (upstream `src/dsv41/` at 0.2.8) built for SYCL through a CUDA-runtime shim
  (`ds4_sycl.hpp`). It starts from the community port in
  [riceharvest/strata-b70-only](https://github.com/riceharvest/strata-b70-only) (branch `b70`), keeps only its Intel fixes
  (no expert pruning, no auto-tune), and merges upstream 0.2.8 on top.
- `strata-dsv41` is an opt-in CMake target (`STRATA_BUILD_DSV41=OFF` by default). `sycl/tools/build.sh strata-dsv41` turns
  it on. No other backend's file is touched: every change is under `sycl/` and `docs/`.
- New switches are off in the engine. `sycl/serve/strata-sycl.sh` turns the measured ones on for you (see "Run").

## Measured on

Arc Pro B60 24 GB (Battlemage, `xe` driver, PCIe 3.0 x8), 2x Xeon E5-2699 v4 (88 threads, 2 NUMA nodes), 314 GB DDR4-2133,
local NVMe. oneAPI 2026.1 in the `strata-sycl-dev` image, AOT build for `bmg-g21`. Model: ISTA-DASLab
DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XXS (2.00 bpw, 138 GB), 6 experts per word, 80 GB RAM store, 4 GB of VRAM kept free.
Dates: 2026-10-31 and 2026-11-01. Other programs shared the machine; every speed-up below comes from runs alternated in
the same slot, never from runs on different days.

| What | Result |
| --- | --- |
| Prompt, 9,352 tokens, before the switches below | 134 to 139 tokens/s |
| The same prompt with `DS4_XMX=1` | 213 tokens/s (2 of 2 runs) |
| ... plus `DS4_FAST_ATTN=1` | 240 tokens/s (2 of 2) |
| ... plus `DS4_COPY_ENGINE=1` | 282 to 285 tokens/s (2 of 2) |
| Prompt, 16,015 tokens, all switches | 353 tokens/s |
| Prompt, 64K and 128K tokens, all switches | 414 tokens/s (154 s and 309 s) |
| Output after a short prompt, quiet machine | 10.7 to 11.1 tokens/s |
| Output after a 16K prompt | 9.9 to 10.2 tokens/s |
| Output with the DSpark drafter (`--dspark on`), code prompt | 10.9 to 11.5 tokens/s, against 9.9 without (+14%) |
| Output, 384 tokens, short prompt, RAM store NUMA-interleaved | 14.97 tokens/s (interleave: +24% on the same binary) |
| A conversation's next turn (prefix reuse, `--serve`) | first word after 1.0 to 1.7 s instead of 87.5 s |

Long context: a note placed 2,000 tokens into a 64K and a 128K prompt was found and quoted exactly both times. The
attention cache took 0.06 GB at 64K and 0.12 GB at 128K.

## Correctness

Logits against the official CUDA engine 0.1.54 (RTX 3070 Ti), 149 positions, the same tokens fed to both (`--n 0
--dump-logits`):

| Pair | Top-1 agreement | Mean KL |
| --- | ---: | ---: |
| CUDA against this port | 100% / 99.3% | 0.0022 / 0.0024 |
| This port against itself (run-to-run noise) | 98.0% to 99.3% | 0.0016 to 0.0017 |
| CUDA against `DS4_XMX=1` | 99.3% | 0.0017 to 0.0018 |
| CUDA against `DS4_XMX=1 DS4_FAST_ATTN=1` | 99.3% | 0.0018 |
| CUDA against all three switches | 97.3% / 99.3% | 0.0018 / 0.0022 |

Upstream's gate for its own matrix-core attention (`DS4_FAST_ATTN` on AMD) is a mean KL of 0.006 or less. Every line above
is under it and inside this port's run-to-run noise. The kernels also have GPU parity tests: `ds4_xmx_parity` (126 of 126
cases, IQ1_M, IQ2_XXS and IQ2_XS against the FMA kernel and an FP64 host reference) and `ds4_attn_parity` (54 of 54 cases).

## Build

In the `strata-sycl-dev` image, never with a host oneAPI of another version:

```sh
DSV41=ON AOT=bmg-g21 sycl/tools/build.sh strata-dsv41      # and ds4_xmx_parity, ds4_attn_parity for the tests
```

## Run

```sh
STRATA_SYCL_BIN=build-sycl-aot/strata-dsv41 STRATA_SYCL_MOUNTS=/path/to/model \
  bash sycl/serve/strata-sycl.sh --pack /path/to/model/maya-ds4 --gpus 0 --max-context 32768 --serve
```

For `strata-dsv41`, `strata-sycl.sh` picks these from the machine and prints each choice with its reason. A variable you
set yourself always wins, and `STRATA_SYCL_AUTO=0` turns the picks off.

| Machine | Pick |
| --- | --- |
| The binary is an AOT build for Battlemage (`bmg-*`) | `DS4_XMX=1`, `DS4_FAST_ATTN=1`, `DS4_COPY_ENGINE=1` |
| More than one NUMA node | `DS4_STORE_NUMA=interleave` (and the container's `SYS_NICE` capability) |
| Free RAM | `DS4_STORE_GB` = a third of it, 80 GB at most, never the whole model |
| `--max-context` | `--prefill-chunk` = the same, 16,384 at most, unless you give one |

The DSpark drafter needs a pack that includes it. Build it with setup's recipe (`tools/dsv41/pack_gguf.py --dspark-gguf
... --dspark-noise-token 128799 --dspark-markov-rank 256`), then add `--dspark on`.

## The switches

| Switch | What it does |
| --- | --- |
| `DS4_XMX=1` | the routed experts' prompt tiles on the XMX units (joint_matrix, FP16 in, FP32 accumulate). On SYCL these fell back to plain FMA tiles, where CUDA uses wmma and HIP its matrix cores |
| `DS4_FAST_ATTN=1` | the chunked prompt's attention on the XMX units, a port of upstream's AMD `sparse_attn_wmma_kernel` |
| `DS4_COPY_ENGINE=1` | the prompt's expert copies on the card's copy engine. The Level Zero v2 adapter runs every in-order queue on the compute engine, so the copies and the kernels ran one after the other (a device timeline showed 16 s of copies and 22 s of kernels in a 39 s prompt). Uses the Qwen port's `make_copy_engine_queue` |
| `DS4_STORE_NUMA=interleave` | spreads the RAM store over every NUMA node (`mbind`) |
| `DS4_FLOW_STAGE=N` | N times more pinned staging slots. Measured: no speed change; kept for diagnosis |
| `DS4_FLOW_WATCHDOG=S`, `DS4_FLOW_POLL=1/2` | diagnosis of a stalled prompt: report a wait longer than S seconds; poll instead of a blocking wait |

## Known limits

- One card, one machine. Not tried: a B70 or B65, two cards, Windows, the other DeepSeek sets.
- A long prompt hung 6 times in 11 runs on 2026-10-31 (a host wait on a copy event that never completed). It has not come
  back in dozens of runs since, including 24 long prompts in a row over `--serve`, so the cause is not known. If you see
  it, run with `DS4_FLOW_WATCHDOG=60` and check `kernel.numa_balancing`: an open upstream report (#4867) ties random GPU
  stalls on AMD to NUMA balancing scanning registered host memory.
- Each prompt pass streams all the routed experts to the card once, about 21 s here whatever the length. A message of a
  few hundred tokens pays it in full.
- Output speed depends on the memory bandwidth the CPU part gets. Another program reading memory hard cost 21% to 27% here.
- A short answer right after a 64K or 128K prompt came out at about 2.4 tokens/s (28 tokens in 12 s). Not understood yet.
