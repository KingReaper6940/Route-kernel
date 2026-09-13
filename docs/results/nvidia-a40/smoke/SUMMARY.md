# NVIDIA GPU run

Status: **passed**

| Step | Status | Seconds | Log |
| --- | --- | ---: | --- |
| preflight | passed | 3.08 | [preflight.log](preflight.log) |
| routing-tests | passed | 2.20 | [routing-tests.log](routing-tests.log) |
| cpu-runtime-tests | passed | 2.77 | [cpu-runtime-tests.log](cpu-runtime-tests.log) |
| numerical-gate | passed | 17.03 | [numerical-gate.log](numerical-gate.log) |
| bench-small-fp32 | passed | 10.09 | [bench-small-fp32.log](bench-small-fp32.log) |
| bench-medium-fp16 | passed | 11.06 | [bench-medium-fp16.log](bench-medium-fp16.log) |

## Benchmark results

| Case | Backend | p50 ms | p95 ms | Speedup vs same-device PyTorch |
| --- | --- | ---: | ---: | ---: |
| [medium-fp16](medium-fp16.json) | pytorch | 1.4752 | 1.5022 | 1.00x |
| [medium-fp16](medium-fp16.json) | grouped | 0.9164 | 0.9241 | 1.61x |
| [medium-fp16](medium-fp16.json) | torch-compiled | 1.7744 | 1.9564 | 0.83x |
| [medium-fp16](medium-fp16.json) | triton | 0.7517 | 0.8040 | 1.96x |
| [small-fp32](small-fp32.json) | pytorch | 1.1905 | 1.2313 | 1.00x |
| [small-fp32](small-fp32.json) | grouped | 0.7749 | 0.7857 | 1.54x |
| [small-fp32](small-fp32.json) | torch-compiled | 1.5045 | 1.5340 | 0.79x |
| [small-fp32](small-fp32.json) | triton | 0.5638 | 0.5832 | 2.11x |

Supplied routes through merged output. Router scoring, validation, initial compilation,
and input generation are excluded. All compared backends run on the same device.
The compiled backend compiles expert arithmetic only. Results are synthetic single-layer
measurements, not full-model inference. Raw samples and timing scope are in each JSON.

**This script does not stop or terminate your rental. Download results, then terminate
the instance in the provider console when finished.**
