# NVIDIA GPU run

Status: **passed**

| Step | Status | Seconds | Log |
| --- | --- | ---: | --- |
| preflight | passed | 3.19 | [preflight.log](preflight.log) |
| routing-tests | passed | 2.52 | [routing-tests.log](routing-tests.log) |
| cpu-runtime-tests | passed | 2.98 | [cpu-runtime-tests.log](cpu-runtime-tests.log) |
| numerical-gate | passed | 13.35 | [numerical-gate.log](numerical-gate.log) |
| bench-t128-d256-uniform | passed | 9.03 | [bench-t128-d256-uniform.log](bench-t128-d256-uniform.log) |
| bench-t128-d256-skewed | passed | 9.34 | [bench-t128-d256-skewed.log](bench-t128-d256-skewed.log) |
| bench-t512-d256-uniform | passed | 10.02 | [bench-t512-d256-uniform.log](bench-t512-d256-uniform.log) |
| bench-t512-d256-skewed | passed | 9.18 | [bench-t512-d256-skewed.log](bench-t512-d256-skewed.log) |
| bench-t128-d512-uniform | passed | 9.29 | [bench-t128-d512-uniform.log](bench-t128-d512-uniform.log) |
| bench-t128-d512-skewed | passed | 8.62 | [bench-t128-d512-skewed.log](bench-t128-d512-skewed.log) |
| bench-t512-d512-uniform | passed | 8.43 | [bench-t512-d512-uniform.log](bench-t512-d512-uniform.log) |
| bench-t512-d512-skewed | passed | 8.45 | [bench-t512-d512-skewed.log](bench-t512-d512-skewed.log) |
| profile | passed | 4.88 | [profile.log](profile.log) |

## Benchmark results

| Case | Backend | p50 ms | p95 ms | Speedup vs same-device PyTorch |
| --- | --- | ---: | ---: | ---: |
| [t128-d256-skewed](t128-d256-skewed.json) | pytorch | 0.7213 | 0.7498 | 1.00x |
| [t128-d256-skewed](t128-d256-skewed.json) | grouped | 0.5621 | 0.5816 | 1.28x |
| [t128-d256-skewed](t128-d256-skewed.json) | torch-compiled | 0.7678 | 0.7844 | 0.94x |
| [t128-d256-skewed](t128-d256-skewed.json) | triton | 0.5801 | 0.5908 | 1.24x |
| [t128-d256-uniform](t128-d256-uniform.json) | pytorch | 1.4771 | 1.5103 | 1.00x |
| [t128-d256-uniform](t128-d256-uniform.json) | grouped | 0.8923 | 0.9172 | 1.66x |
| [t128-d256-uniform](t128-d256-uniform.json) | torch-compiled | 1.7238 | 1.7536 | 0.86x |
| [t128-d256-uniform](t128-d256-uniform.json) | triton | 0.7533 | 0.8363 | 1.96x |
| [t128-d512-skewed](t128-d512-skewed.json) | pytorch | 0.8551 | 0.8750 | 1.00x |
| [t128-d512-skewed](t128-d512-skewed.json) | grouped | 0.6297 | 0.6356 | 1.36x |
| [t128-d512-skewed](t128-d512-skewed.json) | torch-compiled | 0.9531 | 0.9849 | 0.90x |
| [t128-d512-skewed](t128-d512-skewed.json) | triton | 0.5943 | 0.6073 | 1.44x |
| [t128-d512-uniform](t128-d512-uniform.json) | pytorch | 1.5662 | 1.5977 | 1.00x |
| [t128-d512-uniform](t128-d512-uniform.json) | grouped | 0.9611 | 0.9776 | 1.63x |
| [t128-d512-uniform](t128-d512-uniform.json) | torch-compiled | 1.7334 | 1.7668 | 0.90x |
| [t128-d512-uniform](t128-d512-uniform.json) | triton | 0.8992 | 0.9259 | 1.74x |
| [t512-d256-skewed](t512-d256-skewed.json) | pytorch | 0.8291 | 0.8563 | 1.00x |
| [t512-d256-skewed](t512-d256-skewed.json) | grouped | 0.6124 | 0.6445 | 1.35x |
| [t512-d256-skewed](t512-d256-skewed.json) | torch-compiled | 0.9107 | 0.9231 | 0.91x |
| [t512-d256-skewed](t512-d256-skewed.json) | triton | 0.5679 | 0.5858 | 1.46x |
| [t512-d256-uniform](t512-d256-uniform.json) | pytorch | 1.5753 | 1.5896 | 1.00x |
| [t512-d256-uniform](t512-d256-uniform.json) | grouped | 0.9610 | 0.9689 | 1.64x |
| [t512-d256-uniform](t512-d256-uniform.json) | torch-compiled | 1.7406 | 1.7552 | 0.91x |
| [t512-d256-uniform](t512-d256-uniform.json) | triton | 0.5908 | 0.6008 | 2.67x |
| [t512-d512-skewed](t512-d512-skewed.json) | pytorch | 0.8273 | 0.9141 | 1.00x |
| [t512-d512-skewed](t512-d512-skewed.json) | grouped | 0.6065 | 0.6211 | 1.36x |
| [t512-d512-skewed](t512-d512-skewed.json) | torch-compiled | 1.1410 | 1.2557 | 0.73x |
| [t512-d512-skewed](t512-d512-skewed.json) | triton | 0.5585 | 0.5756 | 1.48x |
| [t512-d512-uniform](t512-d512-uniform.json) | pytorch | 1.4376 | 1.4924 | 1.00x |
| [t512-d512-uniform](t512-d512-uniform.json) | grouped | 0.8770 | 0.9123 | 1.64x |
| [t512-d512-uniform](t512-d512-uniform.json) | torch-compiled | 1.8222 | 1.9920 | 0.79x |
| [t512-d512-uniform](t512-d512-uniform.json) | triton | 0.5872 | 0.6879 | 2.45x |

Supplied routes through merged output. Router scoring, validation, initial compilation,
and input generation are excluded. All compared backends run on the same device.
The compiled backend compiles expert arithmetic only. Results are synthetic single-layer
measurements, not full-model inference. Raw samples and timing scope are in each JSON.

**This script does not stop or terminate your rental. Download results, then terminate
the instance in the provider console when finished.**
