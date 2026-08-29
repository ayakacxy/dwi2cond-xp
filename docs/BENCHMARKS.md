# Historical benchmarks

This page contains the performance measurement attributable to `v0.1.0`. It is
a retained release record, not a current hardware claim. The 2026-08-29
documentation refresh did not rerun or reinterpret the benchmark.

## DTI fitting

The private HCP input contained b=0 plus the b=1000 shell: 108 volumes on a
`145x174x145` grid with 881,299 masked voxels. Both implementations used the
same server, selected volumes, WLS/gradient-nonlinearity contract, and output
boundary. The boundary included tensor, FA, MD, MO, L1-L3, V1-V3, S0, SSE,
validity mask, and QA JSON.

| Implementation | Execution | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| dwi2cond-xp `v0.1.0` | 16 workers | 9.76 s | 670% | 767,656 KiB |
| FSL 6.0.4 | 1 process | 108.23 s | 99% | 2,023,996 KiB |

The observed wall-time ratio was `11.09x`, and the recorded Python peak RSS was
about 37.9% of the FSL value. These numbers describe one private input, one
server, and only the DTI-fitting/output boundary above. They do not include raw
preprocessing, tensor-to-T1 registration, CHARM, meshing, FEM, or lead-field
generation, and they must not be used as an end-to-end speedup.

The retained record did not capture the exact CPU model, frequency policy,
memory configuration, repeated-sample distribution, or uncertainty interval.
Accordingly, `11.09x` is reported as the observed historical ratio, not as an
expected ratio on other hardware or datasets.
