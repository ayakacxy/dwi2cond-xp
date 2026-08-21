# Benchmarks

## DTI fitting

The private HCP b0+b1000 input contained 108 volumes on a `145x174x145` grid
with 881,299 masked voxels. The output boundary included tensor, FA, MD, MO,
L1-L3, V1-V3, S0, SSE, validity mask, and QA JSON.

| Implementation | Workers | Wall time | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| dwi2cond-xp Python | 16 | 9.76 s | 670% | 767,656 KiB |
| FSL 6.0.4 | 1 process | 108.23 s | 99% | 2,023,996 KiB |

The same server, DWI, shell selection, WLS/gradient-nonlinearity semantics, and
output set were used. The observed wall-time ratio was `11.09x`; Python peak RSS
was approximately 37.9% of FSL's. This is one system and one DTI-fitting input.
It must not be extrapolated to raw preprocessing, registration, meshing, FEM, or
other hardware.

The run used a many-core server, but the frozen evidence record did not capture
the exact CPU model. Consequently no model-specific performance claim is made.
