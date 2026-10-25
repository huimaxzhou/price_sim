# Packet D Performance Observations

**Recorded:** 2026-10-25
**Command:** `python tests/performance/benchmark_packet_d.py --case all`
**Policy:** Non-gating; no hardware-dependent threshold

| Case | Dimensions `(M,H,N,K)` | End to end | Diagnostics | Serialization | Read-back | Estimated peak | Observed result |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canonical | `(20,63,50,10)` | 0.052 s | 0.034 s | 0.442 s | 0.051 s | 2,782,240 B | 2,785,440 B |
| Maximum | `(100,252,250,100)` | 3.167 s | 1.605 s | 10.300 s | 2.258 s | 293,965,600 B | 294,009,600 B |

The maximum-shape result remains below the approved 1 GB default memory limit.
Serialization is the largest measured stage. In accordance with the approved
Packet D decision, this observation does not activate compression, batching,
parallelism, compiled extensions, or another invasive optimization.
