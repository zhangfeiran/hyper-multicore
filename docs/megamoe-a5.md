# Ascend 950 MegaMoE and PanGu integration

The standalone package exposes `MegaMoeExperts` and `MXFP8MegaMoeExperts` from
`hyper_multicore`. The Ascend 950 port combines the current MegaMoE implementation,
indexed ingress and backward, shared experts, MXFP8, and hot replicas with the
standalone repository's dynamic-token and resource-lifecycle implementation.
The source baseline is Hyper-Parallel commit `62ba15e0691918fb03b97e910c7813e59f3e6d2d`;
PanGu integration is based on `7d7b7fb701f320fa56e7ef2cf9d25105a6aa9b0f`.
Hyper-Parallel may still be needed separately for PanGu's FSDP backend.

## Build

Use the selected CANN and matching Torch/torch_npu environment. The default universal payload supports 950 and
910B/910C using separate SHMEM SDKs for each device architecture. Select
`--soc-list ascend950` to build only the A5 payload.

```bash
source /path/to/cann/set_env.sh
bash build.sh --soc-list ascend950 --jobs 16
python -m pip install --no-build-isolation -e .
source build/native/payload/hyper_multicore/lib/set_env.bash
```

The native payload records its dependencies and hashes, and contains the
selected adapter pair, private SHMEM runtime, vendor kernels and the MXFP8
capability receipt. The editable installation also builds the CPU replica
planner extension. A complete payload is required for actual NPU execution.
SHMEM follows the master baseline at upstream v1.8.0, which already includes
the explicit Mesh fabric, UDMA queue publication and Vector Host RMA fixes.
The private bootstrap disconnect adapter is retained. The dependency lock covers ops-nn, ops-transformer and the 950
nlohmann JSON header as well.

For an uninstalled source tree, put the standalone checkout, the required
Hyper-Parallel checkout, and PanGu checkout on `PYTHONPATH`. PanGu's MegaMoE
backend imports `hyper_multicore` directly; its recompute scope also preserves
the legacy Hyper context when that package is present. The PanGu benchmark and
replica preflight accept separate `MULTICORE_ROOT` and `HYPER_ROOT` settings.
Historical multi-node benchmark and replica autocalibration scripts that still
consume `hyper_parallel.core.multicore` are not part of the migrated runtime;
the replica bundle rejects legacy autocalibration rather than calibrating the
wrong implementation.

The runtime accepts the existing `HYPER_PARALLEL_SHMEM_*` settings as well as
`HYPER_MULTICORE_SHMEM_*` for heap size, timeout, bootstrap endpoint, data
engine, and UDMA QP count. If both names for one setting are supplied with
different values, initialization fails. A reduced UDMA QP count requires
`mte_udma`. Existing `HYPER_PARALLEL_REPLICA_*` and MXFP8 tuning settings
retain their original names.

## Validation scope

The 16-rank, 8-layer, seq4096, MBS1, GAS16, balanced MXFP8 Trainer baseline on
the original source measured B0 at 6024.954 ms versus native at 6504.828 ms:
7.377% lower step time, or 1.07965 times throughput. Native permute and
unpermute fusion were enabled, and runtime observers remained enabled. This is
a cropped-model measurement, not an EP64 cluster or convergence guarantee.
It is evidence for the original source and does not by itself certify the
standalone port.

The port has passed CPU tests and locked-source assembly. Its Ascend 950 native
build, multi-rank tensor checks, and matched Trainer performance must be verified
on the intended DT environment before treating the migration as complete.
