#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# nvblox-defaults sensitivity analysis (reviewer request).
#
# Re-runs the paper's Table 2 (LiDAR TSDF) and Table 5 (LiDAR ESDF)
# sweeps in two configurations:
#
#   Config A ("paper"):    nvblox lidar max integration distance 60 m,
#                          truncation = 3 x voxel — the workload-matched
#                          setup disclosed in the paper's appendix A.2.
#   Config B ("defaults"): nvblox at its UPSTREAM defaults — 10 m lidar
#                          max integration distance, truncation = 4 x
#                          voxel. fvdb is given the equivalently matched
#                          workload (input points pre-clipped to 10 m,
#                          truncation = 4 x voxel).
#
# Both configs run under --gpu-mem-cap-gb 51.5, which emulates the
# paper's 48 GB RTX 6000 Ada (49140 MiB) on this 96 GB RTX PRO 6000
# Blackwell, so OOM thresholds land at the same memory budget.
#
# Usage:
#   FVDB_PYTHON=... NVBLOX_ENV_PYTHON=... ./run_sensitivity_sweeps.sh [mai_city|kitti|all]

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-${HERE}/../data}"
OUT="${OUT:-${HERE}/../results_sensitivity}"
mkdir -p "${OUT}"

FVDB_PYTHON="${FVDB_PYTHON:?set FVDB_PYTHON to the python with the PR-656 fvdb build}"
export NVBLOX_ENV_PYTHON="${NVBLOX_ENV_PYTHON:?set NVBLOX_ENV_PYTHON to the nvblox env python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MEM_CAP="51.5"           # emulated total GB = paper's 48 GB Ada (49140 MiB)
MAI_ROOT="${DATA}/mai_city/mai_city"
KITTI_ROOT="${DATA}/kitti"

# Voxel sweeps. Config B gets finer tail entries because the 10 m
# range clip shrinks the working set, pushing the OOM threshold finer.
TSDF_VOX_A="0.4 0.2 0.1 0.05 0.03 0.02 0.015 0.0125 0.01"
TSDF_VOX_B="0.4 0.2 0.1 0.05 0.03 0.02 0.015 0.0125 0.01 0.0075 0.005"
KITTI_TSDF_VOX_A="0.4 0.2 0.1 0.05 0.03 0.02"
KITTI_TSDF_VOX_B="0.4 0.2 0.1 0.05 0.03 0.02 0.015 0.01"
ESDF_VOX_A="0.2 0.1 0.05 0.03 0.02"
ESDF_VOX_B="0.2 0.1 0.05 0.03 0.02 0.015 0.01"

run() {
    local name="$1"; shift
    local log="${OUT}/${name}.log"
    echo "===== [$(date +%H:%M:%S)] ${name} ====="
    if "$@" 2>&1 | tee "${log}" | grep -E "OK|FAIL|SKIP|OOM|===|config|fvdb:|nvblox:|note|max-range|gpu-mem-cap"; then :; fi
    echo "===== [$(date +%H:%M:%S)] ${name} done (log: ${log}) ====="
}

mai_city() {
    # --- Table 2 rows: Mai City 700 f LiDAR TSDF ---
    # fvdb and nvblox run in SEPARATE driver invocations: at fine
    # voxels fvdb's cached allocator holds tens of GB, which inside
    # the shared 48 GB emulation would starve the nvblox subprocess.
    for sys in fvdb nvblox; do
        run "mai_city_tsdf_A_${sys}" "${FVDB_PYTHON}" "${HERE}/bench_mai_city.py" \
            --root "${MAI_ROOT}" --sequence 00 --n-frames 700 \
            --systems "${sys}" \
            --voxel-sizes ${TSDF_VOX_A} --truncation-multiplier 3.0 \
            --nvblox-max-integration-distance-m 60 \
            --gpu-mem-cap-gb "${MEM_CAP}" \
            --json-out "${OUT}/mai_city_tsdf_A.${sys}.json"

        run "mai_city_tsdf_B_${sys}" "${FVDB_PYTHON}" "${HERE}/bench_mai_city.py" \
            --root "${MAI_ROOT}" --sequence 00 --n-frames 700 \
            --systems "${sys}" \
            --voxel-sizes ${TSDF_VOX_B} --truncation-multiplier 4.0 \
            --max-range-m 10 \
            --gpu-mem-cap-gb "${MEM_CAP}" \
            --json-out "${OUT}/mai_city_tsdf_B.${sys}.json"
    done

    # --- Table 5 rows: Mai City 700 f LiDAR ESDF ---
    run mai_city_esdf_A "${FVDB_PYTHON}" "${HERE}/bench_esdf_vs_nvblox.py" \
        --root "${MAI_ROOT}" --sequence 00 --n-frames 700 \
        --voxel-sizes-m ${ESDF_VOX_A} --trunc-voxel-multiplier 3.0 \
        --max-distance-voxel-multiplier 10.0 \
        --nvblox-max-integration-distance-m 60 \
        --gpu-mem-cap-gb "${MEM_CAP}" \
        --json-out "${OUT}/mai_city_esdf_A.json"

    run mai_city_esdf_B "${FVDB_PYTHON}" "${HERE}/bench_esdf_vs_nvblox.py" \
        --root "${MAI_ROOT}" --sequence 00 --n-frames 700 \
        --voxel-sizes-m ${ESDF_VOX_B} --trunc-voxel-multiplier 4.0 \
        --max-distance-voxel-multiplier 10.0 \
        --max-range-m 10 \
        --gpu-mem-cap-gb "${MEM_CAP}" \
        --json-out "${OUT}/mai_city_esdf_B.json"
}

kitti() {
    # --- Table 2 rows: KITTI full-sequence LiDAR TSDF ---
    # Per-system invocations for the same reason as mai_city().
    for sys in fvdb nvblox; do
        run "kitti_tsdf_A_${sys}" "${FVDB_PYTHON}" "${HERE}/bench_kitti.py" \
            --root "${KITTI_ROOT}" --sequences 00 02 05 \
            --systems "${sys}" \
            --voxel-sizes ${KITTI_TSDF_VOX_A} --truncation-multiplier 3.0 \
            --nvblox-max-integration-distance-m 60 \
            --gpu-mem-cap-gb "${MEM_CAP}" \
            --json-out "${OUT}/kitti_tsdf_A.${sys}.json"

        run "kitti_tsdf_B_${sys}" "${FVDB_PYTHON}" "${HERE}/bench_kitti.py" \
            --root "${KITTI_ROOT}" --sequences 00 02 05 \
            --systems "${sys}" \
            --voxel-sizes ${KITTI_TSDF_VOX_B} --truncation-multiplier 4.0 \
            --max-range-m 10 \
            --gpu-mem-cap-gb "${MEM_CAP}" \
            --json-out "${OUT}/kitti_tsdf_B.${sys}.json"
    done

    # --- Table 5 rows: KITTI 100-frame-prefix LiDAR ESDF ---
    run kitti_esdf_A "${FVDB_PYTHON}" "${HERE}/bench_esdf_kitti.py" \
        --root "${KITTI_ROOT}" --sequences 00 02 05 --n-frames 100 \
        --voxel-sizes-m ${ESDF_VOX_A} --trunc-voxel-multiplier 3.0 \
        --max-distance-voxel-multiplier 10.0 \
        --nvblox-max-integration-distance-m 60 \
        --gpu-mem-cap-gb "${MEM_CAP}" \
        --json-out "${OUT}/kitti_esdf_A.json"

    run kitti_esdf_B "${FVDB_PYTHON}" "${HERE}/bench_esdf_kitti.py" \
        --root "${KITTI_ROOT}" --sequences 00 02 05 --n-frames 100 \
        --voxel-sizes-m ${ESDF_VOX_B} --trunc-voxel-multiplier 4.0 \
        --max-distance-voxel-multiplier 10.0 \
        --max-range-m 10 \
        --gpu-mem-cap-gb "${MEM_CAP}" \
        --json-out "${OUT}/kitti_esdf_B.json"
}

case "${1:-all}" in
    mai_city) mai_city ;;
    kitti)    kitti ;;
    all)      mai_city; kitti ;;
    *) echo "usage: $0 [mai_city|kitti|all]" >&2; exit 2 ;;
esac
echo "ALL SWEEPS COMPLETE $(date)"
