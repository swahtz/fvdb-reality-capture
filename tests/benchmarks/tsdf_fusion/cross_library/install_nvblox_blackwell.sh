#!/usr/bin/env bash
# Adapted from tests/benchmarks/tsdf_fusion/cross_library/install_nvblox.sh
# for jswartz's Blackwell workstation:
#   * CONDA_ROOT=/home/jswartz/miniforge3
#   * RTX PRO 6000 Blackwell (sm_120) -> CUDA 12.8 + torch cu128 wheels
#   * -DCMAKE_CUDA_ARCHITECTURES=120
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/jswartz/miniforge3}"
NVBLOX_SRC="${NVBLOX_SRC:-/tmp/nvblox}"
CUDA_STAGED="${CUDA_STAGED:-/tmp/cuda-root}"

echo "=== [1/5] Create conda env 'nvblox' with CUDA 12.8 ==="
if [[ ! -d "${CONDA_ROOT}/envs/nvblox" ]]; then
    "${CONDA_ROOT}/bin/conda" create -n nvblox -c conda-forge -c nvidia -y \
        python=3.11 cuda-toolkit=12.8 cuda-nvcc=12.8 cmake ninja git
fi
NVBLOX_ENV="${CONDA_ROOT}/envs/nvblox"

echo "=== [2/5] Install torch 2.8 + CUDA-12.8 wheel ==="
if ! "${NVBLOX_ENV}/bin/python" -c "import torch; assert torch.version.cuda.startswith('12.8')" 2>/dev/null; then
    "${NVBLOX_ENV}/bin/pip" install 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' \
        --index-url https://download.pytorch.org/whl/cu128
fi

echo "=== [3/5] Stage conda CUDA toolkit as a FindCUDA-friendly prefix ==="
if [[ ! -f "${CUDA_STAGED}/include/cuda_runtime.h" ]]; then
    rm -rf "${CUDA_STAGED}"
    mkdir -p "${CUDA_STAGED}/include"
    for f in "${NVBLOX_ENV}/targets/x86_64-linux/include/"*; do
        ln -sf "$f" "${CUDA_STAGED}/include/$(basename "$f")"
    done
    ln -sf "${NVBLOX_ENV}/lib/python3.11/site-packages/nvidia/nvtx/include/nvtx3" \
        "${CUDA_STAGED}/include/nvtx3"
    ln -sf "${NVBLOX_ENV}/bin" "${CUDA_STAGED}/bin"
    ln -sf "${NVBLOX_ENV}/lib" "${CUDA_STAGED}/lib"
    ln -sf "${NVBLOX_ENV}/lib" "${CUDA_STAGED}/lib64"
    ln -sf "${NVBLOX_ENV}/nvvm" "${CUDA_STAGED}/nvvm"
fi

echo "=== [4/5] Clone + build nvblox C++ (static lib + pybind) ==="
if [[ ! -d "${NVBLOX_SRC}" ]]; then
    git clone --depth 1 https://github.com/nvidia-isaac/nvblox.git "${NVBLOX_SRC}"
fi
if [[ ! -f "${NVBLOX_SRC}/build/nvblox_torch/cpp/libpy_nvblox.so" ]]; then
    export PATH="${NVBLOX_ENV}/bin:${PATH}"
    export CMAKE_PREFIX_PATH="${NVBLOX_ENV}/lib/python3.11/site-packages/torch/share/cmake:${NVBLOX_ENV}"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_STAGED}"
    export CUDAToolkit_ROOT="${CUDA_STAGED}"
    rm -rf "${NVBLOX_SRC}/build" && mkdir "${NVBLOX_SRC}/build" && cd "${NVBLOX_SRC}/build"
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DBUILD_PYTORCH_WRAPPER=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXPERIMENTS=OFF \
        -DBUILD_RENDERER=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_STAGED}" \
        -DCUDAToolkit_ROOT="${CUDA_STAGED}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -Dnvtx3_dir="${CUDA_STAGED}/include/nvtx3" \
        -DPython_EXECUTABLE="${NVBLOX_ENV}/bin/python" \
        -DCMAKE_CXX_FLAGS="-I${CUDA_STAGED}/include" \
        -DCMAKE_CUDA_FLAGS="-I${CUDA_STAGED}/include"
    ninja py_nvblox
fi

echo "=== [5/5] Install nvblox_torch wheel (+ runtime deps) ==="
if ! "${NVBLOX_ENV}/bin/python" -c "import nvblox_torch" 2>/dev/null; then
    export CUDA_VERSION=12
    "${NVBLOX_ENV}/bin/pip" install --no-cache-dir --no-deps \
        transforms3d imageio opencv-python einops nvtx scipy scikit-learn \
        plotly dash flask werkzeug jinja2 markupsafe itsdangerous click \
        blinker retrying importlib-metadata jupyter_dash nbformat narwhals \
        threadpoolctl joblib pillow
    "${NVBLOX_ENV}/bin/pip" install --no-deps --no-cache-dir \
        "${NVBLOX_SRC}/nvblox_torch/"
    # --resume-retries: pypi CDN was flaky for the 400MB open3d wheel.
    "${NVBLOX_ENV}/bin/pip" install --no-cache-dir --resume-retries 30 'open3d==0.18.*'
    # open3d's resolver drags numpy 2.x + matplotlib 3.11 in, but
    # nvblox_torch pins numpy<1.27 and matplotlib<3.11, and opencv 5.x
    # needs numpy>=2. Pin a mutually consistent set.
    "${NVBLOX_ENV}/bin/pip" install --no-cache-dir 'numpy==1.26.4'
    "${NVBLOX_ENV}/bin/pip" install --no-cache-dir --no-deps \
        'opencv-python==4.10.0.84' 'matplotlib==3.10.7'
fi

echo "=== Verifying import ==="
"${NVBLOX_ENV}/bin/python" -c "
import nvblox_torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.sensor import Sensor
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
m = Mapper(voxel_sizes_m=[0.2], integrator_types=[ProjectiveIntegratorType.TSDF])
s = Sensor.from_lidar(1800, 64, 0.4712, 1.0)
print(f'OK -- nvblox_torch {nvblox_torch.__version__} with {m} and {s}')
"
echo "=== Done ==="
