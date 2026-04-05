#!/bin/bash
# Example DSUB directives; replace them with your site-specific scheduler settings if needed.
#DSUB --mpi path/to/mpi
#DSUB -n polarpic_ad_build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}/..}"
POLARPIC_DIR="${POLARPIC_DIR:-${REPO_ROOT}}"
AMREX_SRC="${AMREX_SRC:-${REPO_ROOT}/deps/amrex-24.07-polarpic}"
PICSAR_SRC="${PICSAR_SRC:-path/to/picsar}"
UNR_INCLUDE_DIR="${UNR_INCLUDE_DIR:-path/to/unr/include}"
UNR_LIB_DIR="${UNR_LIB_DIR:-path/to/unr/lib}"

BUILD_ROOT="${BUILD_ROOT:-${REPO_ROOT}/build}"
INSTALL_ROOT="${INSTALL_ROOT:-${REPO_ROOT}/install}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${BUILD_ROOT}" "${INSTALL_ROOT}"
LOG_FILE="${LOG_DIR}/polarpic_ad_build_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

load_module_if_configured() {
  local module_name="$1"

  if ! type module >/dev/null 2>&1; then
    return 0
  fi

  if [[ -z "${module_name}" || "${module_name}" == path/to/* ]]; then
    echo "[INFO] skip placeholder module: ${module_name:-<empty>}"
    return 0
  fi

  module load "${module_name}"
}

if type module >/dev/null 2>&1; then
  module purge || true
fi
load_module_if_configured "${COMPILER_MODULE:-path/to/compiler}"
load_module_if_configured "${MPI_MODULE:-path/to/mpi}"
load_module_if_configured "${CMAKE_MODULE:-path/to/cmake}"
load_module_if_configured "${IO_MODULE:-path/to/io-stack}"

BUILD_VARIANT="${BUILD_VARIANT:-release}"
BUILD_DIR="${BUILD_ROOT}/${BUILD_VARIANT}"
INSTALL_DIR="${INSTALL_ROOT}/${BUILD_VARIANT}"

VERSION_FLAGS="${VERSION_FLAGS:-}"
COMMON_C_FLAGS="${COMMON_C_FLAGS:--O3 -fopenmp -w}"
COMMON_CXX_FLAGS="${COMMON_CXX_FLAGS:--O3 -fopenmp -w}"
EXTRA_C_FLAGS="${EXTRA_C_FLAGS:-}"
EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS:-}"
EXTRA_LINK_FLAGS="${EXTRA_LINK_FLAGS:-}"
C_COMPILER="${C_COMPILER:-${CC:-path/to/c/compiler}}"
CXX_COMPILER="${CXX_COMPILER:-${CXX:-path/to/cxx/compiler}}"

if [[ ! -d "${POLARPIC_DIR}" ]]; then
  echo "[ERROR] polarpic-ad source dir not found: ${POLARPIC_DIR}"
  exit 1
fi

cd "${POLARPIC_DIR}"

cmake -S . -B "${BUILD_DIR}" \
  -DWarpX_amrex_src="${AMREX_SRC}" \
  -DWarpX_picsar_src="${PICSAR_SRC}" \
  -DWarpX_DIMS=3 \
  -DWarpX_MPI=ON \
  -DWarpX_PYTHON=OFF \
  -DWarpX_IPO=ON \
  -DWarpX_FFT=OFF \
  -DWarpX_COMPUTE=OMP \
  -DWarpX_OPENPMD=OFF \
  -DWarpX_QED=OFF \
  -DWarpX_MPI_THREAD_MULTIPLE=ON \
  -DAMReX_BASE_PROFILE=OFF \
  -DAMReX_TRACE_PROFILE=OFF \
  -DAMReX_COMM_PROFILE=OFF \
  -DAMReX_TINY_PROFILE=ON \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCMAKE_C_FLAGS="${COMMON_C_FLAGS} ${VERSION_FLAGS} ${EXTRA_C_FLAGS} -I${UNR_INCLUDE_DIR}" \
  -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS} ${VERSION_FLAGS} ${EXTRA_CXX_FLAGS} -I${UNR_INCLUDE_DIR}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${UNR_LIB_DIR} -lunr ${EXTRA_LINK_FLAGS}"

cmake --build "${BUILD_DIR}" --target install -j

echo "[OK] Build/install finished"
echo "[OK] INSTALL_DIR=${INSTALL_DIR}"
