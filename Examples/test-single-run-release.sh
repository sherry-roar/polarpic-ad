#!/bin/bash
# Example DSUB directives; replace them with your site-specific scheduler settings if needed.
#DSUB --mpi path/to/mpi
#DSUB -n polarpic_ad_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}/..}"
INSTALL_ROOT="${INSTALL_ROOT:-${REPO_ROOT}/install}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
INPUT_FILE="${INPUT_FILE:-${REPO_ROOT}/Examples/inputs/lia-3d.modify}"
PROGRAM_VARIANT="${PROGRAM_VARIANT:-release}"
RUN_PROGRAM="${RUN_PROGRAM:-${INSTALL_ROOT}/${PROGRAM_VARIANT}/bin/warpx.3d.MPI.OMP.DP.PDP}"
UNR_LIB_DIR="${UNR_LIB_DIR:-path/to/unr/lib}"
EXTRA_LIBRARY_DIRS="${EXTRA_LIBRARY_DIRS:-}"
MPI_RANKS="${MPI_RANKS:-}"

sanitize_input_file() {
  local input_path="$1"
  local tmp_input="${input_path}.tmp"

  awk '
  {
    if ($1 == "warpx.reduced_diags_names" && $2 == "=") {
      printf "warpx.reduced_diags_names ="
      for (i = 3; i <= NF; ++i) {
        if ($i != "PhaseSpaceElectrons") {
          printf " %s", $i
        }
      }
      printf "\n"
    } else if ($0 ~ /^[[:space:]]*PhaseSpaceElectrons\./) {
      next
    } else if ($0 ~ /^[[:space:]]*[A-Za-z0-9_.-]+[[:space:]]*\.[Ff]ormat[[:space:]]*=[[:space:]]*openpmd([[:space:]]|$)/) {
      next
    } else {
      print
    }
  }
  ' "${input_path}" > "${tmp_input}"
  mv "${tmp_input}" "${input_path}"
}

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

prepend_library_path_if_configured() {
  local library_dir="$1"

  if [[ -z "${library_dir}" || "${library_dir}" == path/to/* ]]; then
    [[ -n "${library_dir}" ]] && echo "[INFO] skip placeholder library dir: ${library_dir}"
    return 0
  fi

  export LD_LIBRARY_PATH="${library_dir}:${LD_LIBRARY_PATH:-}"
}

if type module >/dev/null 2>&1; then
  module purge || true
fi
load_module_if_configured "${COMPILER_MODULE:-path/to/compiler}"
load_module_if_configured "${MPI_MODULE:-path/to/mpi}"
load_module_if_configured "${IO_MODULE:-path/to/io-stack}"

prepend_library_path_if_configured "${INSTALL_ROOT}/${PROGRAM_VARIANT}/lib"
prepend_library_path_if_configured "${UNR_LIB_DIR}"
if [[ -n "${EXTRA_LIBRARY_DIRS}" ]]; then
  IFS=':' read -r -a extra_lib_dirs <<< "${EXTRA_LIBRARY_DIRS}"
  for extra_dir in "${extra_lib_dirs[@]}"; do
    prepend_library_path_if_configured "${extra_dir}"
  done
fi

if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS
fi
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

if [[ ! -x "${RUN_PROGRAM}" ]]; then
  echo "[ERROR] executable not found: ${RUN_PROGRAM}"
  echo "[HINT] run Examples/test-single-build-release.sh first or set RUN_PROGRAM explicitly"
  exit 1
fi
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "[ERROR] input file not found: ${INPUT_FILE}"
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${RUN_TS}-polarpic"
OUT_LOG="${LOG_DIR}/run_${RUN_TS}.out.log"
ERR_LOG="${LOG_DIR}/run_${RUN_TS}.err.log"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cp "${INPUT_FILE}" "${RUN_DIR}/"
cd "${RUN_DIR}"
INPUT_LOCAL="$(basename "${INPUT_FILE}")"
sanitize_input_file "${INPUT_LOCAL}"

MPI_LAUNCHER="$(command -v hmpirun || command -v mpirun || true)"
if [[ -z "${MPI_LAUNCHER}" ]]; then
  echo "[ERROR] neither hmpirun nor mpirun found"
  exit 1
fi

MPI_LAUNCH_ARGS=()
if [[ -n "${MPI_RANKS}" ]]; then
  MPI_LAUNCH_ARGS+=( -n "${MPI_RANKS}" )
fi

MPI_ENV_EXPORTS=( -x OMP_PLACES -x OMP_PROC_BIND )
if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
  MPI_ENV_EXPORTS=( -x OMP_NUM_THREADS "${MPI_ENV_EXPORTS[@]}" )
fi

echo "[RUN] launcher=${MPI_LAUNCHER}"
echo "[RUN] MPI_RANKS=${MPI_RANKS:-<launcher-default>}"
echo "[RUN] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "[RUN] RUN_PROGRAM=${RUN_PROGRAM}"
echo "[RUN] INPUT=${INPUT_LOCAL}"
echo "[RUN] RUN_DIR=${RUN_DIR}"
echo "[RUN] STDOUT=${OUT_LOG}"
echo "[RUN] STDERR=${ERR_LOG}"

tmp_err_fifo="${RUN_DIR}/.run_${RUN_TS}.stderr.fifo"
mkfifo "${tmp_err_fifo}"
tee "${ERR_LOG}" < "${tmp_err_fifo}" >&2 &
TEE_ERR_PID=$!

cleanup() {
  if [[ -n "${TEE_ERR_PID:-}" ]]; then
    wait "${TEE_ERR_PID}" 2>/dev/null || true
  fi
  rm -f "${tmp_err_fifo}"
}
trap cleanup EXIT

"${MPI_LAUNCHER}" "${MPI_LAUNCH_ARGS[@]}" \
  --bind-to none \
  "${MPI_ENV_EXPORTS[@]}" \
  "${RUN_PROGRAM}" "${INPUT_LOCAL}" \
  2> "${tmp_err_fifo}" | tee "${OUT_LOG}"

wait "${TEE_ERR_PID}"
trap - EXIT
rm -f "${tmp_err_fifo}"

echo "[OK] run finished, output dir: ${RUN_DIR}"
echo "STDOUT=${OUT_LOG}"
echo "STDERR=${ERR_LOG}"
