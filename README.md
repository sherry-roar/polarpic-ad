# POLAR-PIC AD

[中文说明 / Chinese README](README.zh.md)

## Overview

`polarpic-ad` is a downstream research codebase derived from WarpX and maintained as the artifact-oriented continuation of the MatrixPIC / MatrixPIC_AD line of work.

This repository keeps the source paths, dependency snapshots, and example scripts needed to study the implementation behind the POLAR-PIC paper:

**POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication**

Current paper status:
- Accepted by **HPDC '26 / The 35th ACM International Symposium on High-Performance Parallel and Distributed Computing**
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3806645.3807574-blue.svg)](https://doi.org/10.1145/3806645.3807574)
[![arXiv](https://img.shields.io/badge/arXiv-2604.19337-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.19337)


This repository is not an official WarpX mirror and should be treated as a downstream research branch with platform-specific optimization paths.

## Relationship to MatrixPIC and upstream WarpX

- Upstream simulation framework: WarpX
- Direct predecessor artifact: [MatrixPIC_AD](https://github.com/sherry-roar/MatrixPIC_AD/tree/main)
- Prior paper: [Matrix-PIC: Harnessing Matrix Outer-product for High-Performance Particle-in-Cell Simulations](https://arxiv.org/abs/2601.08277)
- Prior ACM DOI: [10.1145/3767295.3769378](https://doi.org/10.1145/3767295.3769378)

Compared with the MatrixPIC artifact, `polarpic-ad` extends the optimization focus from matrixized compute kernels to a broader co-design space that also includes particle layout management and communication orchestration.

## Repository scope

This repository keeps the code paths that are most relevant to the published downstream implementation:

- matrixized / physically ordered particle push path,
- custom order-3 deposition path,
- communication overlap hooks and one-sided communication integration,
- downstream AMReX metadata changes required by physical sorting and incremental sorting,
- sanitized build/run templates for artifact-style reuse.

## Repository layout

```text
.
├── Source/                     # Core WarpX-derived C++ implementation
├── Examples/                   # Example inputs and portable build/run templates
├── Python/                     # Python bindings and packaging files inherited from WarpX
├── Docs/                       # Inherited documentation snapshot
├── Tools/                      # Utility scripts and developer helpers
├── Regression/                 # Regression-related assets
├── deps/
│   ├── amrex-24.07-polarpic/   # Vendored AMReX 24.07 snapshot with downstream changes
│   └── UNR_AD/                 # Vendored platform-specific one-sided communication snapshot
├── CMakeLists.txt
├── GNUmakefile
├── LICENSE.txt
├── NOTICE.txt
├── LEGAL.txt
├── README.md
└── README.zh.md
```

## Dependencies

### `deps/amrex-24.07-polarpic`

This repository vendors an AMReX snapshot based on **AMReX 24.07**.

The downstream AMReX-side modifications currently retained in this artifact are centered on:

- `deps/amrex-24.07-polarpic/Src/Particle/AMReX_ParticleTile.H`

That file carries the metadata and initialization logic needed by the downstream physical-sorting path, including the sorting-state helpers used by:

- `init_phys_sort(...)`
- `init_incr_sort(...)`

In practical terms, these changes support gather-side physical sorting metadata, incremental sorting state, and the particle-tile bookkeeping expected by the retained POLAR-PIC data-layout path.

### `deps/UNR_AD`

`deps/UNR_AD` is kept as a vendored snapshot of a **platform-specific one-sided communication backend** used by this repository.

For UNR-specific questions, please refer to the original open-source note in:
- [UNR_AD original repository note](https://github.com/NoBugEveryDay/UNR_AD#%E5%8E%9F%E5%A7%8B%E5%BC%80%E6%BA%90%E4%BB%93%E5%BA%93)

If your target platform does not use this exact backend, the communication layer is intended to be replaceable in principle. For example, an `unr put` style one-sided transfer can be replaced by an MPI-based equivalent such as `mpi put`, as long as the surrounding synchronization semantics are preserved.

### External inputs expected by the release scripts

The published release scripts intentionally keep some dependencies external and configurable:

- `PICSAR_SRC`: local PICSAR source tree
- `UNR_INCLUDE_DIR`: headers for the communication backend in use
- `UNR_LIB_DIR`: libraries for the communication backend in use
- `COMPILER_MODULE`, `MPI_MODULE`, `CMAKE_MODULE`, `IO_MODULE`: optional site-specific environment modules
- `C_COMPILER`, `CXX_COMPILER`: concrete compiler paths when modules are not sufficient
- `VERSION_FLAGS`: platform-specific feature flags/macros used to enable the desired VPU/MPU build path

## Main downstream-modified files

The most relevant downstream-edited files in this artifact are:

- `Source/Particles/PhysicalParticleContainer.H`
- `Source/Particles/PhysicalParticleContainer.cpp`
- `Source/Particles/WarpXParticleContainer.cpp`
- `Source/Particles/Deposition/CurrentDeposition.H`
- `Source/Particles/MultiParticleContainer.H`
- `Source/Particles/MultiParticleContainer.cpp`
- `Source/Evolve/WarpXEvolve.cpp`
- `Source/Initialization/WarpXInit.cpp`
- `Source/Initialization/WarpXInitData.cpp`
- `Source/WarpX.cpp`
- `Source/WarpX.H`
- `deps/amrex-24.07-polarpic/Src/Particle/AMReX_ParticleTile.H`
- `Examples/test-single-build-release.sh`
- `Examples/test-single-run-release.sh`

At a high level, these files cover the retained custom push/deposition path, communication overlap hooks, initialization/evolution wiring, downstream particle-layout metadata, and portable artifact scripts.

## Key runtime flow and main functions

The main control flow of the retained implementation can be read from the following entry points:

1. `Source/main.cpp` → `main(...)`
2. `Source/Initialization/WarpXInitData.cpp` → `WarpX::InitData()`
3. `Source/Evolve/WarpXEvolve.cpp` → `WarpX::Evolve(int numsteps)`
4. `Source/Particles/MultiParticleContainer.cpp` → `MultiParticleContainer::Evolve(...)`

The main retained downstream compute / layout / communication functions include:

- `PhysicalParticleContainer::PushPX_vpu_mpu_physort_order3(...)`
- `doDepositionShapeN_vpu_rhocell_mpu_physort_order3(...)`
- `MultiParticleContainer::fusion_unr_put()`
- `MultiParticleContainer::fusion_unr_wait()`
- `MultiParticleContainer::fusion_remote_collect(int lev)`
- `MultiParticleContainer::fusion_unr_remote_collect(int lev)`
- `PhysicalParticleContainer::fusion_unr_put()`
- `PhysicalParticleContainer::fusion_unr_wait()`
- `PhysicalParticleContainer::fusion_remote_collect(int lev)`
- `PhysicalParticleContainer::fusion_unr_remote_collect(int lev)`
- `init_phys_sort(...)`
- `init_incr_sort(...)`

A useful mental model is:

- `WarpX::InitData()` builds the simulation state,
- `WarpX::Evolve()` drives the timestep loop,
- `MultiParticleContainer::Evolve(...)` coordinates species-level particle evolution,
- `PushPX_vpu_mpu_physort_order3(...)` and `doDepositionShapeN_vpu_rhocell_mpu_physort_order3(...)` implement the retained optimized compute/layout path,
- `fusion_*` functions connect that path to the communication side.

## Examples and how to use the repository

### Example inputs

- `Examples/inputs/lia-3d.modify`
  - The real physical LWFA simulation configuration used in the paper.
  - Kept as the default published input in this repository.
  - Uses the retained direct-deposition, shape-order-3 downstream path and serves as the primary reference input for reproducing the paper-oriented workflow.

- `Examples/inputs/test-uni-16`
  - A larger and more complete 3D research-style configuration.
  - Better suited for fuller experimental runs after the environment is correctly configured.

### Build script

Use `Examples/test-single-build-release.sh` as the published build template.

What you usually need to replace before building:

- `PICSAR_SRC`
- `UNR_INCLUDE_DIR`
- `UNR_LIB_DIR`
- `COMPILER_MODULE`, `MPI_MODULE`, `CMAKE_MODULE`, `IO_MODULE` if your cluster uses modules
- `C_COMPILER`, `CXX_COMPILER` if your site needs explicit compiler paths
- `VERSION_FLAGS` if your platform requires specific vector/matrix feature macros or code-generation flags

Typical usage pattern:

```bash
export PICSAR_SRC=/path/to/picsar
export UNR_INCLUDE_DIR=/path/to/unr/include
export UNR_LIB_DIR=/path/to/unr/lib
export COMPILER_MODULE=path/to/compiler
export MPI_MODULE=path/to/mpi
export CMAKE_MODULE=path/to/cmake
export VERSION_FLAGS="<platform-specific flags>"

bash Examples/test-single-build-release.sh
```

By default, the script uses repository-relative `build/`, `install/`, and `logs/` directories.

### Run script

Use `Examples/test-single-run-release.sh` as the published run template.

What you usually need to review or replace before running:

- `RUN_PROGRAM`
- `INPUT_FILE`
- `MPI_RANKS`
- `OMP_NUM_THREADS`
- `UNR_LIB_DIR`
- `EXTRA_LIBRARY_DIRS`
- scheduler directives at the top of the script if your site uses batch submission

Typical usage pattern:

```bash
export RUN_PROGRAM=/path/to/install/release/bin/warpx.3d.MPI.OMP.DP.PDP
export INPUT_FILE=$PWD/Examples/inputs/lia-3d.modify
export MPI_RANKS=1
export OMP_NUM_THREADS=1
export UNR_LIB_DIR=/path/to/unr/lib

bash Examples/test-single-run-release.sh
```

The run script copies the selected input into a run directory, sanitizes some optional output-format settings for portability, and writes stdout/stderr logs into repository-relative `logs/` and `runs/` paths by default.

## Platform adaptation note

The retained optimized source path uses the terms `vpu` and `mpu` as placeholders for the target platform's vector/matrix execution path.

When porting this repository to a different machine or compiler stack, you should expect to adapt:

- platform-specific vector/matrix macros,
- architecture-specific compiler flags,
- backend library paths,
- any build-time options that map the retained `vpu` / `mpu` path onto your local hardware capabilities.

In other words, the source-level algorithmic path is preserved here, but the exact low-level mapping remains platform-dependent.

## License and ownership boundary

- The upstream WarpX license text is retained in [`LICENSE.txt`](LICENSE.txt).
- This repository also contains downstream modifications maintained for the POLAR-PIC artifact.
- Those downstream modifications are additionally marked as portions copyrighted by **Yizhuo Rao**.
- Please also review [`NOTICE.txt`](NOTICE.txt) and [`LEGAL.txt`](LEGAL.txt) for repository-wide notice files inherited from the WarpX-derived codebase.

If you plan to redistribute derived packages or reuse the downstream modifications outside the original WarpX context, review the retained upstream license text and the repository notices together.

## Contributing and questions

This repository is published primarily as a research artifact and downstream code snapshot.

For repository-specific issues, please use the project review / issue workflow associated with this repository.
For UNR backend questions, prefer the original UNR_AD public note linked above.
