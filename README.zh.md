# polarpic-ad

[English README](README.md)

## 项目概述

`polarpic-ad` 是一个基于 WarpX 下游演化的研究代码仓库，可视为 MatrixPIC / MatrixPIC_AD 这条工作线的后续 artifact 版本。

本仓库保留了理解 POLAR-PIC 论文实现所需的核心源码路径、依赖快照和示例脚本。对应论文为：

**POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication**

当前论文状态：
- 已被 **HPDC '26 / The 35th ACM International Symposium on High-Performance Parallel and Distributed Computing** 接收
- 论文链接：**TBD**
- arXiv 链接：**TBD**
- 代码 / artifact 首页：**TBD**

本仓库不是 WarpX 官方镜像，而是带有平台特定优化路径的下游研究分支。

## 与 MatrixPIC 及上游 WarpX 的关系

- 上游模拟框架：WarpX
- 直接前作 artifact：[MatrixPIC_AD](https://github.com/sherry-roar/MatrixPIC_AD/tree/main)
- 前作论文：[Matrix-PIC: Harnessing Matrix Outer-product for High-Performance Particle-in-Cell Simulations](https://arxiv.org/abs/2601.08277)
- 前作 ACM DOI：[10.1145/3767295.3769378](https://doi.org/10.1145/3767295.3769378)

相较于 MatrixPIC artifact，`polarpic-ad` 进一步把优化关注点从矩阵化计算核扩展到了更完整的协同设计范围，包括粒子布局管理与通信编排。

## 仓库范围

本仓库当前保留的内容主要服务于公开后的下游实现理解与复现实验：

- 矩阵化 / 物理排序粒子 push 主路径，
- 自定义 order-3 deposition 主路径，
- 通信重叠钩子与单边通信接线，
- 物理排序与增量排序所需的 AMReX 下游元数据改动，
- 经过脱敏处理的构建 / 运行模板脚本。

## 仓库结构

```text
.
├── Source/                     # 基于 WarpX 的核心 C++ 实现
├── Examples/                   # 示例输入与可移植构建/运行模板
├── Python/                     # 继承自 WarpX 的 Python 绑定与打包文件
├── Docs/                       # 继承文档快照
├── Tools/                      # 工具脚本与开发辅助文件
├── Regression/                 # 回归相关资源
├── deps/
│   ├── amrex-24.07-polarpic/   # 带下游改动的 AMReX 24.07 vendored 快照
│   └── UNR_AD/                 # 平台特定单边通信后端快照
├── CMakeLists.txt
├── GNUmakefile
├── LICENSE.txt
├── NOTICE.txt
├── LEGAL.txt
├── README.md
└── README.zh.md
```

## 依赖说明

### `deps/amrex-24.07-polarpic`

本仓库 vendored 了一份基于 **AMReX 24.07** 的快照。

当前 artifact 中保留的 AMReX 下游改动主要集中在：

- `deps/amrex-24.07-polarpic/Src/Particle/AMReX_ParticleTile.H`

这个文件中保留了下游物理排序路径所需的元数据与初始化逻辑，特别是以下辅助函数所依赖的数据结构：

- `init_phys_sort(...)`
- `init_incr_sort(...)`

从功能上说，这些改动为物理排序路径提供了 gather-side sorting metadata、增量排序状态以及粒子 tile bookkeeping。

### `deps/UNR_AD`

`deps/UNR_AD` 是本仓库保留的一份 **平台特定单边通信后端** vendored 快照。

如果你关心 UNR 本身的问题，请优先参考：
- [UNR_AD 原始开源仓库说明](https://github.com/NoBugEveryDay/UNR_AD#%E5%8E%9F%E5%A7%8B%E5%BC%80%E6%BA%90%E4%BB%93%E5%BA%93)

如果你的目标平台不使用这一套后端，实现原则上可以替换为其他等价通信实现。例如 `unr put` 这类单边传输，在保持外围同步语义不变的前提下，可以替换成 `mpi put` 等 MPI 版本。

### 发布脚本默认假设的外部依赖

公开版 release 脚本刻意把部分依赖保留为外部可配置输入：

- `PICSAR_SRC`：本地 PICSAR 源码路径
- `UNR_INCLUDE_DIR`：当前通信后端头文件路径
- `UNR_LIB_DIR`：当前通信后端库路径
- `COMPILER_MODULE`、`MPI_MODULE`、`CMAKE_MODULE`、`IO_MODULE`：站点环境模块
- `C_COMPILER`、`CXX_COMPILER`：明确的编译器路径
- `VERSION_FLAGS`：用于开启目标 VPU/MPU 路径的平台特定宏与编译选项

## 论文相关的主要修改文件

当前 artifact 中最值得关注的下游修改文件包括：

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

这些文件共同覆盖了保留的 custom push / deposition 路径、通信重叠钩子、初始化与演化接线、下游粒子布局元数据，以及可移植 artifact 脚本。

## 关键执行流程与主要函数

保留实现的主要控制流可以从下面这些入口函数阅读：

1. `Source/main.cpp` → `main(...)`
2. `Source/Initialization/WarpXInitData.cpp` → `WarpX::InitData()`
3. `Source/Evolve/WarpXEvolve.cpp` → `WarpX::Evolve(int numsteps)`
4. `Source/Particles/MultiParticleContainer.cpp` → `MultiParticleContainer::Evolve(...)`

当前保留的关键 compute / layout / communication 函数包括：

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

可以把这条主线理解成：

- `WarpX::InitData()` 负责搭建模拟初始状态，
- `WarpX::Evolve()` 负责驱动时间步推进，
- `MultiParticleContainer::Evolve(...)` 负责 species 级粒子演化协同，
- `PushPX_vpu_mpu_physort_order3(...)` 和 `doDepositionShapeN_vpu_rhocell_mpu_physort_order3(...)` 对应保留的优化 compute/layout 主路径，
- `fusion_*` 系列函数把这条主路径接到通信侧。

## Examples 与使用方法

### 示例输入

- `Examples/inputs/lia-3d.modify`
  - 这是论文中使用的真实物理 LWFA 仿真配置，也是当前仓库默认公开的输入文件。
  - 保留了当前下游主路径使用的 direct deposition 与 shape order 3 设置。
  - 它不是额外的小规模测试参数，而是复现实验工作流时优先参考的主输入配置。

- `Examples/inputs/test-uni-16`
  - 一个更完整、更接近研究配置的 3D 输入。
  - 更适合在环境完全配置正确后做较完整的实验运行。

### 构建脚本

公开版构建模板为 `Examples/test-single-build-release.sh`。

在实际构建前，通常需要先替换或确认：

- `PICSAR_SRC`
- `UNR_INCLUDE_DIR`
- `UNR_LIB_DIR`
- `COMPILER_MODULE`、`MPI_MODULE`、`CMAKE_MODULE`、`IO_MODULE`
- `C_COMPILER`、`CXX_COMPILER`
- `VERSION_FLAGS`（如果你的平台需要特定 vector/matrix 宏或代码生成选项）

典型使用方式：

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

脚本默认使用仓库相对路径下的 `build/`、`install/`、`logs/` 目录。

### 运行脚本

公开版运行模板为 `Examples/test-single-run-release.sh`。

运行前通常需要重点检查或替换：

- `RUN_PROGRAM`
- `INPUT_FILE`
- `MPI_RANKS`
- `OMP_NUM_THREADS`
- `UNR_LIB_DIR`
- `EXTRA_LIBRARY_DIRS`
- 脚本头部的调度器参数（如果你的环境使用 batch 提交）

典型使用方式：

```bash
export RUN_PROGRAM=/path/to/install/release/bin/warpx.3d.MPI.OMP.DP.PDP
export INPUT_FILE=$PWD/Examples/inputs/lia-3d.modify
export MPI_RANKS=1
export OMP_NUM_THREADS=1
export UNR_LIB_DIR=/path/to/unr/lib

bash Examples/test-single-run-release.sh
```

运行脚本会把输入文件复制到 run 目录，对部分可选输出格式做一次可移植性清洗，并把 stdout/stderr 分别写到仓库相对路径下的 `logs/` 与 `runs/` 目录。

## 平台适配说明

当前保留的优化源码路径使用 `vpu` 和 `mpu` 这两个术语，表示目标平台上的向量 / 矩阵执行路径。

如果你要把本仓库迁移到其他机器或编译器栈，通常需要适配：

- 平台特定的 vector/matrix 宏，
- 架构相关编译选项，
- 后端库路径，
- 以及所有把保留的 `vpu` / `mpu` 路径映射到本地硬件能力的 build-time 配置。

也就是说，本仓库保留的是算法与实现路径本身，而底层映射方式仍然依赖目标平台。

## 许可证与版权边界

- 上游 WarpX 的许可证正文保留在 [`LICENSE.txt`](LICENSE.txt) 中。
- 本仓库同时包含为 POLAR-PIC artifact 维护的下游修改。
- 这些下游修改额外标记为 **Yizhuo Rao** 所拥有的 portions copyright。
- 此外，也建议一并阅读 [`NOTICE.txt`](NOTICE.txt) 与 [`LEGAL.txt`](LEGAL.txt) 中继承自 WarpX 派生代码库的公告文件。

如果你计划对本仓库的派生版本做再分发，建议将上游 WarpX 许可证正文、README 中的边界说明以及仓库公告文件一起审阅。

## 贡献与问题反馈

本仓库主要作为研究 artifact 和下游代码快照公开。

与仓库自身相关的问题，请使用当前仓库对应的 review / issue 流程。
与 UNR 后端本身相关的问题，建议优先参考上文给出的 UNR_AD 原始开源说明。
