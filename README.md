# polarpic-ad

## Project overview

`polarpic-ad` is a downstream, research-oriented codebase derived from WarpX.
It is used as an active development branch for architecture and communication-path experiments, rather than a drop-in replacement for upstream WarpX releases.

This repository is maintained as a practical, code-first downstream research branch.

## Relationship to upstream WarpX

- This project started from WarpX code and retains major WarpX-style structure and components.
- It is **not** an official upstream WarpX mirror.
- Upstream project information, release badges, and performance claims do not automatically apply to this fork.
- When behavior differs between this repository and upstream, this repository’s implementation is authoritative for local users.

## Current scope and retained content

At this stage, the repository keeps the core source and project metadata needed for code review and incremental cleanup:

- `Source/`: main C++ implementation
- `Python/`: Python bindings and packaging files
- `Examples/`: example input cases kept for reference and future curation
- `Docs/`: inherited user/developer documentation, still under review
- `Tools/`: utility scripts and developer helpers
- `Regression/`: regression-related assets and scripts
- `deps/`: vendored dependency snapshots used by this repository
- project metadata and build entry files such as `CMakeLists.txt`, `GNUmakefile`, `pyproject.toml`, `setup.py`, `CONTRIBUTING.rst`, `CODE_OF_CONDUCT.rst`, `GOVERNANCE.rst`, `LICENSE.txt`, `NOTICE.txt`, and `LEGAL.txt`

## Repository layout (quick view)

```text
.
├── Source/
├── Python/
├── Examples/
├── Docs/
├── Tools/
├── Regression/
├── deps/
├── CMakeLists.txt
├── GNUmakefile
├── pyproject.toml
└── README.md
```

## Dependencies

This codebase depends on the same class of external HPC/scientific dependencies as WarpX-derived projects (for example AMReX and optional I/O/post-processing stacks).

Dependency details are intentionally kept minimal in this repository snapshot. External HPC dependencies should be reviewed and pinned explicitly before production or redistribution use.

## Example build and run scripts

The release-oriented scripts in `Examples/` are provided as portable templates rather than ready-to-run cluster recipes.

- They now default to repository-relative source, build, install, input, log, and run paths.
- Site-specific modules, compiler names, MPI launchers, UNR locations, and optional dependency roots are intentionally represented as environment variables or `path/to/...` placeholders.
- Fixed process counts, fixed thread counts, platform-specific bind scripts, and architecture-disclosing compiler options have been removed from the published examples.
- Users are expected to supply their own scheduler directives, toolchain modules, optional feature defines, and library paths for their local environment.

## Locally modified files

The current repository snapshot keeps a small set of downstream-edited files that are relevant to code review and future cleanup:

- `Source/Particles/PhysicalParticleContainer.H` and `Source/Particles/PhysicalParticleContainer.cpp`: retained the target physort order-3 push path and removed non-target experiment/test branches.
- `Source/Particles/Deposition/CurrentDeposition.H`: kept the target deposition path while removing non-target test and experiment macros.
- `Source/Particles/WarpXParticleContainer.cpp`: kept the order-3 custom deposition path and collapsed non-target order-1 custom branches back to the baseline flow.
- `Source/Evolve/WarpXEvolve.cpp`, `Source/Initialization/WarpXInit.cpp`, `Source/Initialization/WarpXInitData.cpp`, `Source/WarpX.cpp`, and `Source/WarpX.H`: draft/comment cleanup and minimal gate cleanup around the retained custom path.
- `Source/Particles/MultiParticleContainer.H` and `Source/Particles/MultiParticleContainer.cpp`: removal of leftover test-only interfaces.
- `deps/amrex-24.07-polarpic/Src/Particle/AMReX_ParticleTile.H`: downstream AMReX-side metadata and initialization changes for physical sorting and incremental sorting.
- `Examples/test-single-build-release.sh` and `Examples/test-single-run-release.sh`: sanitized example scripts rewritten with repository-relative defaults and placeholder toolchain/module paths so the build/run flow can be shared without exposing site-specific platform details.

## Code cleanup status

Current cleanup work focuses on:

- removing upstream-specific branding and claims that are not accurate for this fork,
- clarifying repository ownership and maintenance scope,
- preserving functional source history while improving open-source readability.

This process is ongoing. Some inherited docs/scripts may still reflect upstream assumptions and will be normalized incrementally.

## Contributing and issues

Contributions are welcome, especially:

- documentation corrections,
- reproducibility fixes,
- code cleanup with clear, minimal diffs.

Please use the repository's issue tracker and review workflow for bug reports, questions, and proposed changes.
If present, follow repository contribution conventions in `CONTRIBUTING.rst`.

## License and notices

- License: see [`LICENSE.txt`](LICENSE.txt)
- Third-party and legal notices: see [`NOTICE.txt`](NOTICE.txt) and [`LEGAL.txt`](LEGAL.txt)

If you need clarification on licensing boundaries for downstream redistribution, contact the repository maintainers before redistributing derived packages.
