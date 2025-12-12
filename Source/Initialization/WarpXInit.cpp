/* Copyright 2024 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpXInit.H"

#include "Initialization/WarpXAMReXInit.H"

#include <AMReX.H>

#include <ablastr/math/fft/AnyFFT.H>
#include <ablastr/parallelization/MPIInitHelpers.H>
#include <unr.h>
#include <AMReX_Print.H>

void warpx::initialization::initialize_external_libraries(int argc, char* argv[])
{
    ablastr::parallelization::mpi_init(argc, argv);
    warpx::initialization::amrex_init(argc, argv);
    ablastr::math::anyfft::setup();

#ifdef UNROLL_OMP_UNR
    unr_init();
    amrex::Print() << "UNR initialized\n";
#endif
}

void warpx::initialization::finalize_external_libraries()
{
#ifdef UNROLL_OMP_UNR
    unr_finalize();
#endif

    ablastr::math::anyfft::cleanup();
    amrex::Finalize();
    ablastr::parallelization::mpi_finalize();
}
