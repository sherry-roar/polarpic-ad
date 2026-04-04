/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ConservationErrors.H"

#include "FieldSolver/Fields.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>

#include <algorithm>
#include <fstream>
#include <string>

using namespace amrex;
using namespace warpx::fields;

ConservationErrors::ConservationErrors (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
    // Output columns:
    // [0] total_charge(C)
    // [1] particle_energy(J)
    // [2] field_energy(J)
    // [3] total_energy(J)
    // [4] particle_momentum_x(kg*m/s)
    // [5] particle_momentum_y(kg*m/s)
    // [6] particle_momentum_z(kg*m/s)
    // [7] field_momentum_x(kg*m/s)
    // [8] field_momentum_y(kg*m/s)
    // [9] field_momentum_z(kg*m/s)
    // [10] total_momentum_x(kg*m/s)
    // [11] total_momentum_y(kg*m/s)
    // [12] total_momentum_z(kg*m/s)
    m_data.resize(13, 0.0_rt);

    if (ParallelDescriptor::IOProcessor())
    {
        if (m_write_header)
        {
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_charge(C)";
            ofs << m_sep;
            ofs << "[" << c++ << "]particle_energy(J)";
            ofs << m_sep;
            ofs << "[" << c++ << "]field_energy(J)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_energy(J)";
            ofs << m_sep;
            ofs << "[" << c++ << "]particle_momentum_x(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]particle_momentum_y(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]particle_momentum_z(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]field_momentum_x(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]field_momentum_y(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]field_momentum_z(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_momentum_x(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_momentum_y(kg*m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]total_momentum_z(kg*m/s)";
            ofs << std::endl;
            ofs.close();
        }
    }
}

void ConservationErrors::ComputeDiags (int step)
{
    if (!m_intervals.contains(step+1)) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto& mypc = warpx.GetPartContainer();
    const int nSpecies = mypc.nSpecies();

    // -----------------------------
    // Particle contributions
    // -----------------------------
    Real total_charge = 0.0_rt;
    Real particle_energy = 0.0_rt;
    Real particle_px = 0.0_rt;
    Real particle_py = 0.0_rt;
    Real particle_pz = 0.0_rt;

    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        auto& myspc = mypc.GetParticleContainer(i_s);

        // Charge: accumulate local contribution, reduce over MPI once at the end
        total_charge += myspc.sumParticleCharge(true);

        const Real m = myspc.getMass();
        using PType = typename WarpXParticleContainer::SuperParticleType;

        ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;

        if (myspc.AmIA<PhysicalSpecies::photon>())
        {
            auto r = ParticleReduce<ReduceData<Real, Real, Real, Real>>(
                myspc,
                [=] AMREX_GPU_DEVICE (const PType& p) noexcept -> GpuTuple<Real, Real, Real, Real>
                {
                    const Real w = p.rdata(PIdx::w);
                    const Real ux = p.rdata(PIdx::ux);
                    const Real uy = p.rdata(PIdx::uy);
                    const Real uz = p.rdata(PIdx::uz);

                    // For photons, use dedicated kinetic energy and electron-mass convention for momentum
                    const Real m_ph = PhysConst::m_e;
                    return {
                        w * Algorithms::KineticEnergyPhotons(ux, uy, uz),
                        w * m_ph * ux,
                        w * m_ph * uy,
                        w * m_ph * uz
                    };
                },
                reduce_ops);

            particle_energy += amrex::get<0>(r);
            particle_px += amrex::get<1>(r);
            particle_py += amrex::get<2>(r);
            particle_pz += amrex::get<3>(r);
        }
        else
        {
            auto r = ParticleReduce<ReduceData<Real, Real, Real, Real>>(
                myspc,
                [=] AMREX_GPU_DEVICE (const PType& p) noexcept -> GpuTuple<Real, Real, Real, Real>
                {
                    const Real w = p.rdata(PIdx::w);
                    const Real ux = p.rdata(PIdx::ux);
                    const Real uy = p.rdata(PIdx::uy);
                    const Real uz = p.rdata(PIdx::uz);

                    return {
                        w * Algorithms::KineticEnergy(ux, uy, uz, m),
                        w * m * ux,
                        w * m * uy,
                        w * m * uz
                    };
                },
                reduce_ops);

            particle_energy += amrex::get<0>(r);
            particle_px += amrex::get<1>(r);
            particle_py += amrex::get<2>(r);
            particle_pz += amrex::get<3>(r);
        }
    }

    // -----------------------------
    // Field contributions
    // -----------------------------
    Real field_energy = 0.0_rt;
    Real field_px = 0.0_rt;
    Real field_py = 0.0_rt;
    Real field_pz = 0.0_rt;

    const int nLevel = warpx.finestLevel() + 1;
    for (int lev = 0; lev < nLevel; ++lev)
    {
        const MultiFab& Ex = warpx.getField(FieldType::Efield_aux, lev, 0);
        const MultiFab& Ey = warpx.getField(FieldType::Efield_aux, lev, 1);
        const MultiFab& Ez = warpx.getField(FieldType::Efield_aux, lev, 2);
        const MultiFab& Bx = warpx.getField(FieldType::Bfield_aux, lev, 0);
        const MultiFab& By = warpx.getField(FieldType::Bfield_aux, lev, 1);
        const MultiFab& Bz = warpx.getField(FieldType::Bfield_aux, lev, 2);

        const std::array<Real, 3>& dx = WarpX::CellSize(lev);
        const Real dV = dx[0] * dx[1] * dx[2];

        Geometry const& geom = warpx.Geom(lev);

        // Same approach as FieldEnergy for Cartesian geometry
        Real const tmpEx = Ex.norm2(0, geom.periodicity());
        Real const tmpEy = Ey.norm2(0, geom.periodicity());
        Real const tmpEz = Ez.norm2(0, geom.periodicity());
        Real const Es = tmpEx*tmpEx + tmpEy*tmpEy + tmpEz*tmpEz;

        Real const tmpBx = Bx.norm2(0, geom.periodicity());
        Real const tmpBy = By.norm2(0, geom.periodicity());
        Real const tmpBz = Bz.norm2(0, geom.periodicity());
        Real const Bs = tmpBx*tmpBx + tmpBy*tmpBy + tmpBz*tmpBz;

        field_energy += 0.5_rt * Es * PhysConst::ep0 * dV;
        field_energy += 0.5_rt * Bs / PhysConst::mu0 * dV;

        // Same approach as FieldMomentum: interpolate to cell centers then sum E x B
        const GpuArray<int,3> cc{0,0,0};
        const GpuArray<int,3> cr{1,1,1};
        constexpr int comp = 0;

        GpuArray<int,3> Ex_stag{0,0,0};
        GpuArray<int,3> Ey_stag{0,0,0};
        GpuArray<int,3> Ez_stag{0,0,0};
        GpuArray<int,3> Bx_stag{0,0,0};
        GpuArray<int,3> By_stag{0,0,0};
        GpuArray<int,3> Bz_stag{0,0,0};

        for (int i = 0; i < AMREX_SPACEDIM; ++i)
        {
            Ex_stag[i] = Ex.ixType()[i];
            Ey_stag[i] = Ey.ixType()[i];
            Ez_stag[i] = Ez.ixType()[i];
            Bx_stag[i] = Bx.ixType()[i];
            By_stag[i] = By.ixType()[i];
            Bz_stag[i] = Bz.ixType()[i];
        }

        ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
        ReduceData<Real, Real, Real> reduce_data(reduce_ops);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& box = enclosedCells(mfi.nodaltilebox());
            const Array4<const Real>& Ex_arr = Ex[mfi].array();
            const Array4<const Real>& Ey_arr = Ey[mfi].array();
            const Array4<const Real>& Ez_arr = Ez[mfi].array();
            const Array4<const Real>& Bx_arr = Bx[mfi].array();
            const Array4<const Real>& By_arr = By[mfi].array();
            const Array4<const Real>& Bz_arr = Bz[mfi].array();

            reduce_ops.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real, Real, Real>
                {
                    const Real Ex_cc = ablastr::coarsen::sample::Interp(Ex_arr, Ex_stag, cc, cr, i, j, k, comp);
                    const Real Ey_cc = ablastr::coarsen::sample::Interp(Ey_arr, Ey_stag, cc, cr, i, j, k, comp);
                    const Real Ez_cc = ablastr::coarsen::sample::Interp(Ez_arr, Ez_stag, cc, cr, i, j, k, comp);

                    const Real Bx_cc = ablastr::coarsen::sample::Interp(Bx_arr, Bx_stag, cc, cr, i, j, k, comp);
                    const Real By_cc = ablastr::coarsen::sample::Interp(By_arr, By_stag, cc, cr, i, j, k, comp);
                    const Real Bz_cc = ablastr::coarsen::sample::Interp(Bz_arr, Bz_stag, cc, cr, i, j, k, comp);

                    return {
                        Ey_cc * Bz_cc - Ez_cc * By_cc,
                        Ez_cc * Bx_cc - Ex_cc * Bz_cc,
                        Ex_cc * By_cc - Ey_cc * Bx_cc
                    };
                });
        }

        auto r = reduce_data.value();
        field_px += PhysConst::ep0 * amrex::get<0>(r) * dV;
        field_py += PhysConst::ep0 * amrex::get<1>(r) * dV;
        field_pz += PhysConst::ep0 * amrex::get<2>(r) * dV;
    }

    // Global MPI reductions: make diagnostics available on I/O rank
    ParallelDescriptor::ReduceRealSum(total_charge, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(particle_energy, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(particle_px, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(particle_py, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(particle_pz, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(field_energy, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(field_px, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(field_py, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(field_pz, ParallelDescriptor::IOProcessorNumber());

    const Real total_energy = particle_energy + field_energy;
    const Real total_px = particle_px + field_px;
    const Real total_py = particle_py + field_py;
    const Real total_pz = particle_pz + field_pz;

    m_data[0] = total_charge;
    m_data[1] = particle_energy;
    m_data[2] = field_energy;
    m_data[3] = total_energy;
    m_data[4] = particle_px;
    m_data[5] = particle_py;
    m_data[6] = particle_pz;
    m_data[7] = field_px;
    m_data[8] = field_py;
    m_data[9] = field_pz;
    m_data[10] = total_px;
    m_data[11] = total_py;
    m_data[12] = total_pz;
}
