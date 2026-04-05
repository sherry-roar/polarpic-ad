/* Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet
 * Remi Lehe, Weiqun Zhang, Michael Rowan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CURRENTDEPOSITION_H_
#define WARPX_CURRENTDEPOSITION_H_

#include "Particles/Deposition/SharedDepositionUtils.H"
#include "ablastr/parallelization/KernelTimer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#   include "Utils/WarpX_Complex.H"
#endif

#include "WarpX.H" // todo: remove include and pass globals as args

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>

#include <arc_vpu.h>
#include <arc_mpu.h>
using namespace std;

typedef svfloat64_t vpuc __attribute__((arc_vpu_vector_bits(__arc_FEATURE_vpu_BITS)));
typedef svint64_t vpucint __attribute__((arc_vpu_vector_bits(__arc_FEATURE_vpu_BITS)));
typedef svuint64_t vpucuint __attribute__((arc_vpu_vector_bits(__arc_FEATURE_vpu_BITS)));

inline uint64_t rdtscv(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val) : : "memory");
    return val;
}

inline uint64_t rdtscm(void) __arc_preserves("za") __arc_streaming {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val) : : "memory");
    return val;
}

class Vec {
private:
    vpuc v_;

public:

    Vec() = default;


    Vec(svfloat64_t v) : v_(v) {}


    Vec(double v) : v_(svdup_f64(v)) {}


    operator svfloat64_t() const {
        return v_;
    }


    void operator+=(const Vec& rhs) {
        v_ = svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }


    void operator-=(const Vec& rhs) {
        v_ = svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }


    void operator*=(const Vec& rhs) {
        v_ = svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }


    Vec operator+(const Vec &rhs) const {
        return Vec(svadd_f64_x(svptrue_b64(), v_, rhs.v_));
    }


    Vec operator-(const Vec &rhs) const {
        return Vec(svsub_f64_x(svptrue_b64(), v_, rhs.v_));
    }


    Vec operator*(const Vec &rhs) const {
        return Vec(svmul_f64_x(svptrue_b64(), v_, rhs.v_));
    }


    Vec operator/(const Vec &rhs) const {
        return Vec(svdiv_f64_x(svptrue_b64(), v_, rhs.v_));
    }


    Vec operator-() const {
        return Vec(svneg_f64_x(svptrue_b64(), v_));
    }


    Vec Sqrt() const {
        return Vec(svsqrt_f64_x(svptrue_b64(), v_));
    }


    Vec operator+(double rhs) const {
        return *this + Vec(rhs);
    }


    Vec operator-(double rhs) const {
        return *this - Vec(rhs);
    }


    Vec operator*(double rhs) const {
        return *this * Vec(rhs);
    }


    Vec operator/(double rhs) const {
        return *this / Vec(rhs);
    }


    friend Vec operator+(double lhs, const Vec &rhs) {
        return Vec(lhs) + rhs;
    }


    friend Vec operator-(double lhs, const Vec &rhs) {
        return Vec(lhs) - rhs;
    }


    friend Vec operator*(double lhs, const Vec &rhs) {
        return Vec(lhs) * rhs;
    }


    friend Vec operator/(double lhs, const Vec &rhs) {
        return Vec(lhs) / rhs;
    }


    void Store(svbool_t p, double *mem) const {
        svst1_f64(p, mem, v_);
    }


    static Vec Load(svbool_t p, const double *mem) {
        return Vec(svld1_f64(p, mem));
    }

    double GetElement(svbool_t p, int pos) {
        double tmp[svcntd()];
        this->Store(p, tmp);
        return tmp[pos];
    }
};

class intVec {
private:
    vpucint  v_;  

public:

    intVec() = default;


    intVec(svint64_t v) : v_(v) {}


    intVec(int64_t v) : v_(svdup_s64(v)) {}


    operator svint64_t() const {
        return v_;
    }


    void operator+=(const intVec& rhs) {
        v_ = svadd_s64_x(svptrue_b64(), v_, rhs.v_);
    }


    void operator-=(const intVec& rhs) {
        v_ = svsub_s64_x(svptrue_b64(), v_, rhs.v_);
    }


    void operator*=(const intVec& rhs) {
        v_ = svmul_s64_x(svptrue_b64(), v_, rhs.v_);
    }


    intVec operator+(const intVec &rhs) const {
        return intVec(svadd_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    intVec operator-(const intVec &rhs) const {
        return intVec(svsub_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    intVec operator*(const intVec &rhs) const {
        return intVec(svmul_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    intVec operator/(const intVec &rhs) const {
        return intVec(svdiv_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    intVec operator-() const {
        return intVec(svneg_s64_x(svptrue_b64(), v_));
    }


    // intVec Sqrt() const {
    //     return intVec(svsqrt_s64_x(svptrue_b64(), v_));
    // }


    intVec operator+(int64_t rhs) const {
        return *this + intVec(rhs);
    }


    intVec operator-(int64_t rhs) const {
        return *this - intVec(rhs);
    }


    intVec operator*(int64_t rhs) const {
        return *this * intVec(rhs);
    }


    intVec operator/(int64_t rhs) const {
        return *this / intVec(rhs);
    }


    friend intVec operator+(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) + rhs;
    }


    friend intVec operator-(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) - rhs;
    }


    friend intVec operator*(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) * rhs;
    }


    friend intVec operator/(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) / rhs;
    }


    void Store(svbool_t p, int64_t *mem) const {
        svst1_s64(p, mem, v_);
    }


    static intVec Load(svbool_t p, const int64_t *mem) {
        return intVec(svld1_s64(p, mem));
    }

    int64_t GetElement(svbool_t p, int pos) {
        int64_t tmp[svcntd()];
        this->Store(p, tmp);
        return tmp[pos];
    }
};

class MVec {
private:
    vpuc v_;
    
public:
    MVec() __arc_preserves("za") __arc_streaming = default;
    

    MVec(vpuc v) __arc_preserves("za") __arc_streaming : v_(v) {}
    MVec(double v) __arc_preserves("za") __arc_streaming : v_(svdup_f64(v)) {}

    operator vpuc() const __arc_preserves("za") __arc_streaming {
        return v_;
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    void operator+=(const MVec &rhs) __arc_preserves("za") __arc_streaming {
        v_ = svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // void operator+=(const MVec &rhs) __arc_preserves("za") __arc_streaming {
    //     v_ = svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    // }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    void operator-=(const MVec &rhs) __arc_preserves("za") __arc_streaming {
        v_ = svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    void operator*=(const MVec &rhs) __arc_preserves("za") __arc_streaming {
        v_ = svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator+(const MVec &rhs) const __arc_preserves("za") __arc_streaming {
        return svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator-(const MVec &rhs) const __arc_preserves("za") __arc_streaming {
        return svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator*(const MVec &rhs) const __arc_preserves("za") __arc_streaming {
        return svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator-() const __arc_preserves("za") __arc_streaming {
        return svneg_f64_x(svptrue_b64(), v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator+(double rhs) const __arc_preserves("za") __arc_streaming {
        return *this + MVec(rhs);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator-(double rhs) const __arc_preserves("za") __arc_streaming {
        return *this - MVec(rhs);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    MVec operator*(double rhs) const __arc_preserves("za") __arc_streaming {
        return *this * MVec(rhs);
    }
    
    MVec operator/(const MVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MVec(svdiv_f64_x(svptrue_b64(), v_, rhs.v_));
    }
    
    MVec operator/(double rhs) const __arc_preserves("za") __arc_streaming {
        return *this / MVec(rhs);
    }
    
    friend MVec operator/(double lhs, const MVec &rhs) __arc_preserves("za") __arc_streaming {
        return MVec(lhs) / rhs;
    }
    
    MVec Sqrt() const __arc_preserves("za") __arc_streaming {
        return MVec(svsqrt_f64_x(svptrue_b64(), v_));
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    friend MVec operator+(double lhs, const MVec &rhs) __arc_preserves("za") __arc_streaming {
        return MVec(lhs) + rhs;
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    friend MVec operator-(double lhs, const MVec &rhs) __arc_preserves("za") __arc_streaming {
        return MVec(lhs) - rhs;
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    friend MVec operator*(double lhs, const MVec &rhs) __arc_preserves("za") __arc_streaming {
        return MVec(lhs) * rhs;
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    void Store(svbool_t p, double *mem) const __arc_preserves("za") __arc_streaming {
        svst1_f64(p, mem, v_);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    static MVec Load(svbool_t p, const double *mem) __arc_preserves("za") __arc_streaming {
        return svld1_f64(p, mem);
    }
    
    // __attribute__((arc_preserves("za")))
    // __attribute__((arc_streaming_compatible))
    double ReduceSum() const __arc_preserves("za") __arc_streaming {
        return svaddv_f64(svptrue_b64(), v_);
    }

    double GetElement(svbool_t p, int pos) __arc_preserves("za") __arc_streaming {
        double tmp[svcntd()];
        this->Store(p, tmp);
        return tmp[pos];
    }
};


class MintVec {
    private:
    vpucint  v_;  

public:

    MintVec() __arc_preserves("za") __arc_streaming = default;


    MintVec(svint64_t v) __arc_preserves("za") __arc_streaming : v_(v) {}


    MintVec(int64_t v) __arc_preserves("za") __arc_streaming : v_(svdup_s64(v)) {}


    operator svint64_t() const __arc_preserves("za") __arc_streaming {
        return v_;
    }


    void operator+=(const MintVec& rhs) __arc_preserves("za") __arc_streaming {
        v_ = svadd_s64_x(svptrue_b64(), v_, rhs.v_);
    }


    void operator-=(const MintVec& rhs) __arc_preserves("za") __arc_streaming {
        v_ = svsub_s64_x(svptrue_b64(), v_, rhs.v_);
    }


    void operator*=(const MintVec& rhs) __arc_preserves("za") __arc_streaming {
        v_ = svmul_s64_x(svptrue_b64(), v_, rhs.v_);
    }


    MintVec operator+(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svadd_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    MintVec operator-(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svsub_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    MintVec operator*(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svmul_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    MintVec operator/(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svdiv_s64_x(svptrue_b64(), v_, rhs.v_));
    }


    MintVec operator-() const __arc_preserves("za") __arc_streaming {
        return MintVec(svneg_s64_x(svptrue_b64(), v_));
    }


    // MintVec Sqrt() const {
    //     return MintVec(svsqrt_s64_x(svptrue_b64(), v_));
    // }


    MintVec operator+(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this + MintVec(rhs);
    }


    MintVec operator-(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this - MintVec(rhs);
    }


    MintVec operator*(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this * MintVec(rhs);
    }


    MintVec operator/(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this / MintVec(rhs);
    }


    friend MintVec operator+(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) + rhs;
    }


    friend MintVec operator-(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) - rhs;
    }


    friend MintVec operator*(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) * rhs;
    }


    friend MintVec operator/(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) / rhs;
    }

    static MintVec Mvec2Mint(svbool_t p, vpuc mvec) __arc_preserves("za") __arc_streaming {
        return svcvt_s64_f64_z(p, mvec);
    }


    void Store(svbool_t p, int64_t *mem) const __arc_preserves("za") __arc_streaming {
        svst1_s64(p, mem, v_);
    }


    static MintVec Load(svbool_t p, const int64_t *mem) __arc_preserves("za") __arc_streaming {
        return MintVec(svld1_s64(p, mem));
    }
};

class UintVec {
private:
    vpucuint v_;  
    
public:

    UintVec() = default;
    

    UintVec(svuint64_t v) : v_(v) {}
    

    UintVec(uint64_t v) : v_(svdup_u64(v)) {}
    

    operator svuint64_t() const {
        return v_;
    }
    

    void operator+=(const UintVec& rhs) {
        v_ = svadd_u64_x(svptrue_b64(), v_, rhs.v_);
    }
    

    void operator-=(const UintVec& rhs) {
        v_ = svsub_u64_x(svptrue_b64(), v_, rhs.v_);
    }
    

    void operator*=(const UintVec& rhs) {
        v_ = svmul_u64_x(svptrue_b64(), v_, rhs.v_);
    }
    

    UintVec operator+(const UintVec &rhs) const {
        return UintVec(svadd_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    

    UintVec operator-(const UintVec &rhs) const {
        return UintVec(svsub_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    

    UintVec operator*(const UintVec &rhs) const {
        return UintVec(svmul_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    

    UintVec operator/(const UintVec &rhs) const {
        return UintVec(svdiv_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    

    // UintVec operator-() const {
    //     return UintVec(svneg_u64_x(svptrue_b64(), v_));
    // }
    

    // UintVec Sqrt() const {
    //     return UintVec(svsqrt_u64_x(svptrue_b64(), v_));
    // }
    

    UintVec operator+(uint64_t rhs) const {
        return *this + UintVec(rhs);
    }
    

    UintVec operator-(uint64_t rhs) const {
        return *this - UintVec(rhs);
    }
    

    UintVec operator*(uint64_t rhs) const {
        return *this * UintVec(rhs);
    }
    

    UintVec operator/(uint64_t rhs) const {
        return *this / UintVec(rhs);
    }
    

    friend UintVec operator+(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) + rhs;
    }
    

    friend UintVec operator-(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) - rhs;
    }
    

    friend UintVec operator*(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) * rhs;
    }
    

    friend UintVec operator/(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) / rhs;
    }
    

    void Store(svbool_t p, uint64_t *mem) const {
        svst1_u64(p, mem, v_);
    }
    

    static UintVec Load(svbool_t p, const uint64_t *mem) {
        return UintVec(svld1_u64(p, mem));
    }
};

/**
 * \brief Kernel for the direct current deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param xp, yp, zp    The particle positions.
 * \param wq            The charge of the macroparticle
 * \param vx,vy,vz      The particle velocities
 * \param jx_arr,jy_arr,jz_arr Array4 of current density, either full array or tile.
 * \param jx_type,jy_type,jz_type The grid types along each direction, either NODE or CELL
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv          3D cell size inverse
 * \param xyzmin        The lower bounds of the domain
 * \param invvol        The inverse volume of a grid cell
 * \param lo            Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doDepositionShapeNKernel([[maybe_unused]] const amrex::ParticleReal xp,
                              [[maybe_unused]] const amrex::ParticleReal yp,
                              [[maybe_unused]] const amrex::ParticleReal zp,
                              const amrex::ParticleReal wq,
                              const amrex::ParticleReal vx,
                              const amrex::ParticleReal vy,
                              const amrex::ParticleReal vz,
                              amrex::Array4<amrex::Real> const& jx_arr,
                              amrex::Array4<amrex::Real> const& jy_arr,
                              amrex::Array4<amrex::Real> const& jz_arr,
                              amrex::IntVect const& jx_type,
                              amrex::IntVect const& jy_type,
                              amrex::IntVect const& jz_type,
                              const amrex::Real relative_time,
                              const amrex::XDim3 & dinv,
                              const amrex::XDim3 & xyzmin,
                              const amrex::Real invvol,
                              const amrex::Dim3 lo,
                              [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // wqx, wqy wqz are particle current in each direction
#if defined(WARPX_DIM_RZ)
    // In RZ, wqx is actually wqr, and wqy is wqtheta
    // Convert to cylindrical at the mid point
    const amrex::Real xpmid = xp + relative_time*vx;
    const amrex::Real ypmid = yp + relative_time*vy;
    const amrex::Real rpmid = std::sqrt(xpmid*xpmid + ypmid*ypmid);
    const amrex::Real costheta = (rpmid > 0._rt ? xpmid/rpmid : 1._rt);
    const amrex::Real sintheta = (rpmid > 0._rt ? ypmid/rpmid : 0._rt);
    const Complex xy0 = Complex{costheta, sintheta};
    const amrex::Real wqx = wq*invvol*(+vx*costheta + vy*sintheta);
    const amrex::Real wqy = wq*invvol*(-vx*sintheta + vy*costheta);
#else
    const amrex::Real wqx = wq*invvol*vx;
    const amrex::Real wqy = wq*invvol*vy;
#endif
    const amrex::Real wqz = wq*invvol*vz;

    // --- Compute shape factors
    Compute_shape_factor< depos_order > const compute_shape_factor;
#if (AMREX_SPACEDIM >= 2)
    // x direction
    // Get particle position after 1/2 push back in position
    // Keep these double to avoid bug in single precision
#if defined(WARPX_DIM_RZ)
    const double xmid = (rpmid - xyzmin.x)*dinv.x;
#else
    const double xmid = ((xp - xyzmin.x) + relative_time*vx)*dinv.x;
#endif

    // j_j[xyz] leftmost grid point in x that the particle touches for the centering of each current
    // sx_j[xyz] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    // Keep these double to avoid bug in single precision
    double sx_node[depos_order + 1] = {0.};
    double sx_cell[depos_order + 1] = {0.};
    int j_node = 0;
    int j_cell = 0;
    if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE) {
        j_node = compute_shape_factor(sx_node, xmid);
    }
    if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL) {
        j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
    }

    amrex::Real sx_jx[depos_order + 1] = {0._rt};
    amrex::Real sx_jy[depos_order + 1] = {0._rt};
    amrex::Real sx_jz[depos_order + 1] = {0._rt};
    for (int ix=0; ix<=depos_order; ix++)
    {
        sx_jx[ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jy[ix] = ((jy_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        sx_jz[ix] = ((jz_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
    }

    int const j_jx = ((jx_type[0] == NODE) ? j_node : j_cell);
    int const j_jy = ((jy_type[0] == NODE) ? j_node : j_cell);
    int const j_jz = ((jz_type[0] == NODE) ? j_node : j_cell);
#endif //AMREX_SPACEDIM >= 2

#if defined(WARPX_DIM_3D)
    // y direction
    // Keep these double to avoid bug in single precision
    const double ymid = ((yp - xyzmin.y) + relative_time*vy)*dinv.y;
    double sy_node[depos_order + 1] = {0.};
    double sy_cell[depos_order + 1] = {0.};
    int k_node = 0;
    int k_cell = 0;
    if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE) {
        k_node = compute_shape_factor(sy_node, ymid);
    }
    if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL) {
        k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
    }
    amrex::Real sy_jx[depos_order + 1] = {0._rt};
    amrex::Real sy_jy[depos_order + 1] = {0._rt};
    amrex::Real sy_jz[depos_order + 1] = {0._rt};
    for (int iy=0; iy<=depos_order; iy++)
    {
        sy_jx[iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jy[iy] = ((jy_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        sy_jz[iy] = ((jz_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
    }
    int const k_jx = ((jx_type[1] == NODE) ? k_node : k_cell);
    int const k_jy = ((jy_type[1] == NODE) ? k_node : k_cell);
    int const k_jz = ((jz_type[1] == NODE) ? k_node : k_cell);
#endif

    // z direction
    // Keep these double to avoid bug in single precision
    constexpr int zdir = WARPX_ZINDEX;
    const double zmid = ((zp - xyzmin.z) + relative_time*vz)*dinv.z;
    double sz_node[depos_order + 1] = {0.};
    double sz_cell[depos_order + 1] = {0.};
    int l_node = 0;
    int l_cell = 0;
    if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE) {
        l_node = compute_shape_factor(sz_node, zmid);
    }
    if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL) {
        l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
    }
    amrex::Real sz_jx[depos_order + 1] = {0._rt};
    amrex::Real sz_jy[depos_order + 1] = {0._rt};
    amrex::Real sz_jz[depos_order + 1] = {0._rt};
    for (int iz=0; iz<=depos_order; iz++)
    {
        sz_jx[iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jy[iz] = ((jy_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        sz_jz[iz] = ((jz_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
    }
    int const l_jx = ((jx_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jy = ((jy_type[zdir] == NODE) ? l_node : l_cell);
    int const l_jz = ((jz_type[zdir] == NODE) ? l_node : l_cell);

    // Deposit current into jx_arr, jy_arr and jz_arr
#if defined(WARPX_DIM_1D_Z)
    for (int iz=0; iz<=depos_order; iz++){
        amrex::Gpu::Atomic::AddNoRet(
            &jx_arr(lo.x+l_jx+iz, 0, 0, 0),
            sz_jx[iz]*wqx);
        amrex::Gpu::Atomic::AddNoRet(
            &jy_arr(lo.x+l_jy+iz, 0, 0, 0),
            sz_jy[iz]*wqy);
        amrex::Gpu::Atomic::AddNoRet(
            &jz_arr(lo.x+l_jz+iz, 0, 0, 0),
            sz_jz[iz]*wqz);
    }
#endif
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            amrex::Gpu::Atomic::AddNoRet(
                &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 0),
                sx_jx[ix]*sz_jx[iz]*wqx);
            amrex::Gpu::Atomic::AddNoRet(
                &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 0),
                sx_jy[ix]*sz_jy[iz]*wqy);
            amrex::Gpu::Atomic::AddNoRet(
                &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 0),
                sx_jz[ix]*sz_jz[iz]*wqz);
#if defined(WARPX_DIM_RZ)
            Complex xy = xy0; // Note that xy is equal to e^{i m theta}
            for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                // The factor 2 on the weighting comes from the normalization of the modes
                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode-1), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode  ), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.imag());
                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode-1), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode  ), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.imag());
                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode-1), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.real());
                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode  ), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.imag());
                xy = xy*xy0;
            }
#endif
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                amrex::Gpu::Atomic::AddNoRet(
                    &jx_arr(lo.x+j_jx+ix, lo.y+k_jx+iy, lo.z+l_jx+iz),
                    sx_jx[ix]*sy_jx[iy]*sz_jx[iz]*wqx);
                amrex::Gpu::Atomic::AddNoRet(
                    &jy_arr(lo.x+j_jy+ix, lo.y+k_jy+iy, lo.z+l_jy+iz),
                    sx_jy[ix]*sy_jy[iy]*sz_jy[iz]*wqy);
                amrex::Gpu::Atomic::AddNoRet(
                    &jz_arr(lo.x+j_jz+ix, lo.y+k_jz+iy, lo.z+l_jz+iz),
                    sx_jz[ix]*sy_jz[iy]*sz_jz[iz]*wqz);
            }
        }
    }
#endif
}

/**
 * \brief Current Deposition for thread thread_num
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // --- Get particle quantities
            const amrex::Real gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                        + uyp[ip]*uyp[ip]*clightsq
                                                        + uzp[ip]*uzp[ip]*clightsq);
            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            doDepositionShapeNKernel<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}


template <int depos_order>
void doDepositionShapeN_vpu (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    
    constexpr int nshapes = 1 + 1;

    const int block_size = 8;
    long n_blocks = np_to_deposit / block_size;
    long remainder = np_to_deposit - n_blocks * block_size;
    const int vl = svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;
    

    // Strides used to map compact rho-cell buffers back to J-array tiles.
    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    auto compute_shape_factor_vpu = [](double* sx[], Vec xmid, svbool_t p) {
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        sx0.Store(p, sx[0]);
        sx1.Store(p, sx[1]);
        return i_newv;
    };

    for(int block_id = 0; block_id < n_blocks; block_id++){
        std::vector<int64_t> j_cell(block_size, 0);
        std::vector<int64_t> k_cell(block_size, 0);
        std::vector<int64_t> l_cell(block_size, 0);
        std::vector<amrex::Real> wqx(block_size, 0.0);
        std::vector<amrex::Real> wqy(block_size, 0.0);
        std::vector<amrex::Real> wqz(block_size, 0.0);
        double sx_m[nshapes][block_size] = {0.};
        double sy_m[nshapes][block_size] = {0.};
        double sz_m[nshapes][block_size] = {0.};
        
        for(int iip = 0; iip < block_size; iip += vl){
            long ip = iip + block_id * block_size;
            svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
            
            Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
            Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
            Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
            Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                        + uyp_v * uyp_v * clightsq
                                        + uzp_v * uzp_v * clightsq).Sqrt();
            Vec vx_v = uxp_v * gaminv;
            Vec vy_v = uyp_v * gaminv;
            Vec vz_v = uzp_v * gaminv;

            Vec wp_v = Vec::Load(p_ip, &wp[ip]);
            Vec wq_v = q * wp_v;
            if (do_ionization) 
            {
                wq_v *= ion_lev[ip];
            }

            Vec wqx_v = wq_v * invvol * vx_v;
            Vec wqy_v = wq_v * invvol * vy_v;
            Vec wqz_v = wq_v * invvol * vz_v;

            wqx_v.Store(p_ip, &wqx[iip]);
            wqy_v.Store(p_ip, &wqy[iip]);
            wqz_v.Store(p_ip, &wqz[iip]);

            Vec xp = Vec::Load(p_ip, &mx[ip]);
            Vec yp = Vec::Load(p_ip, &my[ip]);
            Vec zp = Vec::Load(p_ip, &mz[ip]);

            Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
            double* sx_temp[nshapes] = {&sx_m[0][iip], &sx_m[1][iip]};
            intVec j_cellv = compute_shape_factor_vpu(sx_temp, xmid, p_ip);
            j_cellv.Store(p_ip, &j_cell[iip]);

            Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
            double* sy_temp[nshapes] = {&sy_m[0][iip], &sy_m[1][iip]};
            intVec k_cellv = compute_shape_factor_vpu(sy_temp, ymid, p_ip);
            k_cellv.Store(p_ip, &k_cell[iip]);
            
            Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
            double* sz_temp[nshapes] = {&sz_m[0][iip], &sz_m[1][iip]};
            intVec l_cellv = compute_shape_factor_vpu(sz_temp, zmid, p_ip);
            l_cellv.Store(p_ip, &l_cell[iip]);            
        }

        alignas(64) long xoffsets[8];
        alignas(64) long yoffsets[8];
        alignas(64) long zoffsets[8];
        alignas(64) double jx_vals[8] = {0};
        alignas(64) double jy_vals[8] = {0};
        alignas(64) double jz_vals[8] = {0};

        for (int iz = 0; iz < nshapes; ++iz) {
            double* sz=sz_m[iz];
            for (int iy = 0; iy < nshapes; ++iy) {
                double* sy=sy_m[iy];
                for (int ix = 0; ix < nshapes; ++ix) {
                    double* sx = sx_m[ix];

                    for (long iip = 0; iip < block_size; iip += vl) {
                        long ip = iip + block_id * block_size;
                        svbool_t p = svwhilelt_b64(ip, np_to_deposit);

                        intVec j_cell_v = intVec::Load(p, &j_cell[iip]);
                        intVec j = lo.x + j_cell_v + ix;
                        intVec k_cell_v = intVec::Load(p, &k_cell[iip]);
                        intVec k = lo.y + k_cell_v + iy;
                        intVec l_cell_v = intVec::Load(p, &l_cell[iip]);
                        intVec l = lo.z + l_cell_v + iz;

                        intVec xoffsets_v = j - jx_arr.begin.x + (k - jx_arr.begin.y) * xjstride + (l - jx_arr.begin.z) * xkstride;
                        xoffsets_v.Store(p, &xoffsets[0]);
                        intVec yoffsets_v = j - jy_arr.begin.x + (k - jy_arr.begin.y) * yjstride + (l - jy_arr.begin.z) * ykstride;
                        yoffsets_v.Store(p, &yoffsets[0]);
                        intVec zoffsets_v = j - jz_arr.begin.x + (k - jz_arr.begin.y) * zjstride + (l - jz_arr.begin.z) * zkstride;
                        zoffsets_v.Store(p, &zoffsets[0]);

                        Vec sx_v = Vec::Load(p, &sx[iip]);
                        Vec sy_v = Vec::Load(p, &sy[iip]);
                        Vec sz_v = Vec::Load(p, &sz[iip]);
                        Vec s = sx_v * sy_v * sz_v;

                        Vec wqx_v = Vec::Load(p, &wqx[iip]);
                        Vec jx_arr_v = s * wqx_v;
                        jx_arr_v.Store(p,&jx_vals[0]);
                        Vec wqy_v = Vec::Load(p, &wqy[iip]);
                        Vec jy_arr_v=s* wqy_v;
                        jy_arr_v.Store(p,&jy_vals[0]);
                        Vec wqz_v = Vec::Load(p, &wqz[iip]);
                        Vec jz_arr_v=s* wqz_v;
                        jz_arr_v.Store(p,&jz_vals[0]);

                        const int vec_len = std::min(vl, static_cast<int>(np_to_deposit - ip));
                        for (int iv = 0; iv < vec_len; ++iv) {
                            amrex::Gpu::Atomic::AddNoRet(&jx_arr.p[xoffsets[iv]], jx_vals[iv]);
                            amrex::Gpu::Atomic::AddNoRet(&jy_arr.p[yoffsets[iv]], jy_vals[iv]);
                            amrex::Gpu::Atomic::AddNoRet(&jz_arr.p[zoffsets[iv]], jz_vals[iv]);
                        }                       
                    }
                }
            }
        }    
    }
    
    std::vector<int64_t> j_cell(remainder, 0);
    std::vector<int64_t> k_cell(remainder, 0);
    std::vector<int64_t> l_cell(remainder, 0);
    std::vector<amrex::Real> wqx(remainder, 0.0);
    std::vector<amrex::Real> wqy(remainder, 0.0);
    std::vector<amrex::Real> wqz(remainder, 0.0);
    double sx_m[nshapes][block_size] = {0.};
    double sy_m[nshapes][block_size] = {0.};
    double sz_m[nshapes][block_size] = {0.};
    for(long iip = 0; iip < remainder; iip += vl)
    {
        long ip = iip + n_blocks * block_size;
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);

        Vec uxp_v=Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v=Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v=Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();

        Vec wp_v=Vec::Load(p_ip,&wp[ip]);
        Vec wq_v = q * wp_v;

        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[iip]);
        wqy_v.Store(p_ip, &wqy[iip]);
        wqz_v.Store(p_ip, &wqz[iip]);
        
        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;

        double* sx_temp[nshapes] = {&sx_m[0][iip], &sx_m[1][iip]};
        intVec j_cellv = compute_shape_factor_vpu( sx_temp, xmid, p_ip);
        j_cellv.Store(p_ip,&j_cell[iip]);
        double* sy_temp[nshapes] = {&sy_m[0][iip], &sy_m[1][iip]};
        intVec k_cellv = compute_shape_factor_vpu( sy_temp, ymid, p_ip);
        k_cellv.Store(p_ip,&k_cell[iip]);
        double* sz_temp[nshapes] = {&sz_m[0][iip], &sz_m[1][iip]};
        intVec l_cellv = compute_shape_factor_vpu( sz_temp, zmid, p_ip);
        l_cellv.Store(p_ip,&l_cell[iip]);
    }

    alignas(64) long   xoffsets[8];
    alignas(64) long   yoffsets[8];
    alignas(64) long   zoffsets[8];
    alignas(64) double jx_vals[8] = {0};
    alignas(64) double jy_vals[8] = {0};
    alignas(64) double jz_vals[8] = {0};

    for (int iz = 0; iz < nshapes; ++iz) {
        double* sz=sz_m[iz];
        for (int iy = 0; iy < nshapes; ++iy) {
            double* sy=sy_m[iy];
            for (int ix = 0; ix < nshapes; ++ix) {
                double* sx=sx_m[ix];

                for (long iip = 0; iip < remainder; iip += vl) {
                    long ip = iip + n_blocks * block_size;
                    svbool_t p = svwhilelt_b64(ip,np_to_deposit);

                    intVec j_cell_v = intVec::Load(p, &j_cell[iip]);
                    intVec j = lo.x + j_cell_v + ix;
                    intVec k_cell_v = intVec::Load(p, &k_cell[iip]);
                    intVec k = lo.y + k_cell_v + iy;
                    intVec l_cell_v = intVec::Load(p, &l_cell[iip]);
                    intVec l = lo.z + l_cell_v + iz;

                    intVec xoffsets_v = j - jx_arr.begin.x + (k - jx_arr.begin.y) * xjstride + (l - jx_arr.begin.z) * xkstride;
                    xoffsets_v.Store(p,&xoffsets[0]);
                    intVec yoffsets_v = j - jy_arr.begin.x + (k - jy_arr.begin.y) * yjstride + (l - jy_arr.begin.z) * ykstride;
                    yoffsets_v.Store(p,&yoffsets[0]);
                    intVec zoffsets_v = j - jz_arr.begin.x + (k - jz_arr.begin.y) * zjstride + (l - jz_arr.begin.z) * zkstride;
                    zoffsets_v.Store(p,&zoffsets[0]);

                    Vec sx_v = Vec::Load(p, &sx[iip]);
                    Vec sy_v = Vec::Load(p, &sy[iip]);
                    Vec sz_v = Vec::Load(p, &sz[iip]);
                    Vec s = sx_v * sy_v * sz_v;

                    Vec wqx_v = Vec::Load(p, &wqx[iip]);
                    Vec jx_arr_v=s* wqx_v;
                    jx_arr_v.Store(p,&jx_vals[0]);
                    Vec wqy_v = Vec::Load(p, &wqy[iip]);
                    Vec jy_arr_v=s* wqy_v;
                    jy_arr_v.Store(p,&jy_vals[0]);
                    Vec wqz_v = Vec::Load(p, &wqz[iip]);
                    Vec jz_arr_v=s* wqz_v;
                    jz_arr_v.Store(p,&jz_vals[0]);

                    const int vec_len = std::min(vl, static_cast<int>(np_to_deposit - ip));
                    for (int iv = 0; iv < vec_len; ++iv) {
                        amrex::Gpu::Atomic::AddNoRet(&jx_arr.p[xoffsets[iv]], jx_vals[iv]);
                        amrex::Gpu::Atomic::AddNoRet(&jy_arr.p[yoffsets[iv]], jy_vals[iv]);
                        amrex::Gpu::Atomic::AddNoRet(&jz_arr.p[zoffsets[iv]], jz_vals[iv]);
                    }                   
                }
            }
        }
    }    
}


template <int depos_order>
void doDepositionShapeN_org_vpu (const GetParticlePosition<PIdx>& GetPosition,
                         const amrex::ParticleReal * const wp,
                         const amrex::ParticleReal * const uxp,
                         const amrex::ParticleReal * const uyp,
                         const amrex::ParticleReal * const uzp,
                         const int* ion_lev,
                         amrex::FArrayBox& jx_fab,
                         amrex::FArrayBox& jy_fab,
                         amrex::FArrayBox& jz_fab,
                         long np_to_deposit,
                         amrex::Real relative_time,
                         const amrex::XDim3 & dinv,
                         const amrex::XDim3 & xyzmin,
                         amrex::Dim3 lo,
                         amrex::Real q,
                         [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0_rt/PhysConst::c/PhysConst::c;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    
    constexpr int nshapes = 1 + 1;
    const int block_size = 8;
    long n_blocks = np_to_deposit / block_size;
    long remainder = np_to_deposit - n_blocks * block_size;
    const int vl = svcntd();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;
    

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    auto compute_shape_factor_vpu_scatter = [](double* sx, Vec xmid, svbool_t p) {
        const svuint64_t sx_index = svindex_u64(0, 2);
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        svst1_scatter_index(p, &sx[0], sx_index, sx0);
        svst1_scatter_index(p, &sx[1], sx_index, sx1);
        return i_newv;
    };

    int sx_step[2] = {0, 1};

    std::vector<int64_t> j_cell(block_size, 0);
    std::vector<int64_t> k_cell(block_size, 0);
    std::vector<int64_t> l_cell(block_size, 0);
    std::vector<amrex::Real> wqx(block_size, 0.0);
    std::vector<amrex::Real> wqy(block_size, 0.0);
    std::vector<amrex::Real> wqz(block_size, 0.0);
    double sx_m[nshapes * block_size] = {0.};
    double sy_m[nshapes * block_size] = {0.};
    double sz_m[nshapes * block_size] = {0.};

    alignas(64) long xoffsets[8];
    alignas(64) long yoffsets[8];
    alignas(64) long zoffsets[8];
    alignas(64) double jx_vals[8] = {0};
    alignas(64) double jy_vals[8] = {0};
    alignas(64) double jz_vals[8] = {0};
    svbool_t p_ix = svwhilelt_b64(0, nshapes);

    for(long ip = 0; ip < np_to_deposit; ip += vl){    
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wp_v = Vec::Load(p_ip, &wp[ip]);
        
        Vec wq_v = q * wp_v;
        if (do_ionization) 
        {
            wq_v *= ion_lev[ip];
        }

        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[0]);
        wqy_v.Store(p_ip, &wqy[0]);
        wqz_v.Store(p_ip, &wqz[0]);

        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        intVec j_cellv = compute_shape_factor_vpu_scatter(&sx_m[0], xmid, p_ip);
        j_cellv.Store(p_ip, &j_cell[0]);

        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        intVec k_cellv = compute_shape_factor_vpu_scatter(&sy_m[0], ymid, p_ip);
        k_cellv.Store(p_ip, &k_cell[0]);
        
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec l_cellv = compute_shape_factor_vpu_scatter(&sz_m[0], zmid, p_ip);
        l_cellv.Store(p_ip, &l_cell[0]);


        int end = (np_to_deposit - ip >= 8) ? block_size : np_to_deposit - ip;        
        for (long iip = 0; iip < end; iip++)
        {
            for (int iz = 0; iz < nshapes; ++iz)
            {
                for (int iy = 0; iy < nshapes; ++iy)
                {
                    long idx = 2 * iy + 4 * iz;

                    intVec mx_index = svld1sw_s64(p_ix, sx_step);
                    Vec sx_v = Vec::Load(p_ix, &sx_m[iip * 2]);
                    Vec tmp_v = sx_v * sy_m[iip * 2 + iy] * sz_m[iip * 2 + iz];

                    Vec jx_val_v = tmp_v * wqx[iip];
                    Vec jy_val_v = tmp_v * wqy[iip];
                    Vec jz_val_v = tmp_v * wqz[iip];
                    jx_val_v.Store(p_ix, &jx_vals[idx]);
                    jy_val_v.Store(p_ix, &jy_vals[idx]);
                    jz_val_v.Store(p_ix, &jz_vals[idx]);

                    int j = lo.x + j_cell[iip];
                    int k = lo.y + k_cell[iip] + iy;
                    int l = lo.z + l_cell[iip] + iz;

                    long xoffset = j - jx_arr.begin.x + (k - jx_arr.begin.y) * xjstride + (l - jx_arr.begin.z) * xkstride;
                    long yoffset = j - jy_arr.begin.x + (k - jy_arr.begin.y) * yjstride + (l - jy_arr.begin.z) * ykstride;
                    long zoffset = j - jz_arr.begin.x + (k - jz_arr.begin.y) * zjstride + (l - jz_arr.begin.z) * zkstride;
                    for (int ix = 0; ix < nshapes; ix++){
                        amrex::Gpu::Atomic::AddNoRet(&jx_arr.p[xoffset + ix], jx_vals[idx + ix]);
                        amrex::Gpu::Atomic::AddNoRet(&jy_arr.p[yoffset + ix], jy_vals[idx + ix]);
                        amrex::Gpu::Atomic::AddNoRet(&jz_arr.p[zoffset + ix], jz_vals[idx + ix]);
                    }
                }
            }
        }
    }
}


template <int depos_order>
void doDepositionShapeN_rhocell (const GetParticlePosition<PIdx> &GetPosition,
                             const amrex::ParticleReal *const wp,
                             const amrex::ParticleReal *const uxp,
                             const amrex::ParticleReal *const uyp,
                             const amrex::ParticleReal *const uzp,
                             const int *ion_lev,
                             amrex::FArrayBox &jx_fab,
                             amrex::FArrayBox &jy_fab,
                             amrex::FArrayBox &jz_fab,
                             long np_to_deposit,
                             amrex::Real relative_time,
                             const amrex::XDim3 &dinv,
                             const amrex::XDim3 &xyzmin,
                             amrex::Dim3 len,
                             amrex::Real q,
                             [[maybe_unused]] int n_rz_azimuthal_modes)
{
    using namespace std::chrono;
    using namespace amrex::literals;

    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // std::vector<amrex::Real> wqx(np_to_deposit, 0.0_rt), wqy(np_to_deposit, 0.0_rt), wqz(np_to_deposit, 0.0_rt);
    amrex::Real wqx, wqy, wqz;
    
    std::vector<amrex::Real> sx_m(1 + depos_order, 0);
    std::vector<amrex::Real> sy_m(1 + depos_order, 0);
    std::vector<amrex::Real> sz_m(1 + depos_order, 0);

    alignas(64) std::vector<int> newbin(np_to_deposit, -1);

    std::vector<amrex::Real> xrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8 * numcell, 0.0);

    int arr_step[8] = {0, 1, xjstride, xjstride + 1, xkstride, xkstride + 1, xjstride + xkstride, xjstride + xkstride + 1};
    int sx_step[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    int sy_step[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    int sz_step[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    #pragma omp simd
    for (int ip = 0; ip < np_to_deposit; ip++)
    {
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        const amrex::Real gaminv = 1.0_rt / std::sqrt(1.0_rt + uxp[ip] * uxp[ip] * clightsq + uyp[ip] * uyp[ip] * clightsq + uzp[ip] * uzp[ip] * clightsq);
        const amrex::Real vx = uxp[ip] * gaminv;
        const amrex::Real vy = uyp[ip] * gaminv;
        const amrex::Real vz = uzp[ip] * gaminv;

        amrex::Real wq = q * wp[ip];
        if (do_ionization)
        {
            wq *= ion_lev[ip];
        }

        constexpr int NODE = amrex::IndexType::NODE;
        constexpr int CELL = amrex::IndexType::CELL;

        wqx = wq * invvol * vx;
        wqy = wq * invvol * vy;
        wqz = wq * invvol * vz;

        Compute_shape_factor<depos_order> const compute_shape_factor;

        const double xmid = ((xp - xyzmin.x) + relative_time * vx) * dinv.x;

        double sx_node[depos_order + 1] = {0.};
        double sx_cell[depos_order + 1] = {0.};
        int j_node = 0;
        int j_cell = 0;
        if (jx_type[0] == NODE || jy_type[0] == NODE || jz_type[0] == NODE)
        {
            j_node = compute_shape_factor(sx_node, xmid);
        }
        if (jx_type[0] == CELL || jy_type[0] == CELL || jz_type[0] == CELL)
        {
            j_cell = compute_shape_factor(sx_cell, xmid - 0.5);
        }

        for (int ix = 0; ix <= depos_order; ix++)
        {
            sx_m[ix] = ((jx_type[0] == NODE) ? amrex::Real(sx_node[ix]) : amrex::Real(sx_cell[ix]));
        }

        int j_m = ((jz_type[0] == NODE) ? j_node : j_cell);

        const double ymid = ((yp - xyzmin.y) + relative_time * vy) * dinv.y;
        double sy_node[depos_order + 1] = {0.};
        double sy_cell[depos_order + 1] = {0.};
        int k_node = 0;
        int k_cell = 0;

        if (jx_type[1] == NODE || jy_type[1] == NODE || jz_type[1] == NODE)
        {
            k_node = compute_shape_factor(sy_node, ymid);
        }
        if (jx_type[1] == CELL || jy_type[1] == CELL || jz_type[1] == CELL)
        {
            k_cell = compute_shape_factor(sy_cell, ymid - 0.5);
        }
        for (int iy = 0; iy <= depos_order; iy++)
        {
            sy_m[iy] = ((jx_type[1] == NODE) ? amrex::Real(sy_node[iy]) : amrex::Real(sy_cell[iy]));
        }
        
        int k_m = ((jx_type[1] == NODE) ? k_node : k_cell);

        constexpr int zdir = WARPX_ZINDEX;
        const double zmid = ((zp - xyzmin.z) + relative_time * vz) * dinv.z;
        double sz_node[depos_order + 1] = {0.};
        double sz_cell[depos_order + 1] = {0.};
        int l_node = 0;
        int l_cell = 0;
        if (jx_type[zdir] == NODE || jy_type[zdir] == NODE || jz_type[zdir] == NODE)
        {
            l_node = compute_shape_factor(sz_node, zmid);
        }
        if (jx_type[zdir] == CELL || jy_type[zdir] == CELL || jz_type[zdir] == CELL)
        {
            l_cell = compute_shape_factor(sz_cell, zmid - 0.5);
        }
        for (int iz = 0; iz <= depos_order; iz++)
        {
            sz_m[iz] = ((jx_type[zdir] == NODE) ? amrex::Real(sz_node[iz]) : amrex::Real(sz_cell[iz]));
        }
        
        int l_m = ((jz_type[zdir] == NODE) ? l_node : l_cell);
        int ip_deposit_idx = j_m + k_m * xjstride + l_m * xkstride;

        #pragma omp simd
        for(int rho_idx = 0; rho_idx < 8; rho_idx++){
            double sx_m_sy_m_sz = sx_m[sx_step[rho_idx]] * sy_m[sy_step[rho_idx]] * sz_m[sz_step[rho_idx]];
            xrhocells[8 * ip_deposit_idx + rho_idx] += sx_m_sy_m_sz * wqx;
            yrhocells[8 * ip_deposit_idx + rho_idx] += sx_m_sy_m_sz * wqy;
            zrhocells[8 * ip_deposit_idx + rho_idx] += sx_m_sy_m_sz * wqz;
        }
    }

    for (int iz = 0; iz < len.z; ++iz)
    {
        for (int iy = 0; iy < len.y; ++iy)
        {
            #pragma omp simd
            for(int ix = 0; ix < len.x; ix++){
                int arr_offset = ix + iy * xjstride + iz * xkstride;
                int rho_offset = 8 * arr_offset;
                
                jx_arr.p[arr_offset + arr_step[0]] += xrhocells[0 + rho_offset];
                jx_arr.p[arr_offset + arr_step[1]] += xrhocells[1 + rho_offset];
                jx_arr.p[arr_offset + arr_step[2]] += xrhocells[2 + rho_offset];
                jx_arr.p[arr_offset + arr_step[3]] += xrhocells[3 + rho_offset];
                jx_arr.p[arr_offset + arr_step[4]] += xrhocells[4 + rho_offset];
                jx_arr.p[arr_offset + arr_step[5]] += xrhocells[5 + rho_offset];
                jx_arr.p[arr_offset + arr_step[6]] += xrhocells[6 + rho_offset];
                jx_arr.p[arr_offset + arr_step[7]] += xrhocells[7 + rho_offset];
                
                jy_arr.p[arr_offset + arr_step[0]] += yrhocells[0 + rho_offset];
                jy_arr.p[arr_offset + arr_step[1]] += yrhocells[1 + rho_offset];
                jy_arr.p[arr_offset + arr_step[2]] += yrhocells[2 + rho_offset];
                jy_arr.p[arr_offset + arr_step[3]] += yrhocells[3 + rho_offset];
                jy_arr.p[arr_offset + arr_step[4]] += yrhocells[4 + rho_offset];
                jy_arr.p[arr_offset + arr_step[5]] += yrhocells[5 + rho_offset];
                jy_arr.p[arr_offset + arr_step[6]] += yrhocells[6 + rho_offset];
                jy_arr.p[arr_offset + arr_step[7]] += yrhocells[7 + rho_offset];

                jz_arr.p[arr_offset + arr_step[0]] += zrhocells[0 + rho_offset];
                jz_arr.p[arr_offset + arr_step[1]] += zrhocells[1 + rho_offset];
                jz_arr.p[arr_offset + arr_step[2]] += zrhocells[2 + rho_offset];
                jz_arr.p[arr_offset + arr_step[3]] += zrhocells[3 + rho_offset];
                jz_arr.p[arr_offset + arr_step[4]] += zrhocells[4 + rho_offset];
                jz_arr.p[arr_offset + arr_step[5]] += zrhocells[5 + rho_offset];
                jz_arr.p[arr_offset + arr_step[6]] += zrhocells[6 + rho_offset];
                jz_arr.p[arr_offset + arr_step[7]] += zrhocells[7 + rho_offset];
            }
        }
    }
}


template <int depos_order>
void doDepositionShapeN_vpu_rhocell (const GetParticlePosition<PIdx> &GetPosition,
                             const amrex::ParticleReal *const wp,
                             const amrex::ParticleReal *const uxp,
                             const amrex::ParticleReal *const uyp,
                             const amrex::ParticleReal *const uzp,
                             const int *ion_lev,
                             amrex::FArrayBox &jx_fab,
                             amrex::FArrayBox &jy_fab,
                             amrex::FArrayBox &jz_fab,
                             long np_to_deposit,
                             amrex::Real relative_time,
                             const amrex::XDim3 &dinv,
                             const amrex::XDim3 &xyzmin,
                             amrex::Dim3 len,
                             amrex::Real q,
                             [[maybe_unused]] int n_rz_azimuthal_modes)
{
    using namespace std::chrono;
    using namespace amrex::literals;

    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;

    auto compute_shape_factor_vpu = [](double* sx[], Vec xmid, svbool_t p) {
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        sx0.Store(p, sx[0]);
        sx1.Store(p, sx[1]);
        return i_newv;
    };

    int vl = svcntd();

    std::vector<amrex::Real> xrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8 * numcell, 0.0);
    
    constexpr int RHOCELL_SIZE = 8;

    int arr_step[8] = {0, 1, xjstride, xjstride + 1, xkstride, xkstride + 1, xjstride + xkstride, xjstride + xkstride + 1};
    int sx_step[8] = {0, RHOCELL_SIZE, 0, RHOCELL_SIZE, 0, RHOCELL_SIZE, 0, RHOCELL_SIZE};
    int sy_step[8] = {0, 0, RHOCELL_SIZE, RHOCELL_SIZE, 0, 0, RHOCELL_SIZE, RHOCELL_SIZE};
    int sz_step[8] = {0, 0, 0, 0, RHOCELL_SIZE, RHOCELL_SIZE, RHOCELL_SIZE, RHOCELL_SIZE};
    svbool_t p = svwhilelt_b64(0, 8);

    std::vector<amrex::Real> wqx(RHOCELL_SIZE, 0.0_rt), wqy(RHOCELL_SIZE, 0.0_rt), wqz(RHOCELL_SIZE, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(2 * RHOCELL_SIZE, 0);
    std::vector<amrex::Real> sy_m(2 * RHOCELL_SIZE, 0);
    std::vector<amrex::Real> sz_m(2 * RHOCELL_SIZE, 0);

    alignas(64) long newbin[RHOCELL_SIZE]; 

    #pragma omp unroll
    for(long ip = 0; ip < np_to_deposit; ip += vl){
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wp_v=Vec::Load(p_ip,&wp[ip]);
        Vec wq_v = q * wp_v;
        if (do_ionization) 
        {
            wq_v *= ion_lev[ip];
        }
        
        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[0]);
        wqy_v.Store(p_ip, &wqy[0]);
        wqz_v.Store(p_ip, &wqz[0]);

        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        double* sx_temp[2] = {&sx_m[0], &sx_m[RHOCELL_SIZE]};
        double* sy_temp[2] = {&sy_m[0], &sy_m[RHOCELL_SIZE]};
        double* sz_temp[2] = {&sz_m[0], &sz_m[RHOCELL_SIZE]};
        intVec j_cellv = compute_shape_factor_vpu(sx_temp, xmid, p_ip);
        intVec k_cellv = compute_shape_factor_vpu(sy_temp, ymid, p_ip);
        intVec l_cellv = compute_shape_factor_vpu(sz_temp, zmid, p_ip); 

        intVec newbin_v = j_cellv + k_cellv * xjstride + l_cellv * xkstride;
        newbin_v.Store(p_ip, &newbin[0]);

        int rho_end = RHOCELL_SIZE < np_to_deposit  - ip ? RHOCELL_SIZE : np_to_deposit - ip;
        for (long rho_ip = 0; rho_ip < rho_end; rho_ip++)
        {
            int idx = rho_ip;
            int ip_deposit_idx = newbin[idx];

            intVec mx_index = svld1sw_s64(p, sx_step);
            const double* sx_base = &sx_m[idx];
            Vec sx_v = svld1_gather_s64index_f64(p, sx_base, mx_index);

            intVec my_index = svld1sw_s64(p, sy_step);
            const double* sy_base = &sy_m[idx];
            Vec sy_v = svld1_gather_s64index_f64(p, sy_base, my_index);

            intVec mz_index = svld1sw_s64(p, sz_step);
            const double* sz_base = &sz_m[idx];
            Vec sz_v = svld1_gather_s64index_f64(p, sz_base, mz_index);
            Vec sx_m_sy_m_sz_v = sx_v * sy_v * sz_v;

            Vec xrhocells_v = Vec::Load(p, &xrhocells[8 * ip_deposit_idx]);
            xrhocells_v += sx_m_sy_m_sz_v * wqx[idx];
            xrhocells_v.Store(p, &xrhocells[8 * ip_deposit_idx]);
            Vec yrhocells_v = Vec::Load(p, &yrhocells[8 * ip_deposit_idx]);
            yrhocells_v += sx_m_sy_m_sz_v * wqy[idx];
            yrhocells_v.Store(p, &yrhocells[8 * ip_deposit_idx]);
            Vec zrhocells_v = Vec::Load(p, &zrhocells[8 * ip_deposit_idx]);
            zrhocells_v += sx_m_sy_m_sz_v * wqz[idx];
            zrhocells_v.Store(p, &zrhocells[8 * ip_deposit_idx]);
        }
    }

    const long cell_stepx = len.x;
    const long cell_stepxy = cell_stepx * len.y;
    #pragma omp unroll
    for (int iz = 0; iz < len.z; ++iz) {
        #pragma omp unroll
        for (int iy = 0; iy < len.y; ++iy) { 
            #pragma omp unroll      
            for (int ix = 0; ix < len.x; ix+=vl) {
                svbool_t p = svwhilelt_b64(ix, len.x);
                svuint64_t rho_index = svindex_u64(0ULL, 8ULL);

                long arr_offset = ix + iy * xjstride + iz * xkstride;
                long cell_idx = ix + iy * cell_stepx + iz * cell_stepxy + 1;
                long rho_offset = 8 * arr_offset;
                
                Vec jx0 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[0]]);
                svfloat64_t rhox0 = svld1_gather_u64index_f64(p, &xrhocells[0 + rho_offset], rho_index);
                jx0 += rhox0;
                jx0.Store(p, &jx_arr.p[arr_offset + arr_step[0]]);

                Vec jx1 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[1]]);
                svfloat64_t rhox1 = svld1_gather_u64index_f64(p, &xrhocells[1 + rho_offset], rho_index);
                jx1 += rhox1;
                jx1.Store(p, &jx_arr.p[arr_offset + arr_step[1]]);

                Vec jx2 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[2]]);
                svfloat64_t rhox2 = svld1_gather_u64index_f64(p, &xrhocells[2 + rho_offset], rho_index);
                jx2 += rhox2;
                jx2.Store(p, &jx_arr.p[arr_offset + arr_step[2]]);

                Vec jx3 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[3]]);
                svfloat64_t rhox3 = svld1_gather_u64index_f64(p, &xrhocells[3 + rho_offset], rho_index);
                jx3 += rhox3;
                jx3.Store(p, &jx_arr.p[arr_offset + arr_step[3]]);

                Vec jx4 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[4]]);
                svfloat64_t rhox4 = svld1_gather_u64index_f64(p, &xrhocells[4 + rho_offset], rho_index);
                jx4 += rhox4;
                jx4.Store(p, &jx_arr.p[arr_offset + arr_step[4]]);

                Vec jx5 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[5]]);
                svfloat64_t rhox5 = svld1_gather_u64index_f64(p, &xrhocells[5 + rho_offset], rho_index);
                jx5 += rhox5;
                jx5.Store(p, &jx_arr.p[arr_offset + arr_step[5]]);

                Vec jx6 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[6]]);
                svfloat64_t rhox6 = svld1_gather_u64index_f64(p, &xrhocells[6 + rho_offset], rho_index);
                jx6 += rhox6;
                jx6.Store(p, &jx_arr.p[arr_offset + arr_step[6]]);

                Vec jx7 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[7]]);
                svfloat64_t rhox7 = svld1_gather_u64index_f64(p, &xrhocells[7 + rho_offset], rho_index);
                jx7 += rhox7;
                jx7.Store(p, &jx_arr.p[arr_offset + arr_step[7]]);
                
                // y
                Vec jy0 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[0]]);
                svfloat64_t rhoy0 = svld1_gather_u64index_f64(p, &yrhocells[0 + rho_offset], rho_index);
                jy0 += rhoy0;
                jy0.Store(p, &jy_arr.p[arr_offset + arr_step[0]]);

                Vec jy1 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[1]]);
                svfloat64_t rhoy1 = svld1_gather_u64index_f64(p, &yrhocells[1 + rho_offset], rho_index);
                jy1 += rhoy1;
                jy1.Store(p, &jy_arr.p[arr_offset + arr_step[1]]);

                Vec jy2 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[2]]);
                svfloat64_t rhoy2 = svld1_gather_u64index_f64(p, &yrhocells[2 + rho_offset], rho_index);
                jy2 += rhoy2;
                jy2.Store(p, &jy_arr.p[arr_offset + arr_step[2]]);

                Vec jy3 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[3]]);
                svfloat64_t rhoy3 = svld1_gather_u64index_f64(p, &yrhocells[3 + rho_offset], rho_index);
                jy3 += rhoy3;
                jy3.Store(p, &jy_arr.p[arr_offset + arr_step[3]]);

                Vec jy4 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[4]]);
                svfloat64_t rhoy4 = svld1_gather_u64index_f64(p, &yrhocells[4 + rho_offset], rho_index);
                jy4 += rhoy4;
                jy4.Store(p, &jy_arr.p[arr_offset + arr_step[4]]);

                Vec jy5 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[5]]);
                svfloat64_t rhoy5 = svld1_gather_u64index_f64(p, &yrhocells[5 + rho_offset], rho_index);
                jy5 += rhoy5;
                jy5.Store(p, &jy_arr.p[arr_offset + arr_step[5]]);

                Vec jy6 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[6]]);
                svfloat64_t rhoy6 = svld1_gather_u64index_f64(p, &yrhocells[6 + rho_offset], rho_index);
                jy6 += rhoy6;
                jy6.Store(p, &jy_arr.p[arr_offset + arr_step[6]]);

                Vec jy7 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[7]]);
                svfloat64_t rhoy7 = svld1_gather_u64index_f64(p, &yrhocells[7 + rho_offset], rho_index);
                jy7 += rhoy7;
                jy7.Store(p, &jy_arr.p[arr_offset + arr_step[7]]);

                // z
                Vec jz0 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[0]]);
                svfloat64_t rhoz0 = svld1_gather_u64index_f64(p, &zrhocells[0 + rho_offset], rho_index);
                jz0 += rhoz0;
                jz0.Store(p, &jz_arr.p[arr_offset + arr_step[0]]);

                Vec jz1 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[1]]);
                svfloat64_t rhoz1 = svld1_gather_u64index_f64(p, &zrhocells[1 + rho_offset], rho_index);
                jz1 += rhoz1;
                jz1.Store(p, &jz_arr.p[arr_offset + arr_step[1]]);

                Vec jz2 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[2]]);
                svfloat64_t rhoz2 = svld1_gather_u64index_f64(p, &zrhocells[2 + rho_offset], rho_index);
                jz2 += rhoz2;
                jz2.Store(p, &jz_arr.p[arr_offset + arr_step[2]]);

                Vec jz3 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[3]]);
                svfloat64_t rhoz3 = svld1_gather_u64index_f64(p, &zrhocells[3 + rho_offset], rho_index);
                jz3 += rhoz3;
                jz3.Store(p, &jz_arr.p[arr_offset + arr_step[3]]);

                Vec jz4 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[4]]);
                svfloat64_t rhoz4 = svld1_gather_u64index_f64(p, &zrhocells[4 + rho_offset], rho_index);
                jz4 += rhoz4;
                jz4.Store(p, &jz_arr.p[arr_offset + arr_step[4]]);

                Vec jz5 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[5]]);
                svfloat64_t rhoz5 = svld1_gather_u64index_f64(p, &zrhocells[5 + rho_offset], rho_index);
                jz5 += rhoz5;
                jz5.Store(p, &jz_arr.p[arr_offset + arr_step[5]]);

                Vec jz6 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[6]]);
                svfloat64_t rhoz6 = svld1_gather_u64index_f64(p, &zrhocells[6 + rho_offset], rho_index);
                jz6 += rhoz6;
                jz6.Store(p, &jz_arr.p[arr_offset + arr_step[6]]);

                Vec jz7 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[7]]);
                svfloat64_t rhoz7 = svld1_gather_u64index_f64(p, &zrhocells[7 + rho_offset], rho_index);
                jz7 += rhoz7;
                jz7.Store(p, &jz_arr.p[arr_offset + arr_step[7]]);
            }
        }
    }
}


template <int depos_order>
__arc_new("za") inline void vpu_mpu_kernel (
    long np_to_deposit,
    const long ip,
    const std::vector<int64_t> newbin,
    std::vector<amrex::Real> &sx_m,
    std::vector<amrex::Real> &sy_m,
    std::vector<amrex::Real> &sz_m,
    std::vector<amrex::Real> &wqx,
    std::vector<amrex::Real> &wqy,
    std::vector<amrex::Real> &wqz,
    const amrex::Array4<amrex::Real>& jx_arr,
    const amrex::Array4<amrex::Real>& jy_arr,
    const amrex::Array4<amrex::Real>& jz_arr,
    const amrex::Dim3 lo
    ) __arc_streaming
{
    svbool_t p = svwhilele_b64(0, depos_order); 
    int vl = svcntd();
    MVec vzero(0);

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;

    int arr_step[8] = {0, 1, xjstride, xjstride + 1, xkstride, xkstride + 1, xjstride + xkstride, xjstride + xkstride + 1};

    int end = 8 < np_to_deposit - ip ? 8 : np_to_deposit - ip;
    #pragma omp unroll
    for (int iip = 0; iip < end; iip++){
        MVec sxm_v = MVec::Load(p, &sx_m[iip * 2]);
        MVec sym_v = MVec::Load(p, &sy_m[iip * 2]);
        MVec sxm_vx = sxm_v * wqx[iip];
        MVec sxm_vy = sxm_v * wqy[iip];
        MVec sxm_vz = sxm_v * wqz[iip];
        #pragma omp unroll
        for (int iz = 0; iz <= depos_order; iz++){
            
            double sz = sz_m[iip * 2 + iz];
            MVec syzm_v = sym_v * sz;

            svzero_za();
            svmopa_za64_m(0, p, p, syzm_v, sxm_vx);
            svmopa_za64_m(1, p, p, syzm_v, sxm_vy);
            svmopa_za64_m(2, p, p, syzm_v, sxm_vz);

            #pragma omp unroll
            for (int iy = 0; iy <= depos_order; iy++) {

                
                int arr_step_idx = iy * 2 + iz * 4;
                long ip_to_deposit = newbin[iip] + arr_step[arr_step_idx];

                MVec sxsysz_vx = svread_hor_za64_m(vzero, p, 0, iy);
                double* jx_p = &jx_arr.p[ip_to_deposit];
                MVec jx_v = MVec::Load(p, jx_p);
                (jx_v + sxsysz_vx).Store(p, jx_p);

                MVec sxsysz_vy = svread_hor_za64_m(vzero, p, 1, iy);
                double* jy_p = &jy_arr.p[ip_to_deposit];
                MVec jy_v = MVec::Load(p, jy_p);
                (jy_v + sxsysz_vy).Store(p, jy_p);
                
                MVec sxsysz_vz = svread_hor_za64_m(vzero, p, 2, iy);
                double* jz_p = &jz_arr.p[ip_to_deposit];
                MVec jz_v = MVec::Load(p, jz_p);
                (jz_v + sxsysz_vz).Store(p, jz_p);
            }
        }
    }  
}

template <int depos_order>
void doDepositionShapeN_vpu_mpu (const GetParticlePosition<PIdx> &GetPosition,
                                     const amrex::ParticleReal *const wp,
                                     const amrex::ParticleReal *const uxp,
                                     const amrex::ParticleReal *const uyp,
                                     const amrex::ParticleReal *const uzp,
                                     const int *ion_lev,
                                     amrex::FArrayBox &jx_fab,
                                     amrex::FArrayBox &jy_fab,
                                     amrex::FArrayBox &jz_fab,
                                     long np_to_deposit,
                                     amrex::Real relative_time,
                                     const amrex::XDim3 &dinv,
                                     const amrex::XDim3 &xyzmin,
                                     amrex::Dim3 lo,
                                     amrex::Real q,
                                     [[maybe_unused]] int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();


    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    constexpr int ARR_SIZE = 8;

    std::vector<amrex::Real> wqx(ARR_SIZE, 0.0_rt), wqy(ARR_SIZE, 0.0_rt), wqz(ARR_SIZE, 0.0_rt);

    std::vector<amrex::Real> sx_m(2 * ARR_SIZE, 0);
    std::vector<amrex::Real> sy_m(2 * ARR_SIZE, 0);
    std::vector<amrex::Real> sz_m(2 * ARR_SIZE, 0);

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;

    auto compute_shape_factor_vpu_scatter = [](double* sx, Vec xmid, svbool_t p) {
        const svuint64_t sx_index = svindex_u64(0, 2);
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        svst1_scatter_index(p, &sx[0], sx_index, sx0);
        svst1_scatter_index(p, &sx[1], sx_index, sx1);
        return i_newv;
    };

    int vl = svcntd();

    std::vector<int64_t> newbin(ARR_SIZE, 0);

    #pragma omp unroll
    for(long ip = 0; ip < np_to_deposit; ip += vl){
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);

        Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        
        Vec wp_v = Vec::Load(p_ip, &wp[ip]);
        Vec wq_v = q * wp_v;

        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[0]);
        wqy_v.Store(p_ip, &wqy[0]);
        wqz_v.Store(p_ip, &wqz[0]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec j_cellv = compute_shape_factor_vpu_scatter(&sx_m[0], xmid, p_ip);
        intVec k_cellv = compute_shape_factor_vpu_scatter(&sy_m[0], ymid, p_ip);
        intVec l_cellv = compute_shape_factor_vpu_scatter(&sz_m[0], zmid, p_ip);

        intVec newbin_v = j_cellv + k_cellv * xjstride + l_cellv * xkstride;
        newbin_v.Store(p_ip, &newbin[0]);

        vpu_mpu_kernel<depos_order>(
            np_to_deposit, ip,
            newbin, 
            sx_m, sy_m, sz_m,
            wqx, wqy, wqz,
            jx_arr, jy_arr, jz_arr, lo
        );
    }        
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_kernel (
    long np_to_deposit,
    long ip,
    std::vector<long>& newbin,
    std::vector<amrex::Real> &xrhocells,
    std::vector<amrex::Real> &yrhocells,
    std::vector<amrex::Real> &zrhocells,
    std::vector<amrex::Real> &sx_m,
    std::vector<amrex::Real> &sy_m,
    std::vector<amrex::Real> &sz_m,
    std::vector<amrex::Real> &wqx,
    std::vector<amrex::Real> &wqy,
    std::vector<amrex::Real> &wqz
    ) __arc_streaming
{
    using namespace amrex::literals;
    svbool_t p_true = svptrue_b64();
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    svbool_t p_4_7 = svbic_z(p_true, p_true, p_0_3);
    svbool_t p_0_1 = svwhilelt_b64(0, 2);
    svbool_t p_2_3 = svbic_z(p_0_3, p_0_3, p_0_1);
    MVec vzero(0);
    
    if (np_to_deposit - ip >= 8)
    {
        for (int idx = 0; idx < 8; idx += 2)
        {
            int ip1 = idx;
            int ip2 = idx + 1;

            MVec sz_v = svdup_n_f64(0);
            MVec sz_v_ip1 = svld1(p_0_1, &sz_m[ip1 * 2]);
            MVec sz_v_ip2 = svld1(p_2_3, &sz_m[ip1 * 2]);

            sz_v = svsel(p_0_1, sz_v_ip1, sz_v);
            sz_v = svsel(p_2_3, sz_v_ip2, sz_v);

            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vx2 = MVec::Load(p_0_1, &sx_m[ip2 * 2]);

            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            MVec vy2 = MVec::Load(p_0_1, &sy_m[ip2 * 2]);

            const uint64_t x_indices[8] = {0, 2, 0, 2, 1, 3, 1, 3};
            MVec vx_broadcast = svtbl(svzip1(vx1, vx2), svld1_u64(p_true, x_indices));
            const uint64_t y_indices[8] = {0, 0, 2, 2, 1, 1, 3, 3};
            MVec vy_broadcast = svtbl(svzip1(vy1, vy2), svld1_u64(p_true, y_indices));

            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;

            svzero_za();
            svmopa_za64_m(0, p_0_3, p_true, sz_v, sx_m_sy_v);

            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);

            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);

            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);

            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);

            int ip_to_deposit2 = newbin[ip2];
            rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_4_7, 0, 3);
            MVec sxyz_mv_ip2_tmp = vpuxt(sx_m_sy_m_sz_m1, sx_m_sy_m_sz_m1, 4);
            MVec sxyz_mv_ip2 = vpuxt(sx_m_sy_m_sz_m0, sxyz_mv_ip2_tmp, 4);
            rhox_v += sxyz_mv_ip2 * wqx[ip2];
            rhoy_v += sxyz_mv_ip2 * wqy[ip2];
            rhoz_v += sxyz_mv_ip2 * wqz[ip2];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit2]); 
        }
    }
    else
    {
        int end = np_to_deposit - ip;
        int last = end % 2;
        for (int idx = 0; idx < end - last; idx += 2)
        {
            int ip1 = idx;
            int ip2 = idx + 1;
    
            MVec sz_v = svdup_n_f64(0);
            MVec sz_v_ip1 = svld1(p_0_1, &sz_m[ip1 * 2]);
            MVec sz_v_ip2 = svld1(p_2_3, &sz_m[ip1 * 2]);
    
            sz_v = svsel(p_0_1, sz_v_ip1, sz_v);
            sz_v = svsel(p_2_3, sz_v_ip2, sz_v);
  
            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vx2 = MVec::Load(p_0_1, &sx_m[ip2 * 2]);
    
            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            MVec vy2 = MVec::Load(p_0_1, &sy_m[ip2 * 2]);
    
            const uint64_t x_indices[8] = {0, 2, 0, 2, 1, 3, 1, 3};
            MVec vx_broadcast = svtbl(svzip1(vx1, vx2), svld1_u64(p_true, x_indices));
            const uint64_t y_indices[8] = {0, 0, 2, 2, 1, 1, 3, 3};
            MVec vy_broadcast = svtbl(svzip1(vy1, vy2), svld1_u64(p_true, y_indices));
        
            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;
       
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_true, sz_v, sx_m_sy_v);
    
            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);

            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);
    
            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            int ip_to_deposit2 = newbin[ip2];
            rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_4_7, 0, 3);
            MVec sxyz_mv_ip2_tmp = vpuxt(sx_m_sy_m_sz_m1, sx_m_sy_m_sz_m1, 4);
            MVec sxyz_mv_ip2 = vpuxt(sx_m_sy_m_sz_m0, sxyz_mv_ip2_tmp, 4);
            rhox_v += sxyz_mv_ip2 * wqx[ip2];
            rhoy_v += sxyz_mv_ip2 * wqy[ip2];
            rhoz_v += sxyz_mv_ip2 * wqz[ip2];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit2]); 
        }
    
        if(last)
        {
            svzero_za();
    
            int ip1 = end - 1;
            MVec sz_v = svld1(p_0_1, &sz_m[ip1 * 2]);
    
            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            const uint64_t x_indices[8] = {0, 1, 0, 1, 0, 0, 0, 0};
            const uint64_t y_indices[8] = {0, 0, 1, 1, 0, 0, 0, 0};
            MVec vx_broadcast = svtbl(vx1, svld1_u64(p_true, x_indices));
            MVec vy_broadcast = svtbl(vy1, svld1_u64(p_true, y_indices));
    
            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;
            svmopa_za64_m(0, p_0_1, p_0_3, sz_v, sx_m_sy_v);
            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
    
            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);
            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);
        }
    }
    
}

template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_kernel2 (
    long np_to_deposit,
    long ip,
    std::vector<long>& newbin,
    std::vector<amrex::Real> &xrhocells,
    std::vector<amrex::Real> &yrhocells,
    std::vector<amrex::Real> &zrhocells,
    std::vector<amrex::Real> &sx_m,
    std::vector<amrex::Real> &sy_m,
    std::vector<amrex::Real> &sz_m,
    std::vector<amrex::Real> &wqx,
    std::vector<amrex::Real> &wqy,
    std::vector<amrex::Real> &wqz
    ) __arc_streaming
{
    using namespace amrex::literals;
    svbool_t p_true = svptrue_b64();
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    svbool_t p_4_7 = svbic_z(p_true, p_true, p_0_3);
    svbool_t p_0_1 = svwhilelt_b64(0, 2);
    svbool_t p_2_3 = svbic_z(p_0_3, p_0_3, p_0_1);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_5 = svwhilelt_b64(0, 6);
    svbool_t p_6_7 = svbic_z(p_0_7, p_0_7, p_0_5);
    svbool_t p_4_5 = svbic_z(p_4_7, p_4_7, p_6_7);
    MVec vzero(0);
    
    if (np_to_deposit - ip >= 8)
    {
        for (int idx = 0; idx < 8; idx += 4)
        {
            int ip0 = idx;
            int ip1 = idx + 1;
            int ip2 = idx + 2;
            int ip3 = idx + 3;

            MVec sz_v_0_7 = svld1(p_0_7, &sz_m[ip0 * 2]);
            MVec sy_v_0_7 = svld1(p_0_7, &sy_m[ip0 * 2]);

            svzero_za();
            svmopa_za64_m(0, p_0_7, p_0_7, sy_v_0_7, sz_v_0_7);

            MVec m0_row0 = svread_hor_za64_m(vzero, p_0_1, 0, 0);
            MVec m0_row1 = svread_hor_za64_m(vzero, p_0_1, 0, 1);
            MVec m0_row2 = svread_hor_za64_m(vzero, p_2_3, 0, 2);
            MVec m0_row3 = svread_hor_za64_m(vzero, p_2_3, 0, 3);
            MVec m0_row4 = svread_hor_za64_m(vzero, p_4_5, 0, 4);
            MVec m0_row5 = svread_hor_za64_m(vzero, p_4_5, 0, 5);
            MVec m0_row6 = svread_hor_za64_m(vzero, p_6_7, 0, 6);
            MVec m0_row7 = svread_hor_za64_m(vzero, p_6_7, 0, 7);

            MVec s_yz_ip0_zip = svzip1(m0_row0, m0_row1);
            MVec s_yz_ip1_zip = svzip1(m0_row2, m0_row3);
            MVec s_yz_ip0_ip1 = svsel(p_0_3, s_yz_ip0_zip, s_yz_ip1_zip);

            MVec s_yz_ip2_zip = svzip2(m0_row4, m0_row5);
            MVec s_yz_ip3_zip = svzip2(m0_row6, m0_row7);
            MVec s_yz_ip2_ip3 = svsel(p_0_3, s_yz_ip2_zip, s_yz_ip3_zip);

            MVec s_x_ip0_ip1 = svld1(p_0_3, &sx_m[ip0 * 2]);
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_0_7, s_x_ip0_ip1, s_yz_ip0_ip1);
            MVec m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            MVec m0_xyz_row2 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            MVec m0_xyz_row3 = svread_hor_za64_m(vzero, p_4_7, 0, 3);

            MVec s_xyz_ip0_v = svzip1(m0_xyz_row0, m0_xyz_row1);
            MVec s_xyz_ip1_v = svzip2(m0_xyz_row2, m0_xyz_row3);

            MVec s_x_ip2_ip3 = svld1(p_0_3, &sx_m[ip2 * 2]);
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_0_7, s_x_ip2_ip3, s_yz_ip2_ip3);
            m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            m0_xyz_row2 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            m0_xyz_row3 = svread_hor_za64_m(vzero, p_4_7, 0, 3);

            MVec s_xyz_ip2_v = svzip1(m0_xyz_row0, m0_xyz_row1);
            MVec s_xyz_ip3_v = svzip2(m0_xyz_row2, m0_xyz_row3);

            int ip_to_deposit0 = newbin[ip0];
            int ip_to_deposit1 = newbin[ip1];
            int ip_to_deposit2 = newbin[ip2];
            int ip_to_deposit3 = newbin[ip3];
            
            MVec rhox_v0 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit0]);
            MVec rhoy_v0 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit0]);
            MVec rhoz_v0 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit0]);
            rhox_v0 += s_xyz_ip0_v * wqx[ip0];
            rhoy_v0 += s_xyz_ip0_v * wqy[ip0];
            rhoz_v0 += s_xyz_ip0_v * wqz[ip0];
            rhox_v0.Store(p_true, &xrhocells[8 * ip_to_deposit0]);
            rhoy_v0.Store(p_true, &yrhocells[8 * ip_to_deposit0]);
            rhoz_v0.Store(p_true, &zrhocells[8 * ip_to_deposit0]);

            MVec rhox_v1 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v1 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v1 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
            rhox_v1 += s_xyz_ip1_v * wqx[ip1];
            rhoy_v1 += s_xyz_ip1_v * wqy[ip1];
            rhoz_v1 += s_xyz_ip1_v * wqz[ip1];
            rhox_v1.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v1.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v1.Store(p_true, &zrhocells[8 * ip_to_deposit1]);

            MVec rhox_v2 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            MVec rhoy_v2 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            MVec rhoz_v2 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            rhox_v2 += s_xyz_ip2_v * wqx[ip2];
            rhoy_v2 += s_xyz_ip2_v * wqy[ip2];
            rhoz_v2 += s_xyz_ip2_v * wqz[ip2];
            rhox_v2.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v2.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v2.Store(p_true, &zrhocells[8 * ip_to_deposit2]);

            MVec rhox_v3 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit3]);
            MVec rhoy_v3 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit3]);
            MVec rhoz_v3 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit3]);
            rhox_v3 += s_xyz_ip3_v * wqx[ip3];
            rhoy_v3 += s_xyz_ip3_v * wqy[ip3];
            rhoz_v3 += s_xyz_ip3_v * wqz[ip3];
            rhox_v3.Store(p_true, &xrhocells[8 * ip_to_deposit3]);
            rhoy_v3.Store(p_true, &yrhocells[8 * ip_to_deposit3]);
            rhoz_v3.Store(p_true, &zrhocells[8 * ip_to_deposit3]);

        }
    }
    else
    {
        int idx = 0;
        int end = np_to_deposit - ip;
        int last = end % 2;
        if (end >= 4)
        {
            int ip0 = idx;
            int ip1 = idx + 1;
            int ip2 = idx + 2;
            int ip3 = idx + 3;

            MVec sz_v_0_7 = svld1(p_0_7, &sz_m[ip0 * 2]);
            MVec sy_v_0_7 = svld1(p_0_7, &sy_m[ip0 * 2]);

            svzero_za();
            svmopa_za64_m(0, p_0_7, p_0_7, sy_v_0_7, sz_v_0_7);

            MVec m0_row0 = svread_hor_za64_m(vzero, p_0_1, 0, 0);
            MVec m0_row1 = svread_hor_za64_m(vzero, p_0_1, 0, 1);
            MVec m0_row2 = svread_hor_za64_m(vzero, p_2_3, 0, 2);
            MVec m0_row3 = svread_hor_za64_m(vzero, p_2_3, 0, 3);
            MVec m0_row4 = svread_hor_za64_m(vzero, p_4_5, 0, 4);
            MVec m0_row5 = svread_hor_za64_m(vzero, p_4_5, 0, 5);
            MVec m0_row6 = svread_hor_za64_m(vzero, p_6_7, 0, 6);
            MVec m0_row7 = svread_hor_za64_m(vzero, p_6_7, 0, 7);

            MVec s_yz_ip0_zip = svzip1(m0_row0, m0_row1);
            MVec s_yz_ip1_zip = svzip1(m0_row2, m0_row3);
            MVec s_yz_ip0_ip1 = svsel(p_0_3, s_yz_ip0_zip, s_yz_ip1_zip);

            MVec s_yz_ip2_zip = svzip2(m0_row4, m0_row5);
            MVec s_yz_ip3_zip = svzip2(m0_row6, m0_row7);
            MVec s_yz_ip2_ip3 = svsel(p_0_3, s_yz_ip2_zip, s_yz_ip3_zip);

            MVec s_x_ip0_ip1 = svld1(p_0_3, &sx_m[ip0 * 2]);
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_0_7, s_x_ip0_ip1, s_yz_ip0_ip1);
            MVec m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            MVec m0_xyz_row2 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            MVec m0_xyz_row3 = svread_hor_za64_m(vzero, p_4_7, 0, 3);

            MVec s_xyz_ip0_v = svzip1(m0_xyz_row0, m0_xyz_row1);
            MVec s_xyz_ip1_v = svzip2(m0_xyz_row2, m0_xyz_row3);

            MVec s_x_ip2_ip3 = svld1(p_0_3, &sx_m[ip2 * 2]);
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_0_7, s_x_ip2_ip3, s_yz_ip2_ip3);
            m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            m0_xyz_row2 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            m0_xyz_row3 = svread_hor_za64_m(vzero, p_4_7, 0, 3);

            MVec s_xyz_ip2_v = svzip1(m0_xyz_row0, m0_xyz_row1);
            MVec s_xyz_ip3_v = svzip2(m0_xyz_row2, m0_xyz_row3);

            int ip_to_deposit0 = newbin[ip0];
            int ip_to_deposit1 = newbin[ip1];
            int ip_to_deposit2 = newbin[ip2];
            int ip_to_deposit3 = newbin[ip3];
                
            MVec rhox_v0 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit0]);
            MVec rhoy_v0 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit0]);
            MVec rhoz_v0 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit0]);
            rhox_v0 += s_xyz_ip0_v * wqx[ip0];
            rhoy_v0 += s_xyz_ip0_v * wqy[ip0];
            rhoz_v0 += s_xyz_ip0_v * wqz[ip0];
            rhox_v0.Store(p_true, &xrhocells[8 * ip_to_deposit0]);
            rhoy_v0.Store(p_true, &yrhocells[8 * ip_to_deposit0]);
            rhoz_v0.Store(p_true, &zrhocells[8 * ip_to_deposit0]);

            MVec rhox_v1 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v1 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v1 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
            rhox_v1 += s_xyz_ip1_v * wqx[ip1];
            rhoy_v1 += s_xyz_ip1_v * wqy[ip1];
            rhoz_v1 += s_xyz_ip1_v * wqz[ip1];
            rhox_v1.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v1.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v1.Store(p_true, &zrhocells[8 * ip_to_deposit1]);

            MVec rhox_v2 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            MVec rhoy_v2 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            MVec rhoz_v2 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            rhox_v2 += s_xyz_ip2_v * wqx[ip2];
            rhoy_v2 += s_xyz_ip2_v * wqy[ip2];
            rhoz_v2 += s_xyz_ip2_v * wqz[ip2];
            rhox_v2.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v2.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v2.Store(p_true, &zrhocells[8 * ip_to_deposit2]);

            MVec rhox_v3 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit3]);
            MVec rhoy_v3 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit3]);
            MVec rhoz_v3 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit3]);
            rhox_v3 += s_xyz_ip3_v * wqx[ip3];
            rhoy_v3 += s_xyz_ip3_v * wqy[ip3];
            rhoz_v3 += s_xyz_ip3_v * wqz[ip3];
            rhox_v3.Store(p_true, &xrhocells[8 * ip_to_deposit3]);
            rhoy_v3.Store(p_true, &yrhocells[8 * ip_to_deposit3]);
            rhoz_v3.Store(p_true, &zrhocells[8 * ip_to_deposit3]);

            idx += 4;
            end -= 4;
        }
        
        
        if (end >= 2)
        {
            int ip1 = idx;
            int ip2 = idx + 1;
    
            MVec sz_v = svdup_n_f64(0);
            MVec sz_v_ip1 = svld1(p_0_1, &sz_m[ip1 * 2]);
            MVec sz_v_ip2 = svld1(p_2_3, &sz_m[ip1 * 2]);
    
            sz_v = svsel(p_0_1, sz_v_ip1, sz_v);
            sz_v = svsel(p_2_3, sz_v_ip2, sz_v);
    
            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vx2 = MVec::Load(p_0_1, &sx_m[ip2 * 2]);
    
            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            MVec vy2 = MVec::Load(p_0_1, &sy_m[ip2 * 2]);
    
            const uint64_t x_indices[8] = {0, 2, 0, 2, 1, 3, 1, 3};
            MVec vx_broadcast = svtbl(svzip1(vx1, vx2), svld1_u64(p_true, x_indices));
            const uint64_t y_indices[8] = {0, 0, 2, 2, 1, 1, 3, 3};
            MVec vy_broadcast = svtbl(svzip1(vy1, vy2), svld1_u64(p_true, y_indices));

            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;
    
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_true, sz_v, sx_m_sy_v);
    
            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);

            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);
    
            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            int ip_to_deposit2 = newbin[ip2];
            rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_4_7, 0, 3);
            MVec sxyz_mv_ip2_tmp = vpuxt(sx_m_sy_m_sz_m1, sx_m_sy_m_sz_m1, 4);
            MVec sxyz_mv_ip2 = vpuxt(sx_m_sy_m_sz_m0, sxyz_mv_ip2_tmp, 4);
            rhox_v += sxyz_mv_ip2 * wqx[ip2];
            rhoy_v += sxyz_mv_ip2 * wqy[ip2];
            rhoz_v += sxyz_mv_ip2 * wqz[ip2];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit2]); 

            end -= 2;
            idx += 2;
        }
    
        if(end >= 1)
        {
            svzero_za();
    
            int ip1 = idx;
            MVec sz_v = svld1(p_0_1, &sz_m[ip1 * 2]);
    
            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            const uint64_t x_indices[8] = {0, 1, 0, 1, 0, 0, 0, 0};
            const uint64_t y_indices[8] = {0, 0, 1, 1, 0, 0, 0, 0};
            MVec vx_broadcast = svtbl(vx1, svld1_u64(p_true, x_indices));
            MVec vy_broadcast = svtbl(vy1, svld1_u64(p_true, y_indices));
    
            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;
            svmopa_za64_m(0, p_0_1, p_0_3, sz_v, sx_m_sy_v);
            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
    
            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);
            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);
        }
    }
    
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_kernel3 (
    long np_to_deposit,
    long ip,
    std::vector<long>& newbin,
    std::vector<amrex::Real> &xrhocells,
    std::vector<amrex::Real> &yrhocells,
    std::vector<amrex::Real> &zrhocells,
    std::vector<amrex::Real> &sx_m,
    std::vector<amrex::Real> &sy_m,
    std::vector<amrex::Real> &sz_m,
    std::vector<amrex::Real> &wqx,
    std::vector<amrex::Real> &wqy,
    std::vector<amrex::Real> &wqz
    ) __arc_streaming
{
    using namespace amrex::literals;
    svbool_t p_true = svptrue_b64();
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    svbool_t p_4_7 = svbic_z(p_true, p_true, p_0_3);
    svbool_t p_0_1 = svwhilelt_b64(0, 2);
    svbool_t p_2_3 = svbic_z(p_0_3, p_0_3, p_0_1);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_5 = svwhilelt_b64(0, 6);
    svbool_t p_6_7 = svbic_z(p_0_7, p_0_7, p_0_5);
    svbool_t p_4_5 = svbic_z(p_4_7, p_4_7, p_6_7);
    MVec vzero(0);

    int wq_step[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    intVec wq_index = svld1sw_s64(p_true, wq_step);
    
    if (np_to_deposit - ip >= 8)
    {
        for (int idx = 0; idx < 8; idx += 4)
        {
            int ip0 = idx;
            int ip1 = idx + 1;
            int ip2 = idx + 2;
            int ip3 = idx + 3;

            MVec wqx_v = svld1_gather_s64index_f64(p_0_7, &wqx[ip0], wq_index);
            MVec sx_v_0_7 = svld1(p_0_7, &sx_m[ip0 * 2]);
            MVec sx_0_7_wqx_v = sx_v_0_7 * wqx_v;

            MVec wqy_v = svld1_gather_s64index_f64(p_0_7, &wqy[ip0], wq_index);
            MVec sy_v_0_7 = svld1(p_0_7, &sy_m[ip0 * 2]);
            MVec sx_0_7_wqy_v = sx_v_0_7 * wqy_v;

            MVec wqz_v = svld1_gather_s64index_f64(p_0_7, &wqz[ip0], wq_index);
            MVec sz_v_0_7 = svld1(p_0_7, &sz_m[ip0 * 2]);
            MVec sx_0_7_wqz_v = sx_v_0_7 * wqz_v;

            svzero_za();
            svmopa_za64_m(0, p_0_7, p_0_7, sy_v_0_7, sz_v_0_7);

            MVec m0_row0 = svread_hor_za64_m(vzero, p_0_1, 0, 0);
            MVec m0_row1 = svread_hor_za64_m(vzero, p_0_1, 0, 1);
            MVec m0_row2 = svread_hor_za64_m(vzero, p_2_3, 0, 2);
            MVec m0_row3 = svread_hor_za64_m(vzero, p_2_3, 0, 3);
            MVec m0_row4 = svread_hor_za64_m(vzero, p_4_5, 0, 4);
            MVec m0_row5 = svread_hor_za64_m(vzero, p_4_5, 0, 5);
            MVec m0_row6 = svread_hor_za64_m(vzero, p_6_7, 0, 6);
            MVec m0_row7 = svread_hor_za64_m(vzero, p_6_7, 0, 7);

            MVec s_yz_ip0_zip = svzip1(m0_row0, m0_row1);
            MVec s_yz_ip1_zip = svzip1(m0_row2, m0_row3);
            MVec s_yz_ip0_ip1 = svsel(p_0_3, s_yz_ip0_zip, s_yz_ip1_zip);

            MVec s_yz_ip2_zip = svzip2(m0_row4, m0_row5);
            MVec s_yz_ip3_zip = svzip2(m0_row6, m0_row7);
            MVec s_yz_ip2_ip3 = svsel(p_0_3, s_yz_ip2_zip, s_yz_ip3_zip);

            MVec sx_wqx_wqy_ip0_ip1 = svzip1(sx_0_7_wqx_v, sx_0_7_wqy_v);
            MVec sx_wqx_wqy_ip2_ip3 = svzip2(sx_0_7_wqx_v, sx_0_7_wqy_v);

            svzero_za();
            svmopa_za64_m(0, p_0_7, p_0_7, sx_wqx_wqy_ip0_ip1, s_yz_ip0_ip1);
            MVec m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            MVec m0_xyz_row2 = svread_hor_za64_m(vzero, p_0_3, 0, 2);
            MVec m0_xyz_row3 = svread_hor_za64_m(vzero, p_0_3, 0, 3);
            MVec m0_xyz_row4 = svread_hor_za64_m(vzero, p_4_7, 0, 4);
            MVec m0_xyz_row5 = svread_hor_za64_m(vzero, p_4_7, 0, 5);
            MVec m0_xyz_row6 = svread_hor_za64_m(vzero, p_4_7, 0, 6);
            MVec m0_xyz_row7 = svread_hor_za64_m(vzero, p_4_7, 0, 7);

            MVec s_xyz_wqx_v_ip0 = svzip1(m0_xyz_row0, m0_xyz_row2);
            MVec s_xyz_wqy_v_ip0 = svzip1(m0_xyz_row1, m0_xyz_row3);
            MVec s_xyz_wqx_v_ip1 = svzip2(m0_xyz_row4, m0_xyz_row6);
            MVec s_xyz_wqy_v_ip1 = svzip2(m0_xyz_row5, m0_xyz_row7);

            svzero_za();
            svmopa_za64_m(0, p_0_7, p_0_7, sx_wqx_wqy_ip2_ip3, s_yz_ip2_ip3);
            m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            m0_xyz_row2 = svread_hor_za64_m(vzero, p_0_3, 0, 2);
            m0_xyz_row3 = svread_hor_za64_m(vzero, p_0_3, 0, 3);
            m0_xyz_row4 = svread_hor_za64_m(vzero, p_4_7, 0, 4);
            m0_xyz_row5 = svread_hor_za64_m(vzero, p_4_7, 0, 5);
            m0_xyz_row6 = svread_hor_za64_m(vzero, p_4_7, 0, 6);
            m0_xyz_row7 = svread_hor_za64_m(vzero, p_4_7, 0, 7);

            MVec s_xyz_wqx_v_ip2 = svzip1(m0_xyz_row0, m0_xyz_row2);
            MVec s_xyz_wqy_v_ip2 = svzip1(m0_xyz_row1, m0_xyz_row3);
            MVec s_xyz_wqx_v_ip3 = svzip2(m0_xyz_row4, m0_xyz_row6);
            MVec s_xyz_wqy_v_ip3 = svzip2(m0_xyz_row5, m0_xyz_row7);

            svzero_za();
            svmopa_za64_m(0, p_0_3, p_0_7, sx_0_7_wqz_v, s_yz_ip0_ip1);
            m0_xyz_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            m0_xyz_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            m0_xyz_row2 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            m0_xyz_row3 = svread_hor_za64_m(vzero, p_4_7, 0, 3);

            MVec s_xyz_wqz_v_ip0 = svzip1(m0_xyz_row0, m0_xyz_row1);
            MVec s_xyz_wqz_v_ip1 = svzip2(m0_xyz_row2, m0_xyz_row3);
            
            svzero_za();
            svmopa_za64_m(0, p_4_7, p_0_7, sx_0_7_wqz_v, s_yz_ip2_ip3);
            m0_xyz_row4 = svread_hor_za64_m(vzero, p_0_3, 0, 4);
            m0_xyz_row5 = svread_hor_za64_m(vzero, p_0_3, 0, 5);
            m0_xyz_row6 = svread_hor_za64_m(vzero, p_4_7, 0, 6);
            m0_xyz_row7 = svread_hor_za64_m(vzero, p_4_7, 0, 7);

            MVec s_xyz_wqz_v_ip2 = svzip1(m0_xyz_row4, m0_xyz_row5);
            MVec s_xyz_wqz_v_ip3 = svzip2(m0_xyz_row6, m0_xyz_row7);

            int ip_to_deposit0 = newbin[ip0];
            int ip_to_deposit1 = newbin[ip1];
            int ip_to_deposit2 = newbin[ip2];
            int ip_to_deposit3 = newbin[ip3];
            
            MVec rhox_v0 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit0]);
            MVec rhoy_v0 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit0]);
            MVec rhoz_v0 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit0]);
            rhox_v0 += s_xyz_wqx_v_ip0;
            rhoy_v0 += s_xyz_wqy_v_ip0;
            rhoz_v0 += s_xyz_wqz_v_ip0;

            rhox_v0.Store(p_true, &xrhocells[8 * ip_to_deposit0]);
            rhoy_v0.Store(p_true, &yrhocells[8 * ip_to_deposit0]);
            rhoz_v0.Store(p_true, &zrhocells[8 * ip_to_deposit0]);

            MVec rhox_v1 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v1 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v1 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
            rhox_v1 += s_xyz_wqx_v_ip1;
            rhoy_v1 += s_xyz_wqy_v_ip1;
            rhoz_v1 += s_xyz_wqz_v_ip1;
            rhox_v1.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v1.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v1.Store(p_true, &zrhocells[8 * ip_to_deposit1]);

            MVec rhox_v2 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            MVec rhoy_v2 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            MVec rhoz_v2 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            rhox_v2 += s_xyz_wqx_v_ip2;
            rhoy_v2 += s_xyz_wqy_v_ip2;
            rhoz_v2 += s_xyz_wqz_v_ip2;
            rhox_v2.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v2.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v2.Store(p_true, &zrhocells[8 * ip_to_deposit2]);

            MVec rhox_v3 = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit3]);
            MVec rhoy_v3 = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit3]);
            MVec rhoz_v3 = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit3]);
            rhox_v3 += s_xyz_wqx_v_ip3;
            rhoy_v3 += s_xyz_wqy_v_ip3;
            rhoz_v3 += s_xyz_wqz_v_ip3;
            rhox_v3.Store(p_true, &xrhocells[8 * ip_to_deposit3]);
            rhoy_v3.Store(p_true, &yrhocells[8 * ip_to_deposit3]);
            rhoz_v3.Store(p_true, &zrhocells[8 * ip_to_deposit3]);
        }
    }
    else
    {
        int end = np_to_deposit - ip;
        int last = end % 2;
        for (int idx = 0; idx < end - last; idx += 2)
        {
            int ip1 = idx;
            int ip2 = idx + 1;
    
            MVec sz_v = svdup_n_f64(0);
            MVec sz_v_ip1 = svld1(p_0_1, &sz_m[ip1 * 2]);
            MVec sz_v_ip2 = svld1(p_2_3, &sz_m[ip1 * 2]);
    
            sz_v = svsel(p_0_1, sz_v_ip1, sz_v);
            sz_v = svsel(p_2_3, sz_v_ip2, sz_v);
    
            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vx2 = MVec::Load(p_0_1, &sx_m[ip2 * 2]);
    
            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            MVec vy2 = MVec::Load(p_0_1, &sy_m[ip2 * 2]);
    
            const uint64_t x_indices[8] = {0, 2, 0, 2, 1, 3, 1, 3};
            MVec vx_broadcast = svtbl(svzip1(vx1, vx2), svld1_u64(p_true, x_indices));
            const uint64_t y_indices[8] = {0, 0, 2, 2, 1, 1, 3, 3};
            MVec vy_broadcast = svtbl(svzip1(vy1, vy2), svld1_u64(p_true, y_indices));

            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;
    
            svzero_za();
            svmopa_za64_m(0, p_0_3, p_true, sz_v, sx_m_sy_v);
    
            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);

            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);
    
            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            int ip_to_deposit2 = newbin[ip2];
            rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit2]);
            sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_4_7, 0, 2);
            sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_4_7, 0, 3);
            MVec sxyz_mv_ip2_tmp = vpuxt(sx_m_sy_m_sz_m1, sx_m_sy_m_sz_m1, 4);
            MVec sxyz_mv_ip2 = vpuxt(sx_m_sy_m_sz_m0, sxyz_mv_ip2_tmp, 4);
            rhox_v += sxyz_mv_ip2 * wqx[ip2];
            rhoy_v += sxyz_mv_ip2 * wqy[ip2];
            rhoz_v += sxyz_mv_ip2 * wqz[ip2];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit2]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit2]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit2]); 
        }
    
    
        if(last)
        {
            svzero_za();
    
            int ip1 = end - 1;
            MVec sz_v = svld1(p_0_1, &sz_m[ip1 * 2]);
    
            MVec vx1 = MVec::Load(p_0_1, &sx_m[ip1 * 2]);
            MVec vy1 = MVec::Load(p_0_1, &sy_m[ip1 * 2]);
            const uint64_t x_indices[8] = {0, 1, 0, 1, 0, 0, 0, 0};
            const uint64_t y_indices[8] = {0, 0, 1, 1, 0, 0, 0, 0};
            MVec vx_broadcast = svtbl(vx1, svld1_u64(p_true, x_indices));
            MVec vy_broadcast = svtbl(vy1, svld1_u64(p_true, y_indices));
    
            MVec sx_m_sy_v = vx_broadcast * vy_broadcast;
            svmopa_za64_m(0, p_0_1, p_0_3, sz_v, sx_m_sy_v);
            MVec sx_m_sy_m_sz_m0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec sx_m_sy_m_sz_m1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
    
            int ip_to_deposit1 = newbin[ip1];
            MVec rhox_v = MVec::Load(p_true, &xrhocells[8 * ip_to_deposit1]);
            MVec rhoy_v = MVec::Load(p_true, &yrhocells[8 * ip_to_deposit1]);
            MVec rhoz_v = MVec::Load(p_true, &zrhocells[8 * ip_to_deposit1]);
    
            MVec sxyz_mv_ip1 = svsplice(p_0_3, sx_m_sy_m_sz_m0, sx_m_sy_m_sz_m1);
            rhox_v += sxyz_mv_ip1 * wqx[ip1];
            rhoy_v += sxyz_mv_ip1 * wqy[ip1];
            rhoz_v += sxyz_mv_ip1 * wqz[ip1];
            rhox_v.Store(p_true, &xrhocells[8 * ip_to_deposit1]);
            rhoy_v.Store(p_true, &yrhocells[8 * ip_to_deposit1]);
            rhoz_v.Store(p_true, &zrhocells[8 * ip_to_deposit1]);
        }
    }
    
}

template <int depos_order>
void doDepositionShapeN_vpu_rhocell_mpu(const GetParticlePosition<PIdx> &GetPosition,
                                        const amrex::ParticleReal *const wp,
                                        const amrex::ParticleReal *const uxp,
                                        const amrex::ParticleReal *const uyp,
                                        const amrex::ParticleReal *const uzp,
                                        const int *ion_lev,
                                        amrex::FArrayBox &jx_fab,
                                        amrex::FArrayBox &jy_fab,
                                        amrex::FArrayBox &jz_fab,
                                        long np_to_deposit,
                                        amrex::Real relative_time,
                                        const amrex::XDim3 &dinv,
                                        const amrex::XDim3 &xyzmin,
                                        amrex::Dim3 len,
                                        amrex::Real q,
                                        [[maybe_unused]] int n_rz_azimuthal_modes)
{
    using namespace std::chrono;
    using namespace amrex::literals;

    amrex::Array4<amrex::Real> const &jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const &jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const &jz_arr = jz_fab.array();

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;
    const int yjstride = jy_arr.jstride;
    const int ykstride = jy_arr.kstride;
    const int zjstride = jz_arr.jstride;
    const int zkstride = jz_arr.kstride;

    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x * dinv.y * dinv.z;

    const amrex::Real clightsq = 1.0_rt / PhysConst::c / PhysConst::c;

    const long numcell = jx_fab.box().numPts();

    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    constexpr int RHOCELL_SIZE = 8;

    std::vector<amrex::Real> wqx(RHOCELL_SIZE, 0.0_rt), wqy(RHOCELL_SIZE, 0.0_rt), wqz(RHOCELL_SIZE, 0.0_rt);
    
    std::vector<amrex::Real> sx_m(2 * RHOCELL_SIZE, 0);
    std::vector<amrex::Real> sy_m(2 * RHOCELL_SIZE, 0);
    std::vector<amrex::Real> sz_m(2 * RHOCELL_SIZE, 0);

    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;

    auto compute_shape_factor_vpu_scatter = [](double* sx, Vec xmid, svbool_t p) {
        const svuint64_t sx_index = svindex_u64(0, 2);
        intVec i_newv = svcvt_s64_f64_z(p, xmid);
        Vec j = svcvt_f64_s64_z(p, i_newv);
        Vec xint = xmid - j;
        Vec sx0 = 1.0 - xint;
        Vec sx1 = xint;
        svst1_scatter_index(p, &sx[0], sx_index, sx0);
        svst1_scatter_index(p, &sx[1], sx_index, sx1);
        return i_newv;
    };

    int vl = svcntd();
    std::vector<long> newbin(RHOCELL_SIZE);

    std::vector<amrex::Real> xrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> yrhocells(8 * numcell, 0.0);
    std::vector<amrex::Real> zrhocells(8 * numcell, 0.0);

    #pragma omp unroll
    for(long ip = 0; ip < np_to_deposit; ip += vl){
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        Vec uxp_v = Vec::Load(p_ip, &uxp[ip]);
        Vec uyp_v = Vec::Load(p_ip, &uyp[ip]);
        Vec uzp_v = Vec::Load(p_ip, &uzp[ip]);
        Vec gaminv = 1._rt / (1._rt + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        
        Vec vx_v = uxp_v * gaminv;
        Vec vy_v = uyp_v * gaminv;
        Vec vz_v = uzp_v * gaminv;

        Vec wp_v=Vec::Load(p_ip,&wp[ip]);
        Vec wq_v = q * wp_v;
        if (do_ionization) 
        {
            wq_v *= ion_lev[ip];
        }
        
        Vec wqx_v = wq_v * invvol * vx_v;
        Vec wqy_v = wq_v * invvol * vy_v;
        Vec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[0]);
        wqy_v.Store(p_ip, &wqy[0]);
        wqz_v.Store(p_ip, &wqz[0]);

        Vec xp = Vec::Load(p_ip, &mx[ip]);
        Vec yp = Vec::Load(p_ip, &my[ip]);
        Vec zp = Vec::Load(p_ip, &mz[ip]);

        Vec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        Vec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        Vec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        intVec j_cellv = compute_shape_factor_vpu_scatter(&sx_m[0], xmid, p_ip);
        intVec k_cellv = compute_shape_factor_vpu_scatter(&sy_m[0], ymid, p_ip);
        intVec l_cellv = compute_shape_factor_vpu_scatter(&sz_m[0], zmid, p_ip); 

        intVec newbin_v = j_cellv + k_cellv * xjstride + l_cellv * xkstride;
        newbin_v.Store(p_ip, &newbin[0]);

        vpu_rhocell_mpu_kernel2<1>(np_to_deposit, ip,
            newbin,
            xrhocells, yrhocells, zrhocells,
            sx_m, sy_m, sz_m,
            wqx, wqy, wqz);
    }

    int arr_step[8] = {0, 1, xjstride, xjstride + 1, xkstride, xkstride + 1, xjstride + xkstride, xjstride + xkstride + 1};

    const long cell_stepx = len.x;
    const long cell_stepxy = cell_stepx * len.y;
    #pragma omp unroll
    for (int iz = 0; iz < len.z; ++iz) {
        #pragma omp unroll
        for (int iy = 0; iy < len.y; ++iy) { 
            #pragma omp unroll      
            for (int ix = 0; ix < len.x; ix+=vl) {
                svbool_t p = svwhilelt_b64(ix, len.x);
                svuint64_t rho_index = svindex_u64(0ULL, 8ULL);

                long arr_offset = ix + iy * xjstride + iz * xkstride;
                long cell_idx = ix + iy * cell_stepx + iz * cell_stepxy+1;
                long rho_offset = 8 * arr_offset;
                
                Vec jx0 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[0]]);
                svfloat64_t rhox0 = svld1_gather_u64index_f64(p, &xrhocells[0 + rho_offset], rho_index);
                jx0 += rhox0;
                jx0.Store(p, &jx_arr.p[arr_offset + arr_step[0]]);

                Vec jx1 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[1]]);
                svfloat64_t rhox1 = svld1_gather_u64index_f64(p, &xrhocells[1 + rho_offset], rho_index);
                jx1 += rhox1;
                jx1.Store(p, &jx_arr.p[arr_offset + arr_step[1]]);

                Vec jx2 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[2]]);
                svfloat64_t rhox2 = svld1_gather_u64index_f64(p, &xrhocells[2 + rho_offset], rho_index);
                jx2 += rhox2;
                jx2.Store(p, &jx_arr.p[arr_offset + arr_step[2]]);

                Vec jx3 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[3]]);
                svfloat64_t rhox3 = svld1_gather_u64index_f64(p, &xrhocells[3 + rho_offset], rho_index);
                jx3 += rhox3;
                jx3.Store(p, &jx_arr.p[arr_offset + arr_step[3]]);

                Vec jx4 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[4]]);
                svfloat64_t rhox4 = svld1_gather_u64index_f64(p, &xrhocells[4 + rho_offset], rho_index);
                jx4 += rhox4;
                jx4.Store(p, &jx_arr.p[arr_offset + arr_step[4]]);

                Vec jx5 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[5]]);
                svfloat64_t rhox5 = svld1_gather_u64index_f64(p, &xrhocells[5 + rho_offset], rho_index);
                jx5 += rhox5;
                jx5.Store(p, &jx_arr.p[arr_offset + arr_step[5]]);

                Vec jx6 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[6]]);
                svfloat64_t rhox6 = svld1_gather_u64index_f64(p, &xrhocells[6 + rho_offset], rho_index);
                jx6 += rhox6;
                jx6.Store(p, &jx_arr.p[arr_offset + arr_step[6]]);

                Vec jx7 = Vec::Load(p, &jx_arr.p[arr_offset + arr_step[7]]);
                svfloat64_t rhox7 = svld1_gather_u64index_f64(p, &xrhocells[7 + rho_offset], rho_index);
                jx7 += rhox7;
                jx7.Store(p, &jx_arr.p[arr_offset + arr_step[7]]);
                
                // y
                Vec jy0 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[0]]);
                svfloat64_t rhoy0 = svld1_gather_u64index_f64(p, &yrhocells[0 + rho_offset], rho_index);
                jy0 += rhoy0;
                jy0.Store(p, &jy_arr.p[arr_offset + arr_step[0]]);

                Vec jy1 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[1]]);
                svfloat64_t rhoy1 = svld1_gather_u64index_f64(p, &yrhocells[1 + rho_offset], rho_index);
                jy1 += rhoy1;
                jy1.Store(p, &jy_arr.p[arr_offset + arr_step[1]]);

                Vec jy2 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[2]]);
                svfloat64_t rhoy2 = svld1_gather_u64index_f64(p, &yrhocells[2 + rho_offset], rho_index);
                jy2 += rhoy2;
                jy2.Store(p, &jy_arr.p[arr_offset + arr_step[2]]);

                Vec jy3 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[3]]);
                svfloat64_t rhoy3 = svld1_gather_u64index_f64(p, &yrhocells[3 + rho_offset], rho_index);
                jy3 += rhoy3;
                jy3.Store(p, &jy_arr.p[arr_offset + arr_step[3]]);

                Vec jy4 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[4]]);
                svfloat64_t rhoy4 = svld1_gather_u64index_f64(p, &yrhocells[4 + rho_offset], rho_index);
                jy4 += rhoy4;
                jy4.Store(p, &jy_arr.p[arr_offset + arr_step[4]]);

                Vec jy5 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[5]]);
                svfloat64_t rhoy5 = svld1_gather_u64index_f64(p, &yrhocells[5 + rho_offset], rho_index);
                jy5 += rhoy5;
                jy5.Store(p, &jy_arr.p[arr_offset + arr_step[5]]);

                Vec jy6 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[6]]);
                svfloat64_t rhoy6 = svld1_gather_u64index_f64(p, &yrhocells[6 + rho_offset], rho_index);
                jy6 += rhoy6;
                jy6.Store(p, &jy_arr.p[arr_offset + arr_step[6]]);

                Vec jy7 = Vec::Load(p, &jy_arr.p[arr_offset + arr_step[7]]);
                svfloat64_t rhoy7 = svld1_gather_u64index_f64(p, &yrhocells[7 + rho_offset], rho_index);
                jy7 += rhoy7;
                jy7.Store(p, &jy_arr.p[arr_offset + arr_step[7]]);

                // z
                Vec jz0 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[0]]);
                svfloat64_t rhoz0 = svld1_gather_u64index_f64(p, &zrhocells[0 + rho_offset], rho_index);
                jz0 += rhoz0;
                jz0.Store(p, &jz_arr.p[arr_offset + arr_step[0]]);

                Vec jz1 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[1]]);
                svfloat64_t rhoz1 = svld1_gather_u64index_f64(p, &zrhocells[1 + rho_offset], rho_index);
                jz1 += rhoz1;
                jz1.Store(p, &jz_arr.p[arr_offset + arr_step[1]]);

                Vec jz2 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[2]]);
                svfloat64_t rhoz2 = svld1_gather_u64index_f64(p, &zrhocells[2 + rho_offset], rho_index);
                jz2 += rhoz2;
                jz2.Store(p, &jz_arr.p[arr_offset + arr_step[2]]);

                Vec jz3 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[3]]);
                svfloat64_t rhoz3 = svld1_gather_u64index_f64(p, &zrhocells[3 + rho_offset], rho_index);
                jz3 += rhoz3;
                jz3.Store(p, &jz_arr.p[arr_offset + arr_step[3]]);

                Vec jz4 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[4]]);
                svfloat64_t rhoz4 = svld1_gather_u64index_f64(p, &zrhocells[4 + rho_offset], rho_index);
                jz4 += rhoz4;
                jz4.Store(p, &jz_arr.p[arr_offset + arr_step[4]]);

                Vec jz5 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[5]]);
                svfloat64_t rhoz5 = svld1_gather_u64index_f64(p, &zrhocells[5 + rho_offset], rho_index);
                jz5 += rhoz5;
                jz5.Store(p, &jz_arr.p[arr_offset + arr_step[5]]);

                Vec jz6 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[6]]);
                svfloat64_t rhoz6 = svld1_gather_u64index_f64(p, &zrhocells[6 + rho_offset], rho_index);
                jz6 += rhoz6;
                jz6.Store(p, &jz_arr.p[arr_offset + arr_step[6]]);

                Vec jz7 = Vec::Load(p, &jz_arr.p[arr_offset + arr_step[7]]);
                svfloat64_t rhoz7 = svld1_gather_u64index_f64(p, &zrhocells[7 + rho_offset], rho_index);
                jz7 += rhoz7;
                jz7.Store(p, &jz_arr.p[arr_offset + arr_step[7]]);
            }
        }
    }
}


// >>>>>>>>>>>>>>>>>>>order3<<<<<<<<<<<<<<<<<<<<<<<<


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_order3_kernel (
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    const double* sx_m,
    const double* sy_m,
    const double* sz_m,
    const std::vector<amrex::Real>& wqx,
    const std::vector<amrex::Real>& wqy,
    const std::vector<amrex::Real>& wqz,
    const long* rho_bin,
    int lenx,
    int leny,
    int lenz,
    long ip, 
    long np_to_deposit
) __arc_streaming {
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    svbool_t p_4_7 = svbic_z(p_true, p_true, p_0_3);
    MVec vzero(0);

    int sx_step[8] = {0, 1 * block_size, 2 * block_size, 3 * block_size, 0, 1 * block_size, 2 * block_size, 3 * block_size};
    int sy_step[8] = {0, 0, 0, 0,block_size,block_size,block_size,block_size};
    int sm_step[8] = {0, 1 * block_size, 2 * block_size, 3 * block_size, 1, 1 + 1 * block_size, 1 + 2 * block_size, 1 + 3 * block_size};
    const uint64_t xy_indices[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    const int nrho = lenx * leny * lenz;

    int rho_end = block_size < np_to_deposit  - ip ? block_size : np_to_deposit - ip;
    if (rho_end == 8)
    {
        for (long iip = 0; iip < rho_end; iip += 2) 
        {
            int ip0 = iip;
            int ip1 = iip + 1;
            int rho_bin_idx0 = rho_bin[ip0];
            int rho_bin_idx1 = rho_bin[ip1];

            intVec sm_index = svld1sw_s64(p_true, sm_step);
            MVec sx_v = svld1_gather_s64index_f64(p_true, &sx_m[ip0], sm_index);
            MVec sy_v = svld1_gather_s64index_f64(p_true, &sy_m[ip0], sm_index);

            svzero_za();
            svmopa_za64_m(0, p_0_7, p_0_7, sy_v, sx_v);
            MVec m0_row0 = svread_hor_za64_m(vzero, p_0_3, 0, 0);
            MVec m0_row1 = svread_hor_za64_m(vzero, p_0_3, 0, 1);
            MVec m0_row2 = svread_hor_za64_m(vzero, p_0_3, 0, 2);
            MVec m0_row3 = svread_hor_za64_m(vzero, p_0_3, 0, 3);
            MVec m0_row4 = svread_hor_za64_m(vzero, p_4_7, 0, 4);
            MVec m0_row5 = svread_hor_za64_m(vzero, p_4_7, 0, 5);
            MVec m0_row6 = svread_hor_za64_m(vzero, p_4_7, 0, 6);
            MVec m0_row7 = svread_hor_za64_m(vzero, p_4_7, 0, 7);
            
            MVec s_x_y_01_ip0 = svtbl(svzip1(m0_row0, m0_row1), svld1_u64(p_true, xy_indices));
            MVec s_x_y_23_ip0 = svtbl(svzip1(m0_row2, m0_row3), svld1_u64(p_true, xy_indices));
            MVec s_x_y_01_ip1 = svtbl(svzip2(m0_row4, m0_row5), svld1_u64(p_true, xy_indices));
            MVec s_x_y_23_ip1 = svtbl(svzip2(m0_row6, m0_row7), svld1_u64(p_true, xy_indices));

            for (int iz = 0; iz < 4; ++iz) 
            {
                double sz_wqx_tmp0 = sz_m[iz * block_size + ip0] * wqx[ip0];
                double sz_wqy_tmp0 = sz_m[iz * block_size + ip0] * wqy[ip0];
                double sz_wqz_tmp0 = sz_m[iz * block_size + ip0] * wqz[ip0];

                double sz_wqx_tmp1 = sz_m[iz * block_size + ip1] * wqx[ip1];
                double sz_wqy_tmp1 = sz_m[iz * block_size + ip1] * wqy[ip1];
                double sz_wqz_tmp1 = sz_m[iz * block_size + ip1] * wqz[ip1];

                int rho_offset0_y_01 = (rho_bin_idx0 + 0 * lenx + iz * lenx * leny) * 8;

                MVec xrhocells_v_ip0_y_01 = MVec::Load(p_true, &xrhocells[rho_offset0_y_01]);
                MVec yrhocells_v_ip0_y_01 = MVec::Load(p_true, &yrhocells[rho_offset0_y_01]);
                MVec zrhocells_v_ip0_y_01 = MVec::Load(p_true, &zrhocells[rho_offset0_y_01]);

                xrhocells_v_ip0_y_01 += s_x_y_01_ip0 * sz_wqx_tmp0;
                yrhocells_v_ip0_y_01 += s_x_y_01_ip0 * sz_wqy_tmp0;
                zrhocells_v_ip0_y_01 += s_x_y_01_ip0 * sz_wqz_tmp0;

                xrhocells_v_ip0_y_01.Store(p_true, &xrhocells[rho_offset0_y_01]);
                yrhocells_v_ip0_y_01.Store(p_true, &yrhocells[rho_offset0_y_01]);
                zrhocells_v_ip0_y_01.Store(p_true, &zrhocells[rho_offset0_y_01]);

                int rho_offset_ip0_y_23 = (rho_bin_idx0 + 2 * lenx + iz * lenx * leny) * 8;
                MVec xrhocells_v_ip0_y_23 = MVec::Load(p_true, &xrhocells[rho_offset_ip0_y_23]);
                MVec yrhocells_v_ip0_y_23 = MVec::Load(p_true, &yrhocells[rho_offset_ip0_y_23]);
                MVec zrhocells_v_ip0_y_23 = MVec::Load(p_true, &zrhocells[rho_offset_ip0_y_23]);

                xrhocells_v_ip0_y_23 += s_x_y_23_ip0 * sz_wqx_tmp0;
                yrhocells_v_ip0_y_23 += s_x_y_23_ip0 * sz_wqy_tmp0;
                zrhocells_v_ip0_y_23 += s_x_y_23_ip0 * sz_wqz_tmp0;

                xrhocells_v_ip0_y_23.Store(p_true, &xrhocells[rho_offset_ip0_y_23]);
                yrhocells_v_ip0_y_23.Store(p_true, &yrhocells[rho_offset_ip0_y_23]);
                zrhocells_v_ip0_y_23.Store(p_true, &zrhocells[rho_offset_ip0_y_23]);

                int rho_offset_ip1_y_01 = (rho_bin_idx1 + 0 * lenx + iz * lenx * leny) * 8;

                MVec xrhocells_v_ip1_y_01 = MVec::Load(p_true, &xrhocells[rho_offset_ip1_y_01]);
                MVec yrhocells_v_ip1_y_01 = MVec::Load(p_true, &yrhocells[rho_offset_ip1_y_01]);
                MVec zrhocells_v_ip1_y_01 = MVec::Load(p_true, &zrhocells[rho_offset_ip1_y_01]);

                xrhocells_v_ip1_y_01 += s_x_y_01_ip1 * sz_wqx_tmp1;
                yrhocells_v_ip1_y_01 += s_x_y_01_ip1 * sz_wqy_tmp1;
                zrhocells_v_ip1_y_01 += s_x_y_01_ip1 * sz_wqz_tmp1;

                xrhocells_v_ip1_y_01.Store(p_true, &xrhocells[rho_offset_ip1_y_01]);
                yrhocells_v_ip1_y_01.Store(p_true, &yrhocells[rho_offset_ip1_y_01]);
                zrhocells_v_ip1_y_01.Store(p_true, &zrhocells[rho_offset_ip1_y_01]);

                int rho_offset_ip1_y_23 = (rho_bin_idx1 + 2 * lenx + iz * lenx * leny) * 8;

                MVec xrhocells_v_ip1_y_23 = MVec::Load(p_true, &xrhocells[rho_offset_ip1_y_23]);
                MVec yrhocells_v_ip1_y_23 = MVec::Load(p_true, &yrhocells[rho_offset_ip1_y_23]);
                MVec zrhocells_v_ip1_y_23 = MVec::Load(p_true, &zrhocells[rho_offset_ip1_y_23]);

                xrhocells_v_ip1_y_23 += s_x_y_23_ip1 * sz_wqx_tmp1;
                yrhocells_v_ip1_y_23 += s_x_y_23_ip1 * sz_wqy_tmp1;
                zrhocells_v_ip1_y_23 += s_x_y_23_ip1 * sz_wqz_tmp1;

                xrhocells_v_ip1_y_23.Store(p_true, &xrhocells[rho_offset_ip1_y_23]);
                yrhocells_v_ip1_y_23.Store(p_true, &yrhocells[rho_offset_ip1_y_23]);
                zrhocells_v_ip1_y_23.Store(p_true, &zrhocells[rho_offset_ip1_y_23]);
            }
        }
    }
    else 
    {
        for (long iip = 0; iip < rho_end; iip++) 
        {
            int rho_bin_idx = rho_bin[iip];
            for (int iz = 0; iz < 4; ++iz) 
            {
                double sz_wqx_tmp = sz_m[iz * block_size + iip] * wqx[iip];
                double sz_wqy_tmp = sz_m[iz * block_size + iip] * wqy[iip];
                double sz_wqz_tmp = sz_m[iz * block_size + iip] * wqz[iip];
                for (int iy = 0; iy < 4; iy += 2) 
                {
                    int rho_offset = (rho_bin_idx + iy * lenx + iz * lenx * leny) * 8;
    
                    intVec mx_index = svld1sw_s64(p_true, sx_step);
                    intVec my_index = svld1sw_s64(p_true, sy_step);
    
                    Vec sx_v = svld1_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    Vec sy_v = svld1_gather_s64index_f64(p_true, &sy_m[iy * block_size + iip], my_index);
    
                    Vec sx_m_sy_v = sx_v * sy_v;
    
                    Vec xrhocells_v = Vec::Load(p_true, &xrhocells[rho_offset]);
                    Vec yrhocells_v = Vec::Load(p_true, &yrhocells[rho_offset]);
                    Vec zrhocells_v = Vec::Load(p_true, &zrhocells[rho_offset]);
    
                    xrhocells_v += sx_m_sy_v * sz_wqx_tmp;
                    yrhocells_v += sx_m_sy_v * sz_wqy_tmp;
                    zrhocells_v += sx_m_sy_v * sz_wqz_tmp;
    
                    xrhocells_v.Store(p_true, &xrhocells[rho_offset]);
                    yrhocells_v.Store(p_true, &yrhocells[rho_offset]);
                    zrhocells_v.Store(p_true, &zrhocells[rho_offset]);
                }
            }
        }
    }
}

template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_order3_kernel2 (
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    const double* sx_m,
    const double* sy_m,
    const double* sz_m,
    const std::vector<amrex::Real>& wqx,
    const std::vector<amrex::Real>& wqy,
    const std::vector<amrex::Real>& wqz,
    const long* rho_bin,
    int lenx,
    int leny,
    int lenz,
    long ip, 
    long np_to_deposit
) __arc_streaming {
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    int sx_step[8] = {0, 1 * block_size, 2 * block_size, 3 * block_size, 0, 1 * block_size, 2 * block_size, 3 * block_size};
    int sy_step[8] = {0, 0, 0, 0,block_size,block_size,block_size,block_size};
    int sm_step[8] = {0, 1 * block_size, 2 * block_size, 3 * block_size, 1, 1 + 1 * block_size, 1 + 2 * block_size, 1 + 3 * block_size};

    const svuint64_t sm_index = svindex_u64(0, 8);

    const int nrho = lenx * leny * lenz;

    int rho_end = block_size < np_to_deposit  - ip ? block_size : np_to_deposit - ip;
    
    for (int iip = 0; iip < rho_end; ++iip)
    {
        svzero_za();
        int rho_bin_idx = rho_bin[iip];

        MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);
        MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
        MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
        MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

        svmopa_za64_m(0, p_0_3, p_0_7, sz_v, sx_y_01_v);
        MVec sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
        MVec sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
        MVec sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
        MVec sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

        int rho_offset_iy01_z0 = (rho_bin_idx + 0 * lenx + 0 * lenx * leny) * 8;
        MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
        MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
        MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

        xrhocells_iy01_z0_v += sx_y01_z0 * wqx[iip];
        yrhocells_iy01_z0_v += sx_y01_z0 * wqy[iip];
        zrhocells_iy01_z0_v += sx_y01_z0 * wqz[iip];

        xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
        yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
        zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

        int rho_offset_iy01_z1 = (rho_bin_idx + 0 * lenx + 1 * lenx * leny) * 8;
        MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
        MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
        MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);

        xrhocells_iy01_z1_v += sx_y01_z1 * wqx[iip];
        yrhocells_iy01_z1_v += sx_y01_z1 * wqy[iip];
        zrhocells_iy01_z1_v += sx_y01_z1 * wqz[iip];

        xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
        yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
        zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);

        int rho_offset_iy01_z2 = (rho_bin_idx + 0 * lenx + 2 * lenx * leny) * 8;
        MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
        MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
        MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);

        xrhocells_iy01_z2_v += sx_y01_z2 * wqx[iip];
        yrhocells_iy01_z2_v += sx_y01_z2 * wqy[iip];
        zrhocells_iy01_z2_v += sx_y01_z2 * wqz[iip];

        xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
        yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
        zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);

        int rho_offset_iy01_z3 = (rho_bin_idx + 0 * lenx + 3 * lenx * leny) * 8;
        MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
        MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
        MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);

        xrhocells_iy01_z3_v += sx_y01_z3 * wqx[iip];
        yrhocells_iy01_z3_v += sx_y01_z3 * wqy[iip];
        zrhocells_iy01_z3_v += sx_y01_z3 * wqz[iip];

        xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
        yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
        zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);

        MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
        MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
        svmopa_za64_m(1, p_0_3, p_0_7, sz_v, sx_y_23_v);
        MVec sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
        MVec sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
        MVec sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
        MVec sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

        int rho_offset_iy23_z0 = (rho_bin_idx + 2 * lenx + 0 * lenx * leny) * 8;
        MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
        MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
        MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);

        xrhocells_iy23_z0_v += sx_y23_z0 * wqx[iip];
        yrhocells_iy23_z0_v += sx_y23_z0 * wqy[iip];
        zrhocells_iy23_z0_v += sx_y23_z0 * wqz[iip];

        xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
        yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
        zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);

        int rho_offset_iy23_z1 = (rho_bin_idx + 2 * lenx + 1 * lenx * leny) * 8;
        MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
        MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
        MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);

        xrhocells_iy23_z1_v += sx_y23_z1 * wqx[iip];
        yrhocells_iy23_z1_v += sx_y23_z1 * wqy[iip];
        zrhocells_iy23_z1_v += sx_y23_z1 * wqz[iip];

        xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
        yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
        zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);

        int rho_offset_iy23_z2 = (rho_bin_idx + 2 * lenx + 2 * lenx * leny) * 8;
        MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
        MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
        MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);

        xrhocells_iy23_z2_v += sx_y23_z2 * wqx[iip];
        yrhocells_iy23_z2_v += sx_y23_z2 * wqy[iip];
        zrhocells_iy23_z2_v += sx_y23_z2 * wqz[iip];

        xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
        yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
        zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);

        int rho_offset_iy23_z3 = (rho_bin_idx + 2 * lenx + 3 * lenx * leny) * 8;
        MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
        MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
        MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);

        xrhocells_iy23_z3_v += sx_y23_z3 * wqx[iip];
        yrhocells_iy23_z3_v += sx_y23_z3 * wqy[iip];
        zrhocells_iy23_z3_v += sx_y23_z3 * wqz[iip];

        xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
        yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
        zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
    }
}


inline MintVec Mvec_compute_shape_factor_vpu_order3(double* sx, MVec xmid, svbool_t p) __arc_preserves("za") __arc_streaming
{
    constexpr int VEC_LEN = 8;
    MintVec i_newv = svcvt_s64_f64_z(p, xmid);

    MVec j = svrintz_x(p, xmid);
    MVec xint = xmid - j;
    MVec one_minus_xint = 1.0 - xint;

    MVec sx0 = (1.0 / 6.0) * one_minus_xint * one_minus_xint * one_minus_xint;
    sx0.Store(p, &sx[0 * VEC_LEN]);
    MVec sx1 = (2.0 / 3.0) - xint * xint * (1.0 - xint * 0.5);
    sx1.Store(p, &sx[1 * VEC_LEN]);
    MVec sx2 = (2.0 / 3.0) - one_minus_xint * one_minus_xint * (1.0 - 0.5 *one_minus_xint);
    sx2.Store(p, &sx[2 * VEC_LEN]);
    MVec sx3 = (1.0 / 6.0) * xint * xint * xint;
    sx3.Store(p, &sx[3 * VEC_LEN]);

    return i_newv - 1;
}

template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_order3_large_kernel (
    const amrex::ParticleReal * const wp,
    const amrex::ParticleReal * const uxp,
    const amrex::ParticleReal * const uyp,
    const amrex::ParticleReal * const uzp,
    const amrex::ParticleReal* mx,
    const amrex::ParticleReal* my,
    const amrex::ParticleReal* mz,
    amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    amrex::Real q,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    int lenx,
    int leny,
    int lenz,
    long np_to_deposit
) __arc_streaming {
    constexpr int nshapes = 1 + 3;
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    const svuint64_t sm_index = svindex_u64(0, 8);

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0/PhysConst::c/PhysConst::c;

    std::vector<int64_t> j_cell(block_size, 0);
    std::vector<int64_t> k_cell(block_size, 0);
    std::vector<int64_t> l_cell(block_size, 0);
    std::vector<amrex::Real> wqx(block_size, 0.0);
    std::vector<amrex::Real> wqy(block_size, 0.0);
    std::vector<amrex::Real> wqz(block_size, 0.0);
    double sx_m[nshapes * block_size] = {0.};
    double sy_m[nshapes * block_size] = {0.};
    double sz_m[nshapes * block_size] = {0.};
    alignas(64) long rho_bin[8];

    const int vl = svcntd();

    for(long ip = 0; ip < np_to_deposit; ip += vl){    
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        MVec uxp_v = MVec::Load(p_ip, &uxp[ip]);
        MVec uyp_v = MVec::Load(p_ip, &uyp[ip]);
        MVec uzp_v = MVec::Load(p_ip, &uzp[ip]);
        MVec gaminv = 1. / (1. + uxp_v * uxp_v * clightsq
                                    + uyp_v * uyp_v * clightsq
                                    + uzp_v * uzp_v * clightsq).Sqrt();
        MVec vx_v = uxp_v * gaminv;
        MVec vy_v = uyp_v * gaminv;
        MVec vz_v = uzp_v * gaminv;

        MVec wp_v = MVec::Load(p_ip, &wp[ip]);
        
        MVec wq_v = q * wp_v;

        MVec wqx_v = wq_v * invvol * vx_v;
        MVec wqy_v = wq_v * invvol * vy_v;
        MVec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[0]);
        wqy_v.Store(p_ip, &wqy[0]);
        wqz_v.Store(p_ip, &wqz[0]);

        MVec xp = MVec::Load(p_ip, &mx[ip]);
        MVec yp = MVec::Load(p_ip, &my[ip]);
        MVec zp = MVec::Load(p_ip, &mz[ip]);

        MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        
        MintVec j_cellv = Mvec_compute_shape_factor_vpu_order3(&sx_m[0], xmid, p_ip);
        MintVec k_cellv = Mvec_compute_shape_factor_vpu_order3(&sy_m[0], ymid, p_ip);
        MintVec l_cellv = Mvec_compute_shape_factor_vpu_order3(&sz_m[0], zmid, p_ip);

        j_cellv.Store(p_ip, &j_cell[0]);
        k_cellv.Store(p_ip, &k_cell[0]);
        l_cellv.Store(p_ip, &l_cell[0]);

        MintVec rho_bin_v = j_cellv + k_cellv * lenx + l_cellv * lenx * leny;
        rho_bin_v.Store(p_ip, &rho_bin[0]);
        
        int rho_end = block_size < np_to_deposit  - ip ? block_size : np_to_deposit - ip;
    
        for (int iip = 0; iip < rho_end; ++iip)
        {
            svzero_za();
            int rho_bin_idx = rho_bin[iip];
    
            MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);
            MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
            MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
            MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
            MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);
    
            svmopa_za64_m(0, p_0_3, p_0_7, sz_v, sx_y_01_v);
            MVec sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
            MVec sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
            MVec sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
            MVec sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

            int rho_offset_iy01_z0 = (rho_bin_idx + 0 * lenx + 0 * lenx * leny) * 8;
            MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
            MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
            MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);
    
            xrhocells_iy01_z0_v += sx_y01_z0 * wqx[iip];
            yrhocells_iy01_z0_v += sx_y01_z0 * wqy[iip];
            zrhocells_iy01_z0_v += sx_y01_z0 * wqz[iip];
    
            xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
            yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
            zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);
    
            int rho_offset_iy01_z1 = (rho_bin_idx + 0 * lenx + 1 * lenx * leny) * 8;
            MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
            MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
            MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
    
            xrhocells_iy01_z1_v += sx_y01_z1 * wqx[iip];
            yrhocells_iy01_z1_v += sx_y01_z1 * wqy[iip];
            zrhocells_iy01_z1_v += sx_y01_z1 * wqz[iip];
    
            xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
            yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
            zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
    
            int rho_offset_iy01_z2 = (rho_bin_idx + 0 * lenx + 2 * lenx * leny) * 8;
            MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
            MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
            MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
    
            xrhocells_iy01_z2_v += sx_y01_z2 * wqx[iip];
            yrhocells_iy01_z2_v += sx_y01_z2 * wqy[iip];
            zrhocells_iy01_z2_v += sx_y01_z2 * wqz[iip];
    
            xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
            yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
            zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
    
            int rho_offset_iy01_z3 = (rho_bin_idx + 0 * lenx + 3 * lenx * leny) * 8;
            MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
            MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
            MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
    
            xrhocells_iy01_z3_v += sx_y01_z3 * wqx[iip];
            yrhocells_iy01_z3_v += sx_y01_z3 * wqy[iip];
            zrhocells_iy01_z3_v += sx_y01_z3 * wqz[iip];
    
            xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
            yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
            zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
    
            MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
            MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
            MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
            svmopa_za64_m(1, p_0_3, p_0_7, sz_v, sx_y_23_v);
            MVec sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
            MVec sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
            MVec sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
            MVec sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);
    
            int rho_offset_iy23_z0 = (rho_bin_idx + 2 * lenx + 0 * lenx * leny) * 8;
            MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
            MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
            MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
    
            xrhocells_iy23_z0_v += sx_y23_z0 * wqx[iip];
            yrhocells_iy23_z0_v += sx_y23_z0 * wqy[iip];
            zrhocells_iy23_z0_v += sx_y23_z0 * wqz[iip];
    
            xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
            yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
            zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
    
            int rho_offset_iy23_z1 = (rho_bin_idx + 2 * lenx + 1 * lenx * leny) * 8;
            MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
            MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
            MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
    
            xrhocells_iy23_z1_v += sx_y23_z1 * wqx[iip];
            yrhocells_iy23_z1_v += sx_y23_z1 * wqy[iip];
            zrhocells_iy23_z1_v += sx_y23_z1 * wqz[iip];
    
            xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
            yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
            zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
    
            int rho_offset_iy23_z2 = (rho_bin_idx + 2 * lenx + 2 * lenx * leny) * 8;
            MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
            MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
            MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
    
            xrhocells_iy23_z2_v += sx_y23_z2 * wqx[iip];
            yrhocells_iy23_z2_v += sx_y23_z2 * wqy[iip];
            zrhocells_iy23_z2_v += sx_y23_z2 * wqz[iip];
    
            xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
            yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
            zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
    
            int rho_offset_iy23_z3 = (rho_bin_idx + 2 * lenx + 3 * lenx * leny) * 8;
            MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
            MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
            MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
    
            xrhocells_iy23_z3_v += sx_y23_z3 * wqx[iip];
            yrhocells_iy23_z3_v += sx_y23_z3 * wqy[iip];
            zrhocells_iy23_z3_v += sx_y23_z3 * wqz[iip];
    
            xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
            yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
            zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
        }
    }    
}


using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;


inline void increment_deposit_rebuild(ParticleTileType& ptile, amrex::Box& box)
{
    int& max_bin_length = ptile.max_bin_length;
    int max_bin_length_rebuild = 2 * max_bin_length;

    const int d_num_bins = ptile.d_num_bins;
    const int d_real_num_bins = ptile.d_real_num_bins;

    std::vector<int> d_incr_bin_offset_rebuild(d_num_bins, -1);
    std::vector<int> d_incr_bin_length_rebuild(d_num_bins, 0);
    std::vector<int> d_local_index_rebuild(max_bin_length_rebuild * d_real_num_bins, ParticleTileType::INVALID_PARTICLE_ID);

    vector<int>& d_incr_bin_offset = ptile.d_incr_bin_offset;
    std::vector<int>& d_incr_bin_length = ptile.d_incr_bin_length;;
    vector<int>& d_local_index = ptile.d_local_index;

    const amrex::Dim3 len = length(box);
    const int lenx = len.x;
    const int leny = len.y;
    const int lenz = len.z;

    int local_index_head = 0;

    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int bin = j_node + k_node * lenx + l_node * lenx * leny;
                int oldbin_offset = d_incr_bin_offset[bin];
                int oldbin_length = d_incr_bin_length[bin];

                if (oldbin_offset == -1) continue;

                int& rebuildbin_offset = d_incr_bin_offset_rebuild[bin];
                int& rebuildbin_length = d_incr_bin_length_rebuild[bin];

                rebuildbin_offset = local_index_head;
                rebuildbin_length = oldbin_length;

                for (int i = 0; i < rebuildbin_length; ++i)
                {
                    int idx = oldbin_offset + i;
                    int rebuildbin_idx = rebuildbin_offset + i;

                    d_local_index_rebuild[rebuildbin_idx] = d_local_index[idx];
                }

                local_index_head += max_bin_length_rebuild;
            }
        }
    }

    d_incr_bin_offset.swap(d_incr_bin_offset_rebuild);
    d_incr_bin_length.swap(d_incr_bin_length_rebuild);
    d_local_index.swap(d_local_index_rebuild);

    max_bin_length = max_bin_length_rebuild;
}

inline void increment_sort_deposit(ParticleTileType& ptile, 
                                   int* newbin, long np, int lenx, int leny, int lenz, 
                                   amrex::Box& real_box, amrex::Box& box,
                                   vector<vector<int>>& outside_bin_ip)
{
    if (!ptile.is_incr_sort_initialized)
    {
        ptile.init_incr_sort(real_box, box);
    }


    std::vector<int>& d_old_outside_index = ptile.d_old_outside_index;
    int& d_old_np = ptile.d_old_np;

    int thread_num = omp_get_thread_num();
    int m_init_np = WarpX::GetInstance().m_init_np;
    int* pending_moves = WarpX::GetInstance().pending_moves + thread_num * m_init_np;
    int pending_moves_num = 0;
    int num_bins = lenx * leny * lenz;

    int vlf = svcntw();
    svint32_t np_v = svdup_n_s32(np);
    svint32_t invalid_particle_id_v = svdup_n_s32(ParticleTileType::INVALID_PARTICLE_ID);

    // Stage 1: Process Newly Added Particles
    for (int ip = d_old_np; ip < np; ip += vlf)
    {
        svbool_t p_ip = svwhilelt_b32(ip, (int32_t)np);
        svint32_t ip_v = svindex_s32(ip, 1);
        int32_t ip_num = svcntp_b32(p_ip, p_ip);
        svst1_s32(p_ip, pending_moves + pending_moves_num, ip_v);
        pending_moves_num += ip_num;
    }

    // for (int ip = d_old_np; ip < np; ++ip)
    // {
    //     pending_moves[pending_moves_num++] = ip;
    // }

    // Stage 2 & 3.1: Identify Moved Existing Particles and Delete From Old Cell

    int old_outside_num = d_old_outside_index.size();
    for (int i = 0; i < old_outside_num; i += vlf)
    {
        svbool_t p_ip = svwhilelt_b32(i, old_outside_num);
        svint32_t ip_v = svld1_s32(p_ip, &d_old_outside_index[i]);
        
        svbool_t ip_valid = svcmplt_s32(p_ip, ip_v, np_v);
        int32_t valid_num = svcntp_b32(ip_valid, ip_valid);
        svbool_t p_valid = svwhilelt_b32(0, valid_num);

        svint32_t old_outside_ips_v = svcompact(ip_valid, ip_v);
        svst1_s32(p_valid, pending_moves + pending_moves_num, old_outside_ips_v);
        pending_moves_num += valid_num;
    }

    // for (int i = 0; i < d_old_outside_index.size(); ++i)
    // {
    //     int ip = d_old_outside_index[i];

    //     if (ip >= np)
    //     {
    //         continue;
    //     }
    //     else
    //     {
    //         pending_moves[pending_moves_num++] = ip;
    //     }
    // }


    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                vector<int>& d_incr_bin_offset = ptile.d_incr_bin_offset;
                std::vector<int>& d_incr_bin_length = ptile.d_incr_bin_length;;
                vector<int>& d_local_index = ptile.d_local_index;
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int& oldbin_offset = d_incr_bin_offset[old_bin];
                int& oldbin_length = d_incr_bin_length[old_bin];

                svint32_t old_bin_v = svdup_n_s32(old_bin);


                if (oldbin_offset == -1)
                {
                    continue;
                }

                int* invalid_idx = WarpX::GetInstance().invalid_idx + thread_num * m_init_np;
                int invalid_num = 0;


                for (int i = 0; i < oldbin_length; i += vlf)
                {

                    svbool_t p_ip = svwhilelt_b32(i, oldbin_length);
                    int idx = oldbin_offset + i;
                    svint32_t idx_v = svindex_s32(idx, 1);
                    svint32_t ip_v = svld1_s32(p_ip, &d_local_index[idx]);
                    


                    svbool_t ip_invalid = svcmpge_s32(p_ip, ip_v, np_v);
                    

                    svbool_t ip_valid = svbic_b_z(p_ip, p_ip, ip_invalid);
                    svint32_t newbin_v = svld1_gather_s32index_s32(ip_valid, newbin, ip_v);
                    svbool_t ip_move = svcmpne_s32(ip_valid, newbin_v, old_bin_v);
                    


                    svbool_t ip_to_remove = svorr_b_z(p_ip, ip_invalid, ip_move);


                    int32_t ip_move_num = svcntp_b32(ip_valid, ip_move);
                    if (ip_move_num > 0)
                    {
                        svbool_t p_move = svwhilelt_b32(0, ip_move_num);
                        svint32_t move_ips_v = svcompact(ip_move, ip_v);
                        svst1_s32(p_move, pending_moves + pending_moves_num, move_ips_v);
                        pending_moves_num += ip_move_num;
                    }


                    int32_t ip_to_remove_num = svcntp_b32(p_ip, ip_to_remove);
                    if (ip_to_remove_num > 0)
                    {
                        svbool_t p_remove = svwhilelt_b32(0, ip_to_remove_num);
                        svint32_t remove_idx_v = svcompact(ip_to_remove, idx_v);
                        

                        svst1_s32(p_remove, invalid_idx + invalid_num, remove_idx_v);
                        invalid_num += ip_to_remove_num;
                        

                        svst1_scatter_s32index_s32(p_remove, d_local_index.data(), remove_idx_v, invalid_particle_id_v);
                    }
                }

                // for (int i = 0; i < oldbin_length; ++i)
                // {
                //     int idx = oldbin_offset + i;
                //     int ip = d_local_index[idx];

                //     if (ip >= np)
                //     {
                //         invalid_idx[invalid_num++] = idx;

                //     }
                //     else if (newbin[ip] != old_bin)
                //     {
                //         invalid_idx[invalid_num++] = idx;

                //         pending_moves[pending_moves_num++] = ip;
                //     }                    
                // }


                int oldbin_tail = oldbin_offset + oldbin_length - 1;
                int invalid_tail = invalid_num - 1;

                while (invalid_tail >= 0 && oldbin_tail - invalid_idx[invalid_tail] < vlf)
                {
                    int idx = invalid_idx[invalid_tail];
                    d_local_index[idx] = d_local_index[oldbin_tail];
                    d_local_index[oldbin_tail] = ParticleTileType::INVALID_PARTICLE_ID;

                    oldbin_tail--;
                    invalid_tail--;
                }

                while (invalid_tail >= 0)
                {
                    svbool_t p_ip = svwhilele_b32(0, invalid_tail);
                    int32_t num_invalid = svcntp_b32(p_ip, p_ip);

                    svint32_t idx_v = svld1_s32(p_ip, invalid_idx + invalid_tail + 1 - num_invalid);
                    svint32_t ip_v = svld1_s32(p_ip, d_local_index.data() + oldbin_tail + 1 - num_invalid);
                    svst1_scatter_s32index_s32(p_ip, d_local_index.data(), idx_v, ip_v);
                    svst1_s32(p_ip, d_local_index.data() + oldbin_tail + 1 - num_invalid, invalid_particle_id_v);
                    oldbin_tail -= num_invalid;
                    invalid_tail -= num_invalid;
                }

                // for (int i = invalid_num - 1; i >= 0; --i)
                // {
                //     int idx = invalid_idx[i];
                //     d_local_index[idx] = d_local_index[oldbin_tail];
                //     d_local_index[oldbin_tail] = ParticleTileType::INVALID_PARTICLE_ID;

                //     oldbin_tail--;
                // }

                oldbin_length = oldbin_tail + 1 - oldbin_offset;
            }
        }
    }
    
    d_old_np = np;

    d_old_outside_index.clear();

    // Stage 3.2: Insert Operation

    for (int i = 0; i < pending_moves_num; ++i)
    {
        int ip = pending_moves[i];
        int newbin_ip = newbin[ip];

        if (ptile.d_incr_bin_offset[newbin_ip] == -1)
        {
            d_old_outside_index.push_back(ip);
            outside_bin_ip[newbin_ip].push_back(ip);
        }
        else
        {
            if (ptile.d_incr_bin_length[newbin_ip] >= ptile.max_bin_length)
            {
                increment_deposit_rebuild(ptile, box);
            }

            int& newbin_offset = ptile.d_incr_bin_offset[newbin_ip];
            int& newbin_length = ptile.d_incr_bin_length[newbin_ip];
            int newbin_tail = newbin_offset + newbin_length;

            ptile.d_local_index[newbin_tail] = ip;
            newbin_length++;
        }
    }

}

inline void Mvec_compute_shape_factor_part_vpu_order3(double* sx, MVec xmid, svbool_t p) __arc_preserves("za") __arc_streaming
{
    constexpr int VEC_LEN = 8;

    MVec j = svrintz_x(p, xmid);
    MVec xint = xmid - j;
    MVec one_minus_xint = 1.0 - xint;

    MVec sx0 = (1.0 / 6.0) * one_minus_xint * one_minus_xint * one_minus_xint;
    sx0.Store(p, &sx[0 * VEC_LEN]);
    MVec sx1 = (2.0 / 3.0) - xint * xint * (1.0 - xint * 0.5);
    sx1.Store(p, &sx[1 * VEC_LEN]);
    MVec sx2 = (2.0 / 3.0) - one_minus_xint * one_minus_xint * (1.0 - 0.5 *one_minus_xint);
    sx2.Store(p, &sx[2 * VEC_LEN]);
    MVec sx3 = (1.0 / 6.0) * xint * xint * xint;
    sx3.Store(p, &sx[3 * VEC_LEN]);
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_incrsort_order3_kernel (
    const amrex::ParticleReal * const wp,
    const amrex::ParticleReal * const uxp,
    const amrex::ParticleReal * const uyp,
    const amrex::ParticleReal * const uzp,
    const amrex::ParticleReal* mx,
    const amrex::ParticleReal* my,
    const amrex::ParticleReal* mz,
    amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    amrex::Real q,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    int lenx,
    int leny,
    int lenz,
    long np_to_deposit,
    ParticleTileType& ptile,
    vector<vector<int>>& outside_bin_ip
) __arc_streaming {
    constexpr int nshapes = 1 + 3;
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    const svuint64_t sm_index = svindex_u64(0, 8);

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0/PhysConst::c/PhysConst::c;

    amrex::Real wqx[block_size];
    amrex::Real wqy[block_size];
    amrex::Real wqz[block_size];
    double sx_m[nshapes * block_size] = {0.};
    double sy_m[nshapes * block_size] = {0.};
    double sz_m[nshapes * block_size] = {0.};

    vector<int>& d_incr_bin_offset = ptile.d_incr_bin_offset;
    std::vector<int>& d_incr_bin_length = ptile.d_incr_bin_length;;
    vector<int>& d_local_index = ptile.d_local_index;

    int nomoved_ips_vec[8];
    int move_pos = 0;
    const int vl = svcntd();
    long tmpbin[8];

    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int& oldbin_offset = d_incr_bin_offset[old_bin];
                int& oldbin_length = d_incr_bin_length[old_bin];

                int oldbin_outside_length = outside_bin_ip[old_bin].size();
                vector<int>& oldbin_outside_particles = outside_bin_ip[old_bin];

                if (oldbin_length == 0 && oldbin_outside_length == 0)
                {
                    continue;
                }

                int rho_offset_iy01_z0 = (old_bin + 0 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy01_z1 = (old_bin + 0 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy01_z2 = (old_bin + 0 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy01_z3 = (old_bin + 0 * lenx + 3 * lenx * leny) * 8;

                int rho_offset_iy23_z0 = (old_bin + 2 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy23_z1 = (old_bin + 2 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy23_z2 = (old_bin + 2 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy23_z3 = (old_bin + 2 * lenx + 3 * lenx * leny) * 8;

                if (oldbin_offset == -1)
                {
                    svzero_za();
                    for (int i = 0; i < oldbin_outside_length; i += vl)
                    {
                        svbool_t p_ip = svwhilelt_b64(i, oldbin_outside_length);
                        intVec ip_v = svld1sw_s64(p_ip, &oldbin_outside_particles[i]);

                        MVec uxp_v = svld1_gather_s64index_f64(p_ip, uxp, ip_v);
                        MVec uyp_v = svld1_gather_s64index_f64(p_ip, uyp, ip_v);
                        MVec uzp_v = svld1_gather_s64index_f64(p_ip, uzp, ip_v);
                        MVec gaminv = 1 / (1 + uxp_v * uxp_v * clightsq
                                                    + uyp_v * uyp_v * clightsq
                                                    + uzp_v * uzp_v * clightsq).Sqrt();

                        MVec vx_v = uxp_v * gaminv;
                        MVec vy_v = uyp_v * gaminv;
                        MVec vz_v = uzp_v * gaminv;

                        MVec wp_v = svld1_gather_s64index_f64(p_ip, wp, ip_v);
                        MVec wq_v = q * wp_v;

                        MVec wqx_v = wq_v * invvol * vx_v;
                        MVec wqy_v = wq_v * invvol * vy_v;
                        MVec wqz_v = wq_v * invvol * vz_v;

                        wqx_v.Store(p_ip, &wqx[0]);
                        wqy_v.Store(p_ip, &wqy[0]);
                        wqz_v.Store(p_ip, &wqz[0]);

                        MVec xp = svld1_gather_s64index_f64(p_ip, mx, ip_v);
                        MVec yp = svld1_gather_s64index_f64(p_ip, my, ip_v);
                        MVec zp = svld1_gather_s64index_f64(p_ip, mz, ip_v);

                        MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
                        MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
                        MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;

                        Mvec_compute_shape_factor_part_vpu_order3(&sx_m[0], xmid, p_ip);
                        Mvec_compute_shape_factor_part_vpu_order3(&sy_m[0], ymid, p_ip);
                        Mvec_compute_shape_factor_part_vpu_order3(&sz_m[0], zmid, p_ip);

                        int rho_end = block_size < oldbin_outside_length - i ? block_size : oldbin_outside_length - i;
                        for (int iip = 0; iip < rho_end; iip++)
                        {
                            MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);

                            MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
                            MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
                            MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                            
                            MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
                            MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
                            MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                            
                            MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

                            MVec wqxsz_v = sz_v * wqx[iip];
                            MVec wqysz_v = sz_v * wqy[iip];
                            MVec wqzsz_v = sz_v * wqz[iip];

                            MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                            svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                            svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                            svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                            svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                        }                        
                    }

                    MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                    MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                    MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                    MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                    MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                    MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                    MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                    MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                    MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                    MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                    MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                    MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                    MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                    MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                    MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                    MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                    MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                    MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                    MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                    MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                    MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                    MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                    MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                    MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);

                    MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                    MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                    MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                    xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                    yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                    zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                    xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                    yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                    zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                    MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                    MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                    MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                    
                    xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                    yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                    zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                    
                    xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                    yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                    zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                        
                    MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                    MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                    MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                    
                    xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                    yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                    zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                    
                    xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                    yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                    zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                            
                    MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                    MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                    MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                    yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                    zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                    
                    xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                    yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                    zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                    MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                    MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                    yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                    zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                    
                    xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                    yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                    zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                    MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                    MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                    yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                    zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                    
                    xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                    yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                    zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                    MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                    MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                    yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                    zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                    
                    xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                    yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                    zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                    MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                    MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                    
                    xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                    yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                    zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                    
                    xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                    yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                    zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
                }
                else
                {
                    svzero_za();
                    for (int i = 0; i < oldbin_length; i += vl)
                    {
                        svbool_t p_ip = svwhilelt_b64(i, oldbin_length);
                        int idx = oldbin_offset + i;
                        intVec ip_v = svld1sw_s64(p_ip, &d_local_index[idx]);

                        MVec uxp_v = svld1_gather_s64index_f64(p_ip, uxp, ip_v);
                        MVec uyp_v = svld1_gather_s64index_f64(p_ip, uyp, ip_v);
                        MVec uzp_v = svld1_gather_s64index_f64(p_ip, uzp, ip_v);
                        MVec gaminv = 1 / (1 + uxp_v * uxp_v * clightsq
                                                    + uyp_v * uyp_v * clightsq
                                                    + uzp_v * uzp_v * clightsq).Sqrt();

                        MVec vx_v = uxp_v * gaminv;
                        MVec vy_v = uyp_v * gaminv;
                        MVec vz_v = uzp_v * gaminv;

                        MVec wp_v = svld1_gather_s64index_f64(p_ip, wp, ip_v);
                        MVec wq_v = q * wp_v;

                        MVec wqx_v = wq_v * invvol * vx_v;
                        MVec wqy_v = wq_v * invvol * vy_v;
                        MVec wqz_v = wq_v * invvol * vz_v;

                        wqx_v.Store(p_ip, &wqx[0]);
                        wqy_v.Store(p_ip, &wqy[0]);
                        wqz_v.Store(p_ip, &wqz[0]);

                        MVec xp = svld1_gather_s64index_f64(p_ip, mx, ip_v);
                        MVec yp = svld1_gather_s64index_f64(p_ip, my, ip_v);
                        MVec zp = svld1_gather_s64index_f64(p_ip, mz, ip_v);

                        MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
                        MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
                        MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;

                        Mvec_compute_shape_factor_part_vpu_order3(&sx_m[0], xmid, p_ip);
                        Mvec_compute_shape_factor_part_vpu_order3(&sy_m[0], ymid, p_ip);
                        Mvec_compute_shape_factor_part_vpu_order3(&sz_m[0], zmid, p_ip);

                        int rho_end = block_size < oldbin_length - i ? block_size : oldbin_length - i;
                        for (int iip = 0; iip < rho_end; iip++)
                        {
                            MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);

                            MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
                            MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
                            MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                            
                            MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
                            MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
                            MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                            
                            MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

                            MVec wqxsz_v = sz_v * wqx[iip];
                            MVec wqysz_v = sz_v * wqy[iip];
                            MVec wqzsz_v = sz_v * wqz[iip];

                            MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                            svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                            svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                            svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                            svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                        }
                    }

                    MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                    MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                    MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                    MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                    MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                    MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                    MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                    MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                    MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                    MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                    MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                    MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                    MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                    MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                    MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                    MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                    MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                    MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                    MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                    MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                    MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                    MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                    MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                    MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);

                    MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                    MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                    MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                    xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                    yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                    zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                    xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                    yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                    zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                    MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                    MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                    MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                    
                    xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                    yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                    zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                    
                    xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                    yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                    zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                        
                    MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                    MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                    MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                    
                    xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                    yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                    zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                    
                    xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                    yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                    zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                            
                    MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                    MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                    MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                    yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                    zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                    
                    xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                    yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                    zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                    MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                    MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                    yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                    zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                    
                    xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                    yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                    zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                    MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                    MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                    yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                    zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                    
                    xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                    yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                    zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                    MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                    MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                    yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                    zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                    
                    xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                    yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                    zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                    MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                    MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                    
                    xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                    yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                    zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                    
                    xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                    yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                    zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
                }
            }
        }
    }
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_incrsort_order3_eurosys_kernel (
    const amrex::ParticleReal * const wp,
    const amrex::ParticleReal * const uxp,
    const amrex::ParticleReal * const uyp,
    const amrex::ParticleReal * const uzp,
    const amrex::ParticleReal* mx,
    const amrex::ParticleReal* my,
    const amrex::ParticleReal* mz,
    amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    amrex::Real q,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    int lenx,
    int leny,
    int lenz,
    amrex::Real* wqx,
    amrex::Real* wqy,
    amrex::Real* wqz,
    amrex::Real* sx_m,
    amrex::Real* sy_m,
    amrex::Real* sz_m,
    long np_to_deposit,
    ParticleTileType& ptile,
    vector<vector<int>>& outside_bin_ip
) __arc_streaming {
    constexpr int nshapes = 1 + 3;
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    const svuint64_t sm_index = svindex_u64(0, 8);

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0/PhysConst::c/PhysConst::c;

    vector<int>& d_incr_bin_offset = ptile.d_incr_bin_offset;
    std::vector<int>& d_incr_bin_length = ptile.d_incr_bin_length;;
    vector<int>& d_local_index = ptile.d_local_index;

    int nomoved_ips_vec[8];
    int move_pos = 0;
    const int vl = svcntd();
    long tmpbin[8];

    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int& oldbin_offset = d_incr_bin_offset[old_bin];
                int& oldbin_length = d_incr_bin_length[old_bin];

                int oldbin_outside_length = outside_bin_ip[old_bin].size();
                vector<int>& oldbin_outside_particles = outside_bin_ip[old_bin];

                if (oldbin_length == 0 && oldbin_outside_length == 0)
                {
                    continue;
                }

                int rho_offset_iy01_z0 = (old_bin + 0 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy01_z1 = (old_bin + 0 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy01_z2 = (old_bin + 0 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy01_z3 = (old_bin + 0 * lenx + 3 * lenx * leny) * 8;

                int rho_offset_iy23_z0 = (old_bin + 2 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy23_z1 = (old_bin + 2 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy23_z2 = (old_bin + 2 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy23_z3 = (old_bin + 2 * lenx + 3 * lenx * leny) * 8;

                if (oldbin_offset == -1)
                {
                    svzero_za();
                    for (int i = 0; i < oldbin_outside_length; i++)
                    {
                        int ip = oldbin_outside_particles[i];
                        MVec sx_v = MVec::Load(p_0_3, sx_m + 4 * ip);

                        MVec sx_y0_v = sx_v * sy_m[0 + 4 * ip];
                        MVec sx_y1_v = sx_v * sy_m[1 + 4 * ip];
                        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                            
                        MVec sx_y2_v = sx_v * sy_m[2 + 4 * ip];
                        MVec sx_y3_v = sx_v * sy_m[3 + 4 * ip];
                        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                            
                        MVec sz_v = MVec::Load(p_0_3, sz_m + 4 * ip);

                        MVec wqxsz_v = sz_v * wqx[ip];
                        MVec wqysz_v = sz_v * wqy[ip];
                        MVec wqzsz_v = sz_v * wqz[ip];

                        MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                        svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                        svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                        svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                        svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);                   
                    }

                    MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                    MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                    MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                    MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                    MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                    MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                    MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                    MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                    MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                    MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                    MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                    MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                    MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                    MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                    MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                    MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                    MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                    MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                    MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                    MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                    MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                    MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                    MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                    MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);

                    MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                    MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                    MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                    xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                    yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                    zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                    xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                    yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                    zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                    MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                    MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                    MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                    
                    xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                    yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                    zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                    
                    xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                    yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                    zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                        
                    MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                    MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                    MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                    
                    xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                    yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                    zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                    
                    xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                    yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                    zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                            
                    MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                    MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                    MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                    yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                    zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                    
                    xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                    yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                    zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                    MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                    MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                    yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                    zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                    
                    xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                    yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                    zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                    MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                    MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                    yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                    zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                    
                    xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                    yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                    zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                    MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                    MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                    yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                    zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                    
                    xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                    yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                    zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                    MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                    MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                    
                    xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                    yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                    zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                    
                    xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                    yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                    zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
                }
                else
                {
                    svzero_za();
                    for (int i = 0; i < oldbin_length; i++)
                    {                        
                        int ip = d_local_index[oldbin_offset + i];
                        MVec sx_v = MVec::Load(p_0_3, sx_m + 4 * ip);

                        MVec sx_y0_v = sx_v * sy_m[0 + 4 * ip];
                        MVec sx_y1_v = sx_v * sy_m[1 + 4 * ip];
                        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                            
                        MVec sx_y2_v = sx_v * sy_m[2 + 4 * ip];
                        MVec sx_y3_v = sx_v * sy_m[3 + 4 * ip];
                        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                            
                        MVec sz_v = MVec::Load(p_0_3, sz_m + 4 * ip);

                        MVec wqxsz_v = sz_v * wqx[ip];
                        MVec wqysz_v = sz_v * wqy[ip];
                        MVec wqzsz_v = sz_v * wqz[ip];

                        MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                        svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                        svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                        svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                        svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                    }

                    MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                    MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                    MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                    MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                    MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                    MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                    MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                    MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                    MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                    MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                    MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                    MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                    MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                    MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                    MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                    MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                    MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                    MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                    MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                    MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                    MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                    MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                    MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                    MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);

                    MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                    MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                    MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                    xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                    yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                    zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                    xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                    yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                    zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                    MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                    MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                    MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                    
                    xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                    yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                    zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                    
                    xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                    yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                    zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                        
                    MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                    MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                    MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                    
                    xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                    yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                    zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                    
                    xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                    yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                    zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                            
                    MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                    MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                    MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                    yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                    zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                    
                    xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                    yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                    zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                    
                    MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                    MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                    MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                    yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                    zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                    
                    xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                    yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                    zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                    
                    MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                    MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                    MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                    yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                    zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                    
                    xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                    yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                    zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                    
                    MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                    MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                    MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                    yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                    zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                    
                    xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                    yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                    zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                    
                    MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                    MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                    MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                    
                    xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                    yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                    zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                    
                    xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                    yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                    zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
                }
            }
        }
    }
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_binsort_physort_order3_kernel (
    const amrex::ParticleReal * const wp,
    const amrex::ParticleReal * const uxp,
    const amrex::ParticleReal * const uyp,
    const amrex::ParticleReal * const uzp,
    const amrex::ParticleReal* mx,
    const amrex::ParticleReal* my,
    const amrex::ParticleReal* mz,
    amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    amrex::Real q,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    int lenx,
    int leny,
    int lenz,
    long np_to_deposit,
    ParticleTileType& ptile,
    vector<vector<int>>& bin_to_ip
) __arc_streaming {
    constexpr int nshapes = 1 + 3;
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    const svuint64_t sm_index = svindex_u64(0, 8);

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0/PhysConst::c/PhysConst::c;

    amrex::Real wqx[block_size];
    amrex::Real wqy[block_size];
    amrex::Real wqz[block_size];
    double sx_m[nshapes * block_size] = {0.};
    double sy_m[nshapes * block_size] = {0.};
    double sz_m[nshapes * block_size] = {0.};

    const std::vector<int>& g_phys_bin_offset = ptile.g_phys_bin_offset;
    const std::vector<int>& g_phys_bin_length = ptile.g_phys_bin_length;

    const int vl = svcntd();

    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int offset = g_phys_bin_offset[old_bin];
                int binlength = g_phys_bin_length[old_bin];
                vector<int>& tmp_particles = bin_to_ip[old_bin];
                int tmp_binlength = tmp_particles.size();
                
                if (binlength == 0 && tmp_binlength == 0) continue;

                int rho_offset_iy01_z0 = (old_bin + 0 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy01_z1 = (old_bin + 0 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy01_z2 = (old_bin + 0 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy01_z3 = (old_bin + 0 * lenx + 3 * lenx * leny) * 8;

                int rho_offset_iy23_z0 = (old_bin + 2 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy23_z1 = (old_bin + 2 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy23_z2 = (old_bin + 2 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy23_z3 = (old_bin + 2 * lenx + 3 * lenx * leny) * 8;
                
                svzero_za();

                for (int k = 0; k < binlength; k += vl)
                {
                    svbool_t p_ip = svwhilelt_b64(k, binlength);
                    long idx = offset + k;

                    MVec uxp_v = MVec::Load(p_ip, &uxp[idx]);
                    MVec uyp_v = MVec::Load(p_ip, &uyp[idx]);
                    MVec uzp_v = MVec::Load(p_ip, &uzp[idx]);
                    MVec gaminv = 1. / (1. + uxp_v * uxp_v * clightsq
                                                + uyp_v * uyp_v * clightsq
                                                + uzp_v * uzp_v * clightsq).Sqrt();
                    MVec vx_v = uxp_v * gaminv;
                    MVec vy_v = uyp_v * gaminv;
                    MVec vz_v = uzp_v * gaminv;
            
                    MVec wp_v = MVec::Load(p_ip, &wp[idx]);
                    
                    MVec wq_v = q * wp_v;
            
                    MVec wqx_v = wq_v * invvol * vx_v;
                    MVec wqy_v = wq_v * invvol * vy_v;
                    MVec wqz_v = wq_v * invvol * vz_v;
            
                    wqx_v.Store(p_ip, &wqx[0]);
                    wqy_v.Store(p_ip, &wqy[0]);
                    wqz_v.Store(p_ip, &wqz[0]);
            
                    MVec xp = MVec::Load(p_ip, &mx[idx]);
                    MVec yp = MVec::Load(p_ip, &my[idx]);
                    MVec zp = MVec::Load(p_ip, &mz[idx]);
            
                    MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
                    MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
                    MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
                    
                    Mvec_compute_shape_factor_part_vpu_order3(&sx_m[0], xmid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sy_m[0], ymid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sz_m[0], zmid, p_ip);

                    int rho_end = binlength - k < block_size ? binlength - k : block_size;
                    for (long iip = 0; iip < rho_end; ++iip)
                    {       
                        MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);

                        MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
                        MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
                        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                        
                        MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
                        MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
                        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                        
                        MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

                        MVec wqxsz_v = sz_v * wqx[iip];
                        MVec wqysz_v = sz_v * wqy[iip];
                        MVec wqzsz_v = sz_v * wqz[iip];

                        MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                        svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                        svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                        svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                        svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                    }
                }

                for (int k = 0; k < tmp_binlength; k += vl)
                {
                    svbool_t p_ip = svwhilelt_b64(k, tmp_binlength);
                    intVec ip_v = svld1sw_s64(p_ip, &tmp_particles[k]);

                    MVec uxp_v = svld1_gather_s64index_f64(p_ip, uxp, ip_v);
                    MVec uyp_v = svld1_gather_s64index_f64(p_ip, uyp, ip_v);
                    MVec uzp_v = svld1_gather_s64index_f64(p_ip, uzp, ip_v);

                    MVec gaminv = 1 / (1 + uxp_v * uxp_v * clightsq
                                                + uyp_v * uyp_v * clightsq
                                                + uzp_v * uzp_v * clightsq).Sqrt();
                    
                    MVec vx_v = uxp_v * gaminv;
                    MVec vy_v = uyp_v * gaminv;
                    MVec vz_v = uzp_v * gaminv;

                    MVec wp_v = svld1_gather_s64index_f64(p_ip, wp, ip_v);

                    MVec wq_v = q * wp_v;
            
                    MVec wqx_v = wq_v * invvol * vx_v;
                    MVec wqy_v = wq_v * invvol * vy_v;
                    MVec wqz_v = wq_v * invvol * vz_v;
            
                    wqx_v.Store(p_ip, &wqx[0]);
                    wqy_v.Store(p_ip, &wqy[0]);
                    wqz_v.Store(p_ip, &wqz[0]);

                    MVec xp = svld1_gather_s64index_f64(p_ip, mx, ip_v);
                    MVec yp = svld1_gather_s64index_f64(p_ip, my, ip_v);
                    MVec zp = svld1_gather_s64index_f64(p_ip, mz, ip_v);

                    MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
                    MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
                    MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
                    
                    Mvec_compute_shape_factor_part_vpu_order3(&sx_m[0], xmid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sy_m[0], ymid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sz_m[0], zmid, p_ip);

                    int rho_end = tmp_binlength - k < block_size ? tmp_binlength - k : block_size;
                    for (long iip = 0; iip < rho_end; ++iip)
                    {
                        MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);

                        MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
                        MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
                        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                        
                        MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
                        MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
                        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                        
                        MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

                        MVec wqxsz_v = sz_v * wqx[iip];
                        MVec wqysz_v = sz_v * wqy[iip];
                        MVec wqzsz_v = sz_v * wqz[iip];

                        MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                        svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                        svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                        svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                        svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                    }
                }

                MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);
                
                MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                
                xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                
                xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                     
                MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                
                xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                
                xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                        
                MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                
                xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                
                xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                
                MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                
                xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                
                xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                
                MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                
                xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                
                xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                
                MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                
                xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                
                xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                
                MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                
                xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                
                xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
            }
        }
    }
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_binsort_physort_order3_kernel_eurosys (
    const amrex::ParticleReal * const wp,
    const amrex::ParticleReal * const uxp,
    const amrex::ParticleReal * const uyp,
    const amrex::ParticleReal * const uzp,
    const amrex::ParticleReal* mx,
    const amrex::ParticleReal* my,
    const amrex::ParticleReal* mz,
    amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    amrex::Real q,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    int lenx,
    int leny,
    int lenz,
    amrex::ParticleReal* wqx_disorder,
    amrex::ParticleReal* wqy_disorder,
    amrex::ParticleReal* wqz_disorder,
    double* sx_m_disorder,
    double* sy_m_disorder,
    double* sz_m_disorder,
    long np_to_deposit,
    ParticleTileType& ptile,
    vector<vector<int>>& bin_to_ip
) __arc_streaming {
    constexpr int nshapes = 1 + 3;
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    const svuint64_t sm_index = svindex_u64(0, 8);

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0/PhysConst::c/PhysConst::c;

    amrex::Real wqx[block_size];
    amrex::Real wqy[block_size];
    amrex::Real wqz[block_size];
    double sx_m[nshapes * block_size] = {0.};
    double sy_m[nshapes * block_size] = {0.};
    double sz_m[nshapes * block_size] = {0.};

    int sx_step_disorder[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    int sy_step_disorder[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    intVec mx_index_disorder = svld1sw_s64(p_true, sx_step_disorder);
    intVec my_index_disorder = svld1sw_s64(p_true, sy_step_disorder);

    const std::vector<int>& g_phys_bin_offset = ptile.g_phys_bin_offset;
    const std::vector<int>& g_phys_bin_length = ptile.g_phys_bin_length;

    const int vl = svcntd();

    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int offset = g_phys_bin_offset[old_bin];
                int binlength = g_phys_bin_length[old_bin];
                vector<int>& tmp_particles = bin_to_ip[old_bin];
                int tmp_binlength = tmp_particles.size();
                
                if (binlength == 0 && tmp_binlength == 0) continue;

                int rho_offset_iy01_z0 = (old_bin + 0 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy01_z1 = (old_bin + 0 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy01_z2 = (old_bin + 0 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy01_z3 = (old_bin + 0 * lenx + 3 * lenx * leny) * 8;

                int rho_offset_iy23_z0 = (old_bin + 2 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy23_z1 = (old_bin + 2 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy23_z2 = (old_bin + 2 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy23_z3 = (old_bin + 2 * lenx + 3 * lenx * leny) * 8;
                
                svzero_za();

                for (int k = 0; k < binlength; k += vl)
                {
                    svbool_t p_ip = svwhilelt_b64(k, binlength);
                    long idx = offset + k;

                    MVec uxp_v = MVec::Load(p_ip, &uxp[idx]);
                    MVec uyp_v = MVec::Load(p_ip, &uyp[idx]);
                    MVec uzp_v = MVec::Load(p_ip, &uzp[idx]);
                    MVec gaminv = 1. / (1. + uxp_v * uxp_v * clightsq
                                                + uyp_v * uyp_v * clightsq
                                                + uzp_v * uzp_v * clightsq).Sqrt();
                    MVec vx_v = uxp_v * gaminv;
                    MVec vy_v = uyp_v * gaminv;
                    MVec vz_v = uzp_v * gaminv;
            
                    MVec wp_v = MVec::Load(p_ip, &wp[idx]);
                    
                    MVec wq_v = q * wp_v;
            
                    MVec wqx_v = wq_v * invvol * vx_v;
                    MVec wqy_v = wq_v * invvol * vy_v;
                    MVec wqz_v = wq_v * invvol * vz_v;
            
                    wqx_v.Store(p_ip, &wqx[0]);
                    wqy_v.Store(p_ip, &wqy[0]);
                    wqz_v.Store(p_ip, &wqz[0]);
            
                    MVec xp = MVec::Load(p_ip, &mx[idx]);
                    MVec yp = MVec::Load(p_ip, &my[idx]);
                    MVec zp = MVec::Load(p_ip, &mz[idx]);
            
                    MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
                    MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
                    MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
                    
                    Mvec_compute_shape_factor_part_vpu_order3(&sx_m[0], xmid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sy_m[0], ymid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sz_m[0], zmid, p_ip);

                    int rho_end = binlength - k < block_size ? binlength - k : block_size;
                    for (long iip = 0; iip < rho_end; ++iip)
                    {       
                        MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);

                        MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
                        MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
                        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                        
                        MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
                        MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
                        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                        
                        MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

                        MVec wqxsz_v = sz_v * wqx[iip];
                        MVec wqysz_v = sz_v * wqy[iip];
                        MVec wqzsz_v = sz_v * wqz[iip];

                        MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                        svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                        svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                        svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                        svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                    }
                }

                for (int k = 0; k < tmp_binlength; k++)
                {
                    long ip = tmp_particles[k];

                    MVec sx_v = svld1_gather_index(p_0_3, &sx_m_disorder[4 * ip], mx_index_disorder);

                    MVec sx_y0_v = sx_v * sy_m_disorder[0 + 4 * ip];
                    MVec sx_y1_v = sx_v * sy_m_disorder[1 + 4 * ip];
                    MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                        
                    MVec sx_y2_v = sx_v * sy_m_disorder[2 + 4 * ip];
                    MVec sx_y3_v = sx_v * sy_m_disorder[3 + 4 * ip];
                    MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                        
                    MVec sz_v = svld1_gather_index(p_0_3, &sz_m_disorder[4 * ip], mx_index_disorder);

                    MVec wqxsz_v = sz_v * wqx_disorder[ip];
                    MVec wqysz_v = sz_v * wqy_disorder[ip];
                    MVec wqzsz_v = sz_v * wqz_disorder[ip];

                    MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                    svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                    svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                    svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                    svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                }

                MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);
                
                MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                
                xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                
                xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                     
                MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                
                xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                
                xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                        
                MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                
                xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                
                xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                
                MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                
                xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                
                xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                
                MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                
                xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                
                xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                
                MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                
                xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                
                xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                
                MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                
                xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                
                xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
            }
        }
    }
}


template <int depos_order>
__arc_new("za") inline void vpu_rhocell_mpu_physort_order3_kernel (
    const amrex::ParticleReal * const wp,
    const amrex::ParticleReal * const uxp,
    const amrex::ParticleReal * const uyp,
    const amrex::ParticleReal * const uzp,
    const amrex::ParticleReal* mx,
    const amrex::ParticleReal* my,
    const amrex::ParticleReal* mz,
    amrex::Real relative_time,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    amrex::Real q,
    std::vector<amrex::Real>& xrhocells,
    std::vector<amrex::Real>& yrhocells,
    std::vector<amrex::Real>& zrhocells,
    int lenx,
    int leny,
    int lenz,
    long np_to_deposit,
    ParticleTileType& ptile
) __arc_streaming {
    constexpr int nshapes = 1 + 3;
    constexpr int block_size = 8;
    svbool_t p_true = svwhilelt_b64(0, 8);
    svbool_t p_0_7 = svwhilelt_b64(0, 8);
    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    MVec vzero(0);

    const svuint64_t sm_index = svindex_u64(0, 8);

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    const amrex::Real clightsq = 1.0/PhysConst::c/PhysConst::c;

    amrex::Real wqx[block_size];
    amrex::Real wqy[block_size];
    amrex::Real wqz[block_size];
    double sx_m[nshapes * block_size] = {0.};
    double sy_m[nshapes * block_size] = {0.};
    double sz_m[nshapes * block_size] = {0.};

    int& g_move_begin = ptile.g_move_begin;
    const std::vector<int>& g_phys_bin_offset = ptile.g_phys_bin_offset;
    const std::vector<int>& g_phys_bin_length = ptile.g_phys_bin_length;

    int nomoved_ips_vec[8];
    int move_pos = 0;
    const int vl = svcntd();
    long tmpbin[8];

    uint64_t calculate_start = rdtscm();
    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int binlength = g_phys_bin_length[old_bin];
                
                if (binlength == 0) continue;
                
                int offset = g_phys_bin_offset[old_bin];
                MintVec oldbin_v(old_bin);

                int rho_offset_iy01_z0 = (old_bin + 0 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy01_z1 = (old_bin + 0 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy01_z2 = (old_bin + 0 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy01_z3 = (old_bin + 0 * lenx + 3 * lenx * leny) * 8;

                int rho_offset_iy23_z0 = (old_bin + 2 * lenx + 0 * lenx * leny) * 8;
                int rho_offset_iy23_z1 = (old_bin + 2 * lenx + 1 * lenx * leny) * 8;
                int rho_offset_iy23_z2 = (old_bin + 2 * lenx + 2 * lenx * leny) * 8;
                int rho_offset_iy23_z3 = (old_bin + 2 * lenx + 3 * lenx * leny) * 8;

                svzero_za();
                for (int k = 0; k < binlength; k += vl)
                {
                    svbool_t p_ip = svwhilelt_b64(k, binlength);
                    long idx = offset + k;
                    
                    MVec uxp_v = MVec::Load(p_ip, &uxp[idx]);
                    MVec uyp_v = MVec::Load(p_ip, &uyp[idx]);
                    MVec uzp_v = MVec::Load(p_ip, &uzp[idx]);

                    MVec gaminv = 1. / (1. + uxp_v * uxp_v * clightsq
                                            + uyp_v * uyp_v * clightsq
                                            + uzp_v * uzp_v * clightsq).Sqrt();

                    MVec vx_v = uxp_v * gaminv;
                    MVec vy_v = uyp_v * gaminv;
                    MVec vz_v = uzp_v * gaminv;

                    MVec wp_v = MVec::Load(p_ip, &wp[idx]);

                    MVec wq_v = q * wp_v;
            
                    MVec wqx_v = wq_v * invvol * vx_v;
                    MVec wqy_v = wq_v * invvol * vy_v;
                    MVec wqz_v = wq_v * invvol * vz_v;
            
                    wqx_v.Store(p_ip, &wqx[0]);
                    wqy_v.Store(p_ip, &wqy[0]);
                    wqz_v.Store(p_ip, &wqz[0]);

                    MVec xp = MVec::Load(p_ip, &mx[idx]);
                    MVec yp = MVec::Load(p_ip, &my[idx]);
                    MVec zp = MVec::Load(p_ip, &mz[idx]);

                    MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
                    MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
                    MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
            
                    MintVec j_cellv = svcvt_s64_f64_z(p_ip, xmid) - 1;
                    MintVec k_cellv = svcvt_s64_f64_z(p_ip, ymid) - 1;
                    MintVec l_cellv = svcvt_s64_f64_z(p_ip, zmid) - 1;
                    
                    MintVec newbin_v = j_cellv + k_cellv * lenx + l_cellv * lenx * leny;

                    svbool_t ip_move = svcmpne_s64(p_ip, newbin_v, oldbin_v);
                    uint64_t move_num = svcntp_b64(p_ip, ip_move);

                    Mvec_compute_shape_factor_part_vpu_order3(&sx_m[0], xmid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sy_m[0], ymid, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(&sz_m[0], zmid, p_ip);

                    int rho_end = block_size < binlength - k ? block_size : binlength - k;
                    for (int iip = 0; iip < rho_end; ++iip)
                    {               
                        MVec sx_v = svld1_gather_index(p_0_3, &sx_m[iip], sm_index);

                        MVec sx_y0_v = sx_v * sy_m[0 * block_size + iip];
                        MVec sx_y1_v = sx_v * sy_m[1 * block_size + iip];
                        MVec sx_y_01_v = svsplice(p_0_3, sx_y0_v, sx_y1_v);
                        
                        MVec sx_y2_v = sx_v * sy_m[2 * block_size + iip];
                        MVec sx_y3_v = sx_v * sy_m[3 * block_size + iip];
                        MVec sx_y_23_v = svsplice(p_0_3, sx_y2_v, sx_y3_v);
                        
                        MVec sz_v = svld1_gather_index(p_0_3, &sz_m[iip], sm_index);

                        MVec wqxsz_v = sz_v * wqx[iip];
                        MVec wqysz_v = sz_v * wqy[iip];
                        MVec wqzsz_v = sz_v * wqz[iip];

                        MVec wqxysz_v = svsplice(p_0_3, wqxsz_v, wqysz_v);

                        svmopa_za64_m(0, p_0_7, p_0_7, wqxysz_v, sx_y_01_v);
                        svmopa_za64_m(1, p_0_7, p_0_7, wqxysz_v, sx_y_23_v);
                        svmopa_za64_m(2, p_0_3, p_0_7, wqzsz_v, sx_y_01_v);
                        svmopa_za64_m(3, p_0_3, p_0_7, wqzsz_v, sx_y_23_v);
                    }
                }

                MVec wqx_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 0);
                MVec wqx_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 1);
                MVec wqx_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 2);
                MVec wqx_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 3);

                MVec wqy_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 0, 4);
                MVec wqy_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 0, 5);
                MVec wqy_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 0, 6);
                MVec wqy_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 0, 7);

                MVec wqz_sx_y01_z0 = svread_hor_za64_m(vzero, p_0_7, 2, 0);
                MVec wqz_sx_y01_z1 = svread_hor_za64_m(vzero, p_0_7, 2, 1);
                MVec wqz_sx_y01_z2 = svread_hor_za64_m(vzero, p_0_7, 2, 2);
                MVec wqz_sx_y01_z3 = svread_hor_za64_m(vzero, p_0_7, 2, 3);

                MVec wqx_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 0);
                MVec wqx_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 1);
                MVec wqx_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 2);
                MVec wqx_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 3);

                MVec wqy_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 1, 4);
                MVec wqy_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 1, 5);
                MVec wqy_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 1, 6);
                MVec wqy_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 1, 7);

                MVec wqz_sx_y23_z0 = svread_hor_za64_m(vzero, p_0_7, 3, 0);
                MVec wqz_sx_y23_z1 = svread_hor_za64_m(vzero, p_0_7, 3, 1);
                MVec wqz_sx_y23_z2 = svread_hor_za64_m(vzero, p_0_7, 3, 2);
                MVec wqz_sx_y23_z3 = svread_hor_za64_m(vzero, p_0_7, 3, 3);
                
                MVec xrhocells_iy01_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z0]);
                MVec yrhocells_iy01_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z0]);
                MVec zrhocells_iy01_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z0]);

                xrhocells_iy01_z0_v += wqx_sx_y01_z0;
                yrhocells_iy01_z0_v += wqy_sx_y01_z0;
                zrhocells_iy01_z0_v += wqz_sx_y01_z0;

                xrhocells_iy01_z0_v.Store(p_true, &xrhocells[rho_offset_iy01_z0]);
                yrhocells_iy01_z0_v.Store(p_true, &yrhocells[rho_offset_iy01_z0]);
                zrhocells_iy01_z0_v.Store(p_true, &zrhocells[rho_offset_iy01_z0]);

                MVec xrhocells_iy01_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z1]);
                MVec yrhocells_iy01_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z1]);
                MVec zrhocells_iy01_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z1]);
                
                xrhocells_iy01_z1_v += wqx_sx_y01_z1;
                yrhocells_iy01_z1_v += wqy_sx_y01_z1;
                zrhocells_iy01_z1_v += wqz_sx_y01_z1;
                
                xrhocells_iy01_z1_v.Store(p_true, &xrhocells[rho_offset_iy01_z1]);
                yrhocells_iy01_z1_v.Store(p_true, &yrhocells[rho_offset_iy01_z1]);
                zrhocells_iy01_z1_v.Store(p_true, &zrhocells[rho_offset_iy01_z1]);
                     
                MVec xrhocells_iy01_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z2]);
                MVec yrhocells_iy01_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z2]);
                MVec zrhocells_iy01_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z2]);
                
                xrhocells_iy01_z2_v += wqx_sx_y01_z2;
                yrhocells_iy01_z2_v += wqy_sx_y01_z2;
                zrhocells_iy01_z2_v += wqz_sx_y01_z2;
                
                xrhocells_iy01_z2_v.Store(p_true, &xrhocells[rho_offset_iy01_z2]);
                yrhocells_iy01_z2_v.Store(p_true, &yrhocells[rho_offset_iy01_z2]);
                zrhocells_iy01_z2_v.Store(p_true, &zrhocells[rho_offset_iy01_z2]);
                        
                MVec xrhocells_iy01_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy01_z3]);
                MVec yrhocells_iy01_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy01_z3]);
                MVec zrhocells_iy01_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy01_z3]);
                
                xrhocells_iy01_z3_v += wqx_sx_y01_z3;
                yrhocells_iy01_z3_v += wqy_sx_y01_z3;
                zrhocells_iy01_z3_v += wqz_sx_y01_z3;
                
                xrhocells_iy01_z3_v.Store(p_true, &xrhocells[rho_offset_iy01_z3]);
                yrhocells_iy01_z3_v.Store(p_true, &yrhocells[rho_offset_iy01_z3]);
                zrhocells_iy01_z3_v.Store(p_true, &zrhocells[rho_offset_iy01_z3]);
                
                MVec xrhocells_iy23_z0_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z0]);
                MVec yrhocells_iy23_z0_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z0]);
                MVec zrhocells_iy23_z0_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z0]);
                
                xrhocells_iy23_z0_v += wqx_sx_y23_z0;
                yrhocells_iy23_z0_v += wqy_sx_y23_z0;
                zrhocells_iy23_z0_v += wqz_sx_y23_z0;
                
                xrhocells_iy23_z0_v.Store(p_true, &xrhocells[rho_offset_iy23_z0]);
                yrhocells_iy23_z0_v.Store(p_true, &yrhocells[rho_offset_iy23_z0]);
                zrhocells_iy23_z0_v.Store(p_true, &zrhocells[rho_offset_iy23_z0]);
                
                MVec xrhocells_iy23_z1_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z1]);
                MVec yrhocells_iy23_z1_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z1]);
                MVec zrhocells_iy23_z1_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z1]);
                
                xrhocells_iy23_z1_v += wqx_sx_y23_z1;
                yrhocells_iy23_z1_v += wqy_sx_y23_z1;
                zrhocells_iy23_z1_v += wqz_sx_y23_z1;
                
                xrhocells_iy23_z1_v.Store(p_true, &xrhocells[rho_offset_iy23_z1]);
                yrhocells_iy23_z1_v.Store(p_true, &yrhocells[rho_offset_iy23_z1]);
                zrhocells_iy23_z1_v.Store(p_true, &zrhocells[rho_offset_iy23_z1]);
                
                MVec xrhocells_iy23_z2_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z2]);
                MVec yrhocells_iy23_z2_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z2]);
                MVec zrhocells_iy23_z2_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z2]);
                
                xrhocells_iy23_z2_v += wqx_sx_y23_z2;
                yrhocells_iy23_z2_v += wqy_sx_y23_z2;
                zrhocells_iy23_z2_v += wqz_sx_y23_z2;
                
                xrhocells_iy23_z2_v.Store(p_true, &xrhocells[rho_offset_iy23_z2]);
                yrhocells_iy23_z2_v.Store(p_true, &yrhocells[rho_offset_iy23_z2]);
                zrhocells_iy23_z2_v.Store(p_true, &zrhocells[rho_offset_iy23_z2]);
                
                MVec xrhocells_iy23_z3_v = MVec::Load(p_true, &xrhocells[rho_offset_iy23_z3]);
                MVec yrhocells_iy23_z3_v = MVec::Load(p_true, &yrhocells[rho_offset_iy23_z3]);
                MVec zrhocells_iy23_z3_v = MVec::Load(p_true, &zrhocells[rho_offset_iy23_z3]);
                
                xrhocells_iy23_z3_v += wqx_sx_y23_z3;
                yrhocells_iy23_z3_v += wqy_sx_y23_z3;
                zrhocells_iy23_z3_v += wqz_sx_y23_z3;
                
                xrhocells_iy23_z3_v.Store(p_true, &xrhocells[rho_offset_iy23_z3]);
                yrhocells_iy23_z3_v.Store(p_true, &yrhocells[rho_offset_iy23_z3]);
                zrhocells_iy23_z3_v.Store(p_true, &zrhocells[rho_offset_iy23_z3]);
            }
        }
    }

    alignas(64) long rho_bin[8];
    int sx_step[8] = {0, 1 * block_size, 2 * block_size, 3 * block_size, 0, 1 * block_size, 2 * block_size, 3 * block_size};
    int sy_step[8] = {0, 0, 0, 0, block_size, block_size, block_size, block_size};
    for(long ip = g_move_begin; ip < np_to_deposit; ip += vl){  
        svbool_t p_ip = svwhilelt_b64(ip, np_to_deposit);
        
        MVec uxp_v = MVec::Load(p_ip, &uxp[ip]);
        MVec uyp_v = MVec::Load(p_ip, &uyp[ip]);
        MVec uzp_v = MVec::Load(p_ip, &uzp[ip]);
        MVec gaminv = 1. / (1. + uxp_v * uxp_v * clightsq
                               + uyp_v * uyp_v * clightsq
                               + uzp_v * uzp_v * clightsq).Sqrt();
        MVec vx_v = uxp_v * gaminv;
        MVec vy_v = uyp_v * gaminv;
        MVec vz_v = uzp_v * gaminv;

        MVec wp_v = MVec::Load(p_ip, &wp[ip]);
        
        MVec wq_v = q * wp_v;

        MVec wqx_v = wq_v * invvol * vx_v;
        MVec wqy_v = wq_v * invvol * vy_v;
        MVec wqz_v = wq_v * invvol * vz_v;

        wqx_v.Store(p_ip, &wqx[0]);
        wqy_v.Store(p_ip, &wqy[0]);
        wqz_v.Store(p_ip, &wqz[0]);

        MVec xp = MVec::Load(p_ip, &mx[ip]);
        MVec yp = MVec::Load(p_ip, &my[ip]);
        MVec zp = MVec::Load(p_ip, &mz[ip]);

        MVec xmid = ((xp - xyzmin.x) + relative_time * vx_v) * dinv.x;
        MVec ymid = ((yp - xyzmin.y) + relative_time * vy_v) * dinv.y;
        MVec zmid = ((zp - xyzmin.z) + relative_time * vz_v) * dinv.z;
        
        MintVec j_cellv = Mvec_compute_shape_factor_vpu_order3(&sx_m[0], xmid, p_ip);
        MintVec k_cellv = Mvec_compute_shape_factor_vpu_order3(&sy_m[0], ymid, p_ip);
        MintVec l_cellv = Mvec_compute_shape_factor_vpu_order3(&sz_m[0], zmid, p_ip);

        MintVec rho_bin_v = j_cellv + k_cellv * lenx + l_cellv * lenx * leny;
        rho_bin_v.Store(p_ip, &rho_bin[0]);
        
        int rho_end = block_size < np_to_deposit  - ip ? block_size : np_to_deposit - ip;
        for (long iip = 0; iip < rho_end; iip++) 
        {
            int rho_bin_idx = rho_bin[iip];
            for (int iz = 0; iz < 4; ++iz) 
            {
                double sz_wqx_tmp = sz_m[iz * block_size + iip] * wqx[iip];
                double sz_wqy_tmp = sz_m[iz * block_size + iip] * wqy[iip];
                double sz_wqz_tmp = sz_m[iz * block_size + iip] * wqz[iip];
                for (int iy = 0; iy < 4; iy += 2) 
                {
                    int rho_offset = (rho_bin_idx + iy * lenx + iz * lenx * leny) * 8;

                    MintVec mx_index = svld1sw_s64(p_true, sx_step);
                    MintVec my_index = svld1sw_s64(p_true, sy_step);

                    MVec sx_v = svld1_gather_s64index_f64(p_true, &sx_m[iip], mx_index);
                    MVec sy_v = svld1_gather_s64index_f64(p_true, &sy_m[iy * block_size + iip], my_index);

                    MVec sx_m_sy_v = sx_v * sy_v;

                    MVec xrhocells_v = MVec::Load(p_true, &xrhocells[rho_offset]);
                    MVec yrhocells_v = MVec::Load(p_true, &yrhocells[rho_offset]);
                    MVec zrhocells_v = MVec::Load(p_true, &zrhocells[rho_offset]);

                    xrhocells_v += sx_m_sy_v * sz_wqx_tmp;
                    yrhocells_v += sx_m_sy_v * sz_wqy_tmp;
                    zrhocells_v += sx_m_sy_v * sz_wqz_tmp;

                    xrhocells_v.Store(p_true, &xrhocells[rho_offset]);
                    yrhocells_v.Store(p_true, &yrhocells[rho_offset]);
                    zrhocells_v.Store(p_true, &zrhocells[rho_offset]);
                }
            }
        }
    }
}

template <int depos_order>
void doDepositionShapeN_vpu_rhocell_mpu_physort_order3 (const GetParticlePosition<PIdx>& GetPosition,
                        const amrex::ParticleReal * const wp,
                        const amrex::ParticleReal * const uxp,
                        const amrex::ParticleReal * const uyp,
                        const amrex::ParticleReal * const uzp,
                        const int* ion_lev,
                        amrex::FArrayBox& jx_fab,
                        amrex::FArrayBox& jy_fab,
                        amrex::FArrayBox& jz_fab,
                        long np_to_deposit,
                        amrex::Real relative_time,
                        const amrex::XDim3 & dinv,
                        const amrex::XDim3 & xyzmin,
                        amrex::Dim3 len,
                        amrex::Real q,
                        ParticleTileType& ptile,
                        [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    
    const amrex::ParticleReal* mx = GetPosition.m_x;
    const amrex::ParticleReal* my = GetPosition.m_y;
    const amrex::ParticleReal* mz = GetPosition.m_z;
    

    const int xjstride = jx_arr.jstride;
    const int xkstride = jx_arr.kstride;

    const int lenx = len.x;
    const int leny = len.y;
    const int lenz = len.z;
    const int nrho = lenx * leny * lenz;

    // Per-cell temporary accumulation buffers for x/y/z current components.
    std::vector<amrex::Real> xrhocells(8 * nrho, 0.0);
    std::vector<amrex::Real> yrhocells(8 * nrho, 0.0);
    std::vector<amrex::Real> zrhocells(8 * nrho, 0.0);

    vpu_rhocell_mpu_physort_order3_kernel<3>(
        wp, uxp, uyp, uzp,
        mx, my, mz,
        relative_time, dinv, xyzmin, q,
        xrhocells, yrhocells, zrhocells, 
        lenx, leny, lenz,
        np_to_deposit, ptile);

    svbool_t p_0_3 = svwhilelt_b64(0, 4);
    // Flush per-cell buffers to global J arrays with vectorized loads/stores.
    for (int iz = 0; iz < lenz; ++iz)
    {
        for (int iy = 0; iy < leny; ++iy)
        {
            for(int ix = 0; ix < lenx; ix++){
                int arr_offset = ix + iy * xjstride + iz * xkstride;
                int rho_offset = 8 * (ix + iy * lenx + iz * lenx * leny);

                Vec xrhocells_0_3_v = Vec::Load(p_0_3, &xrhocells[0 + rho_offset]);
                Vec xrhocells_4_7_v = Vec::Load(p_0_3, &xrhocells[4 + rho_offset]);
                Vec jx_arr_0_3_v = Vec::Load(p_0_3, &jx_arr.p[0 + arr_offset]);
                Vec jx_arr_4_7_v = Vec::Load(p_0_3, &jx_arr.p[xjstride + arr_offset]);

                Vec yrhocells_0_3_v = Vec::Load(p_0_3, &yrhocells[0 + rho_offset]);
                Vec yrhocells_4_7_v = Vec::Load(p_0_3, &yrhocells[4 + rho_offset]);
                Vec jy_arr_0_3_v = Vec::Load(p_0_3, &jy_arr.p[0 + arr_offset]);
                Vec jy_arr_4_7_v = Vec::Load(p_0_3, &jy_arr.p[xjstride + arr_offset]);
                
                Vec zrhocells_0_3_v = Vec::Load(p_0_3, &zrhocells[0 + rho_offset]);
                Vec zrhocells_4_7_v = Vec::Load(p_0_3, &zrhocells[4 + rho_offset]);
                Vec jz_arr_0_3_v = Vec::Load(p_0_3, &jz_arr.p[0 + arr_offset]);
                Vec jz_arr_4_7_v = Vec::Load(p_0_3, &jz_arr.p[xjstride + arr_offset]);

                jx_arr_0_3_v += xrhocells_0_3_v;
                jy_arr_0_3_v += yrhocells_0_3_v;
                jz_arr_0_3_v += zrhocells_0_3_v;
                jx_arr_4_7_v += xrhocells_4_7_v;
                jy_arr_4_7_v += yrhocells_4_7_v;
                jz_arr_4_7_v += zrhocells_4_7_v;

                jx_arr_0_3_v.Store(p_0_3, &jx_arr.p[0 + arr_offset]);
                jy_arr_0_3_v.Store(p_0_3, &jy_arr.p[0 + arr_offset]);
                jz_arr_0_3_v.Store(p_0_3, &jz_arr.p[0 + arr_offset]);

                jx_arr_4_7_v.Store(p_0_3, &jx_arr.p[xjstride + arr_offset]);
                jy_arr_4_7_v.Store(p_0_3, &jy_arr.p[xjstride + arr_offset]);
                jz_arr_4_7_v.Store(p_0_3, &jz_arr.p[xjstride + arr_offset]);
            }
        }
    }
}

/**
 * \brief Direct current deposition for thread thread_num for the implicit scheme
 *        The only difference from doDepositionShapeN is in how the particle gamma
 *        is calculated.
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time n.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum at time n+1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionShapeNImplicit(const GetParticlePosition<PIdx>& GetPosition,
                                const amrex::ParticleReal * const wp,
                                const amrex::ParticleReal * const uxp_n,
                                const amrex::ParticleReal * const uyp_n,
                                const amrex::ParticleReal * const uzp_n,
                                const amrex::ParticleReal * const uxp,
                                const amrex::ParticleReal * const uyp,
                                const amrex::ParticleReal * const uzp,
                                const int * const ion_lev,
                                amrex::FArrayBox& jx_fab,
                                amrex::FArrayBox& jy_fab,
                                amrex::FArrayBox& jz_fab,
                                const long np_to_deposit,
                                const amrex::XDim3 & dinv,
                                const amrex::XDim3 & xyzmin,
                                const amrex::Dim3 lo,
                                const amrex::Real q,
                                [[maybe_unused]]const int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    amrex::ParallelFor(
            np_to_deposit,
            [=] AMREX_GPU_DEVICE (long ip) {
            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);

            const amrex::Real vx  = uxp[ip]*gaminv;
            const amrex::Real vy  = uyp[ip]*gaminv;
            const amrex::Real vz  = uzp[ip]*gaminv;

            amrex::Real wq  = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            const amrex::Real relative_time = 0._rt;
            doDepositionShapeNKernel<depos_order>(xp, yp, zp, wq, vx, vy, vz, jx_arr, jy_arr, jz_arr,
                                                  jx_type, jy_type, jz_type,
                                                  relative_time, dinv, xyzmin,
                                                  invvol, lo, n_rz_azimuthal_modes);

        }
    );
}

/**
 * \brief Current Deposition for thread thread_num using shared memory
 * \tparam depos_order deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param jx_fab,jy_fab,jz_fab FArrayBox of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param relative_time Time at which to deposit J, relative to the time of the
 *                      current positions of the particles. When different than 0,
 *                      the particle position will be temporarily modified to match
 *                      the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doDepositionSharedShapeN (const GetParticlePosition<PIdx>& GetPosition,
                               const amrex::ParticleReal * const wp,
                               const amrex::ParticleReal * const uxp,
                               const amrex::ParticleReal * const uyp,
                               const amrex::ParticleReal * const uzp,
                               const int*  ion_lev,
                               amrex::FArrayBox& jx_fab,
                               amrex::FArrayBox& jy_fab,
                               amrex::FArrayBox& jz_fab,
                               long np_to_deposit,
                               const amrex::Real relative_time,
                               const amrex::XDim3 & dinv,
                               const amrex::XDim3 & xyzmin,
                               amrex::Dim3 lo,
                               amrex::Real q,
                               int n_rz_azimuthal_modes,
                               const amrex::DenseBins<WarpXParticleContainer::ParticleTileType::ParticleTileDataType>& a_bins,
                               const amrex::Box& box,
                               const amrex::Geometry& geom,
                               const amrex::IntVect& a_tbox_max_size)
{
    using namespace amrex::literals;

#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
    using namespace amrex;

    auto permutation = a_bins.permutationPtr();

    amrex::Array4<amrex::Real> const& jx_arr = jx_fab.array();
    amrex::Array4<amrex::Real> const& jy_arr = jy_fab.array();
    amrex::Array4<amrex::Real> const& jz_arr = jz_fab.array();
    amrex::IntVect const jx_type = jx_fab.box().type();
    amrex::IntVect const jy_type = jy_fab.box().type();
    amrex::IntVect const jz_type = jz_fab.box().type();

    constexpr int zdir = WARPX_ZINDEX;
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // Loop over particles and deposit into jx_fab, jy_fab and jz_fab
    const auto dxiarr = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    amrex::Box sample_tbox(IntVect(AMREX_D_DECL(0,0,0)), a_tbox_max_size - 1);
    sample_tbox.grow(depos_order);

    amrex::Box sample_tbox_x = convert(sample_tbox, jx_type);
    amrex::Box sample_tbox_y = convert(sample_tbox, jy_type);
    amrex::Box sample_tbox_z = convert(sample_tbox, jz_type);

    const auto npts = amrex::max(sample_tbox_x.numPts(), sample_tbox_y.numPts(), sample_tbox_z.numPts());

    const int nblocks = a_bins.numBins();
    const int threads_per_block = WarpX::shared_mem_current_tpb;
    const auto offsets_ptr = a_bins.offsetsPtr();

    const std::size_t shared_mem_bytes = npts*sizeof(amrex::Real);
    const amrex::IntVect bin_size = WarpX::shared_tilesize;
    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(shared_mem_bytes <= max_shared_mem_bytes,
                                     "Tile size too big for GPU shared memory current deposition");

    amrex::ignore_unused(np_to_deposit);
    // Launch one thread-block per bin
    amrex::launch(
            nblocks, threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () noexcept {
        const int bin_id = blockIdx.x;
        const unsigned int bin_start = offsets_ptr[bin_id];
        const unsigned int bin_stop = offsets_ptr[bin_id+1];

        if (bin_start == bin_stop) { return; /*this bin has no particles*/ }

        // These boxes define the index space for the shared memory buffers
        amrex::Box buffer_box;
        {
            ParticleReal xp, yp, zp;
            GetPosition(permutation[bin_start], xp, yp, zp);
#if defined(WARPX_DIM_3D)
            IntVect iv = IntVect(int( amrex::Math::floor((xp-plo[0]) * dxiarr[0]) ),
                                 int( amrex::Math::floor((yp-plo[1]) * dxiarr[1]) ),
                                 int( amrex::Math::floor((zp-plo[2]) * dxiarr[2]) ));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            IntVect iv = IntVect(int( amrex::Math::floor((xp-plo[0]) * dxiarr[0]) ),
                                 int( amrex::Math::floor((zp-plo[1]) * dxiarr[1]) ));
#elif defined(WARPX_DIM_1D_Z)
            IntVect iv = IntVect(int( amrex::Math::floor((zp-plo[0]) * dxiarr[0]) ));
#endif
            iv += domain.smallEnd();
            getTileIndex(iv, box, true, bin_size, buffer_box);
        }

        buffer_box.grow(depos_order);
        Box tbox_x = convert(buffer_box, jx_type);
        Box tbox_y = convert(buffer_box, jy_type);
        Box tbox_z = convert(buffer_box, jz_type);

        Gpu::SharedMemory<amrex::Real> gsm;
        amrex::Real* const shared = gsm.dataPtr();

        amrex::Array4<amrex::Real> const jx_buff(shared,
                amrex::begin(tbox_x), amrex::end(tbox_x), 1);
        amrex::Array4<amrex::Real> const jy_buff(shared,
                amrex::begin(tbox_y), amrex::end(tbox_y), 1);
        amrex::Array4<amrex::Real> const jz_buff(shared,
                amrex::begin(tbox_z), amrex::end(tbox_z), 1);

        // Zero-initialize the temporary array in shared memory
        volatile amrex::Real* vs = shared;
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }
        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jx_buff, jx_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 0);
        }

        __syncthreads();
        addLocalToGlobal(tbox_x, jx_arr, jx_buff);
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }

        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jy_buff, jy_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 1);
        }

        __syncthreads();
        addLocalToGlobal(tbox_y, jy_arr, jy_buff);
        for (int i = threadIdx.x; i < npts; i += blockDim.x){
            vs[i] = 0.0;
        }

        __syncthreads();
        for (unsigned int ip_orig = bin_start+threadIdx.x; ip_orig<bin_stop; ip_orig += blockDim.x)
        {
            const unsigned int ip = permutation[ip_orig];
            depositComponent<depos_order>(GetPosition, wp, uxp, uyp, uzp, ion_lev, jz_buff, jz_type,
                                          relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes,
                                          ip, zdir, NODE, CELL, 2);
        }

        __syncthreads();
        addLocalToGlobal(tbox_z, jz_arr, jz_buff);
    });
#else // not using hip/cuda
    // Note, you should never reach this part of the code. This funcion cannot be called unless
    // using HIP/CUDA, and those things are checked prior
    //don't use any args
    ignore_unused(GetPosition, wp, uxp, uyp, uzp, ion_lev, jx_fab, jy_fab, jz_fab, np_to_deposit, relative_time, dinv, xyzmin, lo, q, n_rz_azimuthal_modes, a_bins, box, geom, a_tbox_max_size);
    WARPX_ABORT_WITH_MESSAGE("Shared memory only implemented for HIP/CUDA");
#endif
}

/**
 * \brief Esirkepov Current Deposition for thread thread_num
 *
 * \tparam depos_order  deposition order
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp,uyp,uzp  Pointer to arrays of particle momentum.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr Array4 of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param[in] relative_time Time at which to deposit J, relative to the time of the
 *                          current positions of the particles. When different than 0,
 *                          the particle position will be temporarily modified to match
 *                          the time of the deposition.
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doEsirkepovDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                                  const amrex::ParticleReal * const wp,
                                  const amrex::ParticleReal * const uxp,
                                  const amrex::ParticleReal * const uyp,
                                  const amrex::ParticleReal * const uzp,
                                  const int* ion_lev,
                                  const amrex::Array4<amrex::Real>& Jx_arr,
                                  const amrex::Array4<amrex::Real>& Jy_arr,
                                  const amrex::Array4<amrex::Real>& Jz_arr,
                                  long np_to_deposit,
                                  amrex::Real dt,
                                  amrex::Real relative_time,
                                  const amrex::XDim3 & dinv,
                                  const amrex::XDim3 & xyzmin,
                                  amrex::Dim3 lo,
                                  amrex::Real q,
                                  [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex;
    using namespace amrex::literals;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;
#if !defined(WARPX_DIM_3D)
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;
#endif

    amrex::XDim3 const invdtd = amrex::XDim3{(1.0_rt/dt)*dinv.y*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.y};

    Real constexpr clightsq = 1.0_rt / ( PhysConst::c * PhysConst::c );

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {
            // --- Get particle quantities
            Real const gaminv = 1.0_rt/std::sqrt(1.0_rt + uxp[ip]*uxp[ip]*clightsq
                                                 + uyp[ip]*uyp[ip]*clightsq
                                                 + uzp[ip]*uzp[ip]*clightsq);

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            Real const xp_new = xp + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv;
            Real const yp_new = yp + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv;
            Real const xp_mid = xp_new - 0.5_rt*dt*uxp[ip]*gaminv;
            Real const yp_mid = yp_new - 0.5_rt*dt*uyp[ip]*gaminv;
            Real const xp_old = xp_new - dt*uxp[ip]*gaminv;
            Real const yp_old = yp_new - dt*uyp[ip]*gaminv;
            Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            Real const rp_mid = std::sqrt(xp_mid*xp_mid + yp_mid*yp_mid);
            Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            const amrex::Real costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
            const amrex::Real sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);
            const amrex::Real costheta_new = (rp_new > 0._rt ? xp_new/rp_new : 1._rt);
            const amrex::Real sintheta_new = (rp_new > 0._rt ? yp_new/rp_new : 0._rt);
            const amrex::Real costheta_old = (rp_old > 0._rt ? xp_old/rp_old : 1._rt);
            const amrex::Real sintheta_old = (rp_old > 0._rt ? yp_old/rp_old : 0._rt);
            const Complex xy_new0 = Complex{costheta_new, sintheta_new};
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
            const Complex xy_old0 = Complex{costheta_old, sintheta_old};
            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp - xyzmin.x + (relative_time + 0.5_rt*dt)*uxp[ip]*gaminv)*dinv.x;
            double const x_old = x_new - dt*dinv.x*uxp[ip]*gaminv;
#endif
#endif
#if defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const y_new = (yp - xyzmin.y + (relative_time + 0.5_rt*dt)*uyp[ip]*gaminv)*dinv.y;
            double const y_old = y_new - dt*dinv.y*uyp[ip]*gaminv;
#endif
            // Keep these double to avoid bug in single precision
            double const z_new = (zp - xyzmin.z + (relative_time + 0.5_rt*dt)*uzp[ip]*gaminv)*dinv.z;
            double const z_old = z_new - dt*dinv.z*uzp[ip]*gaminv;

#if defined(WARPX_DIM_RZ)
            Real const vy = (-uxp[ip]*sintheta_mid + uyp[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            Real const vy = uyp[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            Real const vx = uxp[ip]*gaminv;
            Real const vy = uyp[ip]*gaminv;
#endif

            // --- Compute shape factors
            // Compute shape factors for position as they are now and at old positions
            // [ijk]_new: leftmost grid point that the particle touches
            const Compute_shape_factor< depos_order > compute_shape_factor;
            const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

            // Shape factor arrays
            // Note that there are extra values above and below
            // to possibly hold the factor for the old particle
            // which can be at a different grid location.
            // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
            double sx_new[depos_order + 3] = {0.};
            double sx_old[depos_order + 3] = {0.};
            const int i_new = compute_shape_factor(sx_new+1, x_new);
            const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
            double sy_new[depos_order + 3] = {0.};
            double sy_old[depos_order + 3] = {0.};
            const int j_new = compute_shape_factor(sy_new+1, y_new);
            const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
            double sz_new[depos_order + 3] = {0.};
            double sz_old[depos_order + 3] = {0.};
            const int k_new = compute_shape_factor(sz_new+1, z_new);
            const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

            // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
            int dil = 1, diu = 1;
            if (i_old < i_new) { dil = 0; }
            if (i_old > i_new) { diu = 0; }
#endif
#if defined(WARPX_DIM_3D)
            int djl = 1, dju = 1;
            if (j_old < j_new) { djl = 0; }
            if (j_old > j_new) { dju = 0; }
#endif
            int dkl = 1, dku = 1;
            if (k_old < k_new) { dkl = 0; }
            if (k_old > k_new) { dku = 0; }

#if defined(WARPX_DIM_3D)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int j=djl; j<=depos_order+2-dju; j++) {
                    amrex::Real sdxi = 0._rt;
                    for (int i=dil; i<=depos_order+1-diu; i++) {
                        sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*(
                            one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                           +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
                    }
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdyj = 0._rt;
                    for (int j=djl; j<=depos_order+1-dju; j++) {
                        sdyj += wq*invdtd.y*(sy_old[j] - sy_new[j])*(
                            one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                           +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
                    }
                }
            }
            for (int j=djl; j<=depos_order+2-dju; j++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdzk = 0._rt;
                    for (int k=dkl; k<=depos_order+1-dku; k++) {
                        sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*(
                            one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                           +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
                    }
                }
            }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real sdxi = 0._rt;
                for (int i=dil; i<=depos_order+1-diu; i++) {
                    sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    Real const sdyj = wq*vy*invvol*(
                        one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                       +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
                    Complex const I = Complex{0._rt, 1._rt};
                    Complex xy_new = xy_new0;
                    Complex xy_mid = xy_mid0;
                    Complex xy_old = xy_old0;
                    // Throughout the following loop, xy_ takes the value e^{i m theta_}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        // The minus sign comes from the different convention with respect to Davidson et al.
                        const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xyzmin.x*dinv.x)*wq*invdtd.x/(amrex::Real)imode
                                                  *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                                  + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                        xy_new = xy_new*xy_new0;
                        xy_mid = xy_mid*xy_mid0;
                        xy_old = xy_old*xy_old0;
                    }
#endif
                }
            }
            for (int i=dil; i<=depos_order+2-diu; i++) {
                Real sdzk = 0._rt;
                for (int k=dkl; k<=depos_order+1-dku; k++) {
                    sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
#elif defined(WARPX_DIM_1D_Z)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdxi = wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdyj = wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
            }
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
            }
#endif
        }
    );
}

/**
 * \brief Esirkepov Current Deposition for thread thread_num for implicit scheme
 *        The difference from doEsirkepovDepositionShapeN is in how the old and new
 *        particles positions are determined and in how the particle gamma is calculated.
 *
 * \tparam depos_order  deposition order
 * \param xp_n,yp_n,zp_n  Pointer to arrays of particle position at time level n.
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr Array4 of current density, either full array or tile.
 * \param np_to_deposit Number of particles for which current is deposited.
 * \param dt           Time step for particle level
 * \param dinv         3D cell size inverse
 * \param xyzmin       Physical lower bounds of domain.
 * \param lo           Index lower bounds of domain.
 * \param q            species charge.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doChargeConservingDepositionShapeNImplicit ([[maybe_unused]]const amrex::ParticleReal * const xp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const yp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const zp_n,
                                                 const GetParticlePosition<PIdx>& GetPosition,
                                                 const amrex::ParticleReal * const wp,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uxp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uyp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uzp_n,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uxp_nph,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uyp_nph,
                                                 [[maybe_unused]]const amrex::ParticleReal * const uzp_nph,
                                                 const int * const ion_lev,
                                                 const amrex::Array4<amrex::Real>& Jx_arr,
                                                 const amrex::Array4<amrex::Real>& Jy_arr,
                                                 const amrex::Array4<amrex::Real>& Jz_arr,
                                                 const long np_to_deposit,
                                                 const amrex::Real dt,
                                                 const amrex::XDim3 & dinv,
                                                 const amrex::XDim3 & xyzmin,
                                                 const amrex::Dim3 lo,
                                                 const amrex::Real q,
                                                 [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;

#if !defined(WARPX_DIM_3D)
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;
#endif

    amrex::XDim3 const invdtd = amrex::XDim3{(1.0_rt/dt)*dinv.y*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.z,
                                             (1.0_rt/dt)*dinv.x*dinv.y};

#if !defined(WARPX_DIM_1D_Z)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {
#if !defined(WARPX_DIM_3D)
            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);
#endif

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n[ip];
#endif
            ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n[ip];

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            amrex::Real const xp_new = xp_np1;
            amrex::Real const yp_new = yp_np1;
            amrex::Real const xp_mid = xp_nph;
            amrex::Real const yp_mid = yp_nph;
            amrex::Real const xp_old = xp_n[ip];
            amrex::Real const yp_old = yp_n[ip];
            amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
            const amrex::Real costheta_mid = (rp_mid > 0._rt ? xp_mid/rp_mid : 1._rt);
            const amrex::Real sintheta_mid = (rp_mid > 0._rt ? yp_mid/rp_mid : 0._rt);
            const amrex::Real costheta_new = (rp_new > 0._rt ? xp_new/rp_new : 1._rt);
            const amrex::Real sintheta_new = (rp_new > 0._rt ? yp_new/rp_new : 0._rt);
            const amrex::Real costheta_old = (rp_old > 0._rt ? xp_old/rp_old : 1._rt);
            const amrex::Real sintheta_old = (rp_old > 0._rt ? yp_old/rp_old : 0._rt);
            const Complex xy_new0 = Complex{costheta_new, sintheta_new};
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};
            const Complex xy_old0 = Complex{costheta_old, sintheta_old};
            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
#else
#if !defined(WARPX_DIM_1D_Z)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
#endif
#endif
#if defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
            double const y_old = (yp_n[ip] - xyzmin.y)*dinv.y;
#endif
            // Keep these double to avoid bug in single precision
            double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
            double const z_old = (zp_n[ip] - xyzmin.z)*dinv.z;

#if defined(WARPX_DIM_RZ)
            amrex::Real const vy = (-uxp_nph[ip]*sintheta_mid + uyp_nph[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const vx = uxp_nph[ip]*gaminv;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#endif

            // --- Compute shape factors
            // Compute shape factors for position as they are now and at old positions
            // [ijk]_new: leftmost grid point that the particle touches
            const Compute_shape_factor< depos_order > compute_shape_factor;
            const Compute_shifted_shape_factor< depos_order > compute_shifted_shape_factor;

            // Shape factor arrays
            // Note that there are extra values above and below
            // to possibly hold the factor for the old particle
            // which can be at a different grid location.
            // Keep these double to avoid bug in single precision
#if !defined(WARPX_DIM_1D_Z)
            double sx_new[depos_order + 3] = {0.};
            double sx_old[depos_order + 3] = {0.};
            const int i_new = compute_shape_factor(sx_new+1, x_new);
            const int i_old = compute_shifted_shape_factor(sx_old, x_old, i_new);
#endif
#if defined(WARPX_DIM_3D)
            double sy_new[depos_order + 3] = {0.};
            double sy_old[depos_order + 3] = {0.};
            const int j_new = compute_shape_factor(sy_new+1, y_new);
            const int j_old = compute_shifted_shape_factor(sy_old, y_old, j_new);
#endif
            double sz_new[depos_order + 3] = {0.};
            double sz_old[depos_order + 3] = {0.};
            const int k_new = compute_shape_factor(sz_new+1, z_new);
            const int k_old = compute_shifted_shape_factor(sz_old, z_old, k_new);

            // computes min/max positions of current contributions
#if !defined(WARPX_DIM_1D_Z)
            int dil = 1, diu = 1;
            if (i_old < i_new) { dil = 0; }
            if (i_old > i_new) { diu = 0; }
#endif
#if defined(WARPX_DIM_3D)
            int djl = 1, dju = 1;
            if (j_old < j_new) { djl = 0; }
            if (j_old > j_new) { dju = 0; }
#endif
            int dkl = 1, dku = 1;
            if (k_old < k_new) { dkl = 0; }
            if (k_old > k_new) { dku = 0; }

#if defined(WARPX_DIM_3D)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int j=djl; j<=depos_order+2-dju; j++) {
                    amrex::Real sdxi = 0._rt;
                    for (int i=dil; i<=depos_order+1-diu; i++) {
                        sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*(
                            one_third*(sy_new[j]*sz_new[k] + sy_old[j]*sz_old[k])
                           +one_sixth*(sy_new[j]*sz_old[k] + sy_old[j]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
                    }
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdyj = 0._rt;
                    for (int j=djl; j<=depos_order+1-dju; j++) {
                        sdyj += wq*invdtd.y*(sy_old[j] - sy_new[j])*(
                            one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                           +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
                    }
                }
            }
            for (int j=djl; j<=depos_order+2-dju; j++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    amrex::Real sdzk = 0._rt;
                    for (int k=dkl; k<=depos_order+1-dku; k++) {
                        sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*(
                            one_third*(sx_new[i]*sy_new[j] + sx_old[i]*sy_old[j])
                           +one_sixth*(sx_new[i]*sy_old[j] + sx_old[i]*sy_new[j]));
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
                    }
                }
            }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real sdxi = 0._rt;
                for (int i=dil; i<=depos_order+1-diu; i++) {
                    sdxi += wq*invdtd.x*(sx_old[i] - sx_new[i])*0.5_rt*(sz_new[k] + sz_old[k]);
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djr_cmplx = 2._rt *sdxi*xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                for (int i=dil; i<=depos_order+2-diu; i++) {
                    Real const sdyj = wq*vy*invvol*(
                        one_third*(sx_new[i]*sz_new[k] + sx_old[i]*sz_old[k])
                       +one_sixth*(sx_new[i]*sz_old[k] + sx_old[i]*sz_new[k]));
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
#if defined(WARPX_DIM_RZ)
                    Complex const I = Complex{0._rt, 1._rt};
                    Complex xy_new = xy_new0;
                    Complex xy_mid = xy_mid0;
                    Complex xy_old = xy_old0;
                    // Throughout the following loop, xy_ takes the value e^{i m theta_}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        // The minus sign comes from the different convention with respect to Davidson et al.
                        const Complex djt_cmplx = -2._rt * I*(i_new-1 + i + xyzmin.x*dinv.x)*wq*invdtd.x/(amrex::Real)imode
                                                  *(Complex(sx_new[i]*sz_new[k], 0._rt)*(xy_new - xy_mid)
                                                  + Complex(sx_old[i]*sz_old[k], 0._rt)*(xy_mid - xy_old));
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
                        xy_new = xy_new*xy_new0;
                        xy_mid = xy_mid*xy_mid0;
                        xy_old = xy_old*xy_old0;
                    }
#endif
                }
            }
            for (int i=dil; i<=depos_order+2-diu; i++) {
                Real sdzk = 0._rt;
                for (int k=dkl; k<=depos_order+1-dku; k++) {
                    sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k])*0.5_rt*(sx_new[i] + sx_old[i]);
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
#if defined(WARPX_DIM_RZ)
                    Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 comes from the normalization of the modes
                        const Complex djz_cmplx = 2._rt * sdzk * xy_mid;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
                        xy_mid = xy_mid*xy_mid0;
                    }
#endif
                }
            }
#elif defined(WARPX_DIM_1D_Z)

            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdxi = wq*vx*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
            }
            for (int k=dkl; k<=depos_order+2-dku; k++) {
                amrex::Real const sdyj = wq*vy*invvol*0.5_rt*(sz_old[k] + sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
            }
            amrex::Real sdzk = 0._rt;
            for (int k=dkl; k<=depos_order+1-dku; k++) {
                sdzk += wq*invdtd.z*(sz_old[k] - sz_new[k]);
                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
            }
#endif
        }
    );
}

/**
 * \brief Villasenor and Buneman Current Deposition for thread thread_num for implicit scheme.
 *        The specifics for the implicit scheme are in how gamma is determined. This is a charge-
 *        conserving deposition. The difference from Esirkepov is that the deposit is done segment
 *        by segment, where the segments are determined by cell crossings. In general, this results
 *        in a tighter stencil. The implementation is valid for an arbitrary number of cell crossings.
 *
 * \param depos_order  deposition order
 * \param xp_n,yp_n,zp_n  Pointer to arrays of particle position at time level n.
 * \param GetPosition  A functor for returning the particle position.
 * \param wp           Pointer to array of particle weights.
 * \param uxp_n,uyp_n,uzp_n  Pointer to arrays of particle momentum at time level n.
 * \param uxp_nph,uyp_nph,uzp_nph  Pointer to arrays of particle momentum at time level n + 1/2.
 * \param ion_lev      Pointer to array of particle ionization level. This is
                       required to have the charge of each macroparticle
                       since q is a scalar. For non-ionizable species,
                       ion_lev is a null pointer.
 * \param Jx_arr,Jy_arr,Jz_arr  Array4 of current density, either full array or tile.
 * \param np_to_deposit         Number of particles for which current is deposited.
 * \param dt                    Time step for particle level
 * \param dinv                  3D cell size inverse
 * \param xyzmin                Physical lower bounds of domain.
 * \param lo                    Index lower bounds of domain.
 * \param q                     species charge.
 * \param n_rz_azimuthal_modes  Number of azimuthal modes when using RZ geometry.
 */
template <int depos_order>
void doVillasenorDepositionShapeNImplicit ([[maybe_unused]]const amrex::ParticleReal * const xp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const yp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const zp_n,
                                           const GetParticlePosition<PIdx>& GetPosition,
                                           const amrex::ParticleReal * const wp,
                                           [[maybe_unused]]const amrex::ParticleReal * const uxp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uyp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uzp_n,
                                           [[maybe_unused]]const amrex::ParticleReal * const uxp_nph,
                                           [[maybe_unused]]const amrex::ParticleReal * const uyp_nph,
                                           [[maybe_unused]]const amrex::ParticleReal * const uzp_nph,
                                           const int * const ion_lev,
                                           const amrex::Array4<amrex::Real>& Jx_arr,
                                           const amrex::Array4<amrex::Real>& Jy_arr,
                                           const amrex::Array4<amrex::Real>& Jz_arr,
                                           const long np_to_deposit,
                                           const amrex::Real dt,
                                           const amrex::XDim3 & dinv,
                                           const amrex::XDim3 & xyzmin,
                                           const amrex::Dim3 lo,
                                           const amrex::Real q,
                                           [[maybe_unused]] const int n_rz_azimuthal_modes)
{
    using namespace amrex;

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    bool const do_ionization = ion_lev;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

#if (AMREX_SPACEDIM > 1)
    Real constexpr one_third = 1.0_rt / 3.0_rt;
    Real constexpr one_sixth = 1.0_rt / 6.0_rt;
#endif

    // Loop over particles and deposit into Jx_arr, Jy_arr and Jz_arr
    amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long const ip) {

#if !defined(WARPX_DIM_3D)
            constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

            // Compute inverse Lorentz factor, the average of gamma at time levels n and n+1
            // The uxp,uyp,uzp are the velocities at time level n+1/2
            const amrex::ParticleReal uxp_np1 = 2._prt*uxp_nph[ip] - uxp_n[ip];
            const amrex::ParticleReal uyp_np1 = 2._prt*uyp_nph[ip] - uyp_n[ip];
            const amrex::ParticleReal uzp_np1 = 2._prt*uzp_nph[ip] - uzp_n[ip];
            const amrex::ParticleReal gamma_n = std::sqrt(1._prt + (uxp_n[ip]*uxp_n[ip] + uyp_n[ip]*uyp_n[ip] + uzp_n[ip]*uzp_n[ip])*inv_c2);
            const amrex::ParticleReal gamma_np1 = std::sqrt(1._prt + (uxp_np1*uxp_np1 + uyp_np1*uyp_np1 + uzp_np1*uzp_np1)*inv_c2);
            const amrex::ParticleReal gaminv = 2.0_prt/(gamma_n + gamma_np1);
#endif

            Real wq = q*wp[ip];
            if (do_ionization){
                wq *= ion_lev[ip];
            }

            ParticleReal xp_nph, yp_nph, zp_nph;
            GetPosition(ip, xp_nph, yp_nph, zp_nph);

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const xp_np1 = 2._prt*xp_nph - xp_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const yp_np1 = 2._prt*yp_nph - yp_n[ip];
#endif
            ParticleReal const zp_np1 = 2._prt*zp_nph - zp_n[ip];

            // computes current and old position in grid units
#if defined(WARPX_DIM_RZ)
            amrex::Real const xp_new = xp_np1;
            amrex::Real const yp_new = yp_np1;
            amrex::Real const xp_mid = xp_nph;
            amrex::Real const yp_mid = yp_nph;
            amrex::Real const xp_old = xp_n[ip];
            amrex::Real const yp_old = yp_n[ip];
            amrex::Real const rp_new = std::sqrt(xp_new*xp_new + yp_new*yp_new);
            amrex::Real const rp_old = std::sqrt(xp_old*xp_old + yp_old*yp_old);
            amrex::Real const rp_mid = (rp_new + rp_old)/2._rt;
            amrex::Real costheta_mid, sintheta_mid;
            if (rp_mid > 0._rt) {
                costheta_mid = xp_mid/rp_mid;
                sintheta_mid = yp_mid/rp_mid;
            } else {
                costheta_mid = 1._rt;
                sintheta_mid = 0._rt;
            }
            const Complex xy_mid0 = Complex{costheta_mid, sintheta_mid};

            // Keep these double to avoid bug in single precision
            double const x_new = (rp_new - xyzmin.x)*dinv.x;
            double const x_old = (rp_old - xyzmin.x)*dinv.x;
            amrex::Real const vx = (rp_new - rp_old)/dt;
            amrex::Real const vy = (-uxp_nph[ip]*sintheta_mid + uyp_nph[ip]*costheta_mid)*gaminv;
#elif defined(WARPX_DIM_XZ)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
            amrex::Real const vx = (xp_np1 - xp_n[ip])/dt;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const vx = uxp_nph[ip]*gaminv;
            amrex::Real const vy = uyp_nph[ip]*gaminv;
#elif defined(WARPX_DIM_3D)
            // Keep these double to avoid bug in single precision
            double const x_new = (xp_np1 - xyzmin.x)*dinv.x;
            double const x_old = (xp_n[ip] - xyzmin.x)*dinv.x;
            double const y_new = (yp_np1 - xyzmin.y)*dinv.y;
            double const y_old = (yp_n[ip] - xyzmin.y)*dinv.y;
            amrex::Real const vx = (xp_np1 - xp_n[ip])/dt;
            amrex::Real const vy = (yp_np1 - yp_n[ip])/dt;
#endif

            // Keep these double to avoid bug in single precision
            double const z_new = (zp_np1 - xyzmin.z)*dinv.z;
            double const z_old = (zp_n[ip] - xyzmin.z)*dinv.z;
            amrex::Real const vz = (zp_np1 - zp_n[ip])/dt;

            // Define velocity kernels to deposit
            amrex::Real const wqx = wq*vx*invvol;
            amrex::Real const wqy = wq*vy*invvol;
            amrex::Real const wqz = wq*vz*invvol;

            // 1) Determine the number of segments.
            // 2) Loop over segments and deposit current.

            // cell crossings are defined at cell edges if depos_order is odd
            // cell crossings are defined at cell centers if depos_order is even

            int num_segments = 1;
            double shift = 0.0;
            if ( (depos_order % 2) == 0 ) { shift = 0.5; }

#if defined(WARPX_DIM_3D)

            // compute cell crossings in X-direction
            const auto i_old = static_cast<int>(x_old-shift);
            const auto i_new = static_cast<int>(x_new-shift);
            const int cell_crossings_x = std::abs(i_new-i_old);
            num_segments += cell_crossings_x;

            // compute cell crossings in Y-direction
            const auto j_old = static_cast<int>(y_old-shift);
            const auto j_new = static_cast<int>(y_new-shift);
            const int cell_crossings_y = std::abs(j_new-j_old);
            num_segments += cell_crossings_y;

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 7) ...

            // compute total change in particle position and the initial cell
            // locations in each direction used to find the position at cell crossings.
            const double dxp = x_new - x_old;
            const double dyp = y_new - y_old;
            const double dzp = z_new - z_old;
            const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
            const auto dirY_sign = static_cast<double>(dyp < 0. ? -1. : 1.);
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Xcell = 0., Ycell = 0., Zcell = 0.;
            if (num_segments > 1) {
                Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
                Ycell = static_cast<double>(j_old) + shift + 0.5*(1.-dirY_sign);
                Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
            }

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dxp_seg, dyp_seg, dzp_seg;
            double x0_new, y0_new, z0_new;
            double x0_old = x_old;
            double y0_old = y_old;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment

                    x0_new = x_new;
                    y0_new = y_new;
                    z0_new = z_new;
                    dxp_seg = x0_new - x0_old;
                    dyp_seg = y0_new - y0_old;
                    dzp_seg = z0_new - z0_old;

                }
                else {

                    x0_new = Xcell + dirX_sign;
                    y0_new = Ycell + dirY_sign;
                    z0_new = Zcell + dirZ_sign;
                    dxp_seg = x0_new - x0_old;
                    dyp_seg = y0_new - y0_old;
                    dzp_seg = z0_new - z0_old;

                    if ( (dyp == 0. || std::abs(dxp_seg) < std::abs(dxp/dyp*dyp_seg))
                      && (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) ) {
                        Xcell = x0_new;
                        dyp_seg = dyp/dxp*dxp_seg;
                        dzp_seg = dzp/dxp*dxp_seg;
                        y0_new = y0_old + dyp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else if (dzp == 0. || std::abs(dyp_seg) < std::abs(dyp/dzp*dzp_seg)) {
                        Ycell = y0_new;
                        dxp_seg = dxp/dyp*dyp_seg;
                        dzp_seg = dzp/dyp*dyp_seg;
                        x0_new = x0_old + dxp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else {
                        Zcell = z0_new;
                        dxp_seg = dxp/dzp*dzp_seg;
                        dyp_seg = dyp/dzp*dzp_seg;
                        x0_new = x0_old + dxp_seg;
                        y0_new = y0_old + dyp_seg;
                    }

                }

                // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, dyp, or dzp)
                const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
                const auto seg_factor_y = static_cast<double>(dyp == 0. ? 1. : dyp_seg/dyp);
                const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sx_cell[depos_order] = {0.};
                double sy_cell[depos_order] = {0.};
                double sz_cell[depos_order] = {0.};
                double const x0_bar = (x0_new + x0_old)/2.0;
                double const y0_bar = (y0_new + y0_old)/2.0;
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
                const int j0_cell = compute_shape_factor_cell( sy_cell, y0_bar-0.5 );
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sx_old_cell[depos_order] = {0.};
                    double sx_new_cell[depos_order] = {0.};
                    double sy_old_cell[depos_order] = {0.};
                    double sy_new_cell[depos_order] = {0.};
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
                    const int j0_cell_2 = compute_shape_factors_cell( sy_old_cell, sy_new_cell, y0_old-0.5, y0_new-0.5 );
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(i0_cell_2, j0_cell_2, k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                        sy_cell[m] = (4.0*sy_cell[m] + sy_old_cell[m] + sy_new_cell[m])/6.0;
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sx_old_node[depos_order+1] = {0.};
                double sx_new_node[depos_order+1] = {0.};
                double sy_old_node[depos_order+1] = {0.};
                double sy_new_node[depos_order+1] = {0.};
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
                const int j0_node = compute_shape_factors_node( sy_old_node, sy_new_node, y0_old, y0_new );
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit Jx for this segment
                amrex::Real this_Jx;
                for (int i=0; i<=depos_order-1; i++) {
                    for (int j=0; j<=depos_order; j++) {
                        for (int k=0; k<=depos_order; k++) {
                            this_Jx = wqx*sx_cell[i]*( sy_old_node[j]*sz_old_node[k]*one_third
                                                     + sy_old_node[j]*sz_new_node[k]*one_sixth
                                                     + sy_new_node[j]*sz_old_node[k]*one_sixth
                                                     + sy_new_node[j]*sz_new_node[k]*one_third )*seg_factor_x;
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k), this_Jx);
                        }
                    }
                }

                // deposit Jy for this segment
                amrex::Real this_Jy;
                for (int i=0; i<=depos_order; i++) {
                    for (int j=0; j<=depos_order-1; j++) {
                        for (int k=0; k<=depos_order; k++) {
                            this_Jy = wqy*sy_cell[j]*( sx_old_node[i]*sz_old_node[k]*one_third
                                                     + sx_old_node[i]*sz_new_node[k]*one_sixth
                                                     + sx_new_node[i]*sz_old_node[k]*one_sixth
                                                     + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k), this_Jy);
                        }
                    }
                }

                // deposit Jz for this segment
                amrex::Real this_Jz;
                for (int i=0; i<=depos_order; i++) {
                    for (int j=0; j<=depos_order; j++) {
                        for (int k=0; k<=depos_order-1; k++) {
                            this_Jz = wqz*sz_cell[k]*( sx_old_node[i]*sy_old_node[j]*one_third
                                                     + sx_old_node[i]*sy_new_node[j]*one_sixth
                                                     + sx_new_node[i]*sy_old_node[j]*one_sixth
                                                     + sx_new_node[i]*sy_new_node[j]*one_third )*seg_factor_z;
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k), this_Jz);
                        }
                    }
                }

                // update old segment values
                if (ns < num_segments-1) {
                    x0_old = x0_new;
                    y0_old = y0_new;
                    z0_old = z0_new;
                }

            } // end loop over segments

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)

            // compute cell crossings in X-direction
            const auto i_old = static_cast<int>(x_old-shift);
            const auto i_new = static_cast<int>(x_new-shift);
            const int cell_crossings_x = std::abs(i_new-i_old);
            num_segments += cell_crossings_x;

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 5) ...

            // compute total change in particle position and the initial cell
            // locations in each direction used to find the position at cell crossings.
            const double dxp = x_new - x_old;
            const double dzp = z_new - z_old;
            const auto dirX_sign = static_cast<double>(dxp < 0. ? -1. : 1.);
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Xcell = 0., Zcell = 0.;
            if (num_segments > 1) {
                Xcell = static_cast<double>(i_old) + shift + 0.5*(1.-dirX_sign);
                Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);
            }

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dxp_seg, dzp_seg;
            double x0_new, z0_new;
            double x0_old = x_old;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment

                    x0_new = x_new;
                    z0_new = z_new;
                    dxp_seg = x0_new - x0_old;
                    dzp_seg = z0_new - z0_old;

                }
                else {

                    x0_new = Xcell + dirX_sign;
                    z0_new = Zcell + dirZ_sign;
                    dxp_seg = x0_new - x0_old;
                    dzp_seg = z0_new - z0_old;

                    if (dzp == 0. || std::abs(dxp_seg) < std::abs(dxp/dzp*dzp_seg)) {
                        Xcell = x0_new;
                        dzp_seg = dzp/dxp*dxp_seg;
                        z0_new = z0_old + dzp_seg;
                    }
                    else {
                        Zcell = z0_new;
                        dxp_seg = dxp/dzp*dzp_seg;
                        x0_new = x0_old + dxp_seg;
                    }

                }

                // compute the segment factors (each equal to dt_seg/dt for nonzero dxp, or dzp)
                const auto seg_factor_x = static_cast<double>(dxp == 0. ? 1. : dxp_seg/dxp);
                const auto seg_factor_z = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sx_cell[depos_order] = {0.};
                double sz_cell[depos_order] = {0.};
                double const x0_bar = (x0_new + x0_old)/2.0;
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int i0_cell = compute_shape_factor_cell( sx_cell, x0_bar-0.5 );
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sx_old_cell[depos_order] = {0.};
                    double sx_new_cell[depos_order] = {0.};
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int i0_cell_2 = compute_shape_factors_cell( sx_old_cell, sx_new_cell, x0_old-0.5, x0_new-0.5 );
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(i0_cell_2, k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sx_cell[m] = (4.0*sx_cell[m] + sx_old_cell[m] + sx_new_cell[m])/6.0;
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sx_old_node[depos_order+1] = {0.};
                double sx_new_node[depos_order+1] = {0.};
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int i0_node = compute_shape_factors_node( sx_old_node, sx_new_node, x0_old, x0_new );
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit Jx for this segment
                amrex::Real this_Jx;
                for (int i=0; i<=depos_order-1; i++) {
                    for (int k=0; k<=depos_order; k++) {
                        this_Jx = wqx*sx_cell[i]*(sz_old_node[k] + sz_new_node[k])/2.0_rt*seg_factor_x;
                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 0), this_Jx);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djr_cmplx = 2._rt*this_Jx*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode-1), djr_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode), djr_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // deposit out-of-plane Jy for this segment
                const auto seg_factor_y = std::min(seg_factor_x,seg_factor_z);
                amrex::Real this_Jy;
                for (int i=0; i<=depos_order; i++) {
                    for (int k=0; k<=depos_order; k++) {
                        this_Jy = wqy*( sx_old_node[i]*sz_old_node[k]*one_third
                                      + sx_old_node[i]*sz_new_node[k]*one_sixth
                                      + sx_new_node[i]*sz_old_node[k]*one_sixth
                                      + sx_new_node[i]*sz_new_node[k]*one_third )*seg_factor_y;
                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 0), this_Jy);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0;
                        // Throughout the following loop, xy_ takes the value e^{i m theta_}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djy_cmplx = 2._rt*this_Jy*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode-1), djy_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode), djy_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // deposit Jz for this segment
                amrex::Real this_Jz;
                for (int i=0; i<=depos_order; i++) {
                    for (int k=0; k<=depos_order-1; k++) {
                        this_Jz = wqz*sz_cell[k]*(sx_old_node[i] + sx_new_node[i])/2.0_rt*seg_factor_z;
                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 0), this_Jz);
#if defined(WARPX_DIM_RZ)
                        Complex xy_mid = xy_mid0; // Throughout the following loop, xy_mid takes the value e^{i m theta}
                        for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                            // The factor 2 comes from the normalization of the modes
                            const Complex djz_cmplx = 2._rt*this_Jz*xy_mid;
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode-1), djz_cmplx.real());
                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode), djz_cmplx.imag());
                            xy_mid = xy_mid*xy_mid0;
                        }
#endif
                    }
                }

                // update old segment values
                if (ns < num_segments-1) {
                    x0_old = x0_new;
                    z0_old = z0_new;
                }

            } // end loop over segments

#elif defined(WARPX_DIM_1D_Z)

            // compute cell crossings in Z-direction
            const auto k_old = static_cast<int>(z_old-shift);
            const auto k_new = static_cast<int>(z_new-shift);
            const int cell_crossings_z = std::abs(k_new-k_old);
            num_segments += cell_crossings_z;

            // need to assert that the number of cell crossings in each direction
            // is within the range permitted by the number of guard cells
            // e.g., if (num_segments > 3) ...

            // compute dzp and the initial cell location used to find the cell crossings.
            double const dzp = z_new - z_old;
            const auto dirZ_sign = static_cast<double>(dzp < 0. ? -1. : 1.);
            double Zcell = static_cast<double>(k_old) + shift + 0.5*(1.-dirZ_sign);

            // loop over the number of segments and deposit
            const Compute_shape_factor< depos_order-1 > compute_shape_factor_cell;
            const Compute_shape_factor_pair< depos_order > compute_shape_factors_node;
            double dzp_seg;
            double z0_new;
            double z0_old = z_old;

            for (int ns=0; ns<num_segments; ns++) {

                if (ns == num_segments-1) { // final segment
                    z0_new = z_new;
                    dzp_seg = z0_new - z0_old;
                }
                else {
                    Zcell = Zcell + dirZ_sign;
                    z0_new = Zcell;
                    dzp_seg = z0_new - z0_old;
                }

                // compute the segment factor (equal to dt_seg/dt for nonzero dzp)
                const auto seg_factor = static_cast<double>(dzp == 0. ? 1. : dzp_seg/dzp);

                // compute cell-based weights using the average segment position
                double sz_cell[depos_order] = {0.};
                double const z0_bar = (z0_new + z0_old)/2.0;
                const int k0_cell = compute_shape_factor_cell( sz_cell, z0_bar-0.5 );

                if constexpr (depos_order >= 3) { // higher-order correction to the cell-based weights
                    const Compute_shape_factor_pair<depos_order-1> compute_shape_factors_cell;
                    double sz_old_cell[depos_order] = {0.};
                    double sz_new_cell[depos_order] = {0.};
                    const int k0_cell_2 = compute_shape_factors_cell( sz_old_cell, sz_new_cell, z0_old-0.5, z0_new-0.5 );
                    ignore_unused(k0_cell_2);
                    for (int m=0; m<depos_order; m++) {
                        sz_cell[m] = (4.0*sz_cell[m] + sz_old_cell[m] + sz_new_cell[m])/6.0;
                    }
                }

                // compute node-based weights using the old and new segment positions
                double sz_old_node[depos_order+1] = {0.};
                double sz_new_node[depos_order+1] = {0.};
                const int k0_node = compute_shape_factors_node( sz_old_node, sz_new_node, z0_old, z0_new );

                // deposit out-of-plane Jx and Jy for this segment
                for (int k=0; k<=depos_order; k++) {
                    const amrex::Real weight = 0.5_rt*(sz_old_node[k] + sz_new_node[k])*seg_factor;
                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k0_node+k, 0, 0), wqx*weight);
                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k0_node+k, 0, 0), wqy*weight);
                }

                // deposit Jz for this segment
                for (int k=0; k<=depos_order-1; k++) {
                    const amrex::Real this_Jz = wqz*sz_cell[k]*seg_factor;
                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k0_cell+k, 0, 0), this_Jz);
                }

                // update old segment values
                if (ns < num_segments-1) {
                    z0_old = z0_new;
                }

            }

#endif
        }
    );
}

/**
 * \brief Vay current deposition
 * (<a href="https://doi.org/10.1016/j.jcp.2013.03.010"> Vay et al, 2013</a>)
 * for thread \c thread_num: deposit \c D in real space and store the result in
 * \c Dx_fab, \c Dy_fab, \c Dz_fab
 *
 * \tparam depos_order  deposition order
 * \param[in] GetPosition  Functor that returns the particle position
 * \param[in] wp           Pointer to array of particle weights
 * \param[in] uxp,uyp,uzp  Pointer to arrays of particle momentum along \c x
 * \param[in] ion_lev      Pointer to array of particle ionization level. This is
                           required to have the charge of each macroparticle since \c q
                           is a scalar. For non-ionizable species, \c ion_lev is \c null
 * \param[in,out] Dx_fab,Dy_fab,Dz_fab FArrayBox of Vay current density, either full array or tile
 * \param[in] np_to_deposit Number of particles for which current is deposited
 * \param[in] dt           Time step for particle level
 * \param[in] relative_time Time at which to deposit D, relative to the time of the
 *                          current positions of the particles. When different than 0,
 *                          the particle position will be temporarily modified to match
 *                          the time of the deposition.
 * \param[in] dinv         3D cell size inverse
 * \param[in] xyzmin       3D lower bounds of physical domain
 * \param[in] lo           Dimension-agnostic lower bounds of index domain
 * \param[in] q            Species charge
 * \param[in] n_rz_azimuthal_modes Number of azimuthal modes in RZ geometry
 */
template <int depos_order>
void doVayDepositionShapeN (const GetParticlePosition<PIdx>& GetPosition,
                            const amrex::ParticleReal* const wp,
                            const amrex::ParticleReal* const uxp,
                            const amrex::ParticleReal* const uyp,
                            const amrex::ParticleReal* const uzp,
                            const int* const ion_lev,
                            amrex::FArrayBox& Dx_fab,
                            amrex::FArrayBox& Dy_fab,
                            amrex::FArrayBox& Dz_fab,
                            long np_to_deposit,
                            amrex::Real dt,
                            amrex::Real relative_time,
                            const amrex::XDim3 & dinv,
                            const amrex::XDim3 & xyzmin,
                            amrex::Dim3 lo,
                            amrex::Real q,
                            [[maybe_unused]]int n_rz_azimuthal_modes)
{
    using namespace amrex::literals;

#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(GetPosition,
        wp, uxp, uyp, uzp, ion_lev, Dx_fab, Dy_fab, Dz_fab,
        np_to_deposit, dt, relative_time, dinv, xyzmin, lo, q);
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented in RZ geometry");
#endif

#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(GetPosition,
        wp, uxp, uyp, uzp, ion_lev, Dx_fab, Dy_fab, Dz_fab,
        np_to_deposit, dt, relative_time, dinv, xyzmin, lo, q);
    WARPX_ABORT_WITH_MESSAGE("Vay deposition not implemented in 1D geometry");
#endif

#if !(defined WARPX_DIM_RZ || defined WARPX_DIM_1D_Z)

    // If ion_lev is a null pointer, then do_ionization=0, else do_ionization=1
    const bool do_ionization = ion_lev;

    // Inverse of time step
    const amrex::Real invdt = 1._rt / dt;

    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    // Allocate temporary arrays
#if defined(WARPX_DIM_3D)
    AMREX_ALWAYS_ASSERT(Dx_fab.box() == Dy_fab.box() && Dx_fab.box() == Dz_fab.box());
    amrex::FArrayBox temp_fab{Dx_fab.box(), 4};
#elif defined(WARPX_DIM_XZ)
    AMREX_ALWAYS_ASSERT(Dx_fab.box() == Dz_fab.box());
    amrex::FArrayBox temp_fab{Dx_fab.box(), 2};
#endif
    temp_fab.setVal<amrex::RunOn::Device>(0._rt);
    amrex::Array4<amrex::Real> const& temp_arr = temp_fab.array();

    // Inverse of light speed squared
    const amrex::Real invcsq = 1._rt / (PhysConst::c * PhysConst::c);

    // Arrays where D will be stored
    amrex::Array4<amrex::Real> const& Dx_arr = Dx_fab.array();
    amrex::Array4<amrex::Real> const& Dy_arr = Dy_fab.array();
    amrex::Array4<amrex::Real> const& Dz_arr = Dz_fab.array();

    // Loop over particles and deposit (Dx,Dy,Dz) into Dx_fab, Dy_fab and Dz_fab
    amrex::ParallelFor(np_to_deposit, [=] AMREX_GPU_DEVICE (long ip)
    {
        // Inverse of Lorentz factor gamma
        const amrex::Real invgam = 1._rt / std::sqrt(1._rt + uxp[ip] * uxp[ip] * invcsq
                                                           + uyp[ip] * uyp[ip] * invcsq
                                                           + uzp[ip] * uzp[ip] * invcsq);
        // Product of particle charges and weights
        amrex::Real wq = q * wp[ip];
        if (do_ionization) { wq *= ion_lev[ip]; }

        // Current particle positions (in physical units)
        amrex::ParticleReal xp, yp, zp;
        GetPosition(ip, xp, yp, zp);

        // Particle velocities
        const amrex::Real vx = uxp[ip] * invgam;
        const amrex::Real vy = uyp[ip] * invgam;
        const amrex::Real vz = uzp[ip] * invgam;

        // Modify the particle position to match the time of the deposition
        xp += relative_time * vx;
        yp += relative_time * vy;
        zp += relative_time * vz;

        // Current and old particle positions in grid units
        // Keep these double to avoid bug in single precision.
        double const x_new = (xp - xyzmin.x + 0.5_rt*dt*vx) * dinv.x;
        double const x_old = (xp - xyzmin.x - 0.5_rt*dt*vx) * dinv.x;
#if defined(WARPX_DIM_3D)
        // Keep these double to avoid bug in single precision.
        double const y_new = (yp - xyzmin.y + 0.5_rt*dt*vy) * dinv.y;
        double const y_old = (yp - xyzmin.y - 0.5_rt*dt*vy) * dinv.y;
#endif
        // Keep these double to avoid bug in single precision.
        double const z_new = (zp - xyzmin.z + 0.5_rt*dt*vz) * dinv.z;
        double const z_old = (zp - xyzmin.z - 0.5_rt*dt*vz) * dinv.z;

        // Shape factor arrays for current and old positions (nodal)
        // Keep these double to avoid bug in single precision.
        double sx_new[depos_order+1] = {0.};
        double sx_old[depos_order+1] = {0.};
#if defined(WARPX_DIM_3D)
        // Keep these double to avoid bug in single precision.
        double sy_new[depos_order+1] = {0.};
        double sy_old[depos_order+1] = {0.};
#endif
        // Keep these double to avoid bug in single precision.
        double sz_new[depos_order+1] = {0.};
        double sz_old[depos_order+1] = {0.};

        // Compute shape factors for current positions

        // i_new leftmost grid point in x that the particle touches
        // sx_new shape factor along x for the centering of each current
        Compute_shape_factor< depos_order > const compute_shape_factor;
        const int i_new = compute_shape_factor(sx_new, x_new);
#if defined(WARPX_DIM_3D)
        // j_new leftmost grid point in y that the particle touches
        // sy_new shape factor along y for the centering of each current
        const int j_new = compute_shape_factor(sy_new, y_new);
#endif
        // k_new leftmost grid point in z that the particle touches
        // sz_new shape factor along z for the centering of each current
        const int k_new = compute_shape_factor(sz_new, z_new);

        // Compute shape factors for old positions

        // i_old leftmost grid point in x that the particle touches
        // sx_old shape factor along x for the centering of each current
        const int i_old = compute_shape_factor(sx_old, x_old);
#if defined(WARPX_DIM_3D)
        // j_old leftmost grid point in y that the particle touches
        // sy_old shape factor along y for the centering of each current
        const int j_old = compute_shape_factor(sy_old, y_old);
#endif
        // k_old leftmost grid point in z that the particle touches
        // sz_old shape factor along z for the centering of each current
        const int k_old = compute_shape_factor(sz_old, z_old);

        // Deposit current into Dx_arr, Dy_arr and Dz_arr
#if defined(WARPX_DIM_XZ)

        const amrex::Real wqy = wq * vy * invvol;
        for (int k=0; k<=depos_order; k++) {
            for (int i=0; i<=depos_order; i++) {

                // Re-casting sx_new and sz_new from double to amrex::Real so that
                // Atomic::Add has consistent types in its argument
                auto const sxn_szn = static_cast<amrex::Real>(sx_new[i] * sz_new[k]);
                auto const sxo_szn = static_cast<amrex::Real>(sx_old[i] * sz_new[k]);
                auto const sxn_szo = static_cast<amrex::Real>(sx_new[i] * sz_old[k]);
                auto const sxo_szo = static_cast<amrex::Real>(sx_old[i] * sz_old[k]);

                if (i_new == i_old && k_new == k_old) {
                    // temp arrays for Dx and Dz
                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wq * invvol * invdt * (sxn_szn - sxo_szo));

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 1),
                        wq * invvol * invdt * (sxn_szo - sxo_szn));

                    // Dy
                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * (sxn_szn + sxn_szo + sxo_szn + sxo_szo));
                } else {
                    // temp arrays for Dx and Dz
                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wq * invvol * invdt * sxn_szn);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
                        - wq * invvol * invdt * sxo_szo);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 1),
                        wq * invvol * invdt * sxn_szo);

                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 1),
                        - wq * invvol * invdt * sxo_szn);

                    // Dy
                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * sxn_szn);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 0),
                        wqy * 0.25_rt * sxn_szo);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 0),
                        wqy * 0.25_rt * sxo_szn);

                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
                        wqy * 0.25_rt * sxo_szo);
                }

            }
        }

#elif defined(WARPX_DIM_3D)

        for (int k=0; k<=depos_order; k++) {
            for (int j=0; j<=depos_order; j++) {

                auto const syn_szn = static_cast<amrex::Real>(sy_new[j] * sz_new[k]);
                auto const syo_szn = static_cast<amrex::Real>(sy_old[j] * sz_new[k]);
                auto const syn_szo = static_cast<amrex::Real>(sy_new[j] * sz_old[k]);
                auto const syo_szo = static_cast<amrex::Real>(sy_old[j] * sz_old[k]);

                for (int i=0; i<=depos_order; i++) {

                    auto const sxn_syn_szn = static_cast<amrex::Real>(sx_new[i]) * syn_szn;
                    auto const sxo_syn_szn = static_cast<amrex::Real>(sx_old[i]) * syn_szn;
                    auto const sxn_syo_szn = static_cast<amrex::Real>(sx_new[i]) * syo_szn;
                    auto const sxo_syo_szn = static_cast<amrex::Real>(sx_old[i]) * syo_szn;
                    auto const sxn_syn_szo = static_cast<amrex::Real>(sx_new[i]) * syn_szo;
                    auto const sxo_syn_szo = static_cast<amrex::Real>(sx_old[i]) * syn_szo;
                    auto const sxn_syo_szo = static_cast<amrex::Real>(sx_new[i]) * syo_szo;
                    auto const sxo_syo_szo = static_cast<amrex::Real>(sx_old[i]) * syo_szo;

                    if (i_new == i_old && j_new == j_old && k_new == k_old) {
                        // temp arrays for Dx, Dy and Dz
                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
                            wq * invvol * invdt * (sxn_syn_szn - sxo_syo_szo));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 1),
                            wq * invvol * invdt * (sxn_syn_szo - sxo_syo_szn));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 2),
                            wq * invvol * invdt * (sxn_syo_szn - sxo_syn_szo));

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 3),
                            wq * invvol * invdt * (sxo_syn_szn - sxn_syo_szo));
                    } else {
                        // temp arrays for Dx, Dy and Dz
                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
                            wq * invvol * invdt * sxn_syn_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_old + k, 0),
                            - wq * invvol * invdt * sxo_syo_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_old + k, 1),
                            wq * invvol * invdt * sxn_syn_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_new + k, 1),
                            - wq * invvol * invdt * sxo_syo_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_new + k, 2),
                            wq * invvol * invdt * sxn_syo_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_old + k, 2),
                            - wq * invvol * invdt * sxo_syn_szo);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_new + k, 3),
                            wq * invvol * invdt * sxo_syn_szn);

                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_old + k, 3),
                            - wq * invvol * invdt * sxn_syo_szo);
                    }
                }
            }
        }
#endif
    } );

#if defined(WARPX_DIM_3D)
    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        const amrex::Real t_a = temp_arr(i,j,k,0);
        const amrex::Real t_b = temp_arr(i,j,k,1);
        const amrex::Real t_c = temp_arr(i,j,k,2);
        const amrex::Real t_d = temp_arr(i,j,k,3);
        Dx_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a       + t_b       + t_c - 2._rt*t_d);
        Dy_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a       + t_b - 2._rt*t_c       + t_d);
        Dz_arr(i,j,k) += (1._rt/6._rt)*(2_rt*t_a - 2._rt*t_b       + t_c       + t_d);
    });
#elif defined(WARPX_DIM_XZ)
    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        const amrex::Real t_a = temp_arr(i,j,0,0);
        const amrex::Real t_b = temp_arr(i,j,0,1);
        Dx_arr(i,j,0) += (0.5_rt)*(t_a + t_b);
        Dz_arr(i,j,0) += (0.5_rt)*(t_a - t_b);
    });
#endif
    // Synchronize so that temp_fab can be safely deallocated in its destructor
    amrex::Gpu::streamSynchronize();

#endif // #if !(defined WARPX_DIM_RZ || defined WARPX_DIM_1D_Z)
}
#endif // WARPX_CURRENTDEPOSITION_H_
