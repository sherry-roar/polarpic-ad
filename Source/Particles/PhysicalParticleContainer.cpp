/* Copyright 2019-2020 Andrew Myers, Aurore Blelly, Axel Huebl
 * David Grote, Glenn Richardson, Jean-Luc Vay
 * Ligia Diana Amorim, Luca Fedeli, Maxence Thevenet
 * Michael Rowan, Remi Lehe, Revathi Jambunathan
 * Weiqun Zhang, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PhysicalParticleContainer.H"

#include "Filter/NCIGodfreyFilter.H"
#include "Initialization/InjectorDensity.H"
#include "Initialization/InjectorMomentum.H"
#include "Initialization/InjectorPosition.H"
#include "MultiParticleContainer.H"
#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#   include "Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H"
#endif
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/ParticleCreation/DefaultInitialization.H"
#include "Particles/Pusher/CopyParticleAttribs.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/PushSelector.H"
#include "Particles/Pusher/UpdateMomentumBoris.H"
#include "Particles/Pusher/UpdateMomentumBorisWithRadiationReaction.H"
#include "Particles/Pusher/UpdateMomentumHigueraCary.H"
#include "Particles/Pusher/UpdateMomentumVay.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/Physics/IonizationEnergiesTable.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#ifdef AMREX_USE_EB
#   include "EmbeddedBoundary/ParticleBoundaryProcess.H"
#   include "EmbeddedBoundary/ParticleScraper.H"
#endif
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuElixir.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParGDB.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainerBase.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_SPACE.H>
#include <AMReX_Scan.H>
#include <AMReX_StructOfArrays.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>
#include <AMReX_Parser.H>

#ifdef AMREX_USE_OMP
#   include <omp.h>
#endif

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/openPMD.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

using namespace amrex;

#include <arc_vpu.h>
#include <arc_mpu.h>

#include <unr.h>

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
    vpuc v_;  // Assume vpuc aliases svfloat64_t

public:
    // Default constructor
    Vec() = default;

    // Construct from svfloat64_t
    Vec(svfloat64_t v) : v_(v) {}

    // Construct from double
    Vec(double v) : v_(svdup_f64(v)) {}

    // Convert to svfloat64_t
    operator svfloat64_t() const {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const Vec& rhs) {
        v_ = svadd_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const Vec& rhs) {
        v_ = svsub_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const Vec& rhs) {
        v_ = svmul_f64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    Vec operator+(const Vec &rhs) const {
        return Vec(svadd_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector subtraction
    Vec operator-(const Vec &rhs) const {
        return Vec(svsub_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector multiplication
    Vec operator*(const Vec &rhs) const {
        return Vec(svmul_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector division
    Vec operator/(const Vec &rhs) const {
        return Vec(svdiv_f64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Unary minus
    Vec operator-() const {
        return Vec(svneg_f64_x(svptrue_b64(), v_));
    }

    // Vector square root
    Vec Sqrt() const {
        return Vec(svsqrt_f64_x(svptrue_b64(), v_));
    }

    // Vector plus scalar
    Vec operator+(double rhs) const {
        return *this + Vec(rhs);
    }

    // Vector minus scalar
    Vec operator-(double rhs) const {
        return *this - Vec(rhs);
    }

    // Vector times scalar
    Vec operator*(double rhs) const {
        return *this * Vec(rhs);
    }

    // Vector divided by scalar
    Vec operator/(double rhs) const {
        return *this / Vec(rhs);
    }

    // Friend: scalar plus vector
    friend Vec operator+(double lhs, const Vec &rhs) {
        return Vec(lhs) + rhs;
    }

    // Friend: scalar minus vector
    friend Vec operator-(double lhs, const Vec &rhs) {
        return Vec(lhs) - rhs;
    }

    // Friend: scalar times vector
    friend Vec operator*(double lhs, const Vec &rhs) {
        return Vec(lhs) * rhs;
    }

    // Friend: scalar divided by vector
    friend Vec operator/(double lhs, const Vec &rhs) {
        return Vec(lhs) / rhs;
    }

    // Store vector to memory
    void Store(svbool_t p, double *mem) const {
        svst1_f64(p, mem, v_);
    }

    // Load vector from memory
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
    // Default constructor
    intVec() = default;

    // Construct from svint64_t
    intVec(svint64_t v) : v_(v) {}

    // Construct from int64_t
    intVec(int64_t v) : v_(svdup_s64(v)) {}

    // Convert to svfloat64_t
    operator svint64_t() const {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const intVec& rhs) {
        v_ = svadd_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const intVec& rhs) {
        v_ = svsub_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const intVec& rhs) {
        v_ = svmul_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    intVec operator+(const intVec &rhs) const {
        return intVec(svadd_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector subtraction
    intVec operator-(const intVec &rhs) const {
        return intVec(svsub_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector multiplication
    intVec operator*(const intVec &rhs) const {
        return intVec(svmul_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector division
    intVec operator/(const intVec &rhs) const {
        return intVec(svdiv_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Unary minus
    intVec operator-() const {
        return intVec(svneg_s64_x(svptrue_b64(), v_));
    }

    // Vector square root
    // intVec Sqrt() const {
    //     return intVec(svsqrt_s64_x(svptrue_b64(), v_));
    // }

    // Vector plus scalar
    intVec operator+(int64_t rhs) const {
        return *this + intVec(rhs);
    }

    // Vector minus scalar
    intVec operator-(int64_t rhs) const {
        return *this - intVec(rhs);
    }

    // Vector times scalar
    intVec operator*(int64_t rhs) const {
        return *this * intVec(rhs);
    }

    // Vector divided by scalar
    intVec operator/(int64_t rhs) const {
        return *this / intVec(rhs);
    }

    // Friend: scalar plus vector
    friend intVec operator+(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) + rhs;
    }

    // Friend: scalar minus vector
    friend intVec operator-(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) - rhs;
    }

    // Friend: scalar times vector
    friend intVec operator*(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) * rhs;
    }

    // Friend: scalar divided by vector
    friend intVec operator/(int64_t lhs, const intVec &rhs) {
        return intVec(lhs) / rhs;
    }

    // Store vector to memory
    void Store(svbool_t p, int64_t *mem) const {
        svst1_s64(p, mem, v_);
    }

    // Load vector from memory
    static intVec Load(svbool_t p, const int64_t *mem) {
        return intVec(svld1_s64(p, mem));
    }
};

class MVec {
private:
    vpuc v_;
    
public:
    MVec() __arc_preserves("za") __arc_streaming = default;
    
    // Construct from svfloat64_t
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
        this->Store(p, tmp);  // Reuse the existing Store helper
        return tmp[pos];
    }
};

class MintVec {
private:
    vpucint  v_;  

public:
    // Default constructor
    MintVec() __arc_preserves("za") __arc_streaming = default;

    // Construct from svint64_t
    MintVec(svint64_t v) __arc_preserves("za") __arc_streaming : v_(v) {}

    // Construct from int64_t
    MintVec(int64_t v) __arc_preserves("za") __arc_streaming : v_(svdup_s64(v)) {}

    // Convert to svfloat64_t
    operator svint64_t() const __arc_preserves("za") __arc_streaming {
        return v_;
    }

    // Vector addition assignment
    void operator+=(const MintVec& rhs) __arc_preserves("za") __arc_streaming {
        v_ = svadd_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector subtraction assignment
    void operator-=(const MintVec& rhs) __arc_preserves("za") __arc_streaming {
        v_ = svsub_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector multiplication assignment
    void operator*=(const MintVec& rhs) __arc_preserves("za") __arc_streaming {
        v_ = svmul_s64_x(svptrue_b64(), v_, rhs.v_);
    }

    // Vector addition
    MintVec operator+(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svadd_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector subtraction
    MintVec operator-(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svsub_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector multiplication
    MintVec operator*(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svmul_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Vector division
    MintVec operator/(const MintVec &rhs) const __arc_preserves("za") __arc_streaming {
        return MintVec(svdiv_s64_x(svptrue_b64(), v_, rhs.v_));
    }

    // Unary minus
    MintVec operator-() const __arc_preserves("za") __arc_streaming {
        return MintVec(svneg_s64_x(svptrue_b64(), v_));
    }

    // Vector square root
    // MintVec Sqrt() const {
    //     return MintVec(svsqrt_s64_x(svptrue_b64(), v_));
    // }

    // Vector plus scalar
    MintVec operator+(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this + MintVec(rhs);
    }

    // Vector minus scalar
    MintVec operator-(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this - MintVec(rhs);
    }

    // Vector times scalar
    MintVec operator*(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this * MintVec(rhs);
    }

    // Vector divided by scalar
    MintVec operator/(int64_t rhs) const __arc_preserves("za") __arc_streaming {
        return *this / MintVec(rhs);
    }

    // Friend: scalar plus vector
    friend MintVec operator+(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) + rhs;
    }

    // Friend: scalar minus vector
    friend MintVec operator-(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) - rhs;
    }

    // Friend: scalar times vector
    friend MintVec operator*(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) * rhs;
    }

    // Friend: scalar divided by vector
    friend MintVec operator/(int64_t lhs, const MintVec &rhs) __arc_preserves("za") __arc_streaming {
        return MintVec(lhs) / rhs;
    }

    static MintVec Mvec2Mint(svbool_t p, vpuc mvec) __arc_preserves("za") __arc_streaming {
        return svcvt_s64_f64_z(p, mvec);
    }

    // Store vector to memory
    void Store(svbool_t p, int64_t *mem) const __arc_preserves("za") __arc_streaming {
        svst1_s64(p, mem, v_);
    }

    // Load vector from memory
    static MintVec Load(svbool_t p, const int64_t *mem) __arc_preserves("za") __arc_streaming {
        return MintVec(svld1_s64(p, mem));
    }

    int64_t GetElement(svbool_t p, int pos) __arc_preserves("za") __arc_streaming {
        int64_t tmp[svcntd()];
        this->Store(p, tmp);  // Reuse the existing Store helper
        return tmp[pos];
    }
};

class UintVec {
private:
    vpucuint v_;  
    
public:
    // Default constructor
    UintVec() = default;
    
    // Construct from svuint64_t
    UintVec(svuint64_t v) : v_(v) {}
    
    // Construct from uint64_t
    UintVec(uint64_t v) : v_(svdup_u64(v)) {}
    
    // Convert to svfloat64_t
    operator svuint64_t() const {
        return v_;
    }
    
    // Vector addition assignment
    void operator+=(const UintVec& rhs) {
        v_ = svadd_u64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // Vector subtraction assignment
    void operator-=(const UintVec& rhs) {
        v_ = svsub_u64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // Vector multiplication assignment
    void operator*=(const UintVec& rhs) {
        v_ = svmul_u64_x(svptrue_b64(), v_, rhs.v_);
    }
    
    // Vector addition
    UintVec operator+(const UintVec &rhs) const {
        return UintVec(svadd_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    
    // Vector subtraction
    UintVec operator-(const UintVec &rhs) const {
        return UintVec(svsub_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    
    // Vector multiplication
    UintVec operator*(const UintVec &rhs) const {
        return UintVec(svmul_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    
    // Vector division
    UintVec operator/(const UintVec &rhs) const {
        return UintVec(svdiv_u64_x(svptrue_b64(), v_, rhs.v_));
    }
    
    // Unary minus
    // UintVec operator-() const {
    //     return UintVec(svneg_u64_x(svptrue_b64(), v_));
    // }
    
    // Vector square root
    // UintVec Sqrt() const {
    //     return UintVec(svsqrt_u64_x(svptrue_b64(), v_));
    // }
    
    // Vector plus scalar
    UintVec operator+(uint64_t rhs) const {
        return *this + UintVec(rhs);
    }
    
    // Vector minus scalar
    UintVec operator-(uint64_t rhs) const {
        return *this - UintVec(rhs);
    }
    
    // Vector times scalar
    UintVec operator*(uint64_t rhs) const {
        return *this * UintVec(rhs);
    }
    
    // Vector divided by scalar
    UintVec operator/(uint64_t rhs) const {
        return *this / UintVec(rhs);
    }
    
    // Friend: scalar plus vector
    friend UintVec operator+(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) + rhs;
    }
    
    // Friend: scalar minus vector
    friend UintVec operator-(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) - rhs;
    }
    
    // Friend: scalar times vector
    friend UintVec operator*(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) * rhs;
    }
    
    // Friend: scalar divided by vector
    friend UintVec operator/(uint64_t lhs, const UintVec &rhs) {
        return UintVec(lhs) / rhs;
    }
    
    // Store vector to memory
    void Store(svbool_t p, uint64_t *mem) const {
        svst1_u64(p, mem, v_);
    }
    
    // Load vector from memory
    static UintVec Load(svbool_t p, const uint64_t *mem) {
        return UintVec(svld1_u64(p, mem));
    }
};

namespace
{
    using ParticleType = WarpXParticleContainer::ParticleType;

    // Since the user provides the density distribution
    // at t_lab=0 and in the lab-frame coordinates,
    // we need to find the lab-frame position of this
    // particle at t_lab=0, from its boosted-frame coordinates
    // Assuming ballistic motion, this is given by:
    // z0_lab = gamma*( z_boost*(1-beta*betaz_lab) - ct_boost*(betaz_lab-beta) )
    // where betaz_lab is the speed of the particle in the lab frame
    //
    // In order for this equation to be solvable, betaz_lab
    // is explicitly assumed to have no dependency on z0_lab
    //
    // Note that we use the bulk momentum to perform the ballistic correction
    // Assume no z0_lab dependency
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real applyBallisticCorrection(const XDim3& pos, const InjectorMomentum* inj_mom,
                                  Real gamma_boost, Real beta_boost, Real t) noexcept
    {
        const XDim3 u_bulk = inj_mom->getBulkMomentum(pos.x, pos.y, pos.z);
        const Real gamma_bulk = std::sqrt(1._rt +
                  (u_bulk.x*u_bulk.x+u_bulk.y*u_bulk.y+u_bulk.z*u_bulk.z));
        const Real betaz_bulk = u_bulk.z/gamma_bulk;
        const Real z0 = gamma_boost * ( pos.z*(1.0_rt-beta_boost*betaz_bulk)
                             - PhysConst::c*t*(betaz_bulk-beta_boost) );
        return z0;
    }

    struct PDim3 {
        ParticleReal x, y, z;

        AMREX_GPU_HOST_DEVICE
        PDim3(const amrex::XDim3& a):
            x{static_cast<ParticleReal>(a.x)},
            y{static_cast<ParticleReal>(a.y)},
            z{static_cast<ParticleReal>(a.z)}
        {}

        AMREX_GPU_HOST_DEVICE
        ~PDim3() = default;

        AMREX_GPU_HOST_DEVICE
        PDim3(PDim3 const &)            = default;
        AMREX_GPU_HOST_DEVICE
        PDim3& operator=(PDim3 const &) = default;
        AMREX_GPU_HOST_DEVICE
        PDim3(PDim3&&)                  = default;
        AMREX_GPU_HOST_DEVICE
        PDim3& operator=(PDim3&&)       = default;
    };

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    XDim3 getCellCoords (const GpuArray<Real, AMREX_SPACEDIM>& lo_corner,
                         const GpuArray<Real, AMREX_SPACEDIM>& dx,
                         const XDim3& r, const IntVect& iv) noexcept
    {
        XDim3 pos;
#if defined(WARPX_DIM_3D)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = lo_corner[1] + (iv[1]+r.y)*dx[1];
        pos.z = lo_corner[2] + (iv[2]+r.z)*dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        pos.x = lo_corner[0] + (iv[0]+r.x)*dx[0];
        pos.y = 0.0_rt;
#if   defined WARPX_DIM_XZ
        pos.z = lo_corner[1] + (iv[1]+r.y)*dx[1];
#elif defined WARPX_DIM_RZ
        // Note that for RZ, r.y will be theta
        pos.z = lo_corner[1] + (iv[1]+r.z)*dx[1];
#endif
#else
        pos.x = 0.0_rt;
        pos.y = 0.0_rt;
        pos.z = lo_corner[0] + (iv[0]+r.x)*dx[0];
#endif
        return pos;
    }

    /**
     * \brief This function is called in AddPlasma when we want a particle to be removed at the
     * next call to redistribute. It initializes all the particle properties to zero (to be safe
     * and avoid any possible undefined behavior before the next call to redistribute) and sets
     * the particle id to -1 so that it can be effectively deleted.
     *
     * \param idcpu particle id soa data
     * \param pa particle real soa data
     * \param ip index for soa data
     * \param do_field_ionization whether species has ionization
     * \param pi ionization level data
     * \param has_quantum_sync whether species has quantum synchrotron
     * \param p_optical_depth_QSR quantum synchrotron optical depth data
     * \param has_breit_wheeler whether species has Breit-Wheeler
     * \param p_optical_depth_BW Breit-Wheeler optical depth data
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void ZeroInitializeAndSetNegativeID (
        uint64_t * AMREX_RESTRICT idcpu,
        const GpuArray<ParticleReal*,PIdx::nattribs>& pa, long& ip,
        const bool& do_field_ionization, int* pi
#ifdef WARPX_QED
        ,const bool& has_quantum_sync, amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_QSR
        ,const bool& has_breit_wheeler, amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_BW
#endif
        ) noexcept
    {
        pa[PIdx::z][ip] = 0._rt;
#if (AMREX_SPACEDIM >= 2)
        pa[PIdx::x][ip] = 0._rt;
#endif
#if defined(WARPX_DIM_3D)
        pa[PIdx::y][ip] = 0._rt;
#endif
        pa[PIdx::w ][ip] = 0._rt;
        pa[PIdx::ux][ip] = 0._rt;
        pa[PIdx::uy][ip] = 0._rt;
        pa[PIdx::uz][ip] = 0._rt;
#ifdef WARPX_DIM_RZ
        pa[PIdx::theta][ip] = 0._rt;
#endif
        if (do_field_ionization) {pi[ip] = 0;}
#ifdef WARPX_QED
        if (has_quantum_sync) {p_optical_depth_QSR[ip] = 0._rt;}
        if (has_breit_wheeler) {p_optical_depth_BW[ip] = 0._rt;}
#endif

        idcpu[ip] = amrex::ParticleIdCpus::Invalid;
    }
}

PhysicalParticleContainer::PhysicalParticleContainer (AmrCore* amr_core, int ispecies,
                                                      const std::string& name)
    : WarpXParticleContainer(amr_core, ispecies),
      species_name(name)
{
    BackwardCompatibility();

    const ParmParse pp_species_name(species_name);

    std::string injection_style = "none";
    pp_species_name.query("injection_style", injection_style);
    if (injection_style != "none") {
        // The base plasma injector, whose input parameters have no source prefix.
        // Only created if needed
        plasma_injectors.push_back(std::make_unique<PlasmaInjector>(species_id, species_name, amr_core->Geom(0)));
    }

    std::vector<std::string> injection_sources;
    pp_species_name.queryarr("injection_sources", injection_sources);
    for (auto &source_name : injection_sources) {
        plasma_injectors.push_back(std::make_unique<PlasmaInjector>(species_id, species_name, amr_core->Geom(0),
                                                                    source_name));
    }

    // Setup the charge and mass. There are multiple ways that they can be specified, so checks are needed to
    // ensure that a value is specified and warnings given if multiple values are specified.
    // The ordering is that species.charge and species.mass take precedence over all other values.
    // Next is charge and mass determined from species_type.
    // Last is charge and mass from the plasma injector setup
    bool charge_from_source = false;
    bool mass_from_source = false;
    for (auto const& plasma_injector : plasma_injectors) {
        // For now, use the last value for charge and mass that is found.
        // A check could be added for consistency of multiple values, but it'll probably never be needed
        charge_from_source |= plasma_injector->queryCharge(charge);
        mass_from_source |= plasma_injector->queryMass(mass);
    }

    std::string physical_species_s;
    const bool species_is_specified = pp_species_name.query("species_type", physical_species_s);
    if (species_is_specified) {
        const auto physical_species_from_string = species::from_string( physical_species_s );
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(physical_species_from_string,
            physical_species_s + " does not exist!");
        physical_species = physical_species_from_string.value();
        charge = species::get_charge( physical_species );
        mass = species::get_mass( physical_species );
    }

    // parse charge and mass (overriding values above)
    const bool charge_is_specified = utils::parser::queryWithParser(pp_species_name, "charge", charge);
    const bool mass_is_specified = utils::parser::queryWithParser(pp_species_name, "mass", mass);

    if (charge_is_specified && species_is_specified) {
        ablastr::warn_manager::WMRecordWarning("Species",
            "Both '" + species_name +  ".charge' and " +
                species_name + ".species_type' are specified.\n" +
                species_name + ".charge' will take precedence.\n");
    }
    if (mass_is_specified && species_is_specified) {
        ablastr::warn_manager::WMRecordWarning("Species",
            "Both '" + species_name +  ".mass' and " +
                species_name + ".species_type' are specified.\n" +
                species_name + ".mass' will take precedence.\n");
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        charge_from_source ||
        charge_is_specified ||
        species_is_specified,
        "Need to specify at least one of species_type or charge for species '" +
        species_name + "'."
    );

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mass_from_source ||
        mass_is_specified ||
        species_is_specified,
        "Need to specify at least one of species_type or mass for species '" +
        species_name + "'."
    );

    pp_species_name.query("boost_adjust_tranvpurse_positions", boost_adjust_tranvpurse_positions);
    pp_species_name.query("do_backward_propagation", do_backward_propagation);
    pp_species_name.query("random_theta", m_rz_random_theta);

    // Initialize splitting
    pp_species_name.query("do_splitting", do_splitting);
    pp_species_name.query("split_type", split_type);
    pp_species_name.query("do_not_deposit", do_not_deposit);
    pp_species_name.query("do_not_gather", do_not_gather);
    pp_species_name.query("do_not_push", do_not_push);

    pp_species_name.query("do_continuous_injection", do_continuous_injection);
    pp_species_name.query("initialize_self_fields", initialize_self_fields);
    utils::parser::queryWithParser(
        pp_species_name, "self_fields_required_precision", self_fields_required_precision);
    utils::parser::queryWithParser(
        pp_species_name, "self_fields_absolute_tolerance", self_fields_absolute_tolerance);
    utils::parser::queryWithParser(
        pp_species_name, "self_fields_max_iters", self_fields_max_iters);
    pp_species_name.query("self_fields_verbosity", self_fields_verbosity);

    pp_species_name.query("do_field_ionization", do_field_ionization);

    pp_species_name.query("do_resampling", do_resampling);
    if (do_resampling) { m_resampler = Resampling(species_name); }

    //check if Radiation Reaction is enabled and do consistency checks
    pp_species_name.query("do_classical_radiation_reaction", do_classical_radiation_reaction);
    //if the species is not a lepton, do_classical_radiation_reaction
    //should be false
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!do_classical_radiation_reaction) ||
        AmIA<PhysicalSpecies::electron>() ||
        AmIA<PhysicalSpecies::positron>(),
        "can't enable classical radiation reaction for non lepton species '"
            + species_name + "'.");

    //Only Boris pusher is compatible with radiation reaction
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!do_classical_radiation_reaction) ||
        WarpX::particle_pusher_algo == ParticlePusherAlgo::Boris,
        "Radiation reaction can be enabled only if Boris pusher is used");
    //_____________________________

#ifdef WARPX_QED
    pp_species_name.query("do_qed_quantum_sync", m_do_qed_quantum_sync);
    if (m_do_qed_quantum_sync) {
        AddRealComp("opticalDepthQSR");
    }

    pp_species_name.query("do_qed_breit_wheeler", m_do_qed_breit_wheeler);
    if (m_do_qed_breit_wheeler) {
        AddRealComp("opticalDepthBW");
    }

    if(m_do_qed_quantum_sync){
        pp_species_name.get("qed_quantum_sync_phot_product_species",
            m_qed_quantum_sync_phot_product_name);
    }
#endif

    // User-defined integer attributes
    pp_species_name.queryarr("addIntegerAttributes", m_user_int_attribs);
    const auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    std::vector< std::string > str_int_attrib_function;
    str_int_attrib_function.resize(n_user_int_attribs);
    m_user_int_attrib_parser.resize(n_user_int_attribs);
    for (int i = 0; i < n_user_int_attribs; ++i) {
        utils::parser::Store_parserString(
            pp_species_name, "attribute."+m_user_int_attribs.at(i)+"(x,y,z,ux,uy,uz,t)",
            str_int_attrib_function.at(i));
        m_user_int_attrib_parser.at(i) = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_int_attrib_function.at(i),{"x","y","z","ux","uy","uz","t"}));
        AddIntComp(m_user_int_attribs.at(i));
    }

    // User-defined real attributes
    pp_species_name.queryarr("addRealAttributes", m_user_real_attribs);
    const auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    std::vector< std::string > str_real_attrib_function;
    str_real_attrib_function.resize(n_user_real_attribs);
    m_user_real_attrib_parser.resize(n_user_real_attribs);
    for (int i = 0; i < n_user_real_attribs; ++i) {
        utils::parser::Store_parserString(
            pp_species_name, "attribute."+m_user_real_attribs.at(i)+"(x,y,z,ux,uy,uz,t)",
            str_real_attrib_function.at(i));
        m_user_real_attrib_parser.at(i) = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_real_attrib_function.at(i),{"x","y","z","ux","uy","uz","t"}));
        AddRealComp(m_user_real_attribs.at(i));
    }

    // If old particle positions should be saved add the needed components
    pp_species_name.query("save_previous_position", m_save_previous_position);
    if (m_save_previous_position) {
#if (AMREX_SPACEDIM >= 2)
        AddRealComp("prev_x");
#endif
#if defined(WARPX_DIM_3D)
        AddRealComp("prev_y");
#endif
        AddRealComp("prev_z");
#ifdef WARPX_DIM_RZ
      amrex::Abort("Saving previous particle positions not yet implemented in RZ");
#endif
    }

    // Read reflection models for absorbing boundaries; defaults to a zero
    pp_species_name.query("reflection_model_xlo(E)", m_boundary_conditions.reflection_model_xlo_str);
    pp_species_name.query("reflection_model_xhi(E)", m_boundary_conditions.reflection_model_xhi_str);
    pp_species_name.query("reflection_model_ylo(E)", m_boundary_conditions.reflection_model_ylo_str);
    pp_species_name.query("reflection_model_yhi(E)", m_boundary_conditions.reflection_model_yhi_str);
    pp_species_name.query("reflection_model_zlo(E)", m_boundary_conditions.reflection_model_zlo_str);
    pp_species_name.query("reflection_model_zhi(E)", m_boundary_conditions.reflection_model_zhi_str);
    m_boundary_conditions.BuildReflectionModelParsers();

    const ParmParse pp_boundary("boundary");
    bool flag = false;
    pp_boundary.query("reflect_all_velocities", flag);
    m_boundary_conditions.Set_reflect_all_velocities(flag);

    // currently supports only isotropic thermal distribution
    // same distribution is applied to all boundaries
    const amrex::ParmParse pp_species_boundary("boundary." + species_name);
    if (WarpX::isAnyParticleBoundaryThermal()) {
        amrex::Real boundary_uth;
        utils::parser::getWithParser(pp_species_boundary,"u_th",boundary_uth);
        m_boundary_conditions.SetThermalVelocity(boundary_uth);
    }
}

PhysicalParticleContainer::PhysicalParticleContainer (AmrCore* amr_core)
    : WarpXParticleContainer(amr_core, 0)
{
}

void
PhysicalParticleContainer::BackwardCompatibility ()
{
    const ParmParse pp_species_name(species_name);
    std::vector<std::string> backward_strings;
    if (pp_species_name.queryarr("plot_vars", backward_strings)){
        WARPX_ABORT_WITH_MESSAGE("<species>.plot_vars is not supported anymore. "
                     "Please use the new syntax for diagnostics, see documentation.");
    }

    int backward_int;
    if (pp_species_name.query("plot_species", backward_int)){
        WARPX_ABORT_WITH_MESSAGE("<species>.plot_species is not supported anymore. "
                     "Please use the new syntax for diagnostics, see documentation.");
    }
}

void PhysicalParticleContainer::InitData ()
{
    AddParticles(0); // Note - add on level 0
    Redistribute();  // We then redistribute
}

void PhysicalParticleContainer::MapParticletoBoostedFrame (
    ParticleReal& x, ParticleReal& y, ParticleReal& z, ParticleReal& ux, ParticleReal& uy, ParticleReal& uz, Real t_lab) const
{
    // Map the particles from the lab frame to the boosted frame.
    // This boosts the particle to the lab frame and calculates
    // the particle time in the boosted frame. It then maps
    // the position to the time in the boosted frame.

    // For now, start with the assumption that this will only happen
    // at the start of the simulation.
    const ParticleReal uz_boost = WarpX::gamma_boost*WarpX::beta_boost*PhysConst::c;

    // tpr is the particle's time in the boosted frame
    const ParticleReal tpr = WarpX::gamma_boost*t_lab - uz_boost*z/(PhysConst::c*PhysConst::c);

    // The particle's transformed location in the boosted frame
    const ParticleReal xpr = x;
    const ParticleReal ypr = y;
    const ParticleReal zpr = WarpX::gamma_boost*z - uz_boost*t_lab;

    // transform u and gamma to the boosted frame
    const ParticleReal gamma_lab = std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)/(PhysConst::c*PhysConst::c));
    // ux = ux;
    // uy = uy;
    uz = WarpX::gamma_boost*uz - uz_boost*gamma_lab;
    const ParticleReal gammapr = std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz)/(PhysConst::c*PhysConst::c));

    const ParticleReal vxpr = ux/gammapr;
    const ParticleReal vypr = uy/gammapr;
    const ParticleReal vzpr = uz/gammapr;

    if (do_backward_propagation){
        uz = -uz;
    }

    //Move the particles to where they will be at t = t0, the current simulation time in the boosted frame
    constexpr int lev = 0;
    const amrex::Real t0 = WarpX::GetInstance().gett_new(lev);
    if (boost_adjust_tranvpurse_positions) {
        x = xpr - (tpr-t0)*vxpr;
        y = ypr - (tpr-t0)*vypr;
    }
    z = zpr - (tpr-t0)*vzpr;

}

void
PhysicalParticleContainer::AddGaussianBeam (PlasmaInjector const& plasma_injector){

    const Real x_m = plasma_injector.x_m;
    const Real y_m = plasma_injector.y_m;
    const Real z_m = plasma_injector.z_m;
    const Real x_rms = plasma_injector.x_rms;
    const Real y_rms = plasma_injector.y_rms;
    const Real z_rms = plasma_injector.z_rms;
    const Real x_cut = plasma_injector.x_cut;
    const Real y_cut = plasma_injector.y_cut;
    const Real z_cut = plasma_injector.z_cut;
    const Real q_tot = plasma_injector.q_tot;
    long npart = plasma_injector.npart;
    const int do_symmetrize = plasma_injector.do_symmetrize;
    const int symmetrization_order = plasma_injector.symmetrization_order;
    const Real focal_distance = plasma_injector.focal_distance;

    // Declare temporary vectors on the CPU
    Gpu::HostVector<ParticleReal> particle_x;
    Gpu::HostVector<ParticleReal> particle_y;
    Gpu::HostVector<ParticleReal> particle_z;
    Gpu::HostVector<ParticleReal> particle_ux;
    Gpu::HostVector<ParticleReal> particle_uy;
    Gpu::HostVector<ParticleReal> particle_uz;
    Gpu::HostVector<ParticleReal> particle_w;

    if (ParallelDescriptor::IOProcessor()) {
        // If do_symmetrize, create either 4x or 8x fewer particles, and
        // Replicate each particle either 4 times (x,y) (-x,y) (x,-y) (-x,-y)
        // or 8 times, additionally (y,x), (-y,x), (y,-x), (-y,-x)
        if (do_symmetrize){
            npart /= symmetrization_order;
        }
        for (long i = 0; i < npart; ++i) {
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            const Real weight = q_tot/(npart*charge);
            Real x = amrex::RandomNormal(x_m, x_rms);
            Real y = amrex::RandomNormal(y_m, y_rms);
            Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_XZ)
            const Real weight = q_tot/(npart*charge*y_rms);
            Real x = amrex::RandomNormal(x_m, x_rms);
            constexpr Real y = 0._prt;
            Real z = amrex::RandomNormal(z_m, z_rms);
#elif defined(WARPX_DIM_1D_Z)
            const Real weight = q_tot/(npart*charge*x_rms*y_rms);
            constexpr Real x = 0._prt;
            constexpr Real y = 0._prt;
            Real z = amrex::RandomNormal(z_m, z_rms);
#endif
            if (plasma_injector.insideBounds(x, y, z)  &&
                std::abs( x - x_m ) <= x_cut * x_rms     &&
                std::abs( y - y_m ) <= y_cut * y_rms     &&
                std::abs( z - z_m ) <= z_cut * z_rms   ) {
                XDim3 u = plasma_injector.getMomentum(x, y, z);

            if (plasma_injector.do_focusing){
                const XDim3 u_bulk = plasma_injector.getInjectorMomentumHost()->getBulkMomentum(x,y,z);
                const Real u_bulk_norm = std::sqrt( u_bulk.x*u_bulk.x+u_bulk.y*u_bulk.y+u_bulk.z*u_bulk.z );

                // Compute the position of the focal plane
                // (it is located at a distance `focal_distance` from the beam centroid, in the direction of the bulk velocity)
                const Real n_x = u_bulk.x/u_bulk_norm;
                const Real n_y = u_bulk.y/u_bulk_norm;
                const Real n_z = u_bulk.z/u_bulk_norm;
                const Real x_f = x_m + focal_distance * n_x;
                const Real y_f = y_m + focal_distance * n_y;
                const Real z_f = z_m + focal_distance * n_z;
                const Real gamma = std::sqrt( 1._rt + (u.x*u.x+u.y*u.y+u.z*u.z) );

                const Real v_x = u.x / gamma * PhysConst::c;
                const Real v_y = u.y / gamma * PhysConst::c;
                const Real v_z = u.z / gamma * PhysConst::c;

                // Compute the time at which the particle will cross the focal plane
                const Real v_dot_n = v_x * n_x + v_y * n_y + v_z * n_z;
                const Real t = ((x_f-x)*n_x + (y_f-y)*n_y + (z_f-z)*n_z) / v_dot_n;

                // Displace particles in the direction orthogonal to the beam bulk momentum
                // i.e. orthogonal to (n_x, n_y, n_z)
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                x = x - (v_x - v_dot_n*n_x) * t;
                y = y - (v_y - v_dot_n*n_y) * t;
                z = z - (v_z - v_dot_n*n_z) * t;
#elif defined(WARPX_DIM_XZ)
                x = x - (v_x - v_dot_n*n_x) * t;
                z = z - (v_z - v_dot_n*n_z) * t;
#elif defined(WARPX_DIM_1D_Z)
                z = z - (v_z - v_dot_n*n_z) * t;
#endif
            }
                u.x *= PhysConst::c;
                u.y *= PhysConst::c;
                u.z *= PhysConst::c;

                if (do_symmetrize && symmetrization_order == 8){
                    // Add eight particles to the beam:
                    CheckAndAddParticle(x, y, z, u.x, u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(x, -y, z, u.x, -u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, y, z, -u.x, u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, -y, z, -u.x, -u.y, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(y, x, z, u.y, u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-y, x, z, -u.y, u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(y, -x, z, u.y, -u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-y, -x, z, -u.y, -u.x, u.z, weight/8._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                } else if (do_symmetrize && symmetrization_order == 4){
                    // Add four particles to the beam:
                    CheckAndAddParticle(x, y, z, u.x, u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(x, -y, z, u.x, -u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, y, z, -u.x, u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                    CheckAndAddParticle(-x, -y, z, -u.x, -u.y, u.z, weight/4._rt,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                } else {
                    CheckAndAddParticle(x, y, z, u.x, u.y, u.z, weight,
                                        particle_x,  particle_y,  particle_z,
                                        particle_ux, particle_uy, particle_uz,
                                        particle_w);
                }
            }
        }
    }
    // Add the temporary CPU vectors to the particle structure
    auto const np = static_cast<long>(particle_z.size());

    const amrex::Vector<ParticleReal> xp(particle_x.data(), particle_x.data() + np);
    const amrex::Vector<ParticleReal> yp(particle_y.data(), particle_y.data() + np);
    const amrex::Vector<ParticleReal> zp(particle_z.data(), particle_z.data() + np);
    const amrex::Vector<ParticleReal> uxp(particle_ux.data(), particle_ux.data() + np);
    const amrex::Vector<ParticleReal> uyp(particle_uy.data(), particle_uy.data() + np);
    const amrex::Vector<ParticleReal> uzp(particle_uz.data(), particle_uz.data() + np);

    amrex::Vector<amrex::Vector<ParticleReal>> attr;
    const amrex::Vector<ParticleReal> wp(particle_w.data(), particle_w.data() + np);
    attr.push_back(wp);

    const amrex::Vector<amrex::Vector<int>> attr_int;

    AddNParticles(0, np, xp,  yp,  zp, uxp, uyp, uzp,
                  1, attr, 0, attr_int, 1);
}

void
PhysicalParticleContainer::AddPlasmaFromFile(PlasmaInjector & plasma_injector,
                                             ParticleReal q_tot,
                                             ParticleReal z_shift)
{
    // Declare temporary vectors on the CPU
    Gpu::HostVector<ParticleReal> particle_x;
    Gpu::HostVector<ParticleReal> particle_z;
    Gpu::HostVector<ParticleReal> particle_ux;
    Gpu::HostVector<ParticleReal> particle_uz;
    Gpu::HostVector<ParticleReal> particle_w;
    Gpu::HostVector<ParticleReal> particle_y;
    Gpu::HostVector<ParticleReal> particle_uy;

#ifdef WARPX_USE_OPENPMD
    //TODO: Make changes for read/write in multiple MPI ranks
    if (ParallelDescriptor::IOProcessor()) {
        // take ownership of the series and close it when done
        auto series = std::move(plasma_injector.m_openpmd_input_series);

        // assumption asserts: see PlasmaInjector
        openPMD::Iteration it = series->iterations.begin()->second;
        const ParmParse pp_species_name(species_name);
        pp_species_name.query("impose_t_lab_from_file", impose_t_lab_from_file);
        double t_lab = 0._prt;
        if (impose_t_lab_from_file) {
            // Impose t_lab as being the time stored in the openPMD file
            t_lab = it.time<double>() * it.timeUnitSI();
        }
        std::string const ps_name = it.particles.begin()->first;
        openPMD::ParticleSpecies ps = it.particles.begin()->second;

        auto const npart = ps["position"]["x"].getExtent()[0];
#if !defined(WARPX_DIM_1D_Z)  // 2D, 3D, and RZ
        const std::shared_ptr<ParticleReal> ptr_x = ps["position"]["x"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_x = ps["positionOffset"]["x"].loadChunk<ParticleReal>();
        auto const position_unit_x = static_cast<ParticleReal>(ps["position"]["x"].unitSI());
        auto const position_offset_unit_x = static_cast<ParticleReal>(ps["positionOffset"]["x"].unitSI());
#endif
#if !(defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z))
        const std::shared_ptr<ParticleReal> ptr_y = ps["position"]["y"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_y = ps["positionOffset"]["y"].loadChunk<ParticleReal>();
        auto const position_unit_y = static_cast<ParticleReal>(ps["position"]["y"].unitSI());
        auto const position_offset_unit_y = static_cast<ParticleReal>(ps["positionOffset"]["y"].unitSI());
#endif
        const std::shared_ptr<ParticleReal> ptr_z = ps["position"]["z"].loadChunk<ParticleReal>();
        const std::shared_ptr<ParticleReal> ptr_offset_z = ps["positionOffset"]["z"].loadChunk<ParticleReal>();
        auto const position_unit_z = static_cast<ParticleReal>(ps["position"]["z"].unitSI());
        auto const position_offset_unit_z = static_cast<ParticleReal>(ps["positionOffset"]["z"].unitSI());
        const std::shared_ptr<ParticleReal> ptr_ux = ps["momentum"]["x"].loadChunk<ParticleReal>();
        auto const momentum_unit_x = static_cast<ParticleReal>(ps["momentum"]["x"].unitSI());
        const std::shared_ptr<ParticleReal> ptr_uz = ps["momentum"]["z"].loadChunk<ParticleReal>();
        auto const momentum_unit_z = static_cast<ParticleReal>(ps["momentum"]["z"].unitSI());
        const std::shared_ptr<ParticleReal> ptr_w = ps["weighting"][openPMD::RecordComponent::SCALAR].loadChunk<ParticleReal>();
        auto const w_unit = static_cast<ParticleReal>(ps["weighting"][openPMD::RecordComponent::SCALAR].unitSI());
        std::shared_ptr<ParticleReal> ptr_uy = nullptr;
        auto momentum_unit_y = 1.0_prt;
        if (ps["momentum"].contains("y")) {
            ptr_uy = ps["momentum"]["y"].loadChunk<ParticleReal>();
            momentum_unit_y = static_cast<ParticleReal>(ps["momentum"]["y"].unitSI());
        }
        series->flush();  // shared_ptr data can be read now

        if (q_tot != 0.0) {
            std::stringstream warnMsg;
            warnMsg << " Loading particle species from file. " << ps_name << ".q_tot is ignored.";
            ablastr::warn_manager::WMRecordWarning("AddPlasmaFromFile",
               warnMsg.str(), ablastr::warn_manager::WarnPriority::high);
        }

        for (auto i = decltype(npart){0}; i<npart; ++i){

            ParticleReal const weight = ptr_w.get()[i]*w_unit;

#if !defined(WARPX_DIM_1D_Z)
            ParticleReal const x = ptr_x.get()[i]*position_unit_x + ptr_offset_x.get()[i]*position_offset_unit_x;
#else
            ParticleReal const x = 0.0_prt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            ParticleReal const y = ptr_y.get()[i]*position_unit_y + ptr_offset_y.get()[i]*position_offset_unit_y;
#else
            ParticleReal const y = 0.0_prt;
#endif
            ParticleReal const z = ptr_z.get()[i]*position_unit_z + ptr_offset_z.get()[i]*position_offset_unit_z + z_shift;

            if (plasma_injector.insideBounds(x, y, z)) {
                ParticleReal const ux = ptr_ux.get()[i]*momentum_unit_x/mass;
                ParticleReal const uz = ptr_uz.get()[i]*momentum_unit_z/mass;
                ParticleReal uy = 0.0_prt;
                if (ps["momentum"].contains("y")) {
                    uy = ptr_uy.get()[i]*momentum_unit_y/mass;
                }
                CheckAndAddParticle(x, y, z, ux, uy, uz, weight,
                                    particle_x,  particle_y,  particle_z,
                                    particle_ux, particle_uy, particle_uz,
                                    particle_w, static_cast<amrex::Real>(t_lab));
            }
        }
        auto const np = particle_z.size();
        if (np < npart) {
            ablastr::warn_manager::WMRecordWarning("Species",
                "Simulation box doesn't cover all particles",
                ablastr::warn_manager::WarnPriority::high);
        }
    } // IO Processor
    auto const np = static_cast<long>(particle_z.size());
    const amrex::Vector<ParticleReal> xp(particle_x.data(), particle_x.data() + np);
    const amrex::Vector<ParticleReal> yp(particle_y.data(), particle_y.data() + np);
    const amrex::Vector<ParticleReal> zp(particle_z.data(), particle_z.data() + np);
    const amrex::Vector<ParticleReal> uxp(particle_ux.data(), particle_ux.data() + np);
    const amrex::Vector<ParticleReal> uyp(particle_uy.data(), particle_uy.data() + np);
    const amrex::Vector<ParticleReal> uzp(particle_uz.data(), particle_uz.data() + np);

    amrex::Vector<amrex::Vector<ParticleReal>> attr;
    const amrex::Vector<ParticleReal> wp(particle_w.data(), particle_w.data() + np);
    attr.push_back(wp);

    const amrex::Vector<amrex::Vector<int>> attr_int;

    AddNParticles(0, np, xp,  yp,  zp, uxp, uyp, uzp,
                  1, attr, 0, attr_int, 1);
#endif // WARPX_USE_OPENPMD

    ignore_unused(plasma_injector, q_tot, z_shift);
}

void
PhysicalParticleContainer::DefaultInitializeRuntimeAttributes (
    typename ContainerLike<amrex::PinnedArenaAllocator>::ParticleTileType& pinned_tile,
    int n_external_attr_real,
    int n_external_attr_int)
{
    ParticleCreation::DefaultInitializeRuntimeAttributes(pinned_tile,
                                       n_external_attr_real, n_external_attr_int,
                                       m_user_real_attribs, m_user_int_attribs,
                                       particle_comps, particle_icomps,
                                       amrex::GetVecOfPtrs(m_user_real_attrib_parser),
                                       amrex::GetVecOfPtrs(m_user_int_attrib_parser),
#ifdef WARPX_QED
                                       true,
                                       m_shr_p_bw_engine.get(),
                                       m_shr_p_qs_engine.get(),
#endif
                                       ionization_initial_level,
                                       0,pinned_tile.numParticles());
}


void
PhysicalParticleContainer::CheckAndAddParticle (
    ParticleReal x, ParticleReal y, ParticleReal z,
    ParticleReal ux, ParticleReal uy, ParticleReal uz,
    ParticleReal weight,
    Gpu::HostVector<ParticleReal>& particle_x,
    Gpu::HostVector<ParticleReal>& particle_y,
    Gpu::HostVector<ParticleReal>& particle_z,
    Gpu::HostVector<ParticleReal>& particle_ux,
    Gpu::HostVector<ParticleReal>& particle_uy,
    Gpu::HostVector<ParticleReal>& particle_uz,
    Gpu::HostVector<ParticleReal>& particle_w,
    Real t_lab) const
{
    if (WarpX::gamma_boost > 1.) {
        MapParticletoBoostedFrame(x, y, z, ux, uy, uz, t_lab);
    }
    particle_x.push_back(x);
    particle_y.push_back(y);
    particle_z.push_back(z);
    particle_ux.push_back(ux);
    particle_uy.push_back(uy);
    particle_uz.push_back(uz);
    particle_w.push_back(weight);
}

void
PhysicalParticleContainer::AddParticles (int lev)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddParticles()");

    for (auto const& plasma_injector : plasma_injectors) {

        if (plasma_injector->add_single_particle) {
            if (WarpX::gamma_boost > 1.) {
                MapParticletoBoostedFrame(plasma_injector->single_particle_pos[0],
                                          plasma_injector->single_particle_pos[1],
                                          plasma_injector->single_particle_pos[2],
                                          plasma_injector->single_particle_u[0],
                                          plasma_injector->single_particle_u[1],
                                          plasma_injector->single_particle_u[2]);
            }
            const amrex::Vector<ParticleReal> xp = {plasma_injector->single_particle_pos[0]};
            const amrex::Vector<ParticleReal> yp = {plasma_injector->single_particle_pos[1]};
            const amrex::Vector<ParticleReal> zp = {plasma_injector->single_particle_pos[2]};
            const amrex::Vector<ParticleReal> uxp = {plasma_injector->single_particle_u[0]};
            const amrex::Vector<ParticleReal> uyp = {plasma_injector->single_particle_u[1]};
            const amrex::Vector<ParticleReal> uzp = {plasma_injector->single_particle_u[2]};
            const amrex::Vector<amrex::Vector<ParticleReal>> attr = {{plasma_injector->single_particle_weight}};
            const amrex::Vector<amrex::Vector<int>> attr_int;
            AddNParticles(lev, 1, xp, yp, zp, uxp, uyp, uzp,
                          1, attr, 0, attr_int, 0);
            return;
        }

        if (plasma_injector->add_multiple_particles) {
            if (WarpX::gamma_boost > 1.) {
                for (int i=0 ; i < plasma_injector->multiple_particles_pos_x.size() ; i++) {
                    MapParticletoBoostedFrame(plasma_injector->multiple_particles_pos_x[i],
                                              plasma_injector->multiple_particles_pos_y[i],
                                              plasma_injector->multiple_particles_pos_z[i],
                                              plasma_injector->multiple_particles_ux[i],
                                              plasma_injector->multiple_particles_uy[i],
                                              plasma_injector->multiple_particles_uz[i]);
                }
            }
            amrex::Vector<amrex::Vector<ParticleReal>> attr;
            attr.push_back(plasma_injector->multiple_particles_weight);
            const amrex::Vector<amrex::Vector<int>> attr_int;
            AddNParticles(lev, static_cast<int>(plasma_injector->multiple_particles_pos_x.size()),
                          plasma_injector->multiple_particles_pos_x,
                          plasma_injector->multiple_particles_pos_y,
                          plasma_injector->multiple_particles_pos_z,
                          plasma_injector->multiple_particles_ux,
                          plasma_injector->multiple_particles_uy,
                          plasma_injector->multiple_particles_uz,
                          1, attr, 0, attr_int, 0);
        }

        if (plasma_injector->gaussian_beam) {
            AddGaussianBeam(*plasma_injector);
        }

        if (plasma_injector->external_file) {
            AddPlasmaFromFile(*plasma_injector,
                              plasma_injector->q_tot,
                              plasma_injector->z_shift);
        }

        if ( plasma_injector->doInjection() ) {
            AddPlasma(*plasma_injector, lev);
        }
    }
}

void
PhysicalParticleContainer::AddPlasma (PlasmaInjector const& plasma_injector, int lev, RealBox part_realbox)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddPlasma()");

    // If no part_realbox is provided, initialize particles in the whole domain
    const Geometry& geom = Geom(lev);
    if (!part_realbox.ok()) { part_realbox = geom.ProbDomain(); }

    const int num_ppc = plasma_injector.num_particles_per_cell;
#ifdef WARPX_DIM_RZ
    const Real rmax = std::min(plasma_injector.xmax, part_realbox.hi(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

    defineAllParticleTiles();

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    const int nlevs = numLevels();
    static bool refine_injection = false;
    static Box fine_injection_box;
    static amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    // This does not work if the mesh is dynamic.  But in that case, we should
    // not use refined injected either.  We also assume there is only one fine level.
    if (WarpX::moving_window_active(WarpX::GetInstance().getistep(0)+1) and WarpX::refine_plasma
        and do_continuous_injection and nlevs == 2)
    {
        refine_injection = true;
        fine_injection_box = ParticleBoxArray(1).minimalBox();
        fine_injection_box.setSmall(WarpX::moving_window_dir, std::numeric_limits<int>::lowest()/2);
        fine_injection_box.setBig(WarpX::moving_window_dir, std::numeric_limits<int>::max()/2);
        rrfac = m_gdb->refRatio(0);
        fine_injection_box.coarsen(rrfac);
    }

    InjectorPosition* inj_pos = plasma_injector.getInjectorPosition();
    InjectorDensity*  inj_rho = plasma_injector.getInjectorDensity();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    const Real gamma_boost = WarpX::gamma_boost;
    const Real beta_boost = WarpX::beta_boost;
    const Real t = WarpX::GetInstance().gett_new(lev);
    const Real density_min = plasma_injector.density_min;
    const Real density_max = plasma_injector.density_max;

#ifdef WARPX_DIM_RZ
    const int nmodes = WarpX::n_rz_azimuthal_modes;
    const bool radially_weighted = plasma_injector.radially_weighted;
#endif


    // User-defined integer and real attributes: prepare parsers
    const auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
    const auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
    amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > user_int_attrib_parserexec_pinned(n_user_int_attribs);
    amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > user_real_attrib_parserexec_pinned(n_user_real_attribs);
    for (int ia = 0; ia < n_user_int_attribs; ++ia) {
        user_int_attrib_parserexec_pinned[ia] = m_user_int_attrib_parser[ia]->compile<7>();
    }
    for (int ia = 0; ia < n_user_real_attribs; ++ia) {
        user_real_attrib_parserexec_pinned[ia] = m_user_real_attrib_parser[ia]->compile<7>();
    }

    MFItInfo info;
    if (do_tiling && Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (not WarpX::serialize_initial_conditions)
#endif
    for (MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const Box& tile_box = mfi.tilebox();
        const RealBox tile_realbox = WarpX::getRealBox(tile_box, lev);

        // Find the cells of part_box that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        RealBox overlap_realbox;
        Box overlap_box;
        IntVect shifted;
        bool no_overlap = false;

        for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
            if ( tile_realbox.lo(dir) <= part_realbox.hi(dir) ) {
                const Real ncells_adjust = std::floor( (tile_realbox.lo(dir) - part_realbox.lo(dir))/dx[dir] );
                overlap_realbox.setLo( dir, part_realbox.lo(dir) + std::max(ncells_adjust, 0._rt) * dx[dir]);
            } else {
                no_overlap = true; break;
            }
            if ( tile_realbox.hi(dir) >= part_realbox.lo(dir) ) {
                const Real ncells_adjust = std::floor( (part_realbox.hi(dir) - tile_realbox.hi(dir))/dx[dir] );
                overlap_realbox.setHi( dir, part_realbox.hi(dir) - std::max(ncells_adjust, 0._rt) * dx[dir]);
            } else {
                no_overlap = true; break;
            }
            // Count the number of cells in this direction in overlap_realbox
            overlap_box.setSmall( dir, 0 );
            overlap_box.setBig( dir,
                int( std::round((overlap_realbox.hi(dir)-overlap_realbox.lo(dir))
                                /dx[dir] )) - 1);
            shifted[dir] =
                static_cast<int>(std::round((overlap_realbox.lo(dir)-problo[dir])/dx[dir]));
            // shifted is exact in non-moving-window direction.  That's all we care.
        }
        if (no_overlap == 1) {
            continue; // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const GpuArray<Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        Gpu::DeviceVector<amrex::Long> counts(overlap_box.numPts(), 0);
        Gpu::DeviceVector<amrex::Long> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        const amrex::IntVect lrrfac = rrfac;
        Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }
        amrex::ParallelFor(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const IntVect iv(AMREX_D_DECL(i, j, k));
            auto lo = getCellCoords(overlap_corner, dx, {0._rt, 0._rt, 0._rt}, iv);
            auto hi = getCellCoords(overlap_corner, dx, {1._rt, 1._rt, 1._rt}, iv);

            lo.z = applyBallisticCorrection(lo, inj_mom, gamma_boost, beta_boost, t);
            hi.z = applyBallisticCorrection(hi, inj_mom, gamma_boost, beta_boost, t);

            if (inj_pos->overlapsWith(lo, hi))
            {
                auto index = overlap_box.index(iv);
                const amrex::Long r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv))?
                    (AMREX_D_TERM(lrrfac[0],*lrrfac[1],*lrrfac[2])) : (1);
                pcounts[index] = num_ppc*r;
                // update pcount by checking if cell-corners or cell-center
                // has non-zero density
                const auto xlim = GpuArray<Real, 3>{lo.x,(lo.x+hi.x)/2._rt,hi.x};
                const auto ylim = GpuArray<Real, 3>{lo.y,(lo.y+hi.y)/2._rt,hi.y};
                const auto zlim = GpuArray<Real, 3>{lo.z,(lo.z+hi.z)/2._rt,hi.z};

                const auto checker = [&](){
                    for (const auto& x : xlim) {
                        for (const auto& y : ylim) {
                            for (const auto& z : zlim) {
                                if (inj_pos->insideBounds(x,y,z) and (inj_rho->getDensity(x,y,z) > 0) ) {
                                    return 1;
                                }
                            }
                        }
                    }
                    return 0;
                };
                const int flag_pcount = checker();
                if (flag_pcount == 1) {
                    pcounts[index] = num_ppc*r;
                } else {
                    pcounts[index] = 0;
                }
            }
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::ignore_unused(k);
#endif
#if defined(WARPX_DIM_1D_Z)
            amrex::ignore_unused(j,k);
#endif
        });

        // Max number of new particles. All of them are created,
        // and invalid ones are then discarded
        const amrex::Long max_new_particles = Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (add_plasma_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < LongParticleIds::LastParticleID,
            "ERROR: overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = GetParticles(lev)[std::make_pair(grid_id,tile_id)];

        if ( (NumRuntimeRealComps()>0) || (NumRuntimeIntComps()>0) ) {
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();
        GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;
        // user-defined integer and real attributes
        amrex::Gpu::PinnedVector<int*> pa_user_int_pinned(n_user_int_attribs);
        amrex::Gpu::PinnedVector<ParticleReal*> pa_user_real_pinned(n_user_real_attribs);
        for (int ia = 0; ia < n_user_int_attribs; ++ia) {
            pa_user_int_pinned[ia] = soa.GetIntData(particle_icomps[m_user_int_attribs[ia]]).data() + old_size;
        }
        for (int ia = 0; ia < n_user_real_attribs; ++ia) {
            pa_user_real_pinned[ia] = soa.GetRealData(particle_comps[m_user_real_attribs[ia]]).data() + old_size;
        }
#ifdef AMREX_USE_GPU
        // To avoid using managed memory, we first define pinned memory vector, initialize on cpu,
        // and them memcpy to device from host
        amrex::Gpu::DeviceVector<int*> d_pa_user_int(n_user_int_attribs);
        amrex::Gpu::DeviceVector<ParticleReal*> d_pa_user_real(n_user_real_attribs);
        amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > d_user_int_attrib_parserexec(n_user_int_attribs);
        amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > d_user_real_attrib_parserexec(n_user_real_attribs);
        amrex::Gpu::copyAsync(Gpu::hostToDevice, pa_user_int_pinned.begin(),
                              pa_user_int_pinned.end(), d_pa_user_int.begin());
        amrex::Gpu::copyAsync(Gpu::hostToDevice, pa_user_real_pinned.begin(),
                              pa_user_real_pinned.end(), d_pa_user_real.begin());
        amrex::Gpu::copyAsync(Gpu::hostToDevice, user_int_attrib_parserexec_pinned.begin(),
                              user_int_attrib_parserexec_pinned.end(), d_user_int_attrib_parserexec.begin());
        amrex::Gpu::copyAsync(Gpu::hostToDevice, user_real_attrib_parserexec_pinned.begin(),
                              user_real_attrib_parserexec_pinned.end(), d_user_real_attrib_parserexec.begin());
        int** pa_user_int_data = d_pa_user_int.dataPtr();
        ParticleReal** pa_user_real_data = d_pa_user_real.dataPtr();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = d_user_int_attrib_parserexec.dataPtr();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = d_user_real_attrib_parserexec.dataPtr();
#else
        int** pa_user_int_data = pa_user_int_pinned.dataPtr();
        ParticleReal** pa_user_real_data = pa_user_real_pinned.dataPtr();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = user_int_attrib_parserexec_pinned.dataPtr();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = user_real_attrib_parserexec_pinned.dataPtr();
#endif

        int* pi = nullptr;
        if (do_field_ionization) {
            pi = soa.GetIntData(particle_icomps["ionizationLevel"]).data() + old_size;
        }

#ifdef WARPX_QED
        //Pointer to the optical depth component
        amrex::ParticleReal* p_optical_depth_QSR = nullptr;
        amrex::ParticleReal* p_optical_depth_BW  = nullptr;

        // If a QED effect is enabled, the corresponding optical depth
        // has to be initialized
        const bool loc_has_quantum_sync = has_quantum_sync();
        const bool loc_has_breit_wheeler = has_breit_wheeler();
        if (loc_has_quantum_sync) {
            p_optical_depth_QSR = soa.GetRealData(
                particle_comps["opticalDepthQSR"]).data() + old_size;
        }
        if(loc_has_breit_wheeler) {
            p_optical_depth_BW = soa.GetRealData(
                particle_comps["opticalDepthBW"]).data() + old_size;
        }

        //If needed, get the appropriate functors from the engines
        QuantumSynchrotronGetOpticalDepth quantum_sync_get_opt;
        BreitWheelerGetOpticalDepth breit_wheeler_get_opt;
        if(loc_has_quantum_sync){
            quantum_sync_get_opt =
                m_shr_p_qs_engine->build_optical_depth_functor();
        }
        if(loc_has_breit_wheeler){
            breit_wheeler_get_opt =
                m_shr_p_bw_engine->build_optical_depth_functor();
        }
#endif

        const bool loc_do_field_ionization = do_field_ionization;
        const int loc_ionization_initial_level = ionization_initial_level;

        // Loop over all new particles and inject them (creates too many
        // particles, in particular does not consider xmin, xmax etc.).
        // The invalid ones are given negative ID and are deleted during the
        // next redistribute.
        auto *const poffset = offset.data();
#ifdef WARPX_DIM_RZ
        const bool rz_random_theta = m_rz_random_theta;
#endif
        amrex::ParallelForRNG(overlap_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const IntVect iv = IntVect(AMREX_D_DECL(i, j, k));
            const auto index = overlap_box.index(iv);
#ifdef WARPX_DIM_RZ
            Real theta_offset = 0._rt;
            if (rz_random_theta) { theta_offset = amrex::Random(engine) * 2._rt * MathConst::pi; }
#endif

            Real scale_fac = 0.0_rt;
            if( pcounts[index] != 0) {
#if defined(WARPX_DIM_3D)
                scale_fac = dx[0]*dx[1]*dx[2]/pcounts[index];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                scale_fac = dx[0]*dx[1]/pcounts[index];
#elif defined(WARPX_DIM_1D_Z)
                scale_fac = dx[0]/pcounts[index];
#endif
            }

            for (int i_part = 0; i_part < pcounts[index]; ++i_part)
            {
                long ip = poffset[index] + i_part;
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid+ip, cpuid);
                const XDim3 r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) ?
                  // In the refined injection region: use refinement ratio `lrrfac`
                  inj_pos->getPositionUnitBox(i_part, lrrfac, engine) :
                  // Otherwise: use 1 as the refinement ratio
                  inj_pos->getPositionUnitBox(i_part, amrex::IntVect::TheUnitVector(), engine);
                auto pos = getCellCoords(overlap_corner, dx, r, iv);

#if defined(WARPX_DIM_3D)
                if (!tile_realbox.contains(XDim3{pos.x,pos.y,pos.z})) {
                    ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                    continue;
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ignore_unused(k);
                if (!tile_realbox.contains(XDim3{pos.x,pos.z,0.0_rt})) {
                    ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                    continue;
                }
#else
                amrex::ignore_unused(j,k);
                if (!tile_realbox.contains(XDim3{pos.z,0.0_rt,0.0_rt})) {
                    ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                    continue;
                }
#endif

                // Save the x and y values to use in the insideBounds checks.
                // This is needed with WARPX_DIM_RZ since x and y are modified.
                const Real xb = pos.x;
                const Real yb = pos.y;

#ifdef WARPX_DIM_RZ
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and density
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const Real theta = (nmodes == 1 && rz_random_theta)?
                    (2._rt*MathConst::pi*amrex::Random(engine)):
                    (2._rt*MathConst::pi*r.y + theta_offset);
                pos.x = xb*std::cos(theta);
                pos.y = xb*std::sin(theta);
#endif

                Real dens;
                XDim3 u;
                if (gamma_boost == 1._rt) {
                    // Lab-frame simulation
                    // If the particle is not within the species's
                    // xmin, xmax, ymin, ymax, zmin, zmax, go to
                    // the next generated particle.

                    // include ballistic correction for plasma species with bulk motion
                    const Real z0 = applyBallisticCorrection(pos, inj_mom, gamma_boost,
                                                             beta_boost, t);
                    if (!inj_pos->insideBounds(xb, yb, z0)) {
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                        continue;
                    }

                    u = inj_mom->getMomentum(pos.x, pos.y, z0, engine);
                    dens = inj_rho->getDensity(pos.x, pos.y, z0);

                    // Remove particle if density below threshold
                    if ( dens < density_min ){
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                        continue;
                    }
                    // Cut density if above threshold
                    dens = amrex::min(dens, density_max);
                } else {
                    // Boosted-frame simulation
                    const Real z0_lab = applyBallisticCorrection(pos, inj_mom, gamma_boost,
                                                                 beta_boost, t);

                    // If the particle is not within the lab-frame zmin, zmax, etc.
                    // go to the next generated particle.
                    if (!inj_pos->insideBounds(xb, yb, z0_lab)) {
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                        continue;
                    }
                    // call `getDensity` with lab-frame parameters
                    dens = inj_rho->getDensity(pos.x, pos.y, z0_lab);
                    // Remove particle if density below threshold
                    if ( dens < density_min ){
                        ZeroInitializeAndSetNegativeID(pa_idcpu, pa, ip, loc_do_field_ionization, pi
#ifdef WARPX_QED
                                                   ,loc_has_quantum_sync, p_optical_depth_QSR
                                                   ,loc_has_breit_wheeler, p_optical_depth_BW
#endif
                                                   );
                        continue;
                    }
                    // Cut density if above threshold
                    dens = amrex::min(dens, density_max);

                    // get the full momentum, including thermal motion
                    u = inj_mom->getMomentum(pos.x, pos.y, 0._rt, engine);
                    const Real gamma_lab = std::sqrt( 1._rt+(u.x*u.x+u.y*u.y+u.z*u.z) );
                    const Real betaz_lab = u.z/(gamma_lab);

                    // At this point u and dens are the lab-frame quantities
                    // => Perform Lorentz transform
                    dens = gamma_boost * dens * ( 1.0_rt - beta_boost*betaz_lab );
                    u.z = gamma_boost * ( u.z -beta_boost*gamma_lab );
                }

                if (loc_do_field_ionization) {
                    pi[ip] = loc_ionization_initial_level;
                }

#ifdef WARPX_QED
                if(loc_has_quantum_sync){
                    p_optical_depth_QSR[ip] = quantum_sync_get_opt(engine);
                }

                if(loc_has_breit_wheeler){
                    p_optical_depth_BW[ip] = breit_wheeler_get_opt(engine);
                }
#endif
                // Initialize user-defined integers with user-defined parser
                for (int ia = 0; ia < n_user_int_attribs; ++ia) {
                    pa_user_int_data[ia][ip] = static_cast<int>(user_int_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t));
                }
                // Initialize user-defined real attributes with user-defined parser
                for (int ia = 0; ia < n_user_real_attribs; ++ia) {
                    pa_user_real_data[ia][ip] = user_real_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t);
                }

                u.x *= PhysConst::c;
                u.y *= PhysConst::c;
                u.z *= PhysConst::c;

                Real weight = dens;
                weight *= scale_fac;

#ifdef WARPX_DIM_RZ
                if (radially_weighted) {
                    weight *= 2._rt*MathConst::pi*xb;
                } else {
                    // This is not correct since it might shift the particle
                    // out of the local grid
                    pos.x = std::sqrt(xb*rmax);
                    weight *= dx[0];
                }
#endif
                pa[PIdx::w ][ip] = weight;
                pa[PIdx::ux][ip] = u.x;
                pa[PIdx::uy][ip] = u.y;
                pa[PIdx::uz][ip] = u.z;

#if defined(WARPX_DIM_3D)
                pa[PIdx::x][ip] = pos.x;
                pa[PIdx::y][ip] = pos.y;
                pa[PIdx::z][ip] = pos.z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
#ifdef WARPX_DIM_RZ
                pa[PIdx::theta][ip] = theta;
#endif
                pa[PIdx::x][ip] = xb;
                pa[PIdx::z][ip] = pos.z;
#else
                pa[PIdx::z][ip] = pos.z;
#endif
            }
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Remove particles that are inside the embedded boundaries
#ifdef AMREX_USE_EB
    auto & distance_to_eb = WarpX::GetInstance().GetDistanceToEB();
    scrapeParticlesAtEB( *this, amrex::GetVecOfConstPtrs(distance_to_eb), ParticleBoundaryProcess::Absorb());
#endif

    // The function that calls this is responsible for redistributing particles.
}

void
PhysicalParticleContainer::AddPlasmaFlux (PlasmaInjector const& plasma_injector, amrex::Real dt)
{
    WARPX_PROFILE("PhysicalParticleContainer::AddPlasmaFlux()");

    const Geometry& geom = Geom(0);
    const amrex::RealBox& part_realbox = geom.ProbDomain();

    const amrex::Real num_ppc_real = plasma_injector.num_particles_per_cell_real;
#ifdef WARPX_DIM_RZ
    const Real rmax = std::min(plasma_injector.xmax, geom.ProbDomain().hi(0));
#endif

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

    Real scale_fac = 0._rt;
    // Scale particle weight by the area of the emitting surface, within one cell
#if defined(WARPX_DIM_3D)
    scale_fac = dx[0]*dx[1]*dx[2]/dx[plasma_injector.flux_normal_axis]/num_ppc_real;
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    scale_fac = dx[0]*dx[1]/num_ppc_real;
    // When emission is in the r direction, the emitting surface is a cylinder.
    // The factor 2*pi*r is added later below.
    if (plasma_injector.flux_normal_axis == 0) { scale_fac /= dx[0]; }
    // When emission is in the z direction, the emitting surface is an annulus
    // The factor 2*pi*r is added later below.
    if (plasma_injector.flux_normal_axis == 2) { scale_fac /= dx[1]; }
    // When emission is in the theta direction (flux_normal_axis == 1),
    // the emitting surface is a rectangle, within the plane of the simulation
#elif defined(WARPX_DIM_1D_Z)
    scale_fac = dx[0]/num_ppc_real;
    if (plasma_injector.flux_normal_axis == 2) { scale_fac /= dx[0]; }
#endif

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(0);

    // Create temporary particle container to which particles will be added;
    // we will then call Redistribute on this new container and finally
    // add the new particles to the original container.
    PhysicalParticleContainer tmp_pc(&WarpX::GetInstance());
    for (int ic = 0; ic < NumRuntimeRealComps(); ++ic) { tmp_pc.AddRealComp(false); }
    for (int ic = 0; ic < NumRuntimeIntComps(); ++ic) { tmp_pc.AddIntComp(false); }
    tmp_pc.defineAllParticleTiles();

    const int nlevs = numLevels();
    static bool refine_injection = false;
    static Box fine_injection_box;
    static amrex::IntVect rrfac(AMREX_D_DECL(1,1,1));
    // This does not work if the mesh is dynamic.  But in that case, we should
    // not use refined injected either.  We also assume there is only one fine level.
    if (WarpX::refine_plasma && nlevs == 2)
    {
        refine_injection = true;
        fine_injection_box = ParticleBoxArray(1).minimalBox();
        rrfac = m_gdb->refRatio(0);
        fine_injection_box.coarsen(rrfac);
    }

    InjectorPosition* flux_pos = plasma_injector.getInjectorFluxPosition();
    InjectorFlux*  inj_flux = plasma_injector.getInjectorFlux();
    InjectorMomentum* inj_mom = plasma_injector.getInjectorMomentumDevice();
    constexpr int level_zero = 0;
    const amrex::Real t = WarpX::GetInstance().gett_new(level_zero);

#ifdef WARPX_DIM_RZ
    const int nmodes = WarpX::n_rz_azimuthal_modes;
    const bool rz_random_theta = m_rz_random_theta;
    const bool radially_weighted = plasma_injector.radially_weighted;
#endif

    MFItInfo info;
    if (do_tiling && Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi = MakeMFIter(0, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        const Box& tile_box = mfi.tilebox();
        const RealBox tile_realbox = WarpX::getRealBox(tile_box, 0);

        // Find the cells of part_realbox that overlap with tile_realbox
        // If there is no overlap, just go to the next tile in the loop
        RealBox overlap_realbox;
        Box overlap_box;
        IntVect shifted;
        bool no_overlap = false;

        for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
#if (defined(WARPX_DIM_3D))
            if (dir == plasma_injector.flux_normal_axis) {
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            if (2*dir == plasma_injector.flux_normal_axis) {
            // The above formula captures the following cases:
            // - flux_normal_axis=0 (emission along x/r) and dir=0
            // - flux_normal_axis=2 (emission along z) and dir=1
#elif defined(WARPX_DIM_1D_Z)
            if ( (dir==0) && (plasma_injector.flux_normal_axis==2) ) {
#endif
                if (plasma_injector.flux_direction > 0) {
                    if (plasma_injector.surface_flux_pos <  tile_realbox.lo(dir) ||
                        plasma_injector.surface_flux_pos >= tile_realbox.hi(dir)) {
                            no_overlap = true;
                            break;
                    }
                } else {
                    if (plasma_injector.surface_flux_pos <= tile_realbox.lo(dir) ||
                        plasma_injector.surface_flux_pos >  tile_realbox.hi(dir)) {
                            no_overlap = true;
                            break;
                    }
                }
                overlap_realbox.setLo( dir, plasma_injector.surface_flux_pos );
                overlap_realbox.setHi( dir, plasma_injector.surface_flux_pos );
                overlap_box.setSmall( dir, 0 );
                overlap_box.setBig( dir, 0 );
                shifted[dir] =
                    static_cast<int>(std::round((overlap_realbox.lo(dir)-problo[dir])/dx[dir]));
            } else {
                if ( tile_realbox.lo(dir) <= part_realbox.hi(dir) ) {
                    const Real ncells_adjust = std::floor( (tile_realbox.lo(dir) - part_realbox.lo(dir))/dx[dir] );
                    overlap_realbox.setLo( dir, part_realbox.lo(dir) + std::max(ncells_adjust, 0._rt) * dx[dir]);
                } else {
                    no_overlap = true; break;
                }
                if ( tile_realbox.hi(dir) >= part_realbox.lo(dir) ) {
                    const Real ncells_adjust = std::floor( (part_realbox.hi(dir) - tile_realbox.hi(dir))/dx[dir] );
                    overlap_realbox.setHi( dir, part_realbox.hi(dir) - std::max(ncells_adjust, 0._rt) * dx[dir]);
                } else {
                    no_overlap = true; break;
                }
                // Count the number of cells in this direction in overlap_realbox
                overlap_box.setSmall( dir, 0 );
                overlap_box.setBig( dir,
                    int( std::round((overlap_realbox.hi(dir)-overlap_realbox.lo(dir))
                                    /dx[dir] )) - 1);
                shifted[dir] =
                    static_cast<int>(std::round((overlap_realbox.lo(dir)-problo[dir])/dx[dir]));
                // shifted is exact in non-moving-window direction.  That's all we care.
            }
        }
        if (no_overlap == 1) {
            continue; // Go to the next tile
        }

        const int grid_id = mfi.index();
        const int tile_id = mfi.LocalTileIndex();

        const GpuArray<Real,AMREX_SPACEDIM> overlap_corner
            {AMREX_D_DECL(overlap_realbox.lo(0),
                          overlap_realbox.lo(1),
                          overlap_realbox.lo(2))};

        // count the number of particles that each cell in overlap_box could add
        Gpu::DeviceVector<int> counts(overlap_box.numPts(), 0);
        Gpu::DeviceVector<int> offset(overlap_box.numPts());
        auto *pcounts = counts.data();
        const amrex::IntVect lrrfac = rrfac;
        Box fine_overlap_box; // default Box is NOT ok().
        if (refine_injection) {
            fine_overlap_box = overlap_box & amrex::shift(fine_injection_box, -shifted);
        }
        amrex::ParallelForRNG(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const IntVect iv(AMREX_D_DECL(i, j, k));
            auto lo = getCellCoords(overlap_corner, dx, {0._rt, 0._rt, 0._rt}, iv);
            auto hi = getCellCoords(overlap_corner, dx, {1._rt, 1._rt, 1._rt}, iv);

            const int num_ppc_int = static_cast<int>(num_ppc_real + amrex::Random(engine));

            if (flux_pos->overlapsWith(lo, hi))
            {
                auto index = overlap_box.index(iv);
                int r;
                if (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) {
                    r = AMREX_D_TERM(lrrfac[0],*lrrfac[1],*lrrfac[2]);
                } else {
                    r = 1;
                }
                pcounts[index] = num_ppc_int*r;
            }
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::ignore_unused(k);
#elif defined(WARPX_DIM_1D_Z)
            amrex::ignore_unused(j,k);
#endif
        });

        // Max number of new particles. All of them are created,
        // and invalid ones are then discarded
        const amrex::Long max_new_particles = Scan::ExclusiveSum(counts.size(), counts.data(), offset.data());

        // Update NextID to include particles created in this function
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (add_plasma_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid+max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < LongParticleIds::LastParticleID,
            "overflow on particle id numbers");

        const int cpuid = ParallelDescriptor::MyProc();

        auto& particle_tile = tmp_pc.DefineAndReturnParticleTile(0, grid_id, tile_id);

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto& soa = particle_tile.GetStructOfArrays();
        GpuArray<ParticleReal*,PIdx::nattribs> pa;
        for (int ia = 0; ia < PIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        uint64_t * AMREX_RESTRICT pa_idcpu = soa.GetIdCPUData().data() + old_size;

        // user-defined integer and real attributes
        const auto n_user_int_attribs = static_cast<int>(m_user_int_attribs.size());
        const auto n_user_real_attribs = static_cast<int>(m_user_real_attribs.size());
        amrex::Gpu::PinnedVector<int*> pa_user_int_pinned(n_user_int_attribs);
        amrex::Gpu::PinnedVector<ParticleReal*> pa_user_real_pinned(n_user_real_attribs);
        amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > user_int_attrib_parserexec_pinned(n_user_int_attribs);
        amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > user_real_attrib_parserexec_pinned(n_user_real_attribs);
        for (int ia = 0; ia < n_user_int_attribs; ++ia) {
            pa_user_int_pinned[ia] = soa.GetIntData(particle_icomps[m_user_int_attribs[ia]]).data() + old_size;
            user_int_attrib_parserexec_pinned[ia] = m_user_int_attrib_parser[ia]->compile<7>();
        }
        for (int ia = 0; ia < n_user_real_attribs; ++ia) {
            pa_user_real_pinned[ia] = soa.GetRealData(particle_comps[m_user_real_attribs[ia]]).data() + old_size;
            user_real_attrib_parserexec_pinned[ia] = m_user_real_attrib_parser[ia]->compile<7>();
        }
#ifdef AMREX_USE_GPU
        // To avoid using managed memory, we first define pinned memory vector, initialize on cpu,
        // and them memcpy to device from host
        amrex::Gpu::DeviceVector<int*> d_pa_user_int(n_user_int_attribs);
        amrex::Gpu::DeviceVector<ParticleReal*> d_pa_user_real(n_user_real_attribs);
        amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > d_user_int_attrib_parserexec(n_user_int_attribs);
        amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > d_user_real_attrib_parserexec(n_user_real_attribs);
        amrex::Gpu::copyAsync(Gpu::hostToDevice, pa_user_int_pinned.begin(),
                              pa_user_int_pinned.end(), d_pa_user_int.begin());
        amrex::Gpu::copyAsync(Gpu::hostToDevice, pa_user_real_pinned.begin(),
                              pa_user_real_pinned.end(), d_pa_user_real.begin());
        amrex::Gpu::copyAsync(Gpu::hostToDevice, user_int_attrib_parserexec_pinned.begin(),
                              user_int_attrib_parserexec_pinned.end(), d_user_int_attrib_parserexec.begin());
        amrex::Gpu::copyAsync(Gpu::hostToDevice, user_real_attrib_parserexec_pinned.begin(),
                              user_real_attrib_parserexec_pinned.end(), d_user_real_attrib_parserexec.begin());
        int** pa_user_int_data = d_pa_user_int.dataPtr();
        ParticleReal** pa_user_real_data = d_pa_user_real.dataPtr();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = d_user_int_attrib_parserexec.dataPtr();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = d_user_real_attrib_parserexec.dataPtr();
#else
        int** pa_user_int_data = pa_user_int_pinned.dataPtr();
        ParticleReal** pa_user_real_data = pa_user_real_pinned.dataPtr();
        amrex::ParserExecutor<7> const* user_int_parserexec_data = user_int_attrib_parserexec_pinned.dataPtr();
        amrex::ParserExecutor<7> const* user_real_parserexec_data = user_real_attrib_parserexec_pinned.dataPtr();
#endif

        int* p_ion_level = nullptr;
        if (do_field_ionization) {
            p_ion_level = soa.GetIntData(particle_icomps["ionizationLevel"]).data() + old_size;
        }

#ifdef WARPX_QED
        //Pointer to the optical depth component
        amrex::ParticleReal* p_optical_depth_QSR = nullptr;
        amrex::ParticleReal* p_optical_depth_BW  = nullptr;

        // If a QED effect is enabled, the corresponding optical depth
        // has to be initialized
        const bool loc_has_quantum_sync = has_quantum_sync();
        const bool loc_has_breit_wheeler = has_breit_wheeler();
        if (loc_has_quantum_sync) {
            p_optical_depth_QSR = soa.GetRealData(
                particle_comps["opticalDepthQSR"]).data() + old_size;
        }
        if(loc_has_breit_wheeler) {
            p_optical_depth_BW = soa.GetRealData(
                particle_comps["opticalDepthBW"]).data() + old_size;
        }

        //If needed, get the appropriate functors from the engines
        QuantumSynchrotronGetOpticalDepth quantum_sync_get_opt;
        BreitWheelerGetOpticalDepth breit_wheeler_get_opt;
        if(loc_has_quantum_sync){
            quantum_sync_get_opt =
                m_shr_p_qs_engine->build_optical_depth_functor();
        }
        if(loc_has_breit_wheeler){
            breit_wheeler_get_opt =
                m_shr_p_bw_engine->build_optical_depth_functor();
        }
#endif

        const bool loc_do_field_ionization = do_field_ionization;
        const int loc_ionization_initial_level = ionization_initial_level;
#ifdef WARPX_DIM_RZ
        int const loc_flux_normal_axis = plasma_injector.flux_normal_axis;
#endif

        // Loop over all new particles and inject them (creates too many
        // particles, in particular does not consider xmin, xmax etc.).
        // The invalid ones are given negative ID and are deleted during the
        // next redistribute.
        auto *const poffset = offset.data();
        amrex::ParallelForRNG(overlap_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            const IntVect iv = IntVect(AMREX_D_DECL(i, j, k));
            const auto index = overlap_box.index(iv);
            for (int i_part = 0; i_part < pcounts[index]; ++i_part)
            {
                const long ip = poffset[index] + i_part;
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid+ip, cpuid);

                // This assumes the flux_pos is of type InjectorPositionRandomPlane
                const XDim3 r = (fine_overlap_box.ok() && fine_overlap_box.contains(iv)) ?
                  // In the refined injection region: use refinement ratio `lrrfac`
                  flux_pos->getPositionUnitBox(i_part, lrrfac, engine) :
                  // Otherwise: use 1 as the refinement ratio
                  flux_pos->getPositionUnitBox(i_part, amrex::IntVect::TheUnitVector(), engine);
                auto pos = getCellCoords(overlap_corner, dx, r, iv);
                auto ppos = PDim3(pos);

                // inj_mom would typically be InjectorMomentumGaussianFlux
                XDim3 u;
                u = inj_mom->getMomentum(pos.x, pos.y, pos.z, engine);
                auto pu = PDim3(u);

                pu.x *= PhysConst::c;
                pu.y *= PhysConst::c;
                pu.z *= PhysConst::c;

                // The containsInclusive is used to allow the case of the flux surface
                // being on the boundary of the domain. After the UpdatePosition below,
                // the particles will be within the domain.
#if defined(WARPX_DIM_3D)
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,ppos.y,ppos.z})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ignore_unused(k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.x,ppos.z,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#else
                amrex::ignore_unused(j,k);
                if (!ParticleUtils::containsInclusive(tile_realbox, XDim3{ppos.z,0.0_prt,0.0_prt})) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }
#endif
                // Lab-frame simulation
                // If the particle's initial position is not within or on the species's
                // xmin, xmax, ymin, ymax, zmin, zmax, go to the next generated particle.
                if (!flux_pos->insideBoundsInclusive(ppos.x, ppos.y, ppos.z)) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }

#ifdef WARPX_DIM_RZ
                // Conversion from cylindrical to Cartesian coordinates
                // Replace the x and y, setting an angle theta.
                // These x and y are used to get the momentum and flux
                // With only 1 mode, the angle doesn't matter so
                // choose it randomly.
                const Real theta = (nmodes == 1 && rz_random_theta)?
                    (2._prt*MathConst::pi*amrex::Random(engine)):
                    (2._prt*MathConst::pi*r.y);
                Real const cos_theta = std::cos(theta);
                Real const sin_theta = std::sin(theta);
                // Rotate the position
                const amrex::Real radial_position = ppos.x;
                ppos.x = radial_position*cos_theta;
                ppos.y = radial_position*sin_theta;
                if (loc_flux_normal_axis != 2) {
                    // Rotate the momentum
                    // This because, when the flux direction is e.g. "r"
                    // the `inj_mom` objects generates a v*Gaussian distribution
                    // along the Cartesian "x" direction by default. This
                    // needs to be rotated along "r".
                    const Real ur = pu.x;
                    const Real ut = pu.y;
                    pu.x = cos_theta*ur - sin_theta*ut;
                    pu.y = sin_theta*ur + cos_theta*ut;
                }
#endif
                const Real flux = inj_flux->getFlux(ppos.x, ppos.y, ppos.z, t);
                // Remove particle if flux is negative or 0
                if (flux <= 0) {
                    pa_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                    continue;
                }

                if (loc_do_field_ionization) {
                    p_ion_level[ip] = loc_ionization_initial_level;
                }

#ifdef WARPX_QED
                if (loc_has_quantum_sync) {
                    p_optical_depth_QSR[ip] = quantum_sync_get_opt(engine);
                }

                if(loc_has_breit_wheeler){
                    p_optical_depth_BW[ip] = breit_wheeler_get_opt(engine);
                }
#endif
                // Initialize user-defined integers with user-defined parser
                for (int ia = 0; ia < n_user_int_attribs; ++ia) {
                    pa_user_int_data[ia][ip] = static_cast<int>(user_int_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t));
                }
                // Initialize user-defined real attributes with user-defined parser
                for (int ia = 0; ia < n_user_real_attribs; ++ia) {
                    pa_user_real_data[ia][ip] = user_real_parserexec_data[ia](pos.x, pos.y, pos.z, u.x, u.y, u.z, t);
                }

#ifdef WARPX_DIM_RZ
                // The particle weight is proportional to the user-specified
                // flux and the emission surface within
                // one cell (captured partially by `scale_fac`).
                // For cylindrical emission (flux_normal_axis==0
                // or flux_normal_axis==2), the emission surface depends on
                // the radius ; thus, the calculation is finalized here
                Real t_weight = flux * scale_fac * dt;
                if (loc_flux_normal_axis != 1) {
                    if (radially_weighted) {
                         t_weight *= 2._rt*MathConst::pi*radial_position;
                    } else {
                         // This is not correct since it might shift the particle
                         // out of the local grid
                         ppos.x = std::sqrt(radial_position*rmax);
                         t_weight *= dx[0];
                    }
                }
                const Real weight = t_weight;
#else
                const Real weight = flux * scale_fac * dt;
#endif
                pa[PIdx::w ][ip] = weight;
                pa[PIdx::ux][ip] = pu.x;
                pa[PIdx::uy][ip] = pu.y;
                pa[PIdx::uz][ip] = pu.z;

                // Update particle position by a random `t_fract`
                // so as to produce a continuous-looking flow of particles
                const amrex::Real t_fract = amrex::Random(engine)*dt;
                UpdatePosition(ppos.x, ppos.y, ppos.z, pu.x, pu.y, pu.z, t_fract);

#if defined(WARPX_DIM_3D)
                pa[PIdx::x][ip] = ppos.x;
                pa[PIdx::y][ip] = ppos.y;
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_RZ)
                pa[PIdx::theta][ip] = std::atan2(ppos.y, ppos.x);
                pa[PIdx::x][ip] = std::sqrt(ppos.x*ppos.x + ppos.y*ppos.y);
                pa[PIdx::z][ip] = ppos.z;
#elif defined(WARPX_DIM_XZ)
                pa[PIdx::x][ip] = ppos.x;
                pa[PIdx::z][ip] = ppos.z;
#else
                pa[PIdx::z][ip] = ppos.z;
#endif
            }
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Remove particles that are inside the embedded boundaries
#ifdef AMREX_USE_EB
    auto & distance_to_eb = WarpX::GetInstance().GetDistanceToEB();
    scrapeParticlesAtEB(tmp_pc, amrex::GetVecOfConstPtrs(distance_to_eb), ParticleBoundaryProcess::Absorb());
#endif

    // Redistribute the new particles that were added to the temporary container.
    // (This eliminates invalid particles, and makes sure that particles
    // are in the right tile.)
    tmp_pc.Redistribute();

    // Add the particles to the current container, tile by tile
    for (int lev=0; lev<numLevels(); lev++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
        {
            // Extract tiles
            const int grid_id = mfi.index();
            const int tile_id = mfi.LocalTileIndex();
            auto& src_tile = tmp_pc.DefineAndReturnParticleTile(lev, grid_id, tile_id);
            auto& dst_tile = DefineAndReturnParticleTile(lev, grid_id, tile_id);

            // Resize container and copy particles
            auto old_size = dst_tile.numParticles();
            auto n_new = src_tile.numParticles();
            dst_tile.resize( old_size+n_new );
            amrex::copyParticles(dst_tile, src_tile, 0, old_size, n_new);
        }
    }
}

void
PhysicalParticleContainer::my_BuildRedistributeMask_and_ModifyRemoteSendAllcomps(int lev, int nghost)
{
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = WarpX::GetInstance().warpx_comm_particle_container.at(sname).warpx_comm_comp;

    // For global communication, allocate thread-safe storage for every rank
    if (nghost <= 0)
    {
        std::map<int, std::vector< std::vector<char> > >& remote_send_allcomps = warpx_comm_comp.remote_send_allcomps;
        int max_threads = omp_get_max_threads();
        for (int i = 0; i < ParallelContext::NProcsSub(); ++i) {
            remote_send_allcomps[i].resize(max_threads);
        }
        return;
    }

    std::unique_ptr<iMultiFab>& redistribute_mask_ptr = warpx_comm_comp.redistribute_mask_ptr;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];
    int& redistribute_mask_nghost = warpx_comm_comp.redistribute_mask_nghost;
    if (redistribute_mask_ptr == nullptr ||
        redistribute_mask_nghost < nghost ||
        ! BoxArray::SameRefs(redistribute_mask_ptr->boxArray(), this->ParticleBoxArray(lev)) ||
        ! DistributionMapping::SameRefs(redistribute_mask_ptr->DistributionMap(), this->ParticleDistributionMap(lev)))
    {
        const Geometry& geom = this->Geom(lev);
        const BoxArray& ba = this->ParticleBoxArray(lev);
        const DistributionMapping& dmap = this->ParticleDistributionMap(lev);

        redistribute_mask_nghost = nghost;
        redistribute_mask_ptr = std::make_unique<iMultiFab>(ba, dmap, 2, nghost);
        redistribute_mask_ptr->setVal(-1, nghost);

        const auto tile_size_do = amrex::ParticleContainerBase::do_tiling ? amrex::ParticleContainerBase::tile_size : IntVect::TheZeroVector();

        #pragma omp parallel
        for (MFIter mfi(*redistribute_mask_ptr, tile_size_do); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.tilebox();
            const int grid_id = mfi.index();
            const int tile_id = mfi.LocalTileIndex();
            (*redistribute_mask_ptr)[mfi].template setVal<RunOn::Host>(grid_id, box, 0, 1);
            (*redistribute_mask_ptr)[mfi].template setVal<RunOn::Host>(tile_id, box, 1, 1);
        }

        redistribute_mask_ptr->FillBoundary(geom.periodicity());

        neighbor_procs.clear();
        for (MFIter mfi(*redistribute_mask_ptr, tile_size_do); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.growntilebox();
            for (IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv))
            {
                const int grid = (*redistribute_mask_ptr)[mfi](iv, 0);
                if (grid >= 0)
                {
                    const int global_rank = this->ParticleDistributionMap(lev)[grid];
                    const int rank = ParallelContext::global_to_local_rank(global_rank);
                    if (rank != ParallelContext::MyProcSub()) {
                        neighbor_procs.push_back(rank);
                    }
                }
            }
        }
        RemoveDuplicates(neighbor_procs);
    }

    std::map<int, std::vector< std::vector<char> > >& remote_send_allcomps = warpx_comm_comp.remote_send_allcomps;
    int max_threads = omp_get_max_threads();
    for (int i = 0; i < neighbor_procs.size(); ++i) {
        if (remote_send_allcomps[neighbor_procs[i]].size() != max_threads)       // If the neighbor set changed, resize each [who] thread slot again
        {
            remote_send_allcomps[neighbor_procs[i]].clear();
            remote_send_allcomps[neighbor_procs[i]].resize(max_threads);
        }
        else        // Otherwise just clear each [who, thread] payload
        {
            for (int j = 0; j < max_threads; ++j) {
                remote_send_allcomps[neighbor_procs[i]][j].clear();
            }
        }
    }
}

void
PhysicalParticleContainer::UNR_WarpX_buffer_reg()
{
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];
    int& num_tiles = warpx_comm_particle_container.num_tiles;
    int& m_init_np = WarpX::GetInstance().m_init_np;
    const int& num_event = warpx_comm_particle_container.num_event;

    int max_threads = omp_get_max_threads();
    int neigubor_proc_num = static_cast<int>(neighbor_procs.size());

    WarpX::UNR_WarpX_buffer& unr_send_buffer = warpx_comm_particle_container.unr_send_buffer;
    WarpX::UNR_WarpX_buffer& unr_recv_buffer = warpx_comm_particle_container.unr_recv_buffer;

    // Register send memory for counts and packed data
    void* & unr_send_count_buffer = unr_send_buffer.count_buffer;
    size_t& unr_send_count_buffer_mem_size = unr_send_buffer.count_buffer_mem_size;
    size_t& unr_send_count_buffer_blk_size = unr_send_buffer.count_buffer_blk_size;
    unr_mem_h& unr_send_count_buffer_mem_h = unr_send_buffer.count_buffer_mem_h;

    void* & unr_send_data_buffer = unr_send_buffer.data_buffer;
    size_t& unr_send_data_buffer_mem_size = unr_send_buffer.data_buffer_mem_size;
    size_t& unr_send_data_buffer_blk_size = unr_send_buffer.data_buffer_blk_size;
    unr_mem_h& unr_send_data_buffer_mem_h = unr_send_buffer.data_buffer_mem_h;
    
    unr_send_count_buffer_mem_size = neigubor_proc_num * max_threads * sizeof(long);
    unr_send_count_buffer_blk_size = sizeof(long);

    unr_send_data_buffer_mem_size = 2 * num_tiles * m_init_np * sizeof(amrex::ParticleReal);
    unr_send_data_buffer_blk_size = unr_send_data_buffer_mem_size / neigubor_proc_num / max_threads;

    unr_mem_alloc_reg(&unr_send_count_buffer, unr_send_count_buffer_mem_size, 0, &unr_send_count_buffer_mem_h);
    unr_mem_alloc_reg(&unr_send_data_buffer, unr_send_data_buffer_mem_size, 0, &unr_send_data_buffer_mem_h);

    // Register receive memory for counts and packed data
    void* & unr_recv_count_buffer = unr_recv_buffer.count_buffer;
    size_t& unr_recv_count_buffer_mem_size = unr_recv_buffer.count_buffer_mem_size;
    size_t& unr_recv_count_buffer_blk_size = unr_recv_buffer.count_buffer_blk_size;
    unr_mem_h& unr_recv_count_buffer_mem_h = unr_recv_buffer.count_buffer_mem_h;

    void* & unr_recv_data_buffer = unr_recv_buffer.data_buffer;
    size_t& unr_recv_data_buffer_mem_size = unr_recv_buffer.data_buffer_mem_size;
    size_t& unr_recv_data_buffer_blk_size = unr_recv_buffer.data_buffer_blk_size;
    unr_mem_h& unr_recv_data_buffer_mem_h = unr_recv_buffer.data_buffer_mem_h;

    unr_recv_count_buffer_mem_size = neigubor_proc_num * max_threads * sizeof(long);
    unr_recv_count_buffer_blk_size = sizeof(long);
    
    int m_unr_mem_factor = WarpX::GetInstance().m_unr_mem_factor;
    unr_recv_data_buffer_mem_size = m_unr_mem_factor * num_tiles * m_init_np * sizeof(amrex::ParticleReal);
    unr_recv_data_buffer_blk_size = unr_recv_data_buffer_mem_size / neigubor_proc_num / max_threads;

    unr_mem_alloc_reg(&unr_recv_count_buffer, unr_recv_count_buffer_mem_size, 0, &unr_recv_count_buffer_mem_h);
    unr_mem_alloc_reg(&unr_recv_data_buffer, unr_recv_data_buffer_mem_size, 0, &unr_recv_data_buffer_mem_h);

    // Synchronize memory registration metadata
    unr_mem_reg_sync();

    // Register send/receive blocks for counts and packed data
    std::vector<std::vector<unr_blk_h>>& unr_send_count_buffer_blk_h = unr_send_buffer.count_buffer_blk_h;
    std::vector<std::vector<unr_sig_h>>& unr_send_count_buffer_sig_h = unr_send_buffer.count_buffer_sig_h;
    std::vector<std::vector<unr_blk_h>>& unr_send_data_buffer_blk_h = unr_send_buffer.data_buffer_blk_h;
    std::vector<std::vector<unr_sig_h>>& unr_send_data_buffer_sig_h = unr_send_buffer.data_buffer_sig_h;

    unr_send_count_buffer_blk_h.resize(neigubor_proc_num);
    unr_send_count_buffer_sig_h.resize(neigubor_proc_num);
    unr_send_data_buffer_blk_h.resize(neigubor_proc_num);
    unr_send_data_buffer_sig_h.resize(neigubor_proc_num);

    std::vector<std::vector<unr_blk_h>>& unr_recv_count_buffer_blk_h = unr_recv_buffer.count_buffer_blk_h;
    std::vector<std::vector<unr_sig_h>>& unr_recv_count_buffer_sig_h = unr_recv_buffer.count_buffer_sig_h;
    std::vector<std::vector<unr_blk_h>>& unr_recv_data_buffer_blk_h = unr_recv_buffer.data_buffer_blk_h;
    std::vector<std::vector<unr_sig_h>>& unr_recv_data_buffer_sig_h = unr_recv_buffer.data_buffer_sig_h;

    unr_recv_count_buffer_blk_h.resize(neigubor_proc_num);
    unr_recv_count_buffer_sig_h.resize(neigubor_proc_num);
    unr_recv_data_buffer_blk_h.resize(neigubor_proc_num);
    unr_recv_data_buffer_sig_h.resize(neigubor_proc_num);
    
    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who_idx = i;        // who_to_idx[who]

        std::vector<unr_blk_h>& who_unr_send_count_blk_h = unr_send_count_buffer_blk_h[who_idx];
        std::vector<unr_sig_h>& who_unr_send_count_sig_h = unr_send_count_buffer_sig_h[who_idx];
        std::vector<unr_blk_h>& who_unr_send_data_blk_h = unr_send_data_buffer_blk_h[who_idx];
        std::vector<unr_sig_h>& who_unr_send_data_sig_h = unr_send_data_buffer_sig_h[who_idx];

        who_unr_send_count_blk_h.resize(max_threads);
        who_unr_send_count_sig_h.resize(max_threads);
        who_unr_send_data_blk_h.resize(max_threads);
        who_unr_send_data_sig_h.resize(max_threads);

        std::vector<unr_blk_h>& who_unr_recv_count_blk_h = unr_recv_count_buffer_blk_h[who_idx];
        std::vector<unr_sig_h>& who_unr_recv_count_sig_h = unr_recv_count_buffer_sig_h[who_idx];
        std::vector<unr_blk_h>& who_unr_recv_data_blk_h = unr_recv_data_buffer_blk_h[who_idx];
        std::vector<unr_sig_h>& who_unr_recv_data_sig_h = unr_recv_data_buffer_sig_h[who_idx];

        who_unr_recv_count_blk_h.resize(max_threads);
        who_unr_recv_count_sig_h.resize(max_threads);
        who_unr_recv_data_blk_h.resize(max_threads);
        who_unr_recv_data_sig_h.resize(max_threads);

        for (int j = 0; j < max_threads; ++j) {
            unr_sig_create(&who_unr_send_count_sig_h[j], num_event);
            unr_blk_reg(unr_send_count_buffer_mem_h, (who_idx * max_threads + j) * unr_send_count_buffer_blk_size, unr_send_count_buffer_blk_size, who_unr_send_count_sig_h[j], UNR_NO_SIGNAL, &who_unr_send_count_blk_h[j]);

            unr_sig_create(&who_unr_send_data_sig_h[j], num_event);
            unr_blk_reg(unr_send_data_buffer_mem_h, (who_idx * max_threads + j) * unr_send_data_buffer_blk_size, unr_send_data_buffer_blk_size, who_unr_send_data_sig_h[j], UNR_NO_SIGNAL, &who_unr_send_data_blk_h[j]);

            unr_sig_create(&who_unr_recv_count_sig_h[j], num_event);
            unr_blk_reg(unr_recv_count_buffer_mem_h, (who_idx * max_threads + j) * unr_recv_count_buffer_blk_size, unr_recv_count_buffer_blk_size, UNR_NO_SIGNAL, who_unr_recv_count_sig_h[j], &who_unr_recv_count_blk_h[j]);

            unr_sig_create(&who_unr_recv_data_sig_h[j], num_event);
            unr_blk_reg(unr_recv_data_buffer_mem_h, (who_idx * max_threads + j) * unr_recv_data_buffer_blk_size, unr_recv_data_buffer_blk_size, UNR_NO_SIGNAL, who_unr_recv_data_sig_h[j], &who_unr_recv_data_blk_h[j]);
        }
    }

}

void
PhysicalParticleContainer::UNR_WarpX_blk_sync()
{
    // Exchange remote block handles with every neighbor
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;
    WarpX::UNR_WarpX_buffer& unr_recv_buffer = warpx_comm_particle_container.unr_recv_buffer;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];
    map<int, int>& who_to_idx = WarpX::GetInstance().who_to_idx[sname];
    
    int max_threads = omp_get_max_threads();
    int neigubor_proc_num = static_cast<int>(neighbor_procs.size());

    std::vector<std::vector<unr_blk_h>>& unr_recv_count_buffer_blk_h = unr_recv_buffer.count_buffer_blk_h;
    std::vector<std::vector<unr_blk_h>>& unr_recv_data_buffer_blk_h = unr_recv_buffer.data_buffer_blk_h;


    std::vector<std::vector<unr_blk_h>> mpi_snd_blk(neigubor_proc_num);
    std::vector<std::vector<unr_blk_h>> mpi_rcv_blk(neigubor_proc_num);

    std::vector<MPI_Request> send_blk_reqs(neigubor_proc_num);
    std::vector<MPI_Status> send_blk_status(neigubor_proc_num);
    std::vector<MPI_Request> recv_blk_reqs(neigubor_proc_num);
    std::vector<MPI_Status> recv_blk_status(neigubor_proc_num);

    const int SeqNum = ParallelDescriptor::SeqNum();

    // Pack count/data receive blocks for all local threads of the same neighbor and send them asynchronously
    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who = neighbor_procs[i];
        int who_idx = i;
        
        std::vector<unr_blk_h>& who_mpi_snd_count_blk_h = mpi_snd_blk[who_idx];
        who_mpi_snd_count_blk_h.resize(max_threads * 2);
        for (int j = 0; j < max_threads; ++j) {
            who_mpi_snd_count_blk_h[j * 2] = unr_recv_count_buffer_blk_h[who_idx][j];
            who_mpi_snd_count_blk_h[j * 2 + 1] = unr_recv_data_buffer_blk_h[who_idx][j];
        }

        std::vector<unr_blk_h>& who_mpi_recv_count_blk_h = mpi_rcv_blk[who_idx];
        who_mpi_recv_count_blk_h.resize(max_threads * 2);

        MPI_Irecv(who_mpi_recv_count_blk_h.data(), max_threads * 2, MPI_UNR_BLK_H, who, SeqNum, ParallelDescriptor::Communicator(), &recv_blk_reqs[i]);
        MPI_Isend(who_mpi_snd_count_blk_h.data(), max_threads * 2, MPI_UNR_BLK_H, who, SeqNum, ParallelDescriptor::Communicator(), &send_blk_reqs[i]);
    }


    // Wait for both send and receive completion
    MPI_Waitall(neigubor_proc_num, recv_blk_reqs.data(), recv_blk_status.data());
    MPI_Waitall(neigubor_proc_num, send_blk_reqs.data(), send_blk_status.data());

    // Unpack the received block handles
    std::vector<std::vector<unr_blk_h>>& rmt_count_blk = warpx_comm_particle_container.unr_rmt_blk.rmt_count_blk;
    std::vector<std::vector<unr_blk_h>>& rmt_data_blk = warpx_comm_particle_container.unr_rmt_blk.rmt_data_blk;
    rmt_count_blk.resize(neigubor_proc_num);
    rmt_data_blk.resize(neigubor_proc_num);

    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who_idx = i;

        std::vector<unr_blk_h>& who_rmt_count_blk_h = rmt_count_blk[who_idx];
        std::vector<unr_blk_h>& who_rmt_data_blk_h = rmt_data_blk[who_idx];
        who_rmt_count_blk_h.resize(max_threads);
        who_rmt_data_blk_h.resize(max_threads);

        for (int j = 0; j < max_threads; ++j) {
            who_rmt_count_blk_h[j] = mpi_rcv_blk[who_idx][j * 2];
            who_rmt_data_blk_h[j] = mpi_rcv_blk[who_idx][j * 2 + 1];
        }
    }

}

void
PhysicalParticleContainer::UNR_BuildRedistributeMask_and_ClearSendBuffer(int lev, int nghost)
{
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;
    // For global communication, allocate thread-safe storage for every rank
    if (nghost <= 0)
    {
        std::map<int, std::vector< std::vector<char> > >& remote_send_allcomps = warpx_comm_comp.remote_send_allcomps;
        int max_threads = omp_get_max_threads();
        for (int i = 0; i < ParallelContext::NProcsSub(); ++i) {
            remote_send_allcomps[i].resize(max_threads);
        }
        return;
    }

    std::unique_ptr<iMultiFab>& redistribute_mask_ptr = warpx_comm_comp.redistribute_mask_ptr;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];
    int& redistribute_mask_nghost = warpx_comm_comp.redistribute_mask_nghost;
    if (redistribute_mask_ptr == nullptr ||
        redistribute_mask_nghost < nghost ||
        ! BoxArray::SameRefs(redistribute_mask_ptr->boxArray(), this->ParticleBoxArray(lev)) ||
        ! DistributionMapping::SameRefs(redistribute_mask_ptr->DistributionMap(), this->ParticleDistributionMap(lev)))
    {
        const Geometry& geom = this->Geom(lev);
        const BoxArray& ba = this->ParticleBoxArray(lev);
        const DistributionMapping& dmap = this->ParticleDistributionMap(lev);

        redistribute_mask_nghost = nghost;
        redistribute_mask_ptr = std::make_unique<iMultiFab>(ba, dmap, 2, nghost);
        redistribute_mask_ptr->setVal(-1, nghost);

        const auto tile_size_do = amrex::ParticleContainerBase::do_tiling ? amrex::ParticleContainerBase::tile_size : IntVect::TheZeroVector();

        #pragma omp parallel
        for (MFIter mfi(*redistribute_mask_ptr, tile_size_do); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.tilebox();
            const int grid_id = mfi.index();
            const int tile_id = mfi.LocalTileIndex();
            (*redistribute_mask_ptr)[mfi].template setVal<RunOn::Host>(grid_id, box, 0, 1);
            (*redistribute_mask_ptr)[mfi].template setVal<RunOn::Host>(tile_id, box, 1, 1);
        }

        redistribute_mask_ptr->FillBoundary(geom.periodicity());

        neighbor_procs.clear();
        int& num_tiles = warpx_comm_particle_container.num_tiles;
        num_tiles = 0;
        for (MFIter mfi(*redistribute_mask_ptr, tile_size_do); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.growntilebox();
            for (IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv))
            {
                const int grid = (*redistribute_mask_ptr)[mfi](iv, 0);
                if (grid >= 0)
                {
                    const int global_rank = this->ParticleDistributionMap(lev)[grid];
                    const int rank = ParallelContext::global_to_local_rank(global_rank);
                    if (rank != ParallelContext::MyProcSub()) {
                        neighbor_procs.push_back(rank);
                    }
                }
            }
            num_tiles++;
        }
        RemoveDuplicates(neighbor_procs);

        int max_threads = omp_get_max_threads();
        int neigubor_proc_num = static_cast<int>(neighbor_procs.size());

        if (neigubor_proc_num == 0) {
            return;
        }

        if (max_threads == 0) {
            amrex::Abort("max_threads is 0, cannot proceed with UNR registration");
        }

        // Build who_to_idx
        std::map<int, int>& who_to_idx = WarpX::GetInstance().who_to_idx[sname];
        for (int i = 0; i < neigubor_proc_num; ++i) {
            int who = neighbor_procs[i];
            who_to_idx[who] = i;
        }

        // Register UNR memory/blocks and synchronize them with MPI
        UNR_WarpX_buffer_reg();
        UNR_WarpX_blk_sync();
    }

    WarpX::UNR_WarpX_buffer& unr_send_buffer = warpx_comm_particle_container.unr_send_buffer;
    unr_send_buffer.clear(sname);
}

void
PhysicalParticleContainer::fusion_Isend_and_Irecv()
{
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = WarpX::GetInstance().warpx_comm_particle_container.at(sname).warpx_comm_comp;

    BL_PROFILE_VAR_NS("MyRedistributeMPI::prepare_send_buf", blp_prepare_send_buf);
    BL_PROFILE_VAR_NS("MyRedistributeMPI::handshake_local", blp_handshake_local);
    BL_PROFILE_VAR_NS("MyRedistributeMPI::Send_And_Recv", blp_Send_And_Recv);
    BL_PROFILE_VAR_START(blp_prepare_send_buf);
    constexpr int local = 1;       // TODO: move out

    std::map<int, std::vector< unsigned long long > >& mpi_snd_data = warpx_comm_comp.mpi_snd_data;

    std::map<int, std::vector< std::vector<char> > >& remote_send_allcomps = warpx_comm_comp.remote_send_allcomps;
    std::vector<long>& Rcvs = warpx_comm_comp.Rcvs;
    std::vector<int>& RcvProc = warpx_comm_comp.RcvProc;
    std::vector<std::size_t>& rOffset = warpx_comm_comp.rOffset; // Offset (in bytes) in the receive buffer
    RcvProc.clear();
    rOffset.clear();
    std::size_t& TotRcvBytes = warpx_comm_comp.TotRcvBytes;
    Vector<MPI_Request>& rreqs = warpx_comm_comp.rreqs;
    Vector<MPI_Request>& sreqs = warpx_comm_comp.sreqs;
    Vector<unsigned long long>& recvdata = warpx_comm_comp.recvdata;

    // Send packed particle data to remote neighbors
    std::map<int, std::vector<char> > not_ours;
    for (auto& map_it : remote_send_allcomps) {
        int who = map_it.first;
        not_ours[who];
    }

    std::vector<int> dest_proc_ids;
    std::vector<std::vector<std::vector<char> >* > pbuff_ptrs;
    for (auto& kv : remote_send_allcomps) {
        dest_proc_ids.push_back(kv.first);
        pbuff_ptrs.push_back(&(kv.second));
    }

    int num_threads = omp_get_max_threads();
    #pragma omp parallel for
    for (int pmap_it = 0; pmap_it < static_cast<int>(pbuff_ptrs.size()); ++pmap_it) {
        int who = dest_proc_ids[pmap_it];
        std::vector<std::vector<char> >& tmp = *(pbuff_ptrs[pmap_it]);
        for (int i = 0; i < num_threads; ++i) {
            not_ours[who].insert(not_ours[who].end(), tmp[i].begin(), tmp[i].end());
            tmp[i].erase(tmp[i].begin(), tmp[i].end());
        }
    }

    // Remove empty entries to reduce communication overhead
    for (auto it = not_ours.begin(); it != not_ours.end(); /* no ++ */) {
        if (it->second.empty()) {
            it = not_ours.erase(it);
        } else {
            ++it;
        }
    }

    using buffer_type = unsigned long long;

    // std::map<int, std::vector<buffer_type> > mpi_snd_data;
    mpi_snd_data.clear();
    for (const auto& kv : not_ours)
    {
        auto nbt = (kv.second.size() + sizeof(buffer_type)-1)/sizeof(buffer_type);      // Round up from a char count to a buffer_type count
        mpi_snd_data[kv.first].resize(nbt);
        std::memcpy((char*) mpi_snd_data[kv.first].data(), kv.second.data(), kv.second.size());
    }

    const int NProcs = ParallelContext::NProcsSub();
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];
    const int NNeighborProcs = neighbor_procs.size();
    std::vector<long> Snds(NProcs, 0);
    Rcvs.resize(NProcs, 0);

    BL_PROFILE_VAR_STOP(blp_prepare_send_buf);

    BL_PROFILE_VAR_START(blp_handshake_local);

    long NumSnds = 0;
    {
        // BuildRedistributeMask(0, local);
        for (const auto& kv : not_ours) {
            NumSnds += kv.second.size();
            Snds[kv.first] = kv.second.size();
        }

        const int SeqNum = ParallelDescriptor::SeqNum();

        const int num_rcvs = static_cast<int>(neighbor_procs.size());
        Vector<MPI_Status>  rstats(num_rcvs);
        Vector<MPI_Status>  sstats(num_rcvs);
        Vector<MPI_Request> rreqs(num_rcvs);
        Vector<MPI_Request> sreqs(num_rcvs);

        // Post receives
        for (int i = 0; i < num_rcvs; ++i) {
            const int Who = neighbor_procs[i];
            const Long Cnt = 1;

            rreqs[i] = ParallelDescriptor::Arecv(&Rcvs[Who], Cnt, Who, SeqNum,
                                                ParallelContext::CommunicatorSub()).req();
        }

        // Send.
        for (int i = 0; i < num_rcvs; ++i) {
            const int Who = neighbor_procs[i];
            const Long Cnt = 1;

            // ParallelDescriptor::Send(&Snds[Who], Cnt, Who, SeqNum,
            //                     ParallelContext::CommunicatorSub());
            sreqs[i] = ParallelDescriptor::Asend(&Snds[Who], Cnt, Who, SeqNum,
                                ParallelContext::CommunicatorSub()).req();
        }

        if (num_rcvs > 0) {
            ParallelDescriptor::Waitall(rreqs, rstats);
            ParallelDescriptor::Waitall(sreqs, sstats);
        }
    }

    BL_PROFILE_VAR_STOP(blp_handshake_local);

    BL_PROFILE_VAR_START(blp_Send_And_Recv);

    const int SeqNum = ParallelDescriptor::SeqNum();        // Globally unique MPI message sequence number

    sreqs.clear();

    // Return early if this rank neither sends nor receives anything
    if (local)
    {
        Long tot_snds_this_proc = 0;
        Long tot_rcvs_this_proc = 0;
        for (int i = 0; i < NNeighborProcs; ++i) {
            tot_snds_this_proc += Snds[neighbor_procs[i]];
            tot_rcvs_this_proc += Rcvs[neighbor_procs[i]];
        }
        if ( (tot_snds_this_proc == 0) && (tot_rcvs_this_proc == 0) ) {
            return; // There's no parallel work to do.
        }
    }

    std::size_t TotRcvInts = 0;

    TotRcvBytes = 0;
    for (int i = 0; i < NProcs; ++i) {
        if (Rcvs[i] > 0) {
            RcvProc.push_back(i);
            rOffset.push_back(TotRcvInts);
            TotRcvBytes += Rcvs[i];
            auto nbt = (Rcvs[i] + sizeof(buffer_type)-1)/sizeof(buffer_type);
            TotRcvInts += nbt;
        }
    }

    const auto nrcvs = static_cast<int>(RcvProc.size());
    // Vector<MPI_Status>  stats(nrcvs);
    // Vector<MPI_Request> rreqs(nrcvs);
        
    rreqs.resize(nrcvs);

    // Allocate data for rcvs as one big chunk.
    // Vector<unsigned long long> recvdata(TotRcvInts);
        
    recvdata.resize(TotRcvInts);

    // Post receives.
    for (int i = 0; i < nrcvs; ++i) {
        const auto Who    = RcvProc[i];
        const auto offset = rOffset[i];
        const auto Cnt = (Rcvs[Who] + sizeof(buffer_type)-1)/sizeof(buffer_type);
        // AMREX_ASSERT(Cnt > 0);
        // AMREX_ASSERT(Cnt < size_t(std::numeric_limits<int>::max()));
        // AMREX_ASSERT(Who >= 0 && Who < NProcs);

        rreqs[i] = ParallelDescriptor::Arecv(&recvdata[offset], Cnt, Who, SeqNum,
                                            ParallelContext::CommunicatorSub()).req();
    }

    // Send.
    for (const auto& kv : mpi_snd_data) {
        const auto Who = kv.first;
        const auto Cnt = kv.second.size();

        MPI_Request sreq = ParallelDescriptor::Asend(kv.second.data(), Cnt, Who, SeqNum,
                                ParallelContext::CommunicatorSub()).req();
        sreqs.push_back(sreq);
    }

    BL_PROFILE_VAR_STOP(blp_Send_And_Recv);
}

void
PhysicalParticleContainer::fusion_unr_put()
{
    BL_PROFILE("UNR::fusion_unr_put");
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;

    WarpX::UNR_WarpX_buffer& unr_send_buffer = warpx_comm_particle_container.unr_send_buffer;
    WarpX::UNR_WarpX_buffer& unr_recv_buffer = warpx_comm_particle_container.unr_recv_buffer;
    WarpX::UNR_WarpX_rmt_blk& unr_rmt_blk = warpx_comm_particle_container.unr_rmt_blk;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];

    int max_threads = omp_get_max_threads();
    int neigubor_proc_num = static_cast<int>(neighbor_procs.size());

    // RDMA sends counts and packed data together to adaptive remote blocks
    std::vector<unr_blk_h> loc_blk_h;
    std::vector<unr_sig_h> loc_sig_h;
    std::vector<size_t> loc_offset;
    std::vector<size_t> loc_size;

    std::vector<unr_blk_h> rmt_blk_h;
    std::vector<unr_sig_h> rmt_sig_h;
    std::vector<size_t> rmt_offset;
    std::vector<uru_transfer_t> dma_type;

    // Pack counts first, then pack payload data
    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who_idx = i;

        for (int j = 0; j < max_threads; ++j) {
            // Fill the count block with the packed particle count
            size_t send_buffer_size = unr_send_buffer.size_whoidx(who_idx, j);
            long* send_count_buffer = (long*)unr_send_buffer.get_whoidx_count_buffer(who_idx, j);
            *send_count_buffer = send_buffer_size;

            loc_blk_h.push_back(unr_send_buffer.get_whoidx_count_blk_h(who_idx, j));
            loc_sig_h.push_back(unr_send_buffer.get_whoidx_count_sig_h(who_idx, j));
            loc_offset.push_back(0);
            loc_size.push_back(unr_send_buffer.count_buffer_blk_size);
            
            rmt_blk_h.push_back(unr_rmt_blk.get_whoidx_rmt_count_blk(who_idx, j));
            rmt_sig_h.push_back(UNR_NO_SIGNAL);
            rmt_offset.push_back(0);

            dma_type.push_back(URU_TRANSFER_TYPE_DMA_PUT);
        }

        for (int j = 0; j < max_threads; ++j) {
            long* send_count_buffer = (long*)unr_send_buffer.get_whoidx_count_buffer(who_idx, j);
            long send_buffer_size = *send_count_buffer;

            loc_blk_h.push_back(unr_send_buffer.get_whoidx_data_blk_h(who_idx, j));
            loc_sig_h.push_back(unr_send_buffer.get_whoidx_data_sig_h(who_idx, j));
            loc_offset.push_back(0);
            loc_size.push_back(send_buffer_size);

            rmt_blk_h.push_back(unr_rmt_blk.get_whoidx_rmt_data_blk(who_idx, j));
            rmt_sig_h.push_back(UNR_NO_SIGNAL);
            rmt_offset.push_back(0);

            dma_type.push_back(URU_TRANSFER_TYPE_DMA_PUT);
        }
    }

    unr_blk_part_rdma_batch_start(neigubor_proc_num * max_threads * 2, loc_blk_h.data(), loc_sig_h.data(), loc_offset.data(), loc_size.data(), rmt_blk_h.data(), rmt_sig_h.data(), rmt_offset.data(), dma_type.data());
}

void
PhysicalParticleContainer::fusion_wait()
{
    BL_PROFILE("MyRedistributeMPI::fusion_wait");
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = WarpX::GetInstance().warpx_comm_particle_container.at(sname).warpx_comm_comp;

    std::vector<long>& Rcvs = warpx_comm_comp.Rcvs;
    std::vector<int>& RcvProc = warpx_comm_comp.RcvProc;
    Vector<MPI_Request>& rreqs = warpx_comm_comp.rreqs;
    Vector<MPI_Request>& sreqs = warpx_comm_comp.sreqs;
    const auto nrcvs = static_cast<int>(RcvProc.size());
    const auto nsnds = static_cast<int>(sreqs.size());

    if (nsnds > 0) {
        Vector<MPI_Status>  sstats(sreqs.size());
        int send_completed = 0;
        int send_polls = 0;
        while (!send_completed) {
            int flag;
            MPI_Testall(sreqs.size(), sreqs.data(), &flag, sstats.data());
            send_polls++;
            
            if (flag) {
                send_completed = 1;
                break;
            }
        }

    }

    if (nrcvs > 0) {
        Vector<MPI_Status>  rstats(nrcvs);

        int recv_completed = 0;
        int recv_polls = 0;
        
        while (!recv_completed) {
            int flag;
            MPI_Testall(rreqs.size(), rreqs.data(), &flag, rstats.data());
            recv_polls++;
            
            if (flag) {
                recv_completed = 1;
                break;
            }
        }

    }
}

void
PhysicalParticleContainer::fusion_unr_wait()
{
    BL_PROFILE("UNR::fusion_unr_wait");
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;

    WarpX::UNR_WarpX_buffer& unr_send_buffer = warpx_comm_particle_container.unr_send_buffer;
    WarpX::UNR_WarpX_buffer& unr_recv_buffer = warpx_comm_particle_container.unr_recv_buffer;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];

    int max_threads = omp_get_max_threads();
    int neigubor_proc_num = static_cast<int>(neighbor_procs.size());

    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who_idx = i;

        // Wait for counts first
        for (int j = 0; j < max_threads; ++j) {
            // Wait for send completion first
            unr_sig_wait(unr_send_buffer.get_whoidx_count_sig_h(who_idx, j));
            unr_sig_reset(unr_send_buffer.get_whoidx_count_sig_h(who_idx, j));

            // Then wait for receive completion
            unr_sig_wait(unr_recv_buffer.get_whoidx_count_sig_h(who_idx, j));
            unr_sig_reset(unr_recv_buffer.get_whoidx_count_sig_h(who_idx, j));
        }

        // Then wait for packed data
        for (int j = 0; j < max_threads; ++j) {
            long* send_count_buffer = (long*)unr_send_buffer.get_whoidx_count_buffer(who_idx, j);
            long send_buffer_size = *send_count_buffer;
            long* recv_count_buffer = (long*)unr_recv_buffer.get_whoidx_count_buffer(who_idx, j);
            long recv_buffer_size = *recv_count_buffer;

            if (send_buffer_size > 0) {
                unr_sig_wait(unr_send_buffer.get_whoidx_data_sig_h(who_idx, j));
                unr_sig_reset(unr_send_buffer.get_whoidx_data_sig_h(who_idx, j));
            }

            if (recv_buffer_size > 0) {
                unr_sig_wait(unr_recv_buffer.get_whoidx_data_sig_h(who_idx, j));
                unr_sig_reset(unr_recv_buffer.get_whoidx_data_sig_h(who_idx, j));
            }
        }
    }
}

void
PhysicalParticleContainer::fusion_remote_collect(int lev)
{
    BL_PROFILE_VAR_NS("MyRedistributeMPI::fusion_remote_collect::Locate", blp_locate);
    BL_PROFILE_VAR_NS("MyRedistributeMPI::fusion_remote_collect::Copy", blp_copy);
    const int superparticle_size = WarpX::GetInstance().superparticle_size;
    const std::string& sname = species_name;
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = WarpX::GetInstance().warpx_comm_particle_container.at(sname).warpx_comm_comp;

    std::vector<long>& Rcvs = warpx_comm_comp.Rcvs;
    std::vector<int>& RcvProc = warpx_comm_comp.RcvProc;
    std::vector<std::size_t>& rOffset = warpx_comm_comp.rOffset;
    std::size_t& TotRcvBytes = warpx_comm_comp.TotRcvBytes;
    Vector<MPI_Request>& rreqs = warpx_comm_comp.rreqs;
    Vector<MPI_Request>& sreqs = warpx_comm_comp.sreqs;
    Vector<unsigned long long>& recvdata = warpx_comm_comp.recvdata;
    const auto nrcvs = static_cast<int>(RcvProc.size());
    const auto nsnds = static_cast<int>(sreqs.size());

    if (nrcvs > 0) {
        BL_PROFILE_VAR_START(blp_locate);

        int npart = TotRcvBytes / superparticle_size;

        std::vector<int> rcv_grid(npart);
        std::vector<int> rcv_tile(npart);

        int ipart = 0;
        ParticleLocData pld;
        for (int j = 0; j < nrcvs; ++j) {
            const auto offset = rOffset[j];
            const auto Who    = RcvProc[j];
            const auto Cnt    = Rcvs[Who] / superparticle_size;
            for (int i = 0; i < int(Cnt); ++i) {
                char* pbuf = ((char*) &recvdata[offset]) + i * superparticle_size;

                Particle<NStructReal, NStructInt> p;

                std::memcpy(&p.m_idcpu, pbuf, sizeof(uint64_t));

                ParticleReal pos[AMREX_SPACEDIM];
                std::memcpy(&pos[0], pbuf + sizeof(uint64_t), AMREX_SPACEDIM*sizeof(ParticleReal));
                AMREX_D_TERM(p.pos(0) = pos[0];,
                            p.pos(1) = pos[1];,
                            p.pos(2) = pos[2]);

                bool success = Where(p, pld, 0, 0, 0);
                if (!success)
                {
                    amrex::Abort("MyRedistributeMPI_locate:: invalid particle.");
                }

                rcv_grid[ipart] = pld.m_grid;
                rcv_tile[ipart] = pld.m_tile;

                ++ipart;
            }
        }

        BL_PROFILE_VAR_STOP(blp_locate);

        BL_PROFILE_VAR_START(blp_copy);

        ipart = 0;
        for (int i = 0; i < nrcvs; ++i)
        {
            const auto offset = rOffset[i];
            const auto Who    = RcvProc[i];
            const auto Cnt = Rcvs[Who] / superparticle_size;
            for (int j = 0; j < int(Cnt); ++j)
            {
                // auto& ptile = m_particles[lev][std::make_pair(rcv_grid[ipart], rcv_tile[ipart])];
                auto& ptile = ParticlesAt(lev, rcv_grid[ipart], rcv_tile[ipart]);
                char* pbuf = ((char*) &recvdata[offset]) + j * superparticle_size;

                uint64_t idcpudata;
                std::memcpy(&idcpudata, pbuf, sizeof(uint64_t));
                pbuf += sizeof(uint64_t);
                ParticleReal xp, yp, zp, ux, uy, uz, w;
                std::memcpy(&xp, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&yp, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&zp, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&ux, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&uy, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&uz, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&w, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);

auto& soa = ptile.GetStructOfArrays();
                auto& iddata = soa.GetIdCPUData();
                auto& soa_m_x = soa.GetRealData(PIdx::x);
                auto& soa_m_y = soa.GetRealData(PIdx::y);
                auto& soa_m_z = soa.GetRealData(PIdx::z);
                auto& soa_ux = soa.GetRealData(PIdx::ux);
                auto& soa_uy = soa.GetRealData(PIdx::uy);
                auto& soa_uz = soa.GetRealData(PIdx::uz);
                auto& soa_wp = soa.GetRealData(PIdx::w);
                iddata.push_back(idcpudata);
                soa_m_x.push_back(xp);
                soa_m_y.push_back(yp);
                soa_m_z.push_back(zp);
                soa_ux.push_back(ux);
                soa_uy.push_back(uy);
                soa_uz.push_back(uz);
                soa_wp.push_back(w);

                ++ipart;
            }
        }

        BL_PROFILE_VAR_STOP(blp_copy);
    }
}

void
PhysicalParticleContainer::fusion_unr_remote_collect(int lev)
{
    BL_PROFILE("UNR::fusion_unr_remote_collect");

    const std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;

    WarpX::UNR_WarpX_buffer& unr_send_buffer = warpx_comm_particle_container.unr_send_buffer;
    WarpX::UNR_WarpX_buffer& unr_recv_buffer = warpx_comm_particle_container.unr_recv_buffer;
    amrex::Vector<int>& neighbor_procs = WarpX::GetInstance().neighbor_procs[sname];
    int superparticle_size = WarpX::GetInstance().superparticle_size;

    int max_threads = omp_get_max_threads();
    int neigubor_proc_num = static_cast<int>(neighbor_procs.size());

    std::vector<int> rcv_grid;
    std::vector<int> rcv_tile;

    // Collect received data
    ParticleLocData pld;
    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who_idx = i;
        for (int j = 0; j < max_threads; ++j) {
            long* recev_count_buffer = (long*)unr_recv_buffer.get_whoidx_count_buffer(who_idx, j);
            long recv_buffer_size = *recev_count_buffer;
            long recv_particle_num = recv_buffer_size / superparticle_size;

            for (int k = 0; k < recv_particle_num; ++k) {
                char* pbuf = (char*)unr_recv_buffer.get_whoidx_data_buffer(who_idx, j) + k * superparticle_size;

                Particle<NStructReal, NStructInt> p;

                std::memcpy(&p.m_idcpu, pbuf, sizeof(uint64_t));

                ParticleReal pos[AMREX_SPACEDIM];
                std::memcpy(&pos[0], pbuf + sizeof(uint64_t), AMREX_SPACEDIM*sizeof(ParticleReal));
                AMREX_D_TERM(p.pos(0) = pos[0];,
                            p.pos(1) = pos[1];,
                            p.pos(2) = pos[2]);

                bool success = Where(p, pld, 0, 0, 0);
                if (!success)
                {
                    amrex::Abort("MyRedistributeMPI_locate:: invalid particle.");
                }

                rcv_grid.push_back(pld.m_grid);
                rcv_tile.push_back(pld.m_tile);
            }
        }
    }

    int ipart = 0;
    for (int i = 0; i < neigubor_proc_num; ++i) {
        int who_idx = i;
        for (int j = 0; j < max_threads; ++j) {
            long* recev_count_buffer = (long*)unr_recv_buffer.get_whoidx_count_buffer(who_idx, j);
            long recv_buffer_size = *recev_count_buffer;
            long recv_particle_num = recv_buffer_size / superparticle_size;

            for (int k = 0; k < recv_particle_num; ++k) {
                auto& ptile = ParticlesAt(lev, rcv_grid[ipart], rcv_tile[ipart]);
                char* pbuf = (char*)unr_recv_buffer.get_whoidx_data_buffer(who_idx, j) + k * superparticle_size;

                uint64_t idcpudata;
                std::memcpy(&idcpudata, pbuf, sizeof(uint64_t));
                pbuf += sizeof(uint64_t);
                ParticleReal xp, yp, zp, ux, uy, uz, w;
                std::memcpy(&xp, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&yp, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&zp, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&ux, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&uy, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&uz, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);
                std::memcpy(&w, pbuf, sizeof(ParticleReal));
                pbuf += sizeof(ParticleReal);

auto& soa = ptile.GetStructOfArrays();
                auto& iddata = soa.GetIdCPUData();
                auto& soa_m_x = soa.GetRealData(PIdx::x);
                auto& soa_m_y = soa.GetRealData(PIdx::y);
                auto& soa_m_z = soa.GetRealData(PIdx::z);
                auto& soa_ux = soa.GetRealData(PIdx::ux);
                auto& soa_uy = soa.GetRealData(PIdx::uy);
                auto& soa_uz = soa.GetRealData(PIdx::uz);
                auto& soa_wp = soa.GetRealData(PIdx::w);
                iddata.push_back(idcpudata);
                soa_m_x.push_back(xp);
                soa_m_y.push_back(yp);
                soa_m_z.push_back(zp);
                soa_ux.push_back(ux);
                soa_uy.push_back(uy);
                soa_uz.push_back(uz);
                soa_wp.push_back(w);
                ipart++;
            }
        }
    }
}

/*
 * Local collect runs before remote collect and compacts the destination tiles.
 */
void
PhysicalParticleContainer::fusion_local_collect(int lev)
{
    BL_PROFILE("MyRedistribute::fusion_local_collect()");
    const auto& dm = *m_dummy_mf[lev];
    
    #pragma omp parallel
    for (amrex::MFIter mfi(dm, do_tiling ? tile_size : amrex::IntVect::TheZeroVector()); mfi.isValid(); ++mfi)
    {
        auto& ptile = ParticlesAt(lev, mfi);
        std::size_t np = ptile.numParticles();
        int max_threads = omp_get_max_threads();
        auto& soa = ptile.GetStructOfArrays();
        auto& iddata = soa.GetIdCPUData();
        auto& m_x = soa.GetRealData(PIdx::x);
        auto& m_y = soa.GetRealData(PIdx::y);
        auto& m_z = soa.GetRealData(PIdx::z);
        auto& ux = soa.GetRealData(PIdx::ux);
        auto& uy = soa.GetRealData(PIdx::uy);
        auto& uz = soa.GetRealData(PIdx::uz);
        auto& w = soa.GetRealData(PIdx::w);

        iddata.erase(iddata.begin() + ptile.g_new_particles_begin, iddata.end());
        m_x.erase(m_x.begin() + ptile.g_new_particles_begin, m_x.end());
        m_y.erase(m_y.begin() + ptile.g_new_particles_begin, m_y.end());
        m_z.erase(m_z.begin() + ptile.g_new_particles_begin, m_z.end());
        ux.erase(ux.begin() + ptile.g_new_particles_begin, ux.end());
        uy.erase(uy.begin() + ptile.g_new_particles_begin, uy.end());
        uz.erase(uz.begin() + ptile.g_new_particles_begin, uz.end());
        w.erase(w.begin() + ptile.g_new_particles_begin, w.end());

        for (int i = 0; i < max_threads; ++i)
        {
            std::vector<uint64_t>& local_recv_idcpu = ptile.local_recv_idcpu[i];
            std::vector<amrex::ParticleReal>& local_recv_xp = ptile.local_recv_xp[i];
            std::vector<amrex::ParticleReal>& local_recv_yp = ptile.local_recv_yp[i];
            std::vector<amrex::ParticleReal>& local_recv_zp = ptile.local_recv_zp[i];
            std::vector<amrex::ParticleReal>& local_recv_ux = ptile.local_recv_ux[i];
            std::vector<amrex::ParticleReal>& local_recv_uy = ptile.local_recv_uy[i];
            std::vector<amrex::ParticleReal>& local_recv_uz = ptile.local_recv_uz[i];
            std::vector<amrex::ParticleReal>& local_recv_w = ptile.local_recv_w[i];

iddata.insert(iddata.end(), local_recv_idcpu.data(), local_recv_idcpu.data() + local_recv_idcpu.size());
            m_x.insert(m_x.end(), local_recv_xp.data(), local_recv_xp.data() + local_recv_xp.size());
            m_y.insert(m_y.end(), local_recv_yp.data(), local_recv_yp.data() + local_recv_yp.size());
            m_z.insert(m_z.end(), local_recv_zp.data(), local_recv_zp.data() + local_recv_zp.size());
            ux.insert(ux.end(), local_recv_ux.data(), local_recv_ux.data() + local_recv_ux.size());
            uy.insert(uy.end(), local_recv_uy.data(), local_recv_uy.data() + local_recv_uy.size());
            uz.insert(uz.end(), local_recv_uz.data(), local_recv_uz.data() + local_recv_uz.size());
            w.insert(w.end(), local_recv_w.data(), local_recv_w.data() + local_recv_w.size());
            // Clear per-thread staging vectors after moving data.
            local_recv_idcpu.clear();
            local_recv_xp.clear();
            local_recv_yp.clear();
            local_recv_zp.clear();
            local_recv_ux.clear();
            local_recv_uy.clear();
            local_recv_uz.clear();
            local_recv_w.clear();
        }
    }
}

void
PhysicalParticleContainer::Evolve (int lev,
                                   const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
                                   const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz,
                                   MultiFab& jx, MultiFab& jy, MultiFab& jz,
                                   MultiFab* cjx, MultiFab* cjy, MultiFab* cjz,
                                   MultiFab* rho, MultiFab* crho,
                                   const MultiFab* cEx, const MultiFab* cEy, const MultiFab* cEz,
                                   const MultiFab* cBx, const MultiFab* cBy, const MultiFab* cBz,
                                   Real /*t*/, Real dt, DtType a_dt_type, bool skip_deposition,
                                   PushType push_type)
{

    WARPX_PROFILE("PhysicalParticleContainer::Evolve()");
    WARPX_PROFILE_VAR_NS("PhysicalParticleContainer::Evolve::GatherAndPush", blp_fg);

    BL_ASSERT(OnSameGrids(lev,jx));

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    const iMultiFab* current_masks = WarpX::CurrentBufferMasks(lev);
    const iMultiFab* gather_masks = WarpX::GatherBufferMasks(lev);

    const bool has_buffer = cEx || cjx;

    if (m_do_back_transformed_particles)
    {
        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            const auto np = pti.numParticles();
            const auto t_lev = pti.GetLevel();
            const auto index = pti.GetPairIndex();
            tmp_particle_data.resize(finestLevel()+1);
            for (int i = 0; i < TmpIdx::nattribs; ++i) {
                tmp_particle_data[t_lev][index][i].resize(np);
            }
        }
    }

#ifdef PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3
    const bool is_last_step = WarpX::GetInstance().is_last_step;
    if (!is_last_step) {
        constexpr int local = 1;
#if defined(UNROLL_OMP_UNR)
        // Prepare UNR send state for this fused physort step.
        UNR_BuildRedistributeMask_and_ClearSendBuffer(0, local);
#endif
    }
#endif // PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3


#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
#ifdef AMREX_USE_OMP
        const int thread_num = omp_get_thread_num();
#else
        const int thread_num = 0;
#endif

        FArrayBox filtered_Ex, filtered_Ey, filtered_Ez;
        FArrayBox filtered_Bx, filtered_By, filtered_Bz;

        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            const Box& box = pti.validbox();

            // Extract particle data
            auto& attribs = pti.GetAttribs();
            auto&  wp = attribs[PIdx::w];
            auto& uxp = attribs[PIdx::ux];
            auto& uyp = attribs[PIdx::uy];
            auto& uzp = attribs[PIdx::uz];

            const long np = pti.numParticles();

            // Data on the grid
            FArrayBox const* exfab = &Ex[pti];
            FArrayBox const* eyfab = &Ey[pti];
            FArrayBox const* ezfab = &Ez[pti];
            FArrayBox const* bxfab = &Bx[pti];
            FArrayBox const* byfab = &By[pti];
            FArrayBox const* bzfab = &Bz[pti];

            Elixir exeli, eyeli, ezeli, bxeli, byeli, bzeli;

            if (WarpX::use_fdtd_nci_corr)
            {
                // Filter arrays Ex[pti], store the result in
                // filtered_Ex and update pointer exfab so that it
                // points to filtered_Ex (and do the same for all
                // components of E and B).
                applyNCIFilter(lev, pti.tilebox(), exeli, eyeli, ezeli, bxeli, byeli, bzeli,
                               filtered_Ex, filtered_Ey, filtered_Ez,
                               filtered_Bx, filtered_By, filtered_Bz,
                               Ex[pti], Ey[pti], Ez[pti], Bx[pti], By[pti], Bz[pti],
                               exfab, eyfab, ezfab, bxfab, byfab, bzfab);
            }

            // Determine which particles deposit/gather in the buffer, and
            // which particles deposit/gather in the fine patch
            long nfine_current = np;
            long nfine_gather = np;
            if (has_buffer && !do_not_push) {
                // - Modify `nfine_current` and `nfine_gather` (in place)
                //    so that they correspond to the number of particles
                //    that deposit/gather in the fine patch respectively.
                // - Reorder the particle arrays,
                //    so that the `nfine_current`/`nfine_gather` first particles
                //    deposit/gather in the fine patch
                //    and (thus) the `np-nfine_current`/`np-nfine_gather` last particles
                //    deposit/gather in the buffer
                PartitionParticlesInBuffers( nfine_current, nfine_gather, np,
                    pti, lev, current_masks, gather_masks );
            }

            const long np_current = (cjx) ? nfine_current : np;

            if (rho && ! skip_deposition && ! do_not_deposit) {
                // Deposit charge before particle push, in component 0 of MultiFab rho.

                const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                    pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr():nullptr;

                DepositCharge(pti, wp, ion_lev, rho, 0, 0,
                              np_current, thread_num, lev, lev);
                if (has_buffer){
                    DepositCharge(pti, wp, ion_lev, crho, 0, np_current,
                                  np-np_current, thread_num, lev, lev-1);
                }
            }

            if (! do_not_push)
            {
                const long np_gather = (cEx) ? nfine_gather : np;

                int e_is_nodal = Ex.is_nodal() and Ey.is_nodal() and Ez.is_nodal();
#if defined(PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3)
                Box growbox = pti.tilebox();
            
                const amrex::IntVect ngEB = Ex.nGrowVect();
                growbox.grow(ngEB);

                const Dim3 lo = lbound(growbox);
                const Dim3 len = length(growbox);
                int lenx = len.x;
                int leny = len.y;
                int lenz = len.z;

                amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
                amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
                amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
                amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
                amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
                amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

                int m_box_size = WarpX::GetInstance().m_box_size;

                if (lenx * leny * lenz > m_box_size) {
                    printf("lenx * leny * lenz: %d, m_box_size: %d\n", lenx * leny * lenz, m_box_size);
                    amrex::Abort("lenx * leny * lenz > m_box_size");
                }

                amrex::Real* aos_arr = WarpX::GetInstance().aos_arr + thread_num * 6 * m_box_size;

                for (int l_node = 0; l_node < lenz; ++l_node)
                {
                    for (int k_node = 0; k_node < leny; ++k_node)
                    {
                        for (int j_node = 0; j_node < lenx; ++j_node)
                        {
                            int idx = j_node + k_node * lenx + l_node * lenx * leny;
                            const int offset = (lo.x + j_node - ex_arr.begin.x) + (lo.y + k_node - ex_arr.begin.y) * ex_arr.jstride + (lo.z + l_node - ex_arr.begin.z) * ex_arr.kstride;
                            aos_arr[6 * idx + 0] = ex_arr.p[offset];
                            aos_arr[6 * idx + 1] = ey_arr.p[offset];
                            aos_arr[6 * idx + 2] = ez_arr.p[offset];
                            aos_arr[6 * idx + 3] = bz_arr.p[offset];
                            aos_arr[6 * idx + 4] = by_arr.p[offset];
                            aos_arr[6 * idx + 5] = bx_arr.p[offset];
                        }
                    }
                }
#endif

                //
                // Gather and push for particles not in the buffer
                //
                WARPX_PROFILE_VAR_START(blp_fg);
                const auto np_to_push = np_gather;
                const auto gather_lev = lev;
                if (push_type == PushType::Explicit) {
#ifdef PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3
                    PushPX_vpu_mpu_physort_order3(pti, exfab, eyfab, ezfab,
                           bxfab, byfab, bzfab,
                           Ex.nGrowVect(), e_is_nodal,
                           0, np_to_push, lev, gather_lev, dt, ScaleFields(false),
                           a_dt_type);
                    if (!is_last_step) {
#if defined(UNROLL_OMP_UNR)
                        // Pack outgoing particles for UNR asynchronous transfer.
                        fusion_pack_unr(pti, Ex.nGrowVect(), 0, np_to_push, lev, gather_lev);
#endif
                    }
#else
                    PushPX(pti, exfab, eyfab, ezfab,
                           bxfab, byfab, bzfab,
                           Ex.nGrowVect(), e_is_nodal,
                           0, np_to_push, lev, gather_lev, dt, ScaleFields(false), a_dt_type);
#endif
                } else if (push_type == PushType::Implicit) {
                    ImplicitPushXP(pti, exfab, eyfab, ezfab,
                                   bxfab, byfab, bzfab,
                                   Ex.nGrowVect(), e_is_nodal,
                                   0, np_to_push, lev, gather_lev, dt, ScaleFields(false), a_dt_type);
                }

                if (np_gather < np)
                {
                    const IntVect& ref_ratio = WarpX::RefRatio(lev-1);
                    const Box& cbox = amrex::coarsen(box,ref_ratio);

                    // Data on the grid
                    FArrayBox const* cexfab = &(*cEx)[pti];
                    FArrayBox const* ceyfab = &(*cEy)[pti];
                    FArrayBox const* cezfab = &(*cEz)[pti];
                    FArrayBox const* cbxfab = &(*cBx)[pti];
                    FArrayBox const* cbyfab = &(*cBy)[pti];
                    FArrayBox const* cbzfab = &(*cBz)[pti];

                    if (WarpX::use_fdtd_nci_corr)
                    {
                        // Filter arrays (*cEx)[pti], store the result in
                        // filtered_Ex and update pointer cexfab so that it
                        // points to filtered_Ex (and do the same for all
                        // components of E and B)
                        applyNCIFilter(lev-1, cbox, exeli, eyeli, ezeli, bxeli, byeli, bzeli,
                                       filtered_Ex, filtered_Ey, filtered_Ez,
                                       filtered_Bx, filtered_By, filtered_Bz,
                                       (*cEx)[pti], (*cEy)[pti], (*cEz)[pti],
                                       (*cBx)[pti], (*cBy)[pti], (*cBz)[pti],
                                       cexfab, ceyfab, cezfab, cbxfab, cbyfab, cbzfab);
                    }

                    // Field gather and push for particles in gather buffers
                    e_is_nodal = cEx->is_nodal() and cEy->is_nodal() and cEz->is_nodal();
                    if (push_type == PushType::Explicit) {
                        PushPX(pti, cexfab, ceyfab, cezfab,
                               cbxfab, cbyfab, cbzfab,
                               cEx->nGrowVect(), e_is_nodal,
                               nfine_gather, np-nfine_gather,
                               lev, lev-1, dt, ScaleFields(false), a_dt_type);
                    } else if (push_type == PushType::Implicit) {
                        ImplicitPushXP(pti, cexfab, ceyfab, cezfab,
                                       cbxfab, cbyfab, cbzfab,
                                       cEx->nGrowVect(), e_is_nodal,
                                       nfine_gather, np-nfine_gather,
                                       lev, lev-1, dt, ScaleFields(false), a_dt_type);
                    }
                }

                WARPX_PROFILE_VAR_STOP(blp_fg);


#if defined(PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3) && defined(UNROLL_OMP_UNR)
            }
        }
    }

    if (!is_last_step) {
        // amrex::Print() << "Run UNROLL_OMP_UNR\n";
        fusion_unr_put();
    }

    #pragma omp parallel
    {
        const int thread_num = omp_get_thread_num();
        
        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            auto wt = static_cast<amrex::Real>(amrex::second());
            
            // Extract particle data
            auto& attribs = pti.GetAttribs();
            auto&  wp = attribs[PIdx::w];
            auto& uxp = attribs[PIdx::ux];
            auto& uyp = attribs[PIdx::uy];
            auto& uzp = attribs[PIdx::uz];

            const long np = pti.numParticles();

            long nfine_current = np;
            const long np_current = (cjx) ? nfine_current : np;

            if (! do_not_push)
            {
#endif  // UNROLL_OMP_UNR

                // Current Deposition
                if (!skip_deposition)
                {
                    // Deposit at t_{n+1/2} with explicit push
                    const amrex::Real relative_time = (push_type == PushType::Explicit ? -0.5_rt * dt : 0.0_rt);

                    const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                        pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr():nullptr;

                    // Deposit inside domains
                    DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, &jx, &jy, &jz,
                                   0, np_current, thread_num,
                                   lev, lev, dt, relative_time, push_type);

                    if (has_buffer)
                    {
                        // Deposit in buffers
                        DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, cjx, cjy, cjz,
                                       np_current, np-np_current, thread_num,
                                       lev, lev-1, dt, relative_time, push_type);
                    }
                } // end of "if electrostatic_solver_id == ElectrostaticSolverAlgo::None"
            } // end of "if do_not_push"

            if (rho && ! skip_deposition && ! do_not_deposit) {
                // Deposit charge after particle push, in component 1 of MultiFab rho.
                // (Skipped for electrostatic solver, as this may lead to out-of-bounds)
                if (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::None) {
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho->nComp() >= 2,
                        "Cannot deposit charge in rho component 1: only component 0 is allocated!");

                    const int* const AMREX_RESTRICT ion_lev = (do_field_ionization)?
                        pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr():nullptr;

                    DepositCharge(pti, wp, ion_lev, rho, 1, 0,
                                  np_current, thread_num, lev, lev);
                    if (has_buffer){
                        DepositCharge(pti, wp, ion_lev, crho, 1, np_current,
                                      np-np_current, thread_num, lev, lev-1);
                    }
                }
            }

            amrex::Gpu::synchronize();

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
            }
        }
    }
    // Split particles at the end of the timestep.
    // When subcycling is ON, the splitting is done on the last call to
    // PhysicalParticleContainer::Evolve on the finest level, i.e., at the
    // end of the large timestep. Otherwise, the pushes on different levels
    // are not consistent, and the call to Redistribute (inside
    // SplitParticles) may result in split particles to deposit twice on the
    // coarse level.
    if (do_splitting && (a_dt_type == DtType::SecondHalf || a_dt_type == DtType::Full) ){
        SplitParticles(lev);
    }

#if defined(PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3)
    if (!is_last_step) {
#if defined(UNROLL_OMP_UNR)
        // Complete UNR transfer before boundary handling collects remote particles.
        fusion_unr_wait();
#endif
    }
#endif  // PUSH_vpu_mpu_PHYSORT_FUSION_ORDER3
}

void
PhysicalParticleContainer::applyNCIFilter (
    int lev, const Box& box,
    Elixir& exeli, Elixir& eyeli, Elixir& ezeli,
    Elixir& bxeli, Elixir& byeli, Elixir& bzeli,
    FArrayBox& filtered_Ex, FArrayBox& filtered_Ey, FArrayBox& filtered_Ez,
    FArrayBox& filtered_Bx, FArrayBox& filtered_By, FArrayBox& filtered_Bz,
    const FArrayBox& Ex, const FArrayBox& Ey, const FArrayBox& Ez,
    const FArrayBox& Bx, const FArrayBox& By, const FArrayBox& Bz,
    FArrayBox const * & ex_ptr, FArrayBox const * & ey_ptr,
    FArrayBox const * & ez_ptr, FArrayBox const * & bx_ptr,
    FArrayBox const * & by_ptr, FArrayBox const * & bz_ptr)
{

    // Get instances of NCI Godfrey filters
    const auto& nci_godfrey_filter_exeybz = WarpX::GetInstance().nci_godfrey_filter_exeybz;
    const auto& nci_godfrey_filter_bxbyez = WarpX::GetInstance().nci_godfrey_filter_bxbyez;

#if defined(WARPX_DIM_1D_Z)
    const Box& tbox = amrex::grow(box, static_cast<int>(WarpX::noz));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    const Box& tbox = amrex::grow(box, {static_cast<int>(WarpX::nox),
                static_cast<int>(WarpX::noz)});
#else
    const Box& tbox = amrex::grow(box, {static_cast<int>(WarpX::nox),
                static_cast<int>(WarpX::noy),
                static_cast<int>(WarpX::noz)});
#endif

    // Filter Ex (Both 2D and 3D)
    filtered_Ex.resize(amrex::convert(tbox,Ex.box().ixType()));
    // Safeguard for GPU
    exeli = filtered_Ex.elixir();
    // Apply filter on Ex, result stored in filtered_Ex

    nci_godfrey_filter_exeybz[lev]->ApplyStencil(filtered_Ex, Ex, filtered_Ex.box());
    // Update ex_ptr reference
    ex_ptr = &filtered_Ex;

    // Filter Ez
    filtered_Ez.resize(amrex::convert(tbox,Ez.box().ixType()));
    ezeli = filtered_Ez.elixir();
    nci_godfrey_filter_bxbyez[lev]->ApplyStencil(filtered_Ez, Ez, filtered_Ez.box());
    ez_ptr = &filtered_Ez;

    // Filter By
    filtered_By.resize(amrex::convert(tbox,By.box().ixType()));
    byeli = filtered_By.elixir();
    nci_godfrey_filter_bxbyez[lev]->ApplyStencil(filtered_By, By, filtered_By.box());
    by_ptr = &filtered_By;
#if defined(WARPX_DIM_3D)
    // Filter Ey
    filtered_Ey.resize(amrex::convert(tbox,Ey.box().ixType()));
    eyeli = filtered_Ey.elixir();
    nci_godfrey_filter_exeybz[lev]->ApplyStencil(filtered_Ey, Ey, filtered_Ey.box());
    ey_ptr = &filtered_Ey;

    // Filter Bx
    filtered_Bx.resize(amrex::convert(tbox,Bx.box().ixType()));
    bxeli = filtered_Bx.elixir();
    nci_godfrey_filter_bxbyez[lev]->ApplyStencil(filtered_Bx, Bx, filtered_Bx.box());
    bx_ptr = &filtered_Bx;

    // Filter Bz
    filtered_Bz.resize(amrex::convert(tbox,Bz.box().ixType()));
    bzeli = filtered_Bz.elixir();
    nci_godfrey_filter_exeybz[lev]->ApplyStencil(filtered_Bz, Bz, filtered_Bz.box());
    bz_ptr = &filtered_Bz;
#else
    amrex::ignore_unused(eyeli, bxeli, bzeli,
        filtered_Ey, filtered_Bx, filtered_Bz,
        Ey, Bx, Bz, ey_ptr, bx_ptr, bz_ptr);
#endif
}

// Loop over all particles in the particle container and
// split particles tagged with p.id()=DoSplitParticleID
void
PhysicalParticleContainer::SplitParticles (int lev)
{
    auto& mypc = WarpX::GetInstance().GetPartContainer();
    auto& pctmp_split = mypc.GetPCtmp();
    RealVector psplit_x, psplit_y, psplit_z, psplit_w;
    RealVector psplit_ux, psplit_uy, psplit_uz;
    long np_split_to_add = 0;
    long np_split;
    if(split_type==0)
    {
        #if defined(WARPX_DIM_3D)
           np_split = 8;
        #elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
           np_split = 4;
        #else
           np_split = 2;
        #endif
    } else {
        np_split = 2*AMREX_SPACEDIM;
    }

    // Loop over particle interator
    for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const auto GetPosition = GetParticlePosition<PIdx>(pti);

        const amrex::Vector<int> ppc_nd = plasma_injectors[0]->num_particles_per_cell_each_dim;
        const std::array<Real,3>& dx = WarpX::CellSize(lev);
        amrex::Vector<Real> split_offset = {dx[0]/2._rt,
                                            dx[1]/2._rt,
                                            dx[2]/2._rt};
        if (ppc_nd[0] > 0){
            // offset for split particles is computed as a function of cell size
            // and number of particles per cell, so that a uniform distribution
            // before splitting results in a uniform distribution after splitting
            split_offset[0] /= ppc_nd[0];
            split_offset[1] /= ppc_nd[1];
            split_offset[2] /= ppc_nd[2];
        }
        // particle Struct Of Arrays data
        auto& attribs = pti.GetAttribs();
        auto& wp  = attribs[PIdx::w ];
        auto& uxp = attribs[PIdx::ux];
        auto& uyp = attribs[PIdx::uy];
        auto& uzp = attribs[PIdx::uz];

        ParticleTileType& ptile = ParticlesAt(lev, pti);
        auto& soa = ptile.GetStructOfArrays();
        uint64_t * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

        const long np = pti.numParticles();
        for(int i=0; i<np; i++){
            ParticleReal xp, yp, zp;
            GetPosition(i, xp, yp, zp);
            if (idcpu[i] == LongParticleIds::DoSplitParticleID){
                // If particle is tagged, split it and put the
                // split particles in local arrays psplit_x etc.
                np_split_to_add += np_split;
#if defined(WARPX_DIM_1D_Z)
                // Split particle in two along z axis
                // 2 particles in 1d, split_type doesn't matter? Discuss with Remi
                for (int ishift = -1; ishift < 2; ishift +=2 ){
                    // Add one particle with offset in z
                    psplit_x.push_back( xp );
                    psplit_y.push_back( yp );
                    psplit_z.push_back( zp + ishift*split_offset[2] );
                    psplit_ux.push_back( uxp[i] );
                    psplit_uy.push_back( uyp[i] );
                    psplit_uz.push_back( uzp[i] );
                    psplit_w.push_back( wp[i]/np_split );
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                if (split_type==0){
                    // Split particle in two along each diagonals
                    // 4 particles in 2d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        for (int kshift = -1; kshift < 2; kshift +=2 ){
                            // Add one particle with offset in x and z
                            psplit_x.push_back( xp + ishift*split_offset[0] );
                            psplit_y.push_back( yp );
                            psplit_z.push_back( zp + kshift*split_offset[2] );
                            psplit_ux.push_back( uxp[i] );
                            psplit_uy.push_back( uyp[i] );
                            psplit_uz.push_back( uzp[i] );
                            psplit_w.push_back( wp[i]/np_split );
                        }
                    }
                } else {
                    // Split particle in two along each axis
                    // 4 particles in 2d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        // Add one particle with offset in x
                        psplit_x.push_back( xp + ishift*split_offset[0] );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                        // Add one particle with offset in z
                        psplit_x.push_back( xp );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp + ishift*split_offset[2] );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                    }
                }
#elif defined(WARPX_DIM_3D)
                if (split_type==0){
                    // Split particle in two along each diagonals
                    // 8 particles in 3d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        for (int jshift = -1; jshift < 2; jshift +=2 ){
                            for (int kshift = -1; kshift < 2; kshift +=2 ){
                                // Add one particle with offset in x, y and z
                                psplit_x.push_back( xp + ishift*split_offset[0] );
                                psplit_y.push_back( yp + jshift*split_offset[1] );
                                psplit_z.push_back( zp + kshift*split_offset[2] );
                                psplit_ux.push_back( uxp[i] );
                                psplit_uy.push_back( uyp[i] );
                                psplit_uz.push_back( uzp[i] );
                                psplit_w.push_back( wp[i]/np_split );
                            }
                        }
                    }
                } else {
                    // Split particle in two along each axis
                    // 6 particles in 3d
                    for (int ishift = -1; ishift < 2; ishift +=2 ){
                        // Add one particle with offset in x
                        psplit_x.push_back( xp + ishift*split_offset[0] );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                        // Add one particle with offset in y
                        psplit_x.push_back( xp );
                        psplit_y.push_back( yp + ishift*split_offset[1] );
                        psplit_z.push_back( zp );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                        // Add one particle with offset in z
                        psplit_x.push_back( xp );
                        psplit_y.push_back( yp );
                        psplit_z.push_back( zp + ishift*split_offset[2] );
                        psplit_ux.push_back( uxp[i] );
                        psplit_uy.push_back( uyp[i] );
                        psplit_uz.push_back( uzp[i] );
                        psplit_w.push_back( wp[i]/np_split );
                    }
                }
#endif
                // invalidate the particle
                idcpu[i] = amrex::ParticleIdCpus::Invalid;
            }
        }
    }
    // Add local arrays psplit_x etc. to the temporary
    // particle container pctmp_split. Split particles
    // are tagged with p.id()=NoSplitParticleID so that
    // they are not re-split when entering a higher level
    // AddNParticles calls Redistribute, so that particles
    // in pctmp_split are in the proper grids and tiles
    const amrex::Vector<ParticleReal> xp(psplit_x.data(), psplit_x.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> yp(psplit_y.data(), psplit_y.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> zp(psplit_z.data(), psplit_z.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> uxp(psplit_ux.data(), psplit_ux.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> uyp(psplit_uy.data(), psplit_uy.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> uzp(psplit_uz.data(), psplit_uz.data() + np_split_to_add);
    const amrex::Vector<ParticleReal> wp(psplit_w.data(), psplit_w.data() + np_split_to_add);

    amrex::Vector<amrex::Vector<ParticleReal>> attr;
    attr.push_back(wp);
    const amrex::Vector<amrex::Vector<int>> attr_int;
    pctmp_split.AddNParticles(lev,
                              np_split_to_add,
                              xp,
                              yp,
                              zp,
                              uxp,
                              uyp,
                              uzp,
                              1,
                              attr,
                              0, attr_int,
                              1, LongParticleIds::NoSplitParticleID);
    // Copy particles from tmp to current particle container
    constexpr bool local_flag = true;
    addParticles(pctmp_split,local_flag);
    // Clear tmp container
    pctmp_split.clearParticles();
}

void
PhysicalParticleContainer::PushP (int lev, Real dt,
                                  const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
                                  const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz)
{
    WARPX_PROFILE("PhysicalParticleContainer::PushP()");

    if (do_not_push) { return; }

    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(lev,0));

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
        {
            amrex::Box box = pti.tilebox();
            box.grow(Ex.nGrowVect());

            const long np = pti.numParticles();

            // Data on the grid
            const FArrayBox& exfab = Ex[pti];
            const FArrayBox& eyfab = Ey[pti];
            const FArrayBox& ezfab = Ez[pti];
            const FArrayBox& bxfab = Bx[pti];
            const FArrayBox& byfab = By[pti];
            const FArrayBox& bzfab = Bz[pti];

            const auto getPosition = GetParticlePosition<PIdx>(pti);

            const auto getExternalEB = GetExternalEBField(pti);

            const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
            const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
            const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
            const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
            const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
            const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

            const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

            const Dim3 lo = lbound(box);

            const bool galerkin_interpolation = WarpX::galerkin_interpolation;
            const int nox = WarpX::nox;
            const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

            amrex::Array4<const amrex::Real> const& ex_arr = exfab.array();
            amrex::Array4<const amrex::Real> const& ey_arr = eyfab.array();
            amrex::Array4<const amrex::Real> const& ez_arr = ezfab.array();
            amrex::Array4<const amrex::Real> const& bx_arr = bxfab.array();
            amrex::Array4<const amrex::Real> const& by_arr = byfab.array();
            amrex::Array4<const amrex::Real> const& bz_arr = bzfab.array();

            amrex::IndexType const ex_type = exfab.box().ixType();
            amrex::IndexType const ey_type = eyfab.box().ixType();
            amrex::IndexType const ez_type = ezfab.box().ixType();
            amrex::IndexType const bx_type = bxfab.box().ixType();
            amrex::IndexType const by_type = byfab.box().ixType();
            amrex::IndexType const bz_type = bzfab.box().ixType();

            auto& attribs = pti.GetAttribs();
            ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
            ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
            ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

            int* AMREX_RESTRICT ion_lev = nullptr;
            if (do_field_ionization) {
                ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr();
            }

            // Loop over the particles and update their momentum
            const amrex::ParticleReal q = this->charge;
            const amrex::ParticleReal m = this-> mass;

            const auto pusher_algo = WarpX::particle_pusher_algo;
            const auto do_crr = do_classical_radiation_reaction;

            const auto t_do_not_gather = do_not_gather;

            enum exteb_flags : int { no_exteb, has_exteb };

            const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;

            amrex::ParallelFor(TypeList<CompileTimeOptions<no_exteb,has_exteb>>{},
                               {exteb_runtime_flag},
                               np, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control)
            {
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                amrex::ParticleReal Exp = Ex_external_particle;
                amrex::ParticleReal Eyp = Ey_external_particle;
                amrex::ParticleReal Ezp = Ez_external_particle;
                amrex::ParticleReal Bxp = Bx_external_particle;
                amrex::ParticleReal Byp = By_external_particle;
                amrex::ParticleReal Bzp = Bz_external_particle;

                if (!t_do_not_gather){
                    // first gather E and B to the particle positions
                    doGatherShapeN(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                   ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                   ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                   dinv, xyzmin, lo, n_rz_azimuthal_modes,
                                   nox, galerkin_interpolation);
                }

                // Externally applied E and B-field in Cartesian co-ordinates
                [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
                if constexpr (exteb_control == has_exteb) {
                    getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
                }

                if (do_crr) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev) { qp *= ion_lev[ip]; }
                    UpdateMomentumBorisWithRadiationReaction(ux[ip], uy[ip], uz[ip],
                                                             Exp, Eyp, Ezp, Bxp,
                                                             Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::Boris) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev) { qp *= ion_lev[ip]; }
                    UpdateMomentumBoris( ux[ip], uy[ip], uz[ip],
                                         Exp, Eyp, Ezp, Bxp,
                                         Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::Vay) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev){ qp *= ion_lev[ip]; }
                    UpdateMomentumVay( ux[ip], uy[ip], uz[ip],
                                       Exp, Eyp, Ezp, Bxp,
                                       Byp, Bzp, qp, m, dt);
                } else if (pusher_algo == ParticlePusherAlgo::HigueraCary) {
                    amrex::ParticleReal qp = q;
                    if (ion_lev){ qp *= ion_lev[ip]; }
                    UpdateMomentumHigueraCary( ux[ip], uy[ip], uz[ip],
                                               Exp, Eyp, Ezp, Bxp,
                                               Byp, Bzp, qp, m, dt);
                } else {
                    amrex::Abort("Unknown particle pusher");
                }
            });
        }
    }
}

/* \brief Inject particles during the simulation
 * \param injection_box: domain where particles should be injected.
 */
void
PhysicalParticleContainer::ContinuousInjection (const RealBox& injection_box)
{
    // Inject plasma on level 0. Particles will be redistributed.
    const int lev=0;
    for (auto const& plasma_injector : plasma_injectors) {
        AddPlasma(*plasma_injector, lev, injection_box);
    }
}

/* \brief Inject a flux of particles during the simulation
 */
void
PhysicalParticleContainer::ContinuousFluxInjection (amrex::Real t, amrex::Real dt)
{
    for (auto const& plasma_injector : plasma_injectors) {
        if (plasma_injector->doFluxInjection()){
            // Check the optional parameters for start and stop of injection
            if ( ((plasma_injector->flux_tmin<0) || (t>=plasma_injector->flux_tmin)) &&
                 ((plasma_injector->flux_tmax<0) || (t< plasma_injector->flux_tmax)) ){

                AddPlasmaFlux(*plasma_injector, dt);

            }
        }
    }
}

/* \brief Perform the field gather and particle push operations in one fused kernel
 *
 */
void
PhysicalParticleContainer::PushPX (WarpXParIter& pti,
                                   amrex::FArrayBox const * exfab,
                                   amrex::FArrayBox const * eyfab,
                                   amrex::FArrayBox const * ezfab,
                                   amrex::FArrayBox const * bxfab,
                                   amrex::FArrayBox const * byfab,
                                   amrex::FArrayBox const * bzfab,
                                   const amrex::IntVect ngEB, const int /*e_is_nodal*/,
                                   const long offset,
                                   const long np_to_push,
                                   int lev, int gather_lev,
                                   amrex::Real dt, ScaleFields scaleFields,
                                   DtType a_dt_type)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((gather_lev==(lev-1)) ||
                                     (gather_lev==(lev  )),
                                     "Gather buffers only work for lev-1");
    // If no particles, do not do anything
    if (np_to_push == 0) { return; }

    // Get cell size on gather_lev
    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev,0));

    // Get box from which field is gathered.
    // If not gathering from the finest level, the box is coarsened.
    Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(),ref_ratio);
    }

    // Add guard cells to the box.
    box.grow(ngEB);

    const auto getPosition = GetParticlePosition<PIdx>(pti, offset);
          auto setPosition = SetParticlePosition<PIdx>(pti, offset);

    const auto getExternalEB = GetExternalEBField(pti, offset);

    const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
    const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
    const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
    const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
    const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
    const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

    // Lower corner of tile box physical domain (take into account Galilean shift)
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    const Dim3 lo = lbound(box);

    const bool galerkin_interpolation = WarpX::galerkin_interpolation;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    auto& attribs = pti.GetAttribs();
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

    const int do_copy = (m_do_back_transformed_particles && (a_dt_type!=DtType::SecondHalf) );
    CopyParticleAttribs copyAttribs;
    if (do_copy) {
        copyAttribs = CopyParticleAttribs(pti, tmp_particle_data, offset);
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr() + offset;
    }

    const bool save_previous_position = m_save_previous_position;
    ParticleReal* x_old = nullptr;
    ParticleReal* y_old = nullptr;
    ParticleReal* z_old = nullptr;
    if (save_previous_position) {
#if (AMREX_SPACEDIM >= 2)
        x_old = pti.GetAttribs(particle_comps["prev_x"]).dataPtr() + offset;
#else
    amrex::ignore_unused(x_old);
#endif
#if defined(WARPX_DIM_3D)
        y_old = pti.GetAttribs(particle_comps["prev_y"]).dataPtr() + offset;
#else
    amrex::ignore_unused(y_old);
#endif
        z_old = pti.GetAttribs(particle_comps["prev_z"]).dataPtr() + offset;
    }

    // Loop over the particles and update their momentum
    const amrex::ParticleReal q = this->charge;
    const amrex::ParticleReal m = this-> mass;

    const auto pusher_algo = WarpX::particle_pusher_algo;
    const auto do_crr = do_classical_radiation_reaction;
#ifdef WARPX_QED
    const auto do_sync = m_do_qed_quantum_sync;
    amrex::Real t_chi_max = 0.0;
    if (do_sync) { t_chi_max = m_shr_p_qs_engine->get_minimum_chi_part(); }

    QuantumSynchrotronEvolveOpticalDepth evolve_opt;
    amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_QSR = nullptr;
    const bool local_has_quantum_sync = has_quantum_sync();
    if (local_has_quantum_sync) {
        evolve_opt = m_shr_p_qs_engine->build_evolve_functor();
        p_optical_depth_QSR = pti.GetAttribs(particle_comps["opticalDepthQSR"]).dataPtr()  + offset;
    }
#endif

    const auto t_do_not_gather = do_not_gather;

    enum exteb_flags : int { no_exteb, has_exteb };
    enum qed_flags : int { no_qed, has_qed };

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    int qed_runtime_flag = no_qed;
#endif

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(
        TypeList<CompileTimeOptions<no_exteb,has_exteb>, CompileTimeOptions<no_qed  ,has_qed>>{},
        {exteb_runtime_flag, qed_runtime_flag},
        np_to_push,
        [=] AMREX_GPU_DEVICE (long ip, auto exteb_control, auto qed_control)
    {
        amrex::ParticleReal xp, yp, zp;
        getPosition(ip, xp, yp, zp);

        if (save_previous_position) {
#if (AMREX_SPACEDIM >= 2)
            x_old[ip] = xp;
#endif
#if defined(WARPX_DIM_3D)
            y_old[ip] = yp;
#endif
            z_old[ip] = zp;
        }

        amrex::ParticleReal Exp = Ex_external_particle;
        amrex::ParticleReal Eyp = Ey_external_particle;
        amrex::ParticleReal Ezp = Ez_external_particle;
        amrex::ParticleReal Bxp = Bx_external_particle;
        amrex::ParticleReal Byp = By_external_particle;
        amrex::ParticleReal Bzp = Bz_external_particle;

        if(!t_do_not_gather){
            // first gather E and B to the particle positions
            doGatherShapeN(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                           ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                           ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                           dinv, xyzmin, lo, n_rz_azimuthal_modes,
                           nox, galerkin_interpolation);
        }

        [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
        if constexpr (exteb_control == has_exteb) {
            getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
        }

        scaleFields(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp);

#ifdef WARPX_QED
        if (!do_sync)
#endif
        {
            if (do_copy) {
                //  Copy the old x and u for the BTD
                copyAttribs(ip);
            }

            doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                      Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                      ion_lev ? ion_lev[ip] : 1,
                                      m, q, pusher_algo, do_crr,
#ifdef WARPX_QED
                                      t_chi_max,
#endif
                                      dt);

            UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt);
            setPosition(ip, xp, yp, zp);
        }
#ifdef WARPX_QED
        else {
            if constexpr (qed_control == has_qed) {
                if (do_copy) {
                    //  Copy the old x and u for the BTD
                    copyAttribs(ip);
                }

                doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                          Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                          ion_lev ? ion_lev[ip] : 1,
                                          m, q, pusher_algo, do_crr,
                                          t_chi_max,
                                          dt);

                UpdatePosition(xp, yp, zp, ux[ip], uy[ip], uz[ip], dt);
                setPosition(ip, xp, yp, zp);
            }
        }
#endif

#ifdef WARPX_QED
        [[maybe_unused]] auto foo_local_has_quantum_sync = local_has_quantum_sync;
        [[maybe_unused]] auto *foo_podq = p_optical_depth_QSR;
        [[maybe_unused]] const auto& foo_evolve_opt = evolve_opt; // have to do all these for nvcc
        if constexpr (qed_control == has_qed) {
            if (local_has_quantum_sync) {
                evolve_opt(ux[ip], uy[ip], uz[ip],
                           Exp, Eyp, Ezp,Bxp, Byp, Bzp,
                           dt, p_optical_depth_QSR[ip]);
            }
        }
#else
            amrex::ignore_unused(qed_control);
#endif
    });
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

using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;

inline void Mvec_compute_shape_factor_part_vpu_order3(double* sx, MVec x, svbool_t p) __arc_preserves("za") __arc_streaming
{
    constexpr int VEC_LEN = 8;

    MVec j = svrintz_x(p, x);
    MVec xint = x - j;
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

/* @brief Double max_bin_length.
*/
void increment_rebuild(ParticleTileType& ptile, Box& box)
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

    const Dim3 len = length(box);
    int lenx = len.x;
    int leny = len.y;
    int lenz = len.z;

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

void increment_sort(ParticleTileType& ptile, int* newbin, long np, Box& real_box, Box& box)
{
    if (!ptile.is_incr_sort_initialized)
    {
        ptile.init_incr_sort(real_box, box);
    }

    const Dim3 len = length(box);
    int lenx = len.x;
    int leny = len.y;
    int lenz = len.z;

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

    // Stage 2 & 3.1: Identify moved existing particles and delete from old cell
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

    // Traverse particles in d_local_index within realbox; they may have moved
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

                // oldbin_offset == -1 means particles are outside realbox
                if (oldbin_offset == -1)
                {
                    continue;
                }

                int* invalid_idx = WarpX::GetInstance().invalid_idx + thread_num * m_init_np;
                int invalid_num = 0;

                // Remove invalid particles, move changed particles into pending_moves,
                // and track removed indices in invalid_idx
                for (int i = 0; i < oldbin_length; i += vlf)
                {
                    // Step 1: load data
                    svbool_t p_ip = svwhilelt_b32(i, oldbin_length);
                    int idx = oldbin_offset + i;
                    svint32_t idx_v = svindex_s32(idx, 1);
                    svint32_t ip_v = svld1_s32(p_ip, &d_local_index[idx]);
                    
                    // Step 2: identify invalid and moved particles
                    // Mark invalid particles: ip >= np
                    svbool_t ip_invalid = svcmpge_s32(p_ip, ip_v, np_v);
                    
                    // Identify moved particles among valid ones: newbin[ip] != old_bin
                    svbool_t ip_valid = svbic_b_z(p_ip, p_ip, ip_invalid);
                    svint32_t newbin_v = svld1_gather_s32index_s32(ip_valid, newbin, ip_v);
                    svbool_t ip_move = svcmpne_s32(ip_valid, newbin_v, old_bin_v);
                    
                    // Merge conditions: particles to remove from current bin
                    // (invalid particles OR moved particles)
                    svbool_t ip_to_remove = svorr_b_z(p_ip, ip_invalid, ip_move);

                    // Step 3: process moved particles
                    // Push indices of valid moved particles into pending_moves
                    int32_t ip_move_num = svcntp_b32(ip_valid, ip_move);
                    if (ip_move_num > 0)
                    {
                        svbool_t p_move = svwhilelt_b32(0, ip_move_num);
                        svint32_t move_ips_v = svcompact(ip_move, ip_v);
                        svst1_s32(p_move, pending_moves + pending_moves_num, move_ips_v);
                        pending_moves_num += ip_move_num;
                    }

                    // Step 4: mark positions to remove
                    // Store all removal positions in invalid_idx and mark invalid
                    int32_t ip_to_remove_num = svcntp_b32(p_ip, ip_to_remove);
                    if (ip_to_remove_num > 0)
                    {
                        svbool_t p_remove = svwhilelt_b32(0, ip_to_remove_num);
                        svint32_t remove_idx_v = svcompact(ip_to_remove, idx_v);
                        
                        // Record indices to remove
                        svst1_s32(p_remove, invalid_idx + invalid_num, remove_idx_v);
                        invalid_num += ip_to_remove_num;
                        
                        // Mark these positions with invalid particle IDs
                        svst1_scatter_s32index_s32(p_remove, d_local_index.data(), remove_idx_v, invalid_particle_id_v);
                    }
                }


                // Back-fill invalid positions from tail
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


                oldbin_length = oldbin_tail + 1 - oldbin_offset;
            }
        }
    }
    
    d_old_np = np;

    d_old_outside_index.clear();
    vector<vector<int>> outside_bin_ip(num_bins);

    // Stage 3.2: Insert moved particles into target bins
    for (int i = 0; i < pending_moves_num; ++i)
    {
        int ip = pending_moves[i];
        int newbin_ip = newbin[ip];

        if (ptile.d_incr_bin_offset[newbin_ip] == -1)
        {
            printf("ip: %d, newbin_ip: %d\n", ip, newbin_ip);
            amrex::Abort("[2]GOTO Deposit");
            d_old_outside_index.push_back(ip);
            outside_bin_ip[newbin_ip].push_back(ip);
        }
        else
        {
            if (ptile.d_incr_bin_length[newbin_ip] >= ptile.max_bin_length)
            {
                increment_rebuild(ptile, box);
            }

            int& newbin_offset = ptile.d_incr_bin_offset[newbin_ip];
            int& newbin_length = ptile.d_incr_bin_length[newbin_ip];
            int newbin_tail = newbin_offset + newbin_length;

            ptile.d_local_index[newbin_tail] = ip;
            newbin_length++;
        }
    }

}

__arc_new("za") inline void PushPX_vpu_mpu_physort_order3_kernel(
    amrex::Real* aos_arr,
    amrex::ParticleReal Ex_external_particle,
    amrex::ParticleReal Ey_external_particle,
    amrex::ParticleReal Ez_external_particle,
    amrex::ParticleReal Bx_external_particle,
    amrex::ParticleReal By_external_particle,
    amrex::ParticleReal Bz_external_particle,
    amrex::ParticleReal* const wp,
    amrex::ParticleReal* const ux,
    amrex::ParticleReal* const uy,
    amrex::ParticleReal* const uz,
    amrex::ParticleReal* m_x,
    amrex::ParticleReal* m_y,
    amrex::ParticleReal* m_z,
    const amrex::ParticleReal& econst,
    const amrex::Real& dt,
    const amrex::XDim3& xyzmin,
    const amrex::XDim3& dinv,
    int lenx,
    int leny,
    int lenz,
    long np_to_push,
    ParticleTileType& ptile,
    vector<vector<int>>& bin_to_ip,
    ParticleReal* const mx_buffer_ptr,
    ParticleReal* const my_buffer_ptr,
    ParticleReal* const mz_buffer_ptr,
    ParticleReal* const ux_buffer_ptr,
    ParticleReal* const uy_buffer_ptr,
    ParticleReal* const uz_buffer_ptr,
    ParticleReal* const w_buffer_ptr
    ) __arc_streaming
{
    constexpr int nshapes = 3 + 1;
    int lenxy = lenx * leny;
    svbool_t p_0_5 = svwhilelt_b64(0, 6);
    MVec vzero(0);

    constexpr amrex::ParticleReal inv_c2 = 1._prt / (PhysConst::c * PhysConst::c);

    int vl = svcntd();

    amrex::Real sx_m[32];
    amrex::Real sy_m[32];
    amrex::Real sz_m[32];

    int& g_move_begin = ptile.g_move_begin;
    std::vector<int>& g_phys_bin_offset = ptile.g_phys_bin_offset;
    std::vector<int>& g_phys_bin_length = ptile.g_phys_bin_length;

    int nomove_idx = 0;
    int move_idx = np_to_push;

    for (int l_node = 0; l_node < lenz; ++l_node)
    {
        for (int k_node = 0; k_node < leny; ++k_node)
        {
            for (int j_node = 0; j_node < lenx; ++j_node)
            {
                int old_bin = j_node + k_node * lenx + l_node * lenx * leny;
                int& offset = g_phys_bin_offset[old_bin];
                int& binlength = g_phys_bin_length[old_bin];
                MintVec oldbin_v(old_bin);

                int new_binlength = 0;

                for (int k = 0; k < binlength; k += vl)
                {
                    svbool_t p_ip = svwhilelt_b64(k, binlength);
                    int base = offset + k;
                    MVec xp_v = MVec::Load(p_ip, m_x + base);
                    MVec yp_v = MVec::Load(p_ip, m_y + base);
                    MVec zp_v = MVec::Load(p_ip, m_z + base);

                    const MVec x = (xp_v - xyzmin.x) * dinv.x;
                    const MVec y = (yp_v - xyzmin.y) * dinv.y;
                    const MVec z = (zp_v - xyzmin.z) * dinv.z;

                    Mvec_compute_shape_factor_part_vpu_order3(sx_m, x, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(sy_m, y, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(sz_m, z, p_ip);

                    MVec Exp_v(Ex_external_particle);
                    MVec Eyp_v(Ey_external_particle);
                    MVec Ezp_v(Ez_external_particle);
                    MVec Bzp_v(Bz_external_particle);
                    MVec Byp_v(By_external_particle);
                    MVec Bxp_v(Bx_external_particle);
                 
                    svzero_za();

                    for (int iz = 0; iz < 4; ++iz)
                    {
                        MVec sz_m_v = MVec::Load(p_ip, &sz_m[iz * 8]);
                        
                        for (int iy = 0; iy < 4; iy++)
                        {
                            MVec sy_m_v = MVec::Load(p_ip, &sy_m[iy * 8]);
                            for (int ix = 0; ix < 4; ix++)
                            {
                                MVec sx_m_v = MVec::Load(p_ip, &sx_m[ix * 8]);
                                MVec aos_v = MVec::Load(p_0_5, &aos_arr[6 * (old_bin + ix + iy * lenx + iz * lenx * leny)]);
                                MVec sx_sy_sz_m_v = sx_m_v * sy_m_v * sz_m_v;
                                
                                svmopa_za64_m(0, p_0_5, p_ip, aos_v, sx_sy_sz_m_v);
                            }
                        }
                    }

                    Exp_v += svread_hor_za64_m(vzero, p_ip, 0, 0);
                    Eyp_v += svread_hor_za64_m(vzero, p_ip, 0, 1);
                    Ezp_v += svread_hor_za64_m(vzero, p_ip, 0, 2);
                    Bzp_v += svread_hor_za64_m(vzero, p_ip, 0, 3);
                    Byp_v += svread_hor_za64_m(vzero, p_ip, 0, 4);
                    Bxp_v += svread_hor_za64_m(vzero, p_ip, 0, 5);

                    MVec ux_v = MVec::Load(p_ip, ux + base);
                    MVec uy_v = MVec::Load(p_ip, uy + base);
                    MVec uz_v = MVec::Load(p_ip, uz + base);

                    ux_v += econst * Exp_v;
                    uy_v += econst * Eyp_v;
                    uz_v += econst * Ezp_v;

                    MVec inv_gamma_v = 1. / (1. + (ux_v * ux_v + uy_v * uy_v + uz_v * uz_v) * inv_c2).Sqrt();

                    MVec tx_v = econst * inv_gamma_v * Bxp_v;
                    MVec ty_v = econst * inv_gamma_v * Byp_v;
                    MVec tz_v = econst * inv_gamma_v * Bzp_v;

                    MVec tsqi_v = 2. / (1. + tx_v * tx_v + ty_v * ty_v + tz_v * tz_v);

                    MVec sx_v = tx_v * tsqi_v;
                    MVec sy_v = ty_v * tsqi_v;
                    MVec sz_v = tz_v * tsqi_v;

                    MVec uxp_v = ux_v + uy_v * tz_v - uz_v * ty_v;
                    MVec uyp_v = uy_v + uz_v * tx_v - ux_v * tz_v;
                    MVec uzp_v = uz_v + ux_v * ty_v - uy_v * tx_v;

                    ux_v += uyp_v * sz_v - uzp_v * sy_v;
                    uy_v += uzp_v * sx_v - uxp_v * sz_v;
                    uz_v += uxp_v * sy_v - uyp_v * sx_v;

                    ux_v += econst * Exp_v;
                    uy_v += econst * Eyp_v;
                    uz_v += econst * Ezp_v;

                    // svst1_scatter_index(p_ip, ux, ip_v, ux_v);
                    // svst1_scatter_index(p_ip, uy, ip_v, uy_v);
                    // svst1_scatter_index(p_ip, uz, ip_v, uz_v);                     

                    inv_gamma_v = 1. / (1. + (ux_v * ux_v + uy_v * uy_v + uz_v * uz_v) * inv_c2).Sqrt();

                    xp_v += ux_v * inv_gamma_v * dt;
                    yp_v += uy_v * inv_gamma_v * dt;
                    zp_v += uz_v * inv_gamma_v * dt;
                    
                    MVec wp_v = MVec::Load(p_ip, wp + base);

                    // svst1_scatter_index(p_ip, m_x, ip_v, xp_v);
                    // svst1_scatter_index(p_ip, m_y, ip_v, yp_v);
                    // svst1_scatter_index(p_ip, m_z, ip_v, zp_v);

                    MVec x_new = (xp_v - xyzmin.x) * dinv.x;
                    MintVec j_nodev = svcvt_s64_f64_z(p_ip, x_new) - 1;
            
                    MVec y_new = (yp_v - xyzmin.y) * dinv.y;
                    MintVec k_nodev = svcvt_s64_f64_z(p_ip, y_new) - 1;
            
                    MVec z_new = (zp_v - xyzmin.z) * dinv.z;
                    MintVec l_nodev = svcvt_s64_f64_z(p_ip, z_new) - 1;
            
                    MintVec newbin_v = j_nodev + k_nodev * lenx + l_nodev * lenx * leny;
                    svbool_t ip_move = svcmpne_s64(p_ip, newbin_v, oldbin_v);
                    svbool_t ip_nomove = svnot_b_z(p_ip, ip_move);

                    uint64_t nomove_num = svcntp_b64(p_ip, ip_nomove);
                    uint64_t move_num = svcntp_b64(p_ip, ip_move);

                    int particles_num = binlength - k < 8 ? binlength - k : 8;
                    if (nomove_num == particles_num)
                    {
                        ux_v.Store(p_ip, ux_buffer_ptr + nomove_idx);
                        uy_v.Store(p_ip, uy_buffer_ptr + nomove_idx);
                        uz_v.Store(p_ip, uz_buffer_ptr + nomove_idx);
                        xp_v.Store(p_ip, mx_buffer_ptr + nomove_idx);
                        yp_v.Store(p_ip, my_buffer_ptr + nomove_idx);
                        zp_v.Store(p_ip, mz_buffer_ptr + nomove_idx);
                        wp_v.Store(p_ip, w_buffer_ptr + nomove_idx);

                        nomove_idx += nomove_num;
                    }
                    else if(move_num == particles_num)
                    {
                        move_idx -= move_num;
                        ux_v.Store(p_ip, ux_buffer_ptr + move_idx);
                        uy_v.Store(p_ip, uy_buffer_ptr + move_idx);
                        uz_v.Store(p_ip, uz_buffer_ptr + move_idx);
                        xp_v.Store(p_ip, mx_buffer_ptr + move_idx);
                        yp_v.Store(p_ip, my_buffer_ptr + move_idx);
                        zp_v.Store(p_ip, mz_buffer_ptr + move_idx);
                        wp_v.Store(p_ip, w_buffer_ptr + move_idx);
                    }
                    else
                    {
                        MVec ux_nomove_v = svcompact_f64(ip_nomove, ux_v);
                        MVec uy_nomove_v = svcompact_f64(ip_nomove, uy_v);
                        MVec uz_nomove_v = svcompact_f64(ip_nomove, uz_v);
                        MVec xp_nomove_v = svcompact_f64(ip_nomove, xp_v);
                        MVec yp_nomove_v = svcompact_f64(ip_nomove, yp_v);
                        MVec zp_nomove_v = svcompact_f64(ip_nomove, zp_v);
                        MVec wp_nomove_v = svcompact_f64(ip_nomove, wp_v);
    
                        MVec ux_move_v = svcompact_f64(ip_move, ux_v);
                        MVec uy_move_v = svcompact_f64(ip_move, uy_v);
                        MVec uz_move_v = svcompact_f64(ip_move, uz_v);
                        MVec xp_move_v = svcompact_f64(ip_move, xp_v);
                        MVec yp_move_v = svcompact_f64(ip_move, yp_v);
                        MVec zp_move_v = svcompact_f64(ip_move, zp_v);
                        MVec wp_move_v = svcompact_f64(ip_move, wp_v);
    
                        svbool_t p_nomove = svwhilelt_b64(0ULL, nomove_num);
                        svbool_t p_move = svwhilelt_b64(0ULL, move_num);
    
                        ux_nomove_v.Store(p_nomove, ux_buffer_ptr + nomove_idx);
                        uy_nomove_v.Store(p_nomove, uy_buffer_ptr + nomove_idx);
                        uz_nomove_v.Store(p_nomove, uz_buffer_ptr + nomove_idx);
                        xp_nomove_v.Store(p_nomove, mx_buffer_ptr + nomove_idx);
                        yp_nomove_v.Store(p_nomove, my_buffer_ptr + nomove_idx);
                        zp_nomove_v.Store(p_nomove, mz_buffer_ptr + nomove_idx);
                        wp_nomove_v.Store(p_nomove, w_buffer_ptr + nomove_idx);

                        nomove_idx += nomove_num;

                        move_idx -= move_num;
    
                        ux_move_v.Store(p_move, ux_buffer_ptr + move_idx);
                        uy_move_v.Store(p_move, uy_buffer_ptr + move_idx);
                        uz_move_v.Store(p_move, uz_buffer_ptr + move_idx);
                        xp_move_v.Store(p_move, mx_buffer_ptr + move_idx);
                        yp_move_v.Store(p_move, my_buffer_ptr + move_idx);
                        zp_move_v.Store(p_move, mz_buffer_ptr + move_idx);
                        wp_move_v.Store(p_move, w_buffer_ptr + move_idx);
                    }

                    new_binlength += nomove_num;
                }

                const vector<int>& tmp_particle = bin_to_ip[old_bin];
                int tmp_binlength = tmp_particle.size();

                // int inr_binlength = inr_bin_length[old_bin];
                // int inr_binoffset = inr_bin_offsets[old_bin];

                // for (int k = 0; k < inr_binlength; k += vl)
                for (int k = 0; k < tmp_binlength; k += vl)
                {
                    svbool_t p_ip = svwhilelt_b64(k, tmp_binlength);
                    MintVec ip_v = svld1sw_s64(p_ip, &tmp_particle[k]);
                    // svbool_t p_ip = svwhilelt_b64(k, inr_binlength);
                    // MintVec ip_v = svld1sw_s64(p_ip, &local_index[inr_binoffset + k]);

                    MVec xp_v = svld1_gather_s64index_f64(p_ip, m_x, ip_v);
                    MVec yp_v = svld1_gather_s64index_f64(p_ip, m_y, ip_v);
                    MVec zp_v = svld1_gather_s64index_f64(p_ip, m_z, ip_v);
                    const MVec x = (xp_v - xyzmin.x) * dinv.x;
                    const MVec y = (yp_v - xyzmin.y) * dinv.y;
                    const MVec z = (zp_v - xyzmin.z) * dinv.z;

                    Mvec_compute_shape_factor_part_vpu_order3(sx_m, x, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(sy_m, y, p_ip);
                    Mvec_compute_shape_factor_part_vpu_order3(sz_m, z, p_ip);

                    MVec Exp_v(Ex_external_particle);
                    MVec Eyp_v(Ey_external_particle);
                    MVec Ezp_v(Ez_external_particle);
                    MVec Bzp_v(Bz_external_particle);
                    MVec Byp_v(By_external_particle);
                    MVec Bxp_v(Bx_external_particle);
                 
                    svzero_za();

                    for (int iz = 0; iz < 4; ++iz)
                    {
                        MVec sz_m_v = MVec::Load(p_ip, &sz_m[iz * 8]);
                        
                        for (int iy = 0; iy < 4; iy++)
                        {
                            MVec sy_m_v = MVec::Load(p_ip, &sy_m[iy * 8]);
                            for (int ix = 0; ix < 4; ix++)
                            {
                                MVec sx_m_v = MVec::Load(p_ip, &sx_m[ix * 8]);
                                MVec aos_v = MVec::Load(p_0_5, &aos_arr[6 * (old_bin + ix + iy * lenx + iz * lenx * leny)]);
                                MVec sx_sy_sz_m_v = sx_m_v * sy_m_v * sz_m_v;
                                
                                svmopa_za64_m(0, p_0_5, p_ip, aos_v, sx_sy_sz_m_v);
                            }
                        }
                    }

                    Exp_v += svread_hor_za64_m(vzero, p_ip, 0, 0);
                    Eyp_v += svread_hor_za64_m(vzero, p_ip, 0, 1);
                    Ezp_v += svread_hor_za64_m(vzero, p_ip, 0, 2);
                    Bzp_v += svread_hor_za64_m(vzero, p_ip, 0, 3);
                    Byp_v += svread_hor_za64_m(vzero, p_ip, 0, 4);
                    Bxp_v += svread_hor_za64_m(vzero, p_ip, 0, 5);

                    MVec ux_v = svld1_gather_s64index_f64(p_ip, ux, ip_v);
                    MVec uy_v = svld1_gather_s64index_f64(p_ip, uy, ip_v);
                    MVec uz_v = svld1_gather_s64index_f64(p_ip, uz, ip_v);

                    ux_v += econst * Exp_v;
                    uy_v += econst * Eyp_v;
                    uz_v += econst * Ezp_v;

                    MVec inv_gamma_v = 1. / (1. + (ux_v * ux_v + uy_v * uy_v + uz_v * uz_v) * inv_c2).Sqrt();

                    MVec tx_v = econst * inv_gamma_v * Bxp_v;
                    MVec ty_v = econst * inv_gamma_v * Byp_v;
                    MVec tz_v = econst * inv_gamma_v * Bzp_v;

                    MVec tsqi_v = 2. / (1. + tx_v * tx_v + ty_v * ty_v + tz_v * tz_v);

                    MVec sx_v = tx_v * tsqi_v;
                    MVec sy_v = ty_v * tsqi_v;
                    MVec sz_v = tz_v * tsqi_v;

                    MVec uxp_v = ux_v + uy_v * tz_v - uz_v * ty_v;
                    MVec uyp_v = uy_v + uz_v * tx_v - ux_v * tz_v;
                    MVec uzp_v = uz_v + ux_v * ty_v - uy_v * tx_v;

                    ux_v += uyp_v * sz_v - uzp_v * sy_v;
                    uy_v += uzp_v * sx_v - uxp_v * sz_v;
                    uz_v += uxp_v * sy_v - uyp_v * sx_v;

                    ux_v += econst * Exp_v;
                    uy_v += econst * Eyp_v;
                    uz_v += econst * Ezp_v;

                    // svst1_scatter_index(p_ip, ux, ip_v, ux_v);
                    // svst1_scatter_index(p_ip, uy, ip_v, uy_v);
                    // svst1_scatter_index(p_ip, uz, ip_v, uz_v);                     

                    inv_gamma_v = 1. / (1. + (ux_v * ux_v + uy_v * uy_v + uz_v * uz_v) * inv_c2).Sqrt();

                    xp_v += ux_v * inv_gamma_v * dt;
                    yp_v += uy_v * inv_gamma_v * dt;
                    zp_v += uz_v * inv_gamma_v * dt;
                    
                    MVec wp_v = svld1_gather_s64index_f64(p_ip, wp, ip_v);

                    // svst1_scatter_index(p_ip, m_x, ip_v, xp_v);
                    // svst1_scatter_index(p_ip, m_y, ip_v, yp_v);
                    // svst1_scatter_index(p_ip, m_z, ip_v, zp_v);

                    MVec x_new = (xp_v - xyzmin.x) * dinv.x;
                    MintVec j_nodev = svcvt_s64_f64_z(p_ip, x_new) - 1;
            
                    MVec y_new = (yp_v - xyzmin.y) * dinv.y;
                    MintVec k_nodev = svcvt_s64_f64_z(p_ip, y_new) - 1;
            
                    MVec z_new = (zp_v - xyzmin.z) * dinv.z;
                    MintVec l_nodev = svcvt_s64_f64_z(p_ip, z_new) - 1;
            
                    MintVec newbin_v = j_nodev + k_nodev * lenx + l_nodev * lenx * leny;
                    svbool_t ip_move = svcmpne_s64(p_ip, newbin_v, oldbin_v);
                    svbool_t ip_nomove = svnot_b_z(p_ip, ip_move);

                    uint64_t nomove_num = svcntp_b64(p_ip, ip_nomove);
                    uint64_t move_num = svcntp_b64(p_ip, ip_move);

                    MVec ux_nomove_v = svcompact_f64(ip_nomove, ux_v);
                    MVec uy_nomove_v = svcompact_f64(ip_nomove, uy_v);
                    MVec uz_nomove_v = svcompact_f64(ip_nomove, uz_v);
                    MVec xp_nomove_v = svcompact_f64(ip_nomove, xp_v);
                    MVec yp_nomove_v = svcompact_f64(ip_nomove, yp_v);
                    MVec zp_nomove_v = svcompact_f64(ip_nomove, zp_v);
                    MVec wp_nomove_v = svcompact_f64(ip_nomove, wp_v);

                    MVec ux_move_v = svcompact_f64(ip_move, ux_v);
                    MVec uy_move_v = svcompact_f64(ip_move, uy_v);
                    MVec uz_move_v = svcompact_f64(ip_move, uz_v);
                    MVec xp_move_v = svcompact_f64(ip_move, xp_v);
                    MVec yp_move_v = svcompact_f64(ip_move, yp_v);
                    MVec zp_move_v = svcompact_f64(ip_move, zp_v);
                    MVec wp_move_v = svcompact_f64(ip_move, wp_v);

                    svbool_t p_nomove = svwhilelt_b64(0ULL, nomove_num);
                    svbool_t p_move = svwhilelt_b64(0ULL, move_num);

                    ux_nomove_v.Store(p_nomove, ux_buffer_ptr + nomove_idx);
                    uy_nomove_v.Store(p_nomove, uy_buffer_ptr + nomove_idx);
                    uz_nomove_v.Store(p_nomove, uz_buffer_ptr + nomove_idx);
                    xp_nomove_v.Store(p_nomove, mx_buffer_ptr + nomove_idx);
                    yp_nomove_v.Store(p_nomove, my_buffer_ptr + nomove_idx);
                    zp_nomove_v.Store(p_nomove, mz_buffer_ptr + nomove_idx);
                    wp_nomove_v.Store(p_nomove, w_buffer_ptr + nomove_idx);

                    nomove_idx += nomove_num;

                    move_idx -= move_num;

                    ux_move_v.Store(p_move, ux_buffer_ptr + move_idx);
                    uy_move_v.Store(p_move, uy_buffer_ptr + move_idx);
                    uz_move_v.Store(p_move, uz_buffer_ptr + move_idx);
                    xp_move_v.Store(p_move, mx_buffer_ptr + move_idx);
                    yp_move_v.Store(p_move, my_buffer_ptr + move_idx);
                    zp_move_v.Store(p_move, mz_buffer_ptr + move_idx);
                    wp_move_v.Store(p_move, w_buffer_ptr + move_idx);

                    new_binlength += nomove_num;
                }

                binlength = new_binlength;
                offset = nomove_idx - binlength;
            }
        }
    }

    g_move_begin = move_idx;
}

void
PhysicalParticleContainer::PushPX_vpu_mpu_physort_order3 (WarpXParIter& pti,
                                   amrex::FArrayBox const * exfab,
                                   amrex::FArrayBox const * eyfab,
                                   amrex::FArrayBox const * ezfab,
                                   amrex::FArrayBox const * bxfab,
                                   amrex::FArrayBox const * byfab,
                                   amrex::FArrayBox const * bzfab,
                                   const amrex::IntVect ngEB, const int /*e_is_nodal*/,
                                   const long offset,
                                   const long np_to_push,
                                   int lev, int gather_lev,
                                   amrex::Real dt, ScaleFields scaleFields,
                                   DtType a_dt_type)
{
    uint64_t init_area = 0;
    uint64_t aos_area = 0;
    uint64_t precompute_area = 0;
    uint64_t sort_area = 0;
    uint64_t calculate_area = 0;
    uint64_t reduce_area = 0;

    uint64_t init_start = rdtscv();
    const int thread_num = omp_get_thread_num();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((gather_lev==(lev-1)) ||
                                     (gather_lev==(lev  )),
                                     "Gather buffers only work for lev-1");

    using RType = amrex::ParticleReal;
    using namespace amrex::literals;
    using namespace amrex;

    if (np_to_push == 0) { return; }
    
    auto& soa = pti.GetStructOfArrays();
    RType* AMREX_RESTRICT m_x = soa.GetRealData(PIdx::x).dataPtr();
    RType* AMREX_RESTRICT m_y = soa.GetRealData(PIdx::y).dataPtr();
    RType* AMREX_RESTRICT m_z = soa.GetRealData(PIdx::z).dataPtr();
    
    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev, 0));

    Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(), ref_ratio);
    }

    box.grow(ngEB);

    const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
    const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
    const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
    const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
    const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
    const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    const Dim3 len = length(box);
    int lenx = len.x;
    int leny = len.y;
    int lenz = len.z;
    int num_bin = lenx * leny * lenz;

    init_area += rdtscv() - init_start;

    uint64_t aos_start = rdtscv();
     
    int m_box_size = WarpX::GetInstance().m_box_size;
    amrex::Real* aos_arr = WarpX::GetInstance().aos_arr + thread_num * 6 * m_box_size;

    aos_area += rdtscv() - aos_start;

    init_start = rdtscv();
    auto& attribs = pti.GetAttribs();
    ParticleReal* const AMREX_RESTRICT wp = attribs[PIdx::w].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

    auto& mx_buffer = WarpX::GetInstance().thread_private_mx_buffer_arr[thread_num];
    auto& my_buffer = WarpX::GetInstance().thread_private_my_buffer_arr[thread_num];
    auto& mz_buffer = WarpX::GetInstance().thread_private_mz_buffer_arr[thread_num];
    auto& ux_buffer = WarpX::GetInstance().thread_private_ux_buffer_arr[thread_num];
    auto& uy_buffer = WarpX::GetInstance().thread_private_uy_buffer_arr[thread_num];
    auto& uz_buffer = WarpX::GetInstance().thread_private_uz_buffer_arr[thread_num];
    auto& w_buffer = WarpX::GetInstance().thread_private_w_buffer_arr[thread_num];
    mx_buffer.resize(np_to_push);
    my_buffer.resize(np_to_push);
    mz_buffer.resize(np_to_push);
    ux_buffer.resize(np_to_push);
    uy_buffer.resize(np_to_push);
    uz_buffer.resize(np_to_push);
    w_buffer.resize(np_to_push);
        
    ParticleReal* const AMREX_RESTRICT mx_buffer_ptr = mx_buffer.dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT my_buffer_ptr = my_buffer.dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT mz_buffer_ptr = mz_buffer.dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT ux_buffer_ptr = ux_buffer.dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy_buffer_ptr = uy_buffer.dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz_buffer_ptr = uz_buffer.dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT w_buffer_ptr = w_buffer.dataPtr() + offset;

    const amrex::ParticleReal q = this->charge;
    const amrex::ParticleReal m = this->mass;

    const amrex::ParticleReal econst = 0.5_prt * q * dt / m;

    int vl = svcntd();

    vector<vector<int>> bin_to_ip(num_bin);
    long tmpbin[8];
    // init_area += rdtscv() - init_start;

    auto& ptile = ParticlesAt(lev, pti);
    ptile.init_phys_sort(box);
    int& g_move_begin = ptile.g_move_begin;
    
    for (long ip = g_move_begin; ip < np_to_push; ip += vl)
    {
        uint64_t precompute_start = rdtscv();
        svbool_t p_ip = svwhilelt_b64(ip, np_to_push);
        Vec xp_v = Vec::Load(p_ip, &m_x[ip]);
        Vec yp_v = Vec::Load(p_ip, &m_y[ip]);
        Vec zp_v = Vec::Load(p_ip, &m_z[ip]);

        const Vec x = (xp_v - xyzmin.x) * dinv.x;
        intVec j_nodev = svcvt_s64_f64_z(p_ip, x) - 1;

        const Vec y = (yp_v - xyzmin.y) * dinv.y;
        intVec k_nodev = svcvt_s64_f64_z(p_ip, y) - 1;

        const Vec z = (zp_v - xyzmin.z) * dinv.z;
        intVec l_nodev = svcvt_s64_f64_z(p_ip, z) - 1;

        intVec newbin_v = j_nodev + k_nodev * lenx + l_nodev * lenx * leny;
        precompute_area += rdtscv() - precompute_start;
        
        uint64_t sort_begin = rdtscv();
        newbin_v.Store(p_ip, tmpbin);

        int num_particles = svcntp_b64(p_ip, p_ip);
        for (int k = 0; k < num_particles; ++k)
        {
            bin_to_ip[tmpbin[k]].push_back((int)(ip + k));
        }
        sort_area += rdtscv() - sort_begin;
    }

    uint64_t calculate_start = rdtscv();
    PushPX_vpu_mpu_physort_order3_kernel(
        aos_arr, 
        Ex_external_particle, Ey_external_particle, Ez_external_particle,
        Bx_external_particle, By_external_particle, Bz_external_particle, 
        wp, ux, uy, uz,
        m_x, m_y, m_z,
        econst, dt,
        xyzmin, dinv,
        lenx, leny, lenz, np_to_push,
        ptile, bin_to_ip,
        mx_buffer_ptr, my_buffer_ptr, mz_buffer_ptr, ux_buffer_ptr, uy_buffer_ptr, uz_buffer_ptr, w_buffer_ptr 
    );
    calculate_area += rdtscv() - calculate_start;

    // Swap pointers after completion
    uint64_t reduce_start = rdtscv();
    attribs[PIdx::x].swap(mx_buffer);
    attribs[PIdx::y].swap(my_buffer);
    attribs[PIdx::z].swap(mz_buffer);
    attribs[PIdx::ux].swap(ux_buffer);
    attribs[PIdx::uy].swap(uy_buffer);
    attribs[PIdx::uz].swap(uz_buffer);
    attribs[PIdx::w].swap(w_buffer);
    reduce_area += rdtscv() - reduce_start;

    amrex::ignore_unused(init_area, aos_area, precompute_area, sort_area, calculate_area, reduce_area);
}

bool
PhysicalParticleContainer::my_locateParticle (WarpXParIter& pti,
                                   ParticleLocData& pld,
                                   Particle<0, 0>& p_prime,
                                   int lev, int local_grid, int* is_pml)
{
    const Geometry& geom = Geom(0);
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    const auto rlo = geom.ProbLoArrayInParticleReal();
    const auto rhi = geom.ProbHiArrayInParticleReal();
    const auto is_per = geom.isPeriodicArray();

    ParticleReal& xp = p_prime.pos(0);
    ParticleReal& yp = p_prime.pos(1);
    ParticleReal& zp = p_prime.pos(2);

    bool is_outside_geom = (xp < rlo[0] || xp > rhi[0] || yp < rlo[1] || yp > rhi[1] || zp < rlo[2] || zp > rhi[2]);

    const std::string& sname = species_name;
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = WarpX::GetInstance().warpx_comm_particle_container.at(sname).warpx_comm_comp;
    *is_pml = 0;

    if (is_outside_geom) {
        if (geom.isAnyPeriodic())
        {
            bool shift_success = false;
            for (int idim = 0; idim < 3; ++idim)
            {
                if (!is_per[idim]) continue;
                if (p_prime.pos(idim) > rhi[idim]) {
                    while (p_prime.pos(idim) > rhi[idim]) {
                        p_prime.pos(idim) -= static_cast<ParticleReal>(phi[idim] - plo[idim]);
                    }
                    if (p_prime.pos(idim) < rlo[idim]) {
                        p_prime.pos(idim) = rlo[idim];
                    }
                    shift_success = true;
                }
                else if (p_prime.pos(idim) < rlo[idim]) {
                    while (p_prime.pos(idim) < rlo[idim]) {
                        p_prime.pos(idim) += static_cast<ParticleReal>(phi[idim] - plo[idim]);
                    }
                    if (p_prime.pos(idim) > rhi[idim]) {
                        p_prime.pos(idim) = rhi[idim];
                    }
                    shift_success = true;
                }
            }
    
            if (!shift_success)
            {
                amrex::Abort("Particle is still outside the domain after periodic shift");
            }            
    
            std::vector< std::pair<int,Box> > isects;
            int grid;
            IntVect iv;
            const BoxArray& ba = ParticleBoxArray(lev);
    
            iv = Index<amrex::Particle<0, 0>, DefaultAssignor>(p_prime, lev);
            ba.intersections(Box(iv, iv), isects, true, 0);
            grid = isects.empty() ? -1 : isects[0].first;
    
            if (grid >= 0)
            {
    
                const Box& bx = ba.getCellCenteredBox(grid);
    
                pld.m_lev = lev;
                pld.m_grid = grid;
                pld.m_tile = getTileIndex(iv, bx, do_tiling, tile_size, pld.m_tilebox);
                pld.m_cell = iv;
                pld.m_gridbox = bx;
                pld.m_grown_gridbox = bx;
            }
            else
            {
                amrex::Abort("Particle is still outside the domain after periodic shift");
            }
        }
        else
        {
            *is_pml = 1;
        }
    }
    else
    {
        bool indomain_success = false;

        std::vector< std::pair<int,Box> > isects;
        const IntVect& iv = Index(p_prime, lev);

        int grid;
        const BoxArray& ba = ParticleBoxArray(lev);

        if (local_grid < 0) {
            amrex::Abort("local_grid < 0");
        }
        else
        {
            std::unique_ptr<iMultiFab>& redistribute_mask_ptr = warpx_comm_comp.redistribute_mask_ptr;
            grid = (*redistribute_mask_ptr)[local_grid](iv, 0);
        }

        if (grid >= 0) {
            const Box& bx = ba.getCellCenteredBox(grid);
            pld.m_lev  = lev;
            pld.m_grid = grid;
            pld.m_tile = getTileIndex(iv, bx, do_tiling, tile_size, pld.m_tilebox);
            pld.m_cell = iv;
            pld.m_gridbox = bx;
            pld.m_grown_gridbox = amrex::grow(bx, 0);
            indomain_success = true;
        }
        else
        {
            amrex::Abort("Particle is still outside the domain after indomain");
        }

        if (!indomain_success)
        {
            amrex::Abort("Particle is still outside the domain after indomain");
        }
    }

    return true;
}

void
PhysicalParticleContainer::fusion_pack(WarpXParIter& pti,
                         amrex::IntVect ngEB,
                         long offset,
                         long np_to_push,
                         int lev, int gather_lev)
{
    BL_PROFILE("MyRedistribute::fusion_pack()");
    const int thread_num = omp_get_thread_num();

    using RType = amrex::ParticleReal;

    auto& soa = pti.GetStructOfArrays();
    RType* AMREX_RESTRICT m_x = soa.GetRealData(PIdx::x).dataPtr();
    RType* AMREX_RESTRICT m_y = soa.GetRealData(PIdx::y).dataPtr();
    RType* AMREX_RESTRICT m_z = soa.GetRealData(PIdx::z).dataPtr();

    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev, 0));

    Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(), ref_ratio);
    }

    box.grow(ngEB);

    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    auto& attribs = pti.GetAttribs();
    ParticleReal* const AMREX_RESTRICT wp = attribs[PIdx::w].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

    int vl = svcntd();

    auto& ptile = ParticlesAt(lev, pti);
    int& g_move_begin = ptile.g_move_begin;

    Box realbox = pti.tilebox();
    const Dim3 reallo = lbound(realbox);
    const Dim3 realhi = ubound(realbox);

    const Dim3 growlo = lbound(box);
    const Dim3 growhi = ubound(box);

    int containlox = reallo.x - growlo.x - 1;
    int containloy = reallo.y - growlo.y - 1;
    int containloz = reallo.z - growlo.z - 1;
    int containhix = realhi.x - growlo.x - 1;
    int containhiy = realhi.y - growlo.y - 1;
    int containhiz = realhi.z - growlo.z - 1;

    int m_init_np = WarpX::GetInstance().m_init_np;
    int* outside_particles = WarpX::GetInstance().int_buffer + thread_num * m_init_np;
    int outside_particles_index = 0;

    intVec containlox_v(containlox);
    intVec containloy_v(containloy);
    intVec containloz_v(containloz);
    intVec containhix_v(containhix);
    intVec containhiy_v(containhiy);
    intVec containhiz_v(containhiz);

    for (long ip = g_move_begin; ip < np_to_push; ip += vl)
    {

        svbool_t p_ip = svwhilelt_b64(ip, np_to_push);
        Vec xp_v = Vec::Load(p_ip, &m_x[ip]);
        Vec yp_v = Vec::Load(p_ip, &m_y[ip]);
        Vec zp_v = Vec::Load(p_ip, &m_z[ip]);


        Vec x = (xp_v - xyzmin.x) * dinv.x;
        Vec y = (yp_v - xyzmin.y) * dinv.y;
        Vec z = (zp_v - xyzmin.z) * dinv.z;

        intVec j_nodev = svcvt_s64_f64_z(p_ip, x) - 1;
        intVec k_nodev = svcvt_s64_f64_z(p_ip, y) - 1;
        intVec l_nodev = svcvt_s64_f64_z(p_ip, z) - 1;
        

        svbool_t p_is_lox_or_hix = svorr_b_z(p_ip, svcmplt_s64(p_ip, j_nodev, containlox_v), svcmpgt_s64(p_ip, j_nodev, containhix_v));
        svbool_t p_is_loy_or_hiy = svorr_b_z(p_ip, svcmplt_s64(p_ip, k_nodev, containloy_v), svcmpgt_s64(p_ip, k_nodev, containhiy_v));
        svbool_t p_is_loz_or_hiz = svorr_b_z(p_ip, svcmplt_s64(p_ip, l_nodev, containloz_v), svcmpgt_s64(p_ip, l_nodev, containhiz_v));

        svbool_t p_is_outside = svorr_b_z(p_ip, p_is_lox_or_hix, p_is_loy_or_hiy);
        p_is_outside = svorr_b_z(p_ip, p_is_outside, p_is_loz_or_hiz);

        intVec ip_v = svindex_s64(ip, 1);
        intVec ip_outside_v = svcompact_s64(p_is_outside, ip_v);
        uint64_t num_outside = svcntp_b64(p_is_outside, p_is_outside);
        if (num_outside > 0)
        {
            svbool_t p_outside = svwhilelt_b64(0ULL, num_outside);
            svst1w(p_outside, outside_particles + outside_particles_index, ip_outside_v);
            outside_particles_index += num_outside;
        }
    }

    int tail = np_to_push - 1;
    int tail_outside_index = outside_particles_index - 1;
    while (tail_outside_index >= 0 && tail - outside_particles[tail_outside_index] < vl)
    {
        long ip = outside_particles[tail_outside_index];
        ParticleReal m_x_tmp = m_x[ip];
        ParticleReal m_y_tmp = m_y[ip];
        ParticleReal m_z_tmp = m_z[ip];
        ParticleReal ux_tmp = ux[ip];
        ParticleReal uy_tmp = uy[ip];
        ParticleReal uz_tmp = uz[ip];
        ParticleReal wp_tmp = wp[ip];
        
        m_x[ip] = m_x[tail];
        m_y[ip] = m_y[tail];
        m_z[ip] = m_z[tail];
        ux[ip] = ux[tail];
        uy[ip] = uy[tail];
        uz[ip] = uz[tail];
        wp[ip] = wp[tail];

        m_x[tail] = m_x_tmp;
        m_y[tail] = m_y_tmp;
        m_z[tail] = m_z_tmp;
        ux[tail] = ux_tmp;
        uy[tail] = uy_tmp;
        uz[tail] = uz_tmp;
        wp[tail] = wp_tmp;    
        tail--;

        tail_outside_index--;
    }

    while (tail_outside_index >= 0)
    {
        svbool_t p_ip = svwhilele_b64(0ULL, tail_outside_index);
        uint64_t num_outside = svcntp_b64(p_ip, p_ip);

        intVec ip_v = svld1sw_s64(p_ip, outside_particles + tail_outside_index - num_outside + 1);
        Vec m_x_outside_v = svld1_gather_index(p_ip, m_x, ip_v);
        Vec m_y_outside_v = svld1_gather_index(p_ip, m_y, ip_v);
        Vec m_z_outside_v = svld1_gather_index(p_ip, m_z, ip_v);
        Vec ux_outside_v = svld1_gather_index(p_ip, ux, ip_v);
        Vec uy_outside_v = svld1_gather_index(p_ip, uy, ip_v);
        Vec uz_outside_v = svld1_gather_index(p_ip, uz, ip_v);
        Vec wp_outside_v = svld1_gather_index(p_ip, wp, ip_v);
        
        Vec m_x_inside_v = Vec::Load(p_ip, m_x + tail - num_outside + 1);
        Vec m_y_inside_v = Vec::Load(p_ip, m_y + tail - num_outside + 1);
        Vec m_z_inside_v = Vec::Load(p_ip, m_z + tail - num_outside + 1);
        Vec ux_inside_v = Vec::Load(p_ip, ux + tail - num_outside + 1);
        Vec uy_inside_v = Vec::Load(p_ip, uy + tail - num_outside + 1);
        Vec uz_inside_v = Vec::Load(p_ip, uz + tail - num_outside + 1);
        Vec wp_inside_v = Vec::Load(p_ip, wp + tail - num_outside + 1);

        m_x_outside_v.Store(p_ip, m_x + tail - num_outside + 1);
        m_y_outside_v.Store(p_ip, m_y + tail - num_outside + 1);
        m_z_outside_v.Store(p_ip, m_z + tail - num_outside + 1);
        ux_outside_v.Store(p_ip, ux + tail - num_outside + 1);
        uy_outside_v.Store(p_ip, uy + tail - num_outside + 1);
        uz_outside_v.Store(p_ip, uz + tail - num_outside + 1);
        wp_outside_v.Store(p_ip, wp + tail - num_outside + 1);

        tail -= num_outside;
        svst1_scatter_index(p_ip, m_x, ip_v, m_x_inside_v);
        svst1_scatter_index(p_ip, m_y, ip_v, m_y_inside_v);
        svst1_scatter_index(p_ip, m_z, ip_v, m_z_inside_v);
        svst1_scatter_index(p_ip, ux, ip_v, ux_inside_v);
        svst1_scatter_index(p_ip, uy, ip_v, uy_inside_v);
        svst1_scatter_index(p_ip, uz, ip_v, uz_inside_v);
        svst1_scatter_index(p_ip, wp, ip_v, wp_inside_v);

        tail_outside_index -= num_outside;
    }

    int& g_new_particles_begin = ptile.g_new_particles_begin;
    g_new_particles_begin = tail + 1;

    const int MyProc = ParallelContext::MyProcSub();
    const auto local_index = pti.GetPairIndex();
    const int local_grid = local_index.first;
    const int local_tile = local_index.second;

    const int superparticle_size = WarpX::GetInstance().superparticle_size;

    const std::string& sname = species_name;
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = WarpX::GetInstance().warpx_comm_particle_container.at(sname).warpx_comm_comp;
    std::map<int, std::vector< std::vector<char> > >& remote_send_allcomps = warpx_comm_comp.remote_send_allcomps;

    for (long ip = g_new_particles_begin; ip < np_to_push; ip++)
    {
        ParticleLocData pld;

        Particle<0, 0> p_prime;
        p_prime.pos(0) = m_x[ip];
        p_prime.pos(1) = m_y[ip];
        p_prime.pos(2) = m_z[ip];

        ParticleReal& xp = p_prime.pos(0);
        ParticleReal& yp = p_prime.pos(1);
        ParticleReal& zp = p_prime.pos(2);

        int is_pml;
        my_locateParticle(pti, pld, p_prime, lev, local_grid, &is_pml);
        if (is_pml == 1) { continue; }

        const int who = ParallelContext::global_to_local_rank(ParticleDistributionMap(pld.m_lev)[pld.m_grid]);
        if (who == MyProc)
        {
            DefineAndReturnParticleTile(lev, pld.m_grid, pld.m_tile);
            auto& local_recv_ptile = ParticlesAt(lev, pld.m_grid, pld.m_tile);
            std::vector<uint64_t>& local_recv_idcpu = local_recv_ptile.local_recv_idcpu[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_xp = local_recv_ptile.local_recv_xp[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_yp = local_recv_ptile.local_recv_yp[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_zp = local_recv_ptile.local_recv_zp[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_ux = local_recv_ptile.local_recv_ux[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_uy = local_recv_ptile.local_recv_uy[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_uz = local_recv_ptile.local_recv_uz[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_w = local_recv_ptile.local_recv_w[thread_num];
            local_recv_idcpu.push_back(soa.GetIdCPUData()[ip]);
            local_recv_xp.push_back(xp);
            local_recv_yp.push_back(yp);
            local_recv_zp.push_back(zp);
            local_recv_ux.push_back(ux[ip]);
            local_recv_uy.push_back(uy[ip]);
            local_recv_uz.push_back(uz[ip]);
            local_recv_w.push_back(wp[ip]);
        }
        else
        {
            auto& particles_to_send = remote_send_allcomps[who][thread_num];
            auto old_size = particles_to_send.size();
            auto new_size = old_size + superparticle_size;
            particles_to_send.resize(new_size);
            
            char* dst = &particles_to_send[old_size];
            
            std::memcpy(dst, &soa.GetIdCPUData()[ip], sizeof(uint64_t));
            dst += sizeof(uint64_t);
            std::memcpy(dst, &xp, sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &yp, sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &zp, sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &ux[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &uy[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &uz[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &wp[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
        }
    }
}

void
PhysicalParticleContainer::fusion_pack_unr(WarpXParIter& pti,
                         amrex::IntVect ngEB,
                         long offset,
                         long np_to_push,
                         int lev, int gather_lev)
{
    BL_PROFILE("MyRedistribute::fusion_pack_unr()");
    const int thread_num = omp_get_thread_num();

    using RType = amrex::ParticleReal;

    auto& soa = pti.GetStructOfArrays();
    RType* AMREX_RESTRICT m_x = soa.GetRealData(PIdx::x).dataPtr();
    RType* AMREX_RESTRICT m_y = soa.GetRealData(PIdx::y).dataPtr();
    RType* AMREX_RESTRICT m_z = soa.GetRealData(PIdx::z).dataPtr();

    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev, 0));

    Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(), ref_ratio);
    }

    box.grow(ngEB);

    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    auto& attribs = pti.GetAttribs();
    ParticleReal* const AMREX_RESTRICT wp = attribs[PIdx::w].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

    int vl = svcntd();

    auto& ptile = ParticlesAt(lev, pti);
    int& g_move_begin = ptile.g_move_begin;

    Box realbox = pti.tilebox();
    const Dim3 reallo = lbound(realbox);
    const Dim3 realhi = ubound(realbox);

    const Dim3 growlo = lbound(box);
    const Dim3 growhi = ubound(box);

    int containlox = reallo.x - growlo.x - 1;
    int containloy = reallo.y - growlo.y - 1;
    int containloz = reallo.z - growlo.z - 1;
    int containhix = realhi.x - growlo.x - 1;
    int containhiy = realhi.y - growlo.y - 1;
    int containhiz = realhi.z - growlo.z - 1;

    int m_init_np = WarpX::GetInstance().m_init_np;
    int* outside_particles = WarpX::GetInstance().int_buffer + thread_num * m_init_np;
    int outside_particles_index = 0;

    intVec containlox_v(containlox);
    intVec containloy_v(containloy);
    intVec containloz_v(containloz);
    intVec containhix_v(containhix);
    intVec containhiy_v(containhiy);
    intVec containhiz_v(containhiz);

    for (long ip = g_move_begin; ip < np_to_push; ip += vl)
    {

        svbool_t p_ip = svwhilelt_b64(ip, np_to_push);
        Vec xp_v = Vec::Load(p_ip, &m_x[ip]);
        Vec yp_v = Vec::Load(p_ip, &m_y[ip]);
        Vec zp_v = Vec::Load(p_ip, &m_z[ip]);


        Vec x = (xp_v - xyzmin.x) * dinv.x;
        Vec y = (yp_v - xyzmin.y) * dinv.y;
        Vec z = (zp_v - xyzmin.z) * dinv.z;

        intVec j_nodev = svcvt_s64_f64_z(p_ip, x) - 1;
        intVec k_nodev = svcvt_s64_f64_z(p_ip, y) - 1;
        intVec l_nodev = svcvt_s64_f64_z(p_ip, z) - 1;
        

        svbool_t p_is_lox_or_hix = svorr_b_z(p_ip, svcmplt_s64(p_ip, j_nodev, containlox_v), svcmpgt_s64(p_ip, j_nodev, containhix_v));
        svbool_t p_is_loy_or_hiy = svorr_b_z(p_ip, svcmplt_s64(p_ip, k_nodev, containloy_v), svcmpgt_s64(p_ip, k_nodev, containhiy_v));
        svbool_t p_is_loz_or_hiz = svorr_b_z(p_ip, svcmplt_s64(p_ip, l_nodev, containloz_v), svcmpgt_s64(p_ip, l_nodev, containhiz_v));

        svbool_t p_is_outside = svorr_b_z(p_ip, p_is_lox_or_hix, p_is_loy_or_hiy);
        p_is_outside = svorr_b_z(p_ip, p_is_outside, p_is_loz_or_hiz);

        intVec ip_v = svindex_s64(ip, 1);
        intVec ip_outside_v = svcompact_s64(p_is_outside, ip_v);
        uint64_t num_outside = svcntp_b64(p_is_outside, p_is_outside);
        if (num_outside > 0)
        {
            svbool_t p_outside = svwhilelt_b64(0ULL, num_outside);
            svst1w(p_outside, outside_particles + outside_particles_index, ip_outside_v);
            outside_particles_index += num_outside;
        }
    }

    int tail = np_to_push - 1;
    int tail_outside_index = outside_particles_index - 1;
    while (tail_outside_index >= 0 && tail - outside_particles[tail_outside_index] < vl)
    {
        long ip = outside_particles[tail_outside_index];
        ParticleReal m_x_tmp = m_x[ip];
        ParticleReal m_y_tmp = m_y[ip];
        ParticleReal m_z_tmp = m_z[ip];
        ParticleReal ux_tmp = ux[ip];
        ParticleReal uy_tmp = uy[ip];
        ParticleReal uz_tmp = uz[ip];
        ParticleReal wp_tmp = wp[ip];
        
        m_x[ip] = m_x[tail];
        m_y[ip] = m_y[tail];
        m_z[ip] = m_z[tail];
        ux[ip] = ux[tail];
        uy[ip] = uy[tail];
        uz[ip] = uz[tail];
        wp[ip] = wp[tail];

        m_x[tail] = m_x_tmp;
        m_y[tail] = m_y_tmp;
        m_z[tail] = m_z_tmp;
        ux[tail] = ux_tmp;
        uy[tail] = uy_tmp;
        uz[tail] = uz_tmp;
        wp[tail] = wp_tmp;    
        tail--;

        tail_outside_index--;
    }

    while (tail_outside_index >= 0)
    {
        svbool_t p_ip = svwhilele_b64(0ULL, tail_outside_index);
        uint64_t num_outside = svcntp_b64(p_ip, p_ip);

        intVec ip_v = svld1sw_s64(p_ip, outside_particles + tail_outside_index - num_outside + 1);
        Vec m_x_outside_v = svld1_gather_index(p_ip, m_x, ip_v);
        Vec m_y_outside_v = svld1_gather_index(p_ip, m_y, ip_v);
        Vec m_z_outside_v = svld1_gather_index(p_ip, m_z, ip_v);
        Vec ux_outside_v = svld1_gather_index(p_ip, ux, ip_v);
        Vec uy_outside_v = svld1_gather_index(p_ip, uy, ip_v);
        Vec uz_outside_v = svld1_gather_index(p_ip, uz, ip_v);
        Vec wp_outside_v = svld1_gather_index(p_ip, wp, ip_v);
        
        Vec m_x_inside_v = Vec::Load(p_ip, m_x + tail - num_outside + 1);
        Vec m_y_inside_v = Vec::Load(p_ip, m_y + tail - num_outside + 1);
        Vec m_z_inside_v = Vec::Load(p_ip, m_z + tail - num_outside + 1);
        Vec ux_inside_v = Vec::Load(p_ip, ux + tail - num_outside + 1);
        Vec uy_inside_v = Vec::Load(p_ip, uy + tail - num_outside + 1);
        Vec uz_inside_v = Vec::Load(p_ip, uz + tail - num_outside + 1);
        Vec wp_inside_v = Vec::Load(p_ip, wp + tail - num_outside + 1);

        m_x_outside_v.Store(p_ip, m_x + tail - num_outside + 1);
        m_y_outside_v.Store(p_ip, m_y + tail - num_outside + 1);
        m_z_outside_v.Store(p_ip, m_z + tail - num_outside + 1);
        ux_outside_v.Store(p_ip, ux + tail - num_outside + 1);
        uy_outside_v.Store(p_ip, uy + tail - num_outside + 1);
        uz_outside_v.Store(p_ip, uz + tail - num_outside + 1);
        wp_outside_v.Store(p_ip, wp + tail - num_outside + 1);

        tail -= num_outside;
        svst1_scatter_index(p_ip, m_x, ip_v, m_x_inside_v);
        svst1_scatter_index(p_ip, m_y, ip_v, m_y_inside_v);
        svst1_scatter_index(p_ip, m_z, ip_v, m_z_inside_v);
        svst1_scatter_index(p_ip, ux, ip_v, ux_inside_v);
        svst1_scatter_index(p_ip, uy, ip_v, uy_inside_v);
        svst1_scatter_index(p_ip, uz, ip_v, uz_inside_v);
        svst1_scatter_index(p_ip, wp, ip_v, wp_inside_v);

        tail_outside_index -= num_outside;
    }

    int& g_new_particles_begin = ptile.g_new_particles_begin;
    g_new_particles_begin = tail + 1;

    const int MyProc = ParallelContext::MyProcSub();
    const auto local_index = pti.GetPairIndex();
    const int local_grid = local_index.first;
    const int local_tile = local_index.second;

    const int superparticle_size = WarpX::GetInstance().superparticle_size;
    std::string& sname = species_name;
    WarpX::WarpX_COMM_ParticleContainer& warpx_comm_particle_container = WarpX::GetInstance().warpx_comm_particle_container.at(sname);
    WarpX::WarpX_COMM_Comp& warpx_comm_comp = warpx_comm_particle_container.warpx_comm_comp;
    WarpX::UNR_WarpX_buffer& unr_send_buffer = warpx_comm_particle_container.unr_send_buffer;

    for (long ip = g_new_particles_begin; ip < np_to_push; ip++)
    {
        ParticleLocData pld;

        Particle<0, 0> p_prime;
        p_prime.pos(0) = m_x[ip];
        p_prime.pos(1) = m_y[ip];
        p_prime.pos(2) = m_z[ip];

        ParticleReal& xp = p_prime.pos(0);
        ParticleReal& yp = p_prime.pos(1);
        ParticleReal& zp = p_prime.pos(2);

        int is_pml;
        my_locateParticle(pti, pld, p_prime, lev, local_grid, &is_pml);
        if (is_pml == 1) { continue; }

        const int who = ParallelContext::global_to_local_rank(ParticleDistributionMap(pld.m_lev)[pld.m_grid]);
        if (who == MyProc)
        {
            DefineAndReturnParticleTile(lev, pld.m_grid, pld.m_tile);
            auto& local_recv_ptile = ParticlesAt(lev, pld.m_grid, pld.m_tile);
            std::vector<uint64_t>& local_recv_idcpu = local_recv_ptile.local_recv_idcpu[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_xp = local_recv_ptile.local_recv_xp[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_yp = local_recv_ptile.local_recv_yp[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_zp = local_recv_ptile.local_recv_zp[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_ux = local_recv_ptile.local_recv_ux[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_uy = local_recv_ptile.local_recv_uy[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_uz = local_recv_ptile.local_recv_uz[thread_num];
            std::vector<amrex::ParticleReal>& local_recv_w = local_recv_ptile.local_recv_w[thread_num];
            local_recv_idcpu.push_back(soa.GetIdCPUData()[ip]);
            local_recv_xp.push_back(xp);
            local_recv_yp.push_back(yp);
            local_recv_zp.push_back(zp);
            local_recv_ux.push_back(ux[ip]);
            local_recv_uy.push_back(uy[ip]);
            local_recv_uz.push_back(uz[ip]);
            local_recv_w.push_back(wp[ip]);
        }
        else
        {
            size_t old_size = unr_send_buffer.size_who(who, thread_num, sname);
            size_t new_size = old_size + superparticle_size;
            unr_send_buffer.resize_who(who, thread_num, new_size, sname);

            char* dst = (char*)unr_send_buffer.get_who_data_buffer(who, thread_num, sname) + old_size;
            
            std::memcpy(dst, &soa.GetIdCPUData()[ip], sizeof(uint64_t));
            dst += sizeof(uint64_t);
            std::memcpy(dst, &xp, sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &yp, sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &zp, sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &ux[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &uy[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &uz[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
            std::memcpy(dst, &wp[ip], sizeof(amrex::ParticleReal));
            dst += sizeof(amrex::ParticleReal);
        }
    }
}

/* \brief Perform the implicit particle push operation in one fused kernel
 *        The main difference from PushPX is the order of operations:
 *         - push position by 1/2 dt
 *         - gather fields
 *         - push velocity by dt
 *         - average old and new velocity to get time centered value
 *        The routines ends with both position and velocity at the half time level.
 */
void
PhysicalParticleContainer::ImplicitPushXP (WarpXParIter& pti,
                                           amrex::FArrayBox const * exfab,
                                           amrex::FArrayBox const * eyfab,
                                           amrex::FArrayBox const * ezfab,
                                           amrex::FArrayBox const * bxfab,
                                           amrex::FArrayBox const * byfab,
                                           amrex::FArrayBox const * bzfab,
                                           amrex::IntVect ngEB, int /*e_is_nodal*/,
                                           long offset,
                                           long np_to_push,
                                           int lev, int gather_lev,
                                           amrex::Real dt, ScaleFields scaleFields,
                                           DtType a_dt_type)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE((gather_lev==(lev-1)) ||
                                     (gather_lev==(lev  )),
                                     "Gather buffers only work for lev-1");
    // If no particles, do not do anything
    if (np_to_push == 0) { return; }

    // Get cell size on gather_lev
    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev,0));

    // Get box from which field is gathered.
    // If not gathering from the finest level, the box is coarsened.
    Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(),ref_ratio);
    }

    // Add guard cells to the box.
    box.grow(ngEB);

    auto setPosition = SetParticlePosition(pti, offset);

    const auto getExternalEB = GetExternalEBField(pti, offset);

    const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
    const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
    const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
    const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
    const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
    const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

    // Lower corner of tile box physical domain (take into account Galilean shift)
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    const Dim3 lo = lbound(box);

    const int depos_type = WarpX::current_deposition_algo;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    auto& attribs = pti.GetAttribs();
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

#if (AMREX_SPACEDIM >= 2)
    ParticleReal* x_n = pti.GetAttribs(particle_comps["x_n"]).dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    ParticleReal* y_n = pti.GetAttribs(particle_comps["y_n"]).dataPtr();
#endif
    ParticleReal* z_n = pti.GetAttribs(particle_comps["z_n"]).dataPtr();
    ParticleReal* ux_n = pti.GetAttribs(particle_comps["ux_n"]).dataPtr();
    ParticleReal* uy_n = pti.GetAttribs(particle_comps["uy_n"]).dataPtr();
    ParticleReal* uz_n = pti.GetAttribs(particle_comps["uz_n"]).dataPtr();

    const int do_copy = (m_do_back_transformed_particles && (a_dt_type!=DtType::SecondHalf) );
    CopyParticleAttribs copyAttribs;
    if (do_copy) {
        copyAttribs = CopyParticleAttribs(pti, tmp_particle_data, offset);
    }

    int* AMREX_RESTRICT ion_lev = nullptr;
    if (do_field_ionization) {
        ion_lev = pti.GetiAttribs(particle_icomps["ionizationLevel"]).dataPtr() + offset;
    }

    // Loop over the particles and update their momentum
    const amrex::ParticleReal q = this->charge;
    const amrex::ParticleReal m = this-> mass;

    const auto pusher_algo = WarpX::particle_pusher_algo;
    const auto do_crr = do_classical_radiation_reaction;
#ifdef WARPX_QED
    const auto do_sync = m_do_qed_quantum_sync;
    amrex::Real t_chi_max = 0.0;
    if (do_sync) { t_chi_max = m_shr_p_qs_engine->get_minimum_chi_part(); }

    QuantumSynchrotronEvolveOpticalDepth evolve_opt;
    amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_QSR = nullptr;
    const bool local_has_quantum_sync = has_quantum_sync();
    if (local_has_quantum_sync) {
        evolve_opt = m_shr_p_qs_engine->build_evolve_functor();
        p_optical_depth_QSR = pti.GetAttribs(particle_comps["opticalDepthQSR"]).dataPtr()  + offset;
    }
#endif

    const auto t_do_not_gather = do_not_gather;

    enum exteb_flags : int { no_exteb, has_exteb };
    enum qed_flags : int { no_qed, has_qed };

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_quantum_sync || do_sync) ? has_qed : no_qed;
#else
    const int qed_runtime_flag = no_qed;
#endif

    const int max_iterations = WarpX::max_particle_its_in_implicit_scheme;
    const amrex::ParticleReal particle_tolerance = WarpX::particle_tol_in_implicit_scheme;

    amrex::Gpu::Buffer<amrex::Long> unconverged_particles({0});
    amrex::Long* unconverged_particles_ptr = unconverged_particles.data();

    // Using this version of ParallelFor with compile time options
    // improves performance when qed or external EB are not used by reducing
    // register pressure.
    amrex::ParallelFor(TypeList<CompileTimeOptions<no_exteb,has_exteb>,
                                CompileTimeOptions<no_qed  ,has_qed>>{},
                       {exteb_runtime_flag, qed_runtime_flag},
                       np_to_push, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control,
                                                         auto qed_control)
    {
        // Position advance starts from the position at the start of the step
        // but uses the most recent velocity.

#if (AMREX_SPACEDIM >= 2)
        amrex::ParticleReal xp = x_n[ip];
        const amrex::ParticleReal xp_n = x_n[ip];
#else
        const amrex::ParticleReal xp = 0._rt;
        const amrex::ParticleReal xp_n = 0._rt;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        amrex::ParticleReal yp = y_n[ip];
        const amrex::ParticleReal yp_n = y_n[ip];
#else
        const amrex::ParticleReal yp = 0._rt;
        const amrex::ParticleReal yp_n = 0._rt;
#endif
        amrex::ParticleReal zp = z_n[ip];
        const amrex::ParticleReal zp_n = z_n[ip];

        amrex::ParticleReal dxp, dxp_save;
        amrex::ParticleReal dyp, dyp_save;
        amrex::ParticleReal dzp, dzp_save;
        auto idxg2 = static_cast<amrex::ParticleReal>(dinv.x*dinv.x);
        auto idyg2 = static_cast<amrex::ParticleReal>(dinv.y*dinv.y);
        auto idzg2 = static_cast<amrex::ParticleReal>(dinv.z*dinv.z);

        amrex::ParticleReal step_norm = 1._prt;
        for (int iter=0; iter<max_iterations;) {

            dxp = 0.0;
            dyp = 0.0;
            dzp = 0.0;
            UpdatePositionImplicit(dxp, dyp, dzp, ux_n[ip], uy_n[ip], uz_n[ip], ux[ip], uy[ip], uz[ip], 0.5_rt*dt);
#if !defined(WARPX_DIM_1D_Z)
            xp = xp_n + dxp;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            yp = yp_n + dyp;
#endif
            zp = zp_n + dzp;
            setPosition(ip, xp, yp, zp);

            PositionNorm( dxp, dyp, dzp, dxp_save, dyp_save, dzp_save,
                          idxg2, idyg2, idzg2, step_norm, iter );
            if( step_norm < particle_tolerance ) { break; }

            amrex::ParticleReal Exp = Ex_external_particle;
            amrex::ParticleReal Eyp = Ey_external_particle;
            amrex::ParticleReal Ezp = Ez_external_particle;
            amrex::ParticleReal Bxp = Bx_external_particle;
            amrex::ParticleReal Byp = By_external_particle;
            amrex::ParticleReal Bzp = Bz_external_particle;

            if(!t_do_not_gather){
                // first gather E and B to the particle positions
                doGatherShapeNImplicit(xp_n, yp_n, zp_n, xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                       ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                       ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                       dinv, xyzmin, lo, n_rz_azimuthal_modes, nox,
                                       depos_type );
            }

            // Externally applied E and B-field in Cartesian co-ordinates
            [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB;
            if constexpr (exteb_control == has_exteb) {
                getExternalEB(ip, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
            }

            scaleFields(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp);

            if (do_copy) {
                //  Copy the old x and u for the BTD
                copyAttribs(ip);
            }

            // The momentum push starts with the velocity at the start of the step
            ux[ip] = ux_n[ip];
            uy[ip] = uy_n[ip];
            uz[ip] = uz_n[ip];

#ifdef WARPX_QED
            if (!do_sync)
#endif
            {
                doParticleMomentumPush<0>(ux[ip], uy[ip], uz[ip],
                                          Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                          ion_lev ? ion_lev[ip] : 1,
                                          m, q, pusher_algo, do_crr,
#ifdef WARPX_QED
                                          t_chi_max,
#endif
                                          dt);
            }
#ifdef WARPX_QED
            else {
                if constexpr (qed_control == has_qed) {
                    doParticleMomentumPush<1>(ux[ip], uy[ip], uz[ip],
                                              Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                              ion_lev ? ion_lev[ip] : 1,
                                              m, q, pusher_algo, do_crr,
                                              t_chi_max,
                                              dt);
                }
            }
#endif

#ifdef WARPX_QED
            [[maybe_unused]] auto foo_local_has_quantum_sync = local_has_quantum_sync;
            [[maybe_unused]] auto *foo_podq = p_optical_depth_QSR;
            [[maybe_unused]] const auto& foo_evolve_opt = evolve_opt; // have to do all these for nvcc
            if constexpr (qed_control == has_qed) {
                if (local_has_quantum_sync) {
                    evolve_opt(ux[ip], uy[ip], uz[ip],
                               Exp, Eyp, Ezp,Bxp, Byp, Bzp,
                               dt, p_optical_depth_QSR[ip]);
                }
            }
#else
            amrex::ignore_unused(qed_control);
#endif

            // Take average to get the time centered value
            ux[ip] = 0.5_rt*(ux[ip] + ux_n[ip]);
            uy[ip] = 0.5_rt*(uy[ip] + uy_n[ip]);
            uz[ip] = 0.5_rt*(uz[ip] + uz_n[ip]);

            iter++;

            // particle did not converge
            if ( iter > 1 && iter == max_iterations ) {
#if !defined(AMREX_USE_GPU)
                std::stringstream convergenceMsg;
                convergenceMsg << "Picard solver for particle failed to converge after " <<
                    iter << " iterations. " << std::endl;
                convergenceMsg << "Position step norm is " << step_norm <<
                    " and the tolerance is " << particle_tolerance << std::endl;
                convergenceMsg << " ux = " << ux[ip] << ", uy = " << uy[ip] << ", uz = " << uz[ip] << std::endl;
                convergenceMsg << " xp = " << xp     << ", yp = " << yp     << ", zp = " << zp;
                ablastr::warn_manager::WMRecordWarning("ImplicitPushXP", convergenceMsg.str());
#endif

                // write signaling flag: how many particles did not converge?
                amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
            }

        } // end Picard iterations

    });

    auto const num_unconverged_particles = *(unconverged_particles.copyToHost());
    if (num_unconverged_particles > 0) {
        ablastr::warn_manager::WMRecordWarning("ImplicitPushXP",
            "Picard solver for " +
            std::to_string(num_unconverged_particles) +
            " particles failed to converge after " +
            std::to_string(max_iterations) + " iterations."
         );
    }
}

void
PhysicalParticleContainer::InitIonizationModule ()
{
    if (!do_field_ionization) { return; }
    const ParmParse pp_species_name(species_name);
    if (charge != PhysConst::q_e){
        ablastr::warn_manager::WMRecordWarning("Species",
            "charge != q_e for ionizable species '" +
            species_name + "':" +
            "overriding user value and setting charge = q_e.");
        charge = PhysConst::q_e;
    }
    utils::parser::queryWithParser(pp_species_name, "do_adk_correction", do_adk_correction);

    utils::parser::queryWithParser(
        pp_species_name, "ionization_initial_level", ionization_initial_level);
    pp_species_name.get("ionization_product_species", ionization_product_name);
    pp_species_name.get("physical_element", physical_element);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        physical_element == "H" || !do_adk_correction,
        "Correction to ADK by Zhang et al., PRA 90, 043410 (2014) only works with Hydrogen");
    // Add runtime integer component for ionization level
    AddIntComp("ionizationLevel");
    // Get atomic number and ionization energies from file
    const int ion_element_id = utils::physics::ion_map_ids.at(physical_element);
    ion_atomic_number = utils::physics::ion_atomic_numbers[ion_element_id];
    Vector<Real> h_ionization_energies(ion_atomic_number);
    const int offset = utils::physics::ion_energy_offsets[ion_element_id];
    for(int i=0; i<ion_atomic_number; i++){
        h_ionization_energies[i] =
            utils::physics::table_ionization_energies[i+offset];
    }
    // Compute ADK prefactors (See Chen, JCP 236 (2013), equation (2))
    // For now, we assume l=0 and m=0.
    // The approximate expressions are used,
    // without Gamma function
    constexpr auto a3 = PhysConst::alpha*PhysConst::alpha*PhysConst::alpha;
    constexpr auto a4 = a3 * PhysConst::alpha;
    constexpr Real wa = a3 * PhysConst::c / PhysConst::r_e;
    constexpr Real Ea = PhysConst::m_e * PhysConst::c*PhysConst::c /PhysConst::q_e *
        a4/PhysConst::r_e;
    constexpr Real UH = utils::physics::table_ionization_energies[0];
    const Real l_eff = std::sqrt(UH/h_ionization_energies[0]) - 1._rt;

    const Real dt = WarpX::GetInstance().getdt(0);

    ionization_energies.resize(ion_atomic_number);
    adk_power.resize(ion_atomic_number);
    adk_prefactor.resize(ion_atomic_number);
    adk_exp_prefactor.resize(ion_atomic_number);

    Gpu::copyAsync(Gpu::hostToDevice,
                   h_ionization_energies.begin(), h_ionization_energies.end(),
                   ionization_energies.begin());

    adk_correction_factors.resize(4);
    if (do_adk_correction) {
        Vector<Real> h_correction_factors(4);
        constexpr int offset_corr = 0; // hard-coded: only Hydrogen
        for(int i=0; i<4; i++){
            h_correction_factors[i] = table_correction_factors[i+offset_corr];
        }
        Gpu::copyAsync(Gpu::hostToDevice,
                       h_correction_factors.begin(), h_correction_factors.end(),
                       adk_correction_factors.begin());
    }

    Real const* AMREX_RESTRICT p_ionization_energies = ionization_energies.data();
    Real * AMREX_RESTRICT p_adk_power = adk_power.data();
    Real * AMREX_RESTRICT p_adk_prefactor = adk_prefactor.data();
    Real * AMREX_RESTRICT p_adk_exp_prefactor = adk_exp_prefactor.data();
    amrex::ParallelFor(ion_atomic_number, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        const Real n_eff = (i+1) * std::sqrt(UH/p_ionization_energies[i]);
        const Real C2 = std::pow(2._rt,2._rt*n_eff)/(n_eff*std::tgamma(n_eff+l_eff+1._rt)*std::tgamma(n_eff-l_eff));
        p_adk_power[i] = -(2._rt*n_eff - 1._rt);
        const Real Uion = p_ionization_energies[i];
        p_adk_prefactor[i] = dt * wa * C2 * ( Uion/(2._rt*UH) )
            * std::pow(2._rt*std::pow((Uion/UH),3._rt/2._rt)*Ea,2._rt*n_eff - 1._rt);
        p_adk_exp_prefactor[i] = -2._rt/3._rt * std::pow( Uion/UH,3._rt/2._rt) * Ea;
    });

    Gpu::synchronize();
}

IonizationFilterFunc
PhysicalParticleContainer::getIonizationFunc (const WarpXParIter& pti,
                                              int lev,
                                              amrex::IntVect ngEB,
                                              const amrex::FArrayBox& Ex,
                                              const amrex::FArrayBox& Ey,
                                              const amrex::FArrayBox& Ez,
                                              const amrex::FArrayBox& Bx,
                                              const amrex::FArrayBox& By,
                                              const amrex::FArrayBox& Bz)
{
    WARPX_PROFILE("PhysicalParticleContainer::getIonizationFunc()");

    return {pti, lev, ngEB, Ex, Ey, Ez, Bx, By, Bz,
                                m_E_external_particle, m_B_external_particle,
                                ionization_energies.dataPtr(),
                                adk_prefactor.dataPtr(),
                                adk_exp_prefactor.dataPtr(),
                                adk_power.dataPtr(),
                                adk_correction_factors.dataPtr(),
                                particle_icomps["ionizationLevel"],
                                ion_atomic_number,
                                do_adk_correction};
}

PlasmaInjector* PhysicalParticleContainer::GetPlasmaInjector (int i)
{
    if (i < 0 || i >= static_cast<int>(plasma_injectors.size())) {
        return nullptr;
    } else {
        return plasma_injectors[i].get();
    }
}

void PhysicalParticleContainer::resample (const int timestep, const bool verbose)
{
    // In heavily load imbalanced simulations, MPI processes with few particles will spend most of
    // the time at the MPI synchronization in TotalNumberOfParticles(). Having two profiler entries
    // here is thus useful to avoid confusing time spent waiting for other processes with time
    // spent doing actual resampling.
    WARPX_PROFILE_VAR_NS("MultiParticleContainer::doResampling::MPI_synchronization",
                         blp_resample_synchronization);
    WARPX_PROFILE_VAR_NS("MultiParticleContainer::doResampling::ActualResampling",
                         blp_resample_actual);

    WARPX_PROFILE_VAR_START(blp_resample_synchronization);
    const amrex::Real global_numparts = TotalNumberOfParticles();
    WARPX_PROFILE_VAR_STOP(blp_resample_synchronization);

    WARPX_PROFILE_VAR_START(blp_resample_actual);
    if (m_resampler.triggered(timestep, global_numparts))
    {
        Redistribute();
        for (int lev = 0; lev <= maxLevel(); lev++)
        {
            for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
            {
                m_resampler(pti, lev, this);
            }
        }
        deleteInvalidParticles();
        if (verbose) {
            amrex::Print() << Utils::TextMsg::Info(
                "Resampled " + species_name + " at step " + std::to_string(timestep)
                + ": macroparticle count decreased by "
                + std::to_string(static_cast<int>(global_numparts - TotalNumberOfParticles()))
            );
        }
    }
    WARPX_PROFILE_VAR_STOP(blp_resample_actual);
}


#ifdef WARPX_QED


bool PhysicalParticleContainer::has_quantum_sync () const
{
    return m_do_qed_quantum_sync;
}

bool PhysicalParticleContainer::has_breit_wheeler () const
{
    return m_do_qed_breit_wheeler;
}

void
PhysicalParticleContainer::
set_breit_wheeler_engine_ptr (const std::shared_ptr<BreitWheelerEngine>& ptr)
{
    m_shr_p_bw_engine = ptr;
}

void
PhysicalParticleContainer::
set_quantum_sync_engine_ptr (const std::shared_ptr<QuantumSynchrotronEngine>& ptr)
{
    m_shr_p_qs_engine = ptr;
}

PhotonEmissionFilterFunc
PhysicalParticleContainer::getPhotonEmissionFilterFunc ()
{
    WARPX_PROFILE("PhysicalParticleContainer::getPhotonEmissionFunc()");
    return PhotonEmissionFilterFunc{particle_runtime_comps["opticalDepthQSR"]};
}

PairGenerationFilterFunc
PhysicalParticleContainer::getPairGenerationFilterFunc ()
{
    WARPX_PROFILE("PhysicalParticleContainer::getPairGenerationFunc()");
    return PairGenerationFilterFunc{particle_runtime_comps["opticalDepthBW"]};
}

#endif
