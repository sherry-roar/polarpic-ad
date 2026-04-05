module mod_statistics_postproc
    use mod_type
    use mod_param_postproc, only: stat_info, nx, ny, nz, nhalo
    use mod_mesh_postproc,  only: dx_inv, dy_inv, dzc_inv, dzf_inv
    use mod_dataio_postproc,only: outputLine, outputPlane
    use mod_utils,          only: abort, setZero
    use mod_spaceAvg,       only: avg_plane_t, initAvgPlane, reduceToAvgPlane, freeAvgPlane, &
                                  avg_line_t, initAvgLine, reduceToAvgLine, freeAvgLine
    use mod_mpi,            only: coord_xpen, coord_ypen, xsz, xst, ysz, yst, myrank, neighbor
    use decomp_2d,          only: transpose_y_to_x, transpose_x_to_y
    use, intrinsic :: iso_c_binding
    !$ use omp_lib

    implicit none
    ! make everything private unless declared public

    include 'mpif.h'
    include '../fftw3.f03'

    private

    ! velocity & pressure
    real(fp), allocatable, dimension(:), save, public :: u_stat,  v_stat,  w_stat
    real(fp), allocatable, dimension(:), save, public :: u2_stat, v2_stat, w2_stat
    real(fp), allocatable, dimension(:), save, public :: uv_stat, uw_stat, vw_stat
    real(fp), allocatable, dimension(:), save, public :: p_stat,  p2_stat
    real(fp), allocatable, dimension(:), save, public :: u3_stat, v3_stat, w3_stat
    real(fp), allocatable, dimension(:), save, public :: u4_stat, v4_stat, w4_stat
    real(fp), allocatable, dimension(:), save, public :: p3_stat, p4_stat
    
    ! velocity derivatives
    real(fp), allocatable, dimension(:), save, public :: dudx_stat, dudy_stat, dudz_stat
    real(fp), allocatable, dimension(:), save, public :: dvdx_stat, dvdy_stat, dvdz_stat
    real(fp), allocatable, dimension(:), save, public :: dwdx_stat, dwdy_stat, dwdz_stat
    real(fp), allocatable, dimension(:), save, public :: dudx_pow2_stat, dudy_pow2_stat, dudz_pow2_stat
    real(fp), allocatable, dimension(:), save, public :: dvdx_pow2_stat, dvdy_pow2_stat, dvdz_pow2_stat
    real(fp), allocatable, dimension(:), save, public :: dwdx_pow2_stat, dwdy_pow2_stat, dwdz_pow2_stat
    real(fp), allocatable, dimension(:), save, public :: dudy_dvdx_stat, dudz_dwdx_stat, dvdz_dwdy_stat
    
    ! high order statistics of velocity, pressure and velocity derivatives
    real(fp), allocatable, dimension(:), save, public :: u5_stat, u6_stat, u7_stat, u8_stat, u9_stat, u10_stat
    real(fp), allocatable, dimension(:), save, public :: v5_stat, v6_stat, v7_stat, v8_stat, v9_stat, v10_stat
    real(fp), allocatable, dimension(:), save, public :: w5_stat, w6_stat, w7_stat, w8_stat, w9_stat, w10_stat
    real(fp), allocatable, dimension(:), save, public :: p5_stat, p6_stat, p7_stat, p8_stat, p9_stat, p10_stat
    real(fp), allocatable, dimension(:), save, public :: dudz_pow3_stat, dudz_pow4_stat, dudz_pow5_stat, dudz_pow6_stat
    real(fp), allocatable, dimension(:), save, public :: dudz_pow7_stat, dudz_pow8_stat, dudz_pow9_stat, dudz_pow10_stat
    real(fp), allocatable, dimension(:), save, public :: dvdz_pow3_stat, dvdz_pow4_stat, dvdz_pow5_stat, dvdz_pow6_stat
    real(fp), allocatable, dimension(:), save, public :: dvdz_pow7_stat, dvdz_pow8_stat, dvdz_pow9_stat, dvdz_pow10_stat
    
    ! vorticity
    real(fp), allocatable, dimension(:), save, public :: omega_x_stat,  omega_y_stat,  omega_z_stat
    real(fp), allocatable, dimension(:), save, public :: omega_x2_stat, omega_y2_stat, omega_z2_stat
    real(fp), allocatable, dimension(:), save, public :: omega_xy_stat, omega_xz_stat, omega_yz_stat

    ! terms in Reynolds stress transport equation (RSTE) for channel flow
    ! production
    !   only velocity stat. data is required
    ! pressure strain
    real(fp), allocatable, dimension(:), save, public :: pdudx_stat, pdvdy_stat, pdwdz_stat
    real(fp), allocatable, dimension(:), save, public :: pdudz_stat, pdwdx_stat
    ! pressure transport
    real(fp), allocatable, dimension(:), save, public :: up_stat, vp_stat, wp_stat
    ! turbulent transport
    real(fp), allocatable, dimension(:), save, public :: uuw_stat, vvw_stat, www_stat
    real(fp), allocatable, dimension(:), save, public :: uww_stat
    ! viscous transport
    !   only velocity stat. data is required
    ! viscous dissipation
    ! real(fp), allocatable, dimension(:), save, public :: epsilon_uu_stat, epsilon_vv_stat, epsilon_ww_stat
    ! real(fp), allocatable, dimension(:), save, public :: epsilon_uw_stat
    
    ! energy spectra
    real(fp), allocatable, dimension(:,:), save, public :: euu_kx_stat, evv_kx_stat, eww_kx_stat
    real(fp), allocatable, dimension(:,:), save, public :: euw_kx_stat
    real(fp), allocatable, dimension(:,:), save, public :: euu_ky_stat, evv_ky_stat, eww_ky_stat
    real(fp), allocatable, dimension(:,:), save, public :: euw_ky_stat
    type(C_PTR) :: plan_r2c_xpen_1d, plan_r2c_ypen_1d
    real(fp),    allocatable, dimension(:), save :: wrk_xpen_1d, wrk_ypen_1d
    complex(fp), allocatable, dimension(:), save :: wrk_xpen_1d_c, wrk2_xpen_1d_c
    complex(fp), allocatable, dimension(:), save :: wrk_ypen_1d_c, wrk2_ypen_1d_c
    real(fp),    allocatable, dimension(:,:,:), save :: var_xpen, var_ypen
    real(fp),    allocatable, dimension(:,:,:), save :: var2_xpen, var2_ypen

    interface calcEnergySpectra_kx
        module procedure calcSelfEnergySpectra_kx
        module procedure calcCrossEnergySpectra_kx
        module procedure calcSelfCrossEnergySpectra_kx
    end interface calcEnergySpectra_kx
    
    interface calcEnergySpectra_ky
        module procedure calcSelfEnergySpectra_ky
        module procedure calcCrossEnergySpectra_ky
        module procedure calcSelfCrossEnergySpectra_ky
    end interface calcEnergySpectra_ky

    type(avg_line_t),  save, public :: al_z
    type(avg_plane_t), save, public :: ap_xz, ap_yz

    integer, save :: ie, je, ke

    public :: allocStat, freeStat, initStat, calcStatScheme1, calcStatScheme2, outputStat

contains
    subroutine allocStat(sz)
        implicit none
        integer, dimension(3), intent(in) :: sz

        integer :: istatus

        ie = sz(1)
        je = sz(2)
        ke = sz(3)

        allocate(u_stat (1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v_stat (1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w_stat (1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(uv_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(uw_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(vw_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p_stat (1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(u3_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v3_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w3_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u4_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v4_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w4_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p3_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p4_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(dudx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudy_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdy_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dwdx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dwdy_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dwdz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudx_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudy_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdx_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdy_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dwdx_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dwdy_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dwdz_pow2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudy_dvdx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_dwdx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_dwdy_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(u5_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u6_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u7_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u8_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u9_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(u10_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v5_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v6_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v7_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v8_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v9_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(v10_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w5_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w6_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w7_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w8_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w9_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(w10_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p5_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p6_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p7_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p8_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p9_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(p10_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow3_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow4_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow5_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow6_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow7_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow8_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow9_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dudz_pow10_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow3_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow4_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow5_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow6_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow7_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow8_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow9_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(dvdz_pow10_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(omega_x_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_y_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_z_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_x2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_y2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_z2_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_xy_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_xz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(omega_yz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(pdudx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(pdvdy_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(pdwdz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(pdudz_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(pdwdx_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(up_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(vp_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(wp_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(uuw_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(vvw_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(www_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(uww_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        ! allocate(epsilon_uu_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        ! allocate(epsilon_vv_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        ! allocate(epsilon_ww_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        ! allocate(epsilon_uw_stat(1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(euu_kx_stat(nx, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(evv_kx_stat(nx, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(eww_kx_stat(nx, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(euw_kx_stat(nx, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(euu_ky_stat(ny, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(evv_ky_stat(ny, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(eww_ky_stat(ny, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        allocate(euw_ky_stat(ny, 1:ke), stat=istatus); if (istatus /= 0) goto 999
        
        allocate(wrk_xpen_1d(nx), stat=istatus); if (istatus /= 0) goto 999
        allocate(wrk_ypen_1d(ny), stat=istatus); if (istatus /= 0) goto 999
        allocate(wrk_xpen_1d_c (nx/2+1), stat=istatus); if (istatus /= 0) goto 999
        allocate(wrk2_xpen_1d_c(nx/2+1), stat=istatus); if (istatus /= 0) goto 999
        allocate(wrk_ypen_1d_c (ny/2+1), stat=istatus); if (istatus /= 0) goto 999
        allocate(wrk2_ypen_1d_c(ny/2+1), stat=istatus); if (istatus /= 0) goto 999
        allocate(var_xpen (xsz(1), xsz(2), xsz(3)), stat=istatus); if (istatus /= 0) goto 999
        allocate(var_ypen (ysz(1), ysz(2), ysz(3)), stat=istatus); if (istatus /= 0) goto 999
        allocate(var2_xpen(xsz(1), xsz(2), xsz(3)), stat=istatus); if (istatus /= 0) goto 999
        allocate(var2_ypen(ysz(1), ysz(2), ysz(3)), stat=istatus); if (istatus /= 0) goto 999
#ifdef _SINGLE_PREC
        plan_r2c_xpen_1d = fftwf_plan_dft_r2c_1d(nx, wrk_xpen_1d, wrk_xpen_1d_c, FFTW_MEASURE)
        plan_r2c_ypen_1d = fftwf_plan_dft_r2c_1d(ny, wrk_ypen_1d, wrk_ypen_1d_c, FFTW_MEASURE)
#else
        plan_r2c_xpen_1d = fftw_plan_dft_r2c_1d(nx, wrk_xpen_1d, wrk_xpen_1d_c, FFTW_MEASURE)
        plan_r2c_ypen_1d = fftw_plan_dft_r2c_1d(ny, wrk_ypen_1d, wrk_ypen_1d_c, FFTW_MEASURE)
#endif
        
        ! initialize space-avarage planes & lines
        call initAvgLine (3, xsz, nx*ny, 'xpen', coord_xpen, al_z)
        call initAvgPlane(2, xsz, ny, 'xpen', coord_xpen, ap_xz)
        call initAvgPlane(1, ysz, nx, 'ypen', coord_ypen, ap_yz)
        
        return
    999 call abort(102, "allocStat: Out of memory when allocating statistical variables!")
    end subroutine allocStat

    subroutine freeStat()
        implicit none

        if (allocated(u_stat )) deallocate(u_stat )
        if (allocated(v_stat )) deallocate(v_stat )
        if (allocated(w_stat )) deallocate(w_stat )
        if (allocated(u2_stat)) deallocate(u2_stat)
        if (allocated(v2_stat)) deallocate(v2_stat)
        if (allocated(w2_stat)) deallocate(w2_stat)
        if (allocated(uv_stat)) deallocate(uv_stat)
        if (allocated(uw_stat)) deallocate(uw_stat)
        if (allocated(vw_stat)) deallocate(vw_stat)
        if (allocated(p_stat )) deallocate(p_stat )
        if (allocated(p2_stat)) deallocate(p2_stat)
        
        if (allocated(u3_stat)) deallocate(u3_stat)
        if (allocated(v3_stat)) deallocate(v3_stat)
        if (allocated(w3_stat)) deallocate(w3_stat)
        if (allocated(u4_stat)) deallocate(u4_stat)
        if (allocated(v4_stat)) deallocate(v4_stat)
        if (allocated(w4_stat)) deallocate(w4_stat)
        if (allocated(p3_stat)) deallocate(p3_stat)
        if (allocated(p4_stat)) deallocate(p4_stat)

        if (allocated(dudx_stat)) deallocate(dudx_stat)
        if (allocated(dudy_stat)) deallocate(dudy_stat)
        if (allocated(dudz_stat)) deallocate(dudz_stat)
        if (allocated(dvdx_stat)) deallocate(dvdx_stat)
        if (allocated(dvdy_stat)) deallocate(dvdy_stat)
        if (allocated(dvdz_stat)) deallocate(dvdz_stat)
        if (allocated(dwdx_stat)) deallocate(dwdx_stat)
        if (allocated(dwdy_stat)) deallocate(dwdy_stat)
        if (allocated(dwdz_stat)) deallocate(dwdz_stat)
        if (allocated(dudx_pow2_stat)) deallocate(dudx_pow2_stat)
        if (allocated(dudy_pow2_stat)) deallocate(dudy_pow2_stat)
        if (allocated(dudz_pow2_stat)) deallocate(dudz_pow2_stat)
        if (allocated(dvdx_pow2_stat)) deallocate(dvdx_pow2_stat)
        if (allocated(dvdy_pow2_stat)) deallocate(dvdy_pow2_stat)
        if (allocated(dvdz_pow2_stat)) deallocate(dvdz_pow2_stat)
        if (allocated(dwdx_pow2_stat)) deallocate(dwdx_pow2_stat)
        if (allocated(dwdy_pow2_stat)) deallocate(dwdy_pow2_stat)
        if (allocated(dwdz_pow2_stat)) deallocate(dwdz_pow2_stat)
        if (allocated(dudy_dvdx_stat)) deallocate(dudy_dvdx_stat)
        if (allocated(dudz_dwdx_stat)) deallocate(dudz_dwdx_stat)
        if (allocated(dvdz_dwdy_stat)) deallocate(dvdz_dwdy_stat)
        
        if (allocated(u5_stat)) deallocate(u5_stat)
        if (allocated(u6_stat)) deallocate(u6_stat)
        if (allocated(u7_stat)) deallocate(u7_stat)
        if (allocated(u8_stat)) deallocate(u8_stat)
        if (allocated(u9_stat)) deallocate(u9_stat)
        if (allocated(u10_stat)) deallocate(u10_stat)
        if (allocated(v5_stat)) deallocate(v5_stat)
        if (allocated(v6_stat)) deallocate(v6_stat)
        if (allocated(v7_stat)) deallocate(v7_stat)
        if (allocated(v8_stat)) deallocate(v8_stat)
        if (allocated(v9_stat)) deallocate(v9_stat)
        if (allocated(v10_stat)) deallocate(v10_stat)
        if (allocated(w5_stat)) deallocate(w5_stat)
        if (allocated(w6_stat)) deallocate(w6_stat)
        if (allocated(w7_stat)) deallocate(w7_stat)
        if (allocated(w8_stat)) deallocate(w8_stat)
        if (allocated(w9_stat)) deallocate(w9_stat)
        if (allocated(w10_stat)) deallocate(w10_stat)
        if (allocated(p5_stat)) deallocate(p5_stat)
        if (allocated(p6_stat)) deallocate(p6_stat)
        if (allocated(p7_stat)) deallocate(p7_stat)
        if (allocated(p8_stat)) deallocate(p8_stat)
        if (allocated(p9_stat)) deallocate(p9_stat)
        if (allocated(p10_stat)) deallocate(p10_stat)
        if (allocated(dudz_pow3_stat)) deallocate(dudz_pow3_stat)
        if (allocated(dudz_pow4_stat)) deallocate(dudz_pow4_stat)
        if (allocated(dudz_pow5_stat)) deallocate(dudz_pow5_stat)
        if (allocated(dudz_pow6_stat)) deallocate(dudz_pow6_stat)
        if (allocated(dudz_pow7_stat)) deallocate(dudz_pow7_stat)
        if (allocated(dudz_pow8_stat)) deallocate(dudz_pow8_stat)
        if (allocated(dudz_pow9_stat)) deallocate(dudz_pow9_stat)
        if (allocated(dudz_pow10_stat)) deallocate(dudz_pow10_stat)
        if (allocated(dvdz_pow3_stat)) deallocate(dvdz_pow3_stat)
        if (allocated(dvdz_pow4_stat)) deallocate(dvdz_pow4_stat)
        if (allocated(dvdz_pow5_stat)) deallocate(dvdz_pow5_stat)
        if (allocated(dvdz_pow6_stat)) deallocate(dvdz_pow6_stat)
        if (allocated(dvdz_pow7_stat)) deallocate(dvdz_pow7_stat)
        if (allocated(dvdz_pow8_stat)) deallocate(dvdz_pow8_stat)
        if (allocated(dvdz_pow9_stat)) deallocate(dvdz_pow9_stat)
        if (allocated(dvdz_pow10_stat)) deallocate(dvdz_pow10_stat)

        if (allocated(omega_x_stat)) deallocate(omega_x_stat)
        if (allocated(omega_y_stat)) deallocate(omega_y_stat)
        if (allocated(omega_z_stat)) deallocate(omega_z_stat)
        if (allocated(omega_x2_stat)) deallocate(omega_x2_stat)
        if (allocated(omega_y2_stat)) deallocate(omega_y2_stat)
        if (allocated(omega_z2_stat)) deallocate(omega_z2_stat)
        if (allocated(omega_xy_stat)) deallocate(omega_xy_stat)
        if (allocated(omega_xz_stat)) deallocate(omega_xz_stat)
        if (allocated(omega_yz_stat)) deallocate(omega_yz_stat)

        if (allocated(pdudx_stat)) deallocate(pdudx_stat)
        if (allocated(pdvdy_stat)) deallocate(pdvdy_stat)
        if (allocated(pdwdz_stat)) deallocate(pdwdz_stat)
        if (allocated(pdudz_stat)) deallocate(pdudz_stat)
        if (allocated(pdwdx_stat)) deallocate(pdwdx_stat)
        if (allocated(up_stat)) deallocate(up_stat)
        if (allocated(vp_stat)) deallocate(vp_stat)
        if (allocated(wp_stat)) deallocate(wp_stat)
        if (allocated(uuw_stat)) deallocate(uuw_stat)
        if (allocated(vvw_stat)) deallocate(vvw_stat)
        if (allocated(www_stat)) deallocate(www_stat)
        if (allocated(uww_stat)) deallocate(uww_stat)
        ! if (allocated(epsilon_uu_stat)) deallocate(epsilon_uu_stat)
        ! if (allocated(epsilon_vv_stat)) deallocate(epsilon_vv_stat)
        ! if (allocated(epsilon_ww_stat)) deallocate(epsilon_ww_stat)
        ! if (allocated(epsilon_uw_stat)) deallocate(epsilon_uw_stat)
        
        if (allocated(euu_kx_stat)) deallocate(euu_kx_stat)
        if (allocated(evv_kx_stat)) deallocate(evv_kx_stat)
        if (allocated(eww_kx_stat)) deallocate(eww_kx_stat)
        if (allocated(euw_kx_stat)) deallocate(euw_kx_stat)
        if (allocated(euu_ky_stat)) deallocate(euu_ky_stat)
        if (allocated(evv_ky_stat)) deallocate(evv_ky_stat)
        if (allocated(eww_ky_stat)) deallocate(eww_ky_stat)
        if (allocated(euw_ky_stat)) deallocate(euw_ky_stat)
        
        if (allocated(wrk_xpen_1d)) deallocate(wrk_xpen_1d)
        if (allocated(wrk_ypen_1d)) deallocate(wrk_ypen_1d)
        if (allocated(wrk_xpen_1d_c )) deallocate(wrk_xpen_1d_c )
        if (allocated(wrk2_xpen_1d_c)) deallocate(wrk2_xpen_1d_c)
        if (allocated(wrk_ypen_1d_c )) deallocate(wrk_ypen_1d_c )
        if (allocated(wrk2_ypen_1d_c)) deallocate(wrk2_ypen_1d_c)
        if (allocated(var_xpen )) deallocate(var_xpen )
        if (allocated(var_ypen )) deallocate(var_ypen )
        if (allocated(var2_xpen)) deallocate(var2_xpen)
        if (allocated(var2_ypen)) deallocate(var2_ypen)
#ifdef _SINGLE_PREC
        call fftwf_destroy_plan(plan_r2c_xpen_1d)
	    call fftwf_destroy_plan(plan_r2c_ypen_1d)
#else
        call fftw_destroy_plan(plan_r2c_xpen_1d)
        call fftw_destroy_plan(plan_r2c_ypen_1d)
#endif

        call freeAvgLine (al_z)
        call freeAvgPlane(ap_xz)
        call freeAvgPlane(ap_yz)

        return
    end subroutine freeStat

    subroutine initStat(nt_init_stat)
        implicit none
        integer, intent(in) :: nt_init_stat

        stat_info%nts  = nt_init_stat
        stat_info%nte  = nt_init_stat
        stat_info%nspl = 0

        call setZero(u_stat); call setZero(v_stat); call setZero(w_stat)
        call setZero(u2_stat); call setZero(v2_stat); call setZero(w2_stat)
        call setZero(uv_stat); call setZero(uw_stat); call setZero(vw_stat)
        call setZero(p_stat); call setZero(p2_stat)
        
        call setZero(u3_stat); call setZero(v3_stat); call setZero(w3_stat)
        call setZero(u4_stat); call setZero(v4_stat); call setZero(w4_stat)
        call setZero(p3_stat); call setZero(p4_stat)
        
        call setZero(dudx_stat); call setZero(dudy_stat); call setZero(dudz_stat)
        call setZero(dvdx_stat); call setZero(dvdy_stat); call setZero(dvdz_stat)
        call setZero(dwdx_stat); call setZero(dwdy_stat); call setZero(dwdz_stat)
        call setZero(dudx_pow2_stat); call setZero(dudy_pow2_stat); call setZero(dudz_pow2_stat)
        call setZero(dvdx_pow2_stat); call setZero(dvdy_pow2_stat); call setZero(dvdz_pow2_stat)
        call setZero(dwdx_pow2_stat); call setZero(dwdy_pow2_stat); call setZero(dwdz_pow2_stat)
        call setZero(dudy_dvdx_stat); call setZero(dudz_dwdx_stat); call setZero(dvdz_dwdy_stat)

        call setZero(u5_stat); call setZero(u6_stat); call setZero(u7_stat)
        call setZero(u8_stat); call setZero(u9_stat); call setZero(u10_stat)
        call setZero(v5_stat); call setZero(v6_stat); call setZero(v7_stat)
        call setZero(v8_stat); call setZero(v9_stat); call setZero(v10_stat)
        call setZero(w5_stat); call setZero(w6_stat); call setZero(w7_stat)
        call setZero(w8_stat); call setZero(w9_stat); call setZero(w10_stat)
        call setZero(p5_stat); call setZero(p6_stat); call setZero(p7_stat)
        call setZero(p8_stat); call setZero(p9_stat); call setZero(p10_stat)
        call setZero(dudz_pow3_stat); call setZero(dudz_pow4_stat); call setZero(dudz_pow5_stat); call setZero(dudz_pow6_stat)
        call setZero(dudz_pow7_stat); call setZero(dudz_pow8_stat); call setZero(dudz_pow9_stat); call setZero(dudz_pow10_stat)
        call setZero(dvdz_pow3_stat); call setZero(dvdz_pow4_stat); call setZero(dvdz_pow5_stat); call setZero(dvdz_pow6_stat)
        call setZero(dvdz_pow7_stat); call setZero(dvdz_pow8_stat); call setZero(dvdz_pow9_stat); call setZero(dvdz_pow10_stat)

        call setZero(omega_x_stat); call setZero(omega_y_stat); call setZero(omega_z_stat)
        call setZero(omega_x2_stat); call setZero(omega_y2_stat); call setZero(omega_z2_stat)
        call setZero(omega_xy_stat); call setZero(omega_xz_stat); call setZero(omega_yz_stat)

        call setZero(pdudx_stat)
        call setZero(pdvdy_stat)
        call setZero(pdwdz_stat)
        call setZero(pdudz_stat)
        call setZero(pdwdx_stat)
        call setZero(up_stat)
        call setZero(vp_stat)
        call setZero(wp_stat)
        call setZero(uuw_stat)
        call setZero(vvw_stat)
        call setZero(www_stat)
        call setZero(uww_stat)
        ! call setZero(epsilon_uu_stat)
        ! call setZero(epsilon_vv_stat)
        ! call setZero(epsilon_ww_stat)
        ! call setZero(epsilon_uw_stat)
        
        call setZero(euu_kx_stat)
        call setZero(evv_kx_stat)
        call setZero(eww_kx_stat)
        call setZero(euw_kx_stat)
        call setZero(euu_ky_stat)
        call setZero(evv_ky_stat)
        call setZero(eww_ky_stat)
        call setZero(euw_ky_stat)

        return        
    end subroutine initStat

    subroutine calcStatScheme1(nt, nhalo, u, v, w, nhalo_one, p)
        implicit none
        integer, intent(in) :: nt
        integer, dimension(6), intent(in) :: nhalo, nhalo_one
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: u, v, w
        real(fp), dimension(1-nhalo_one(1):,1-nhalo_one(3):,1-nhalo_one(5):), intent(in) :: p

        real(fp) :: uc, vc, wc, pc, uf, vf, wf, wfm
        real(fp) :: dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
        real(fp) :: dudyc, dudzm, dudzc, dvdxc, dvdzm, dvdzc, dwdxm, dwdxc, dwdym, dwdyc
        real(fp) :: omega_x, omega_y, omega_z
        real(fp) :: omega_xm, omega_ym, omega_xc, omega_yc, omega_zc
        integer :: i, j, k

        stat_info%nte  = nt
        stat_info%nspl = stat_info%nspl + 1

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) &
        !$OMP PRIVATE(i, j, k, uc, vc, wc, pc, uf, vf, wf, wfm, &
        !$OMP         dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, &
        !$OMP         dudyc, dudzm, dudzc, dvdxc, dvdzm, dvdzc, dwdxm, dwdxc, dwdym, dwdyc, &
        !$OMP         omega_x, omega_y, omega_z, omega_xm, omega_ym, &
        !$OMP         omega_xc, omega_yc, omega_zc)
        do k = 1, ke
        do j = 1, je
        do i = 1, ie
            uf  = u(i,j,k)
            vf  = v(i,j,k)
            wf  = w(i,j,k)
            wfm = w(i,j,k-1)
            uc = (u(i-1,j,k)+u(i,j,k))*0.5_fp
            vc = (v(i,j-1,k)+v(i,j,k))*0.5_fp
            wc = (wf + wfm)*0.5_fp
            pc = p(i,j,k)

            ! low order statistics
            u_stat (k) = u_stat (k) + uf
            v_stat (k) = v_stat (k) + vf
            w_stat (k) = w_stat (k) + wc
            u2_stat(k) = u2_stat(k) + uf*uf
            v2_stat(k) = v2_stat(k) + vf*vf
            w2_stat(k) = w2_stat(k) + (wfm*wfm + wf*wf)*0.5_fp
            uv_stat(k) = uv_stat(k) + uc*vc
            uw_stat(k) = uw_stat(k) + uc*wc
            vw_stat(k) = vw_stat(k) + vc*wc
            p_stat (k) = p_stat (k) + pc
            p2_stat(k) = p2_stat(k) + pc*pc

            ! high order statistics for calculation of skewness and flatness
            u3_stat(k) = u3_stat(k) + uf*uf*uf
            v3_stat(k) = v3_stat(k) + vf*vf*vf
            w3_stat(k) = w3_stat(k) + (wfm*wfm*wfm + wf*wf*wf)*0.5_fp
            p3_stat(k) = p3_stat(k) + pc*pc*pc
            u4_stat(k) = u4_stat(k) + uf*uf*uf*uf
            v4_stat(k) = v4_stat(k) + vf*vf*vf*vf
            w4_stat(k) = w4_stat(k) + (wfm*wfm*wfm*wfm + wf*wf*wf*wf)*0.5_fp
            p4_stat(k) = p4_stat(k) + pc*pc*pc*pc

            ! higher order statistics of velocity and pressure
            u5_stat(k) = u5_stat(k) + uc**5
            u6_stat(k) = u6_stat(k) + uc**6
            u7_stat(k) = u7_stat(k) + uc**7
            u8_stat(k) = u8_stat(k) + uc**8
            u9_stat(k) = u9_stat(k) + uc**9
            u10_stat(k) = u10_stat(k) + uc**10
            v5_stat(k) = v5_stat(k) + vc**5
            v6_stat(k) = v6_stat(k) + vc**6
            v7_stat(k) = v7_stat(k) + vc**7
            v8_stat(k) = v8_stat(k) + vc**8
            v9_stat(k) = v9_stat(k) + vc**9
            v10_stat(k) = v10_stat(k) + vc**10
            w5_stat(k) = w5_stat(k) + (wfm**5 + wf**5)*0.5_fp
            w6_stat(k) = w6_stat(k) + (wfm**6 + wf**6)*0.5_fp
            w7_stat(k) = w7_stat(k) + (wfm**7 + wf**7)*0.5_fp
            w8_stat(k) = w8_stat(k) + (wfm**8 + wf**8)*0.5_fp
            w9_stat(k) = w9_stat(k) + (wfm**9 + wf**9)*0.5_fp
            w10_stat(k) = w10_stat(k) + (wfm**10 + wf**10)*0.5_fp
            p5_stat(k) = p5_stat(k) + pc**5
            p6_stat(k) = p6_stat(k) + pc**6
            p7_stat(k) = p7_stat(k) + pc**7
            p8_stat(k) = p8_stat(k) + pc**8
            p9_stat(k) = p9_stat(k) + pc**9
            p10_stat(k) = p10_stat(k) + pc**10
            
            ! RSTE related statistics
            up_stat(k) = up_stat(k) + uc*pc
            vp_stat(k) = vp_stat(k) + vc*pc
            wp_stat(k) = wp_stat(k) + wc*pc
            uuw_stat(k) = uuw_stat(k) + uc*uc*wc
            vvw_stat(k) = vvw_stat(k) + vc*vc*wc
            www_stat(k) = www_stat(k) + (wfm*wfm*wfm + wf*wf*wf)*0.5_fp
            uww_stat(k) = uww_stat(k) + uc*wc*wc

            dudx = (u(i, j, k) - u(i-1, j, k))*dx_inv
            dudy = (u(i, j+1, k) - u(i, j, k))*dy_inv
            dudyc= (u(i, j+1, k) - u(i, j-1, k) + u(i-1, j+1, k) - u(i-1, j-1, k))*0.25_fp*dy_inv
            dudz = (u(i, j, k+1) - u(i, j, k))*dzc_inv(k  )
            dudzm= (u(i, j, k) - u(i, j, k-1))*dzc_inv(k-1)
            dudzc= ((u(i-1, j, k+1) - u(i-1, j, k  ))*dzc_inv(k  ) + dudz + &
                    (u(i-1, j, k  ) - u(i-1, j, k-1))*dzc_inv(k-1) + dudzm)*0.25_fp
            dvdx = (v(i+1, j, k) - v(i, j, k))*dx_inv
            dvdxc= (v(i+1, j, k) - v(i-1, j, k) + v(i+1, j-1, k) - v(i-1, j-1, k))*0.25_fp*dx_inv
            dvdy = (v(i, j, k) - v(i, j-1, k))*dy_inv
            dvdz = (v(i, j, k+1) - v(i, j, k))*dzc_inv(k  )
            dvdzm= (v(i, j, k) - v(i, j, k-1))*dzc_inv(k-1)
            dvdzc= ((v(i, j-1, k+1) - v(i, j-1, k  ))*dzc_inv(k  ) + dvdz + &
                    (v(i, j-1, k  ) - v(i, j-1, k-1))*dzc_inv(k-1) + dvdzm)*0.25_fp
            dwdx = (w(i+1, j, k  ) - w(i, j, k  ))*dx_inv
            dwdxm= (w(i+1, j, k-1) - w(i, j, k-1))*dx_inv
            dwdxc= (w(i+1, j, k) - w(i-1, j, k) + w(i+1, j, k-1) - w(i-1, j, k-1))*0.25_fp*dx_inv
            dwdy = (w(i, j+1, k  ) - w(i, j, k  ))*dy_inv
            dwdym= (w(i, j+1, k-1) - w(i, j, k-1))*dy_inv
            dwdyc= (w(i, j+1, k) - w(i, j-1, k) + w(i, j+1, k-1) - w(i, j-1, k-1))*0.25_fp*dy_inv
            dwdz = (w(i, j, k) - w(i, j, k-1))*dzf_inv(k)
            
            ! velocity derivatives statistics
            dudx_stat(k) = dudx_stat(k) + dudx
            dudy_stat(k) = dudy_stat(k) + dudy
            dudz_stat(k) = dudz_stat(k) + (dudz + dudzm)*0.5_fp
            dvdx_stat(k) = dvdx_stat(k) + dvdx
            dvdy_stat(k) = dvdy_stat(k) + dvdy
            dvdz_stat(k) = dvdz_stat(k) + (dvdz + dvdzm)*0.5_fp
            dwdx_stat(k) = dwdx_stat(k) + dwdx
            dwdy_stat(k) = dwdy_stat(k) + dwdy
            dwdz_stat(k) = dwdz_stat(k) + dwdz
            dudx_pow2_stat(k) = dudx_pow2_stat(k) + dudx*dudx
            dudy_pow2_stat(k) = dudy_pow2_stat(k) + dudy*dudy
            dudz_pow2_stat(k) = dudz_pow2_stat(k) + (dudz*dudz + dudzm*dudzm)*0.5_fp
            dvdx_pow2_stat(k) = dvdx_pow2_stat(k) + dvdx*dvdx
            dvdy_pow2_stat(k) = dvdy_pow2_stat(k) + dvdy*dvdy
            dvdz_pow2_stat(k) = dvdz_pow2_stat(k) + (dvdz*dvdz + dvdzm*dvdzm)*0.5_fp
            dwdx_pow2_stat(k) = dwdx_pow2_stat(k) + dwdx*dwdx
            dwdy_pow2_stat(k) = dwdy_pow2_stat(k) + dwdy*dwdy
            dwdz_pow2_stat(k) = dwdz_pow2_stat(k) + dwdz*dwdz
            dudy_dvdx_stat(k) = dudy_dvdx_stat(k) + dudy*dvdx
            dudz_dwdx_stat(k) = dudz_dwdx_stat(k) + dudz*dwdx
            dvdz_dwdy_stat(k) = dvdz_dwdy_stat(k) + dvdz*dwdy
            
            ! higher order statistics of velocity derivatives
            dudz_pow3_stat(k) = dudz_pow3_stat(k) + (dudz**3 + dudzm**3)*0.5_fp
            dudz_pow4_stat(k) = dudz_pow4_stat(k) + (dudz**4 + dudzm**4)*0.5_fp
            dudz_pow5_stat(k) = dudz_pow5_stat(k) + (dudz**5 + dudzm**5)*0.5_fp
            dudz_pow6_stat(k) = dudz_pow6_stat(k) + (dudz**6 + dudzm**6)*0.5_fp
            dudz_pow7_stat(k) = dudz_pow7_stat(k) + (dudz**7 + dudzm**7)*0.5_fp
            dudz_pow8_stat(k) = dudz_pow8_stat(k) + (dudz**8 + dudzm**8)*0.5_fp
            dudz_pow9_stat(k) = dudz_pow9_stat(k) + (dudz**9 + dudzm**9)*0.5_fp
            dudz_pow10_stat(k) = dudz_pow10_stat(k) + (dudz**10 + dudzm**10)*0.5_fp
            dvdz_pow3_stat(k) = dvdz_pow3_stat(k) + (dvdz**3 + dvdzm**3)*0.5_fp
            dvdz_pow4_stat(k) = dvdz_pow4_stat(k) + (dvdz**4 + dvdzm**4)*0.5_fp
            dvdz_pow5_stat(k) = dvdz_pow5_stat(k) + (dvdz**5 + dvdzm**5)*0.5_fp
            dvdz_pow6_stat(k) = dvdz_pow6_stat(k) + (dvdz**6 + dvdzm**6)*0.5_fp
            dvdz_pow7_stat(k) = dvdz_pow7_stat(k) + (dvdz**7 + dvdzm**7)*0.5_fp
            dvdz_pow8_stat(k) = dvdz_pow8_stat(k) + (dvdz**8 + dvdzm**8)*0.5_fp
            dvdz_pow9_stat(k) = dvdz_pow9_stat(k) + (dvdz**9 + dvdzm**9)*0.5_fp
            dvdz_pow10_stat(k) = dvdz_pow10_stat(k) + (dvdz**10 + dvdzm**10)*0.5_fp

            ! voticity statistics
            omega_x = dwdy - dvdz
            omega_xm = dwdym - dvdzm
            omega_y = dudz - dwdx
            omega_ym = dudzm - dwdxm
            omega_z = dvdx - dudy
            omega_x_stat(k) = omega_x_stat(k) + (omega_x + omega_xm)*0.5_fp
            omega_y_stat(k) = omega_y_stat(k) + (omega_y + omega_ym)*0.5_fp
            omega_z_stat(k) = omega_z_stat(k) + omega_z
            omega_x2_stat(k) = omega_x2_stat(k) + (omega_x*omega_x + omega_xm*omega_xm)*0.5_fp
            omega_y2_stat(k) = omega_y2_stat(k) + (omega_y*omega_y + omega_ym*omega_ym)*0.5_fp
            omega_z2_stat(k) = omega_z2_stat(k) + omega_z*omega_z
            omega_xc = dwdyc - dvdzc
            omega_yc = dudzc - dwdxc
            omega_zc = dvdxc - dudyc
            omega_xy_stat(k) = omega_xy_stat(k) + omega_xc*omega_yc
            omega_xz_stat(k) = omega_xz_stat(k) + omega_xc*omega_zc
            omega_yz_stat(k) = omega_yz_stat(k) + omega_yc*omega_zc                
            
            ! RSTE related statistics
            pdudx_stat(k) = pdudx_stat(k) + pc*dudx
            pdvdy_stat(k) = pdvdy_stat(k) + pc*dvdy
            pdwdz_stat(k) = pdwdz_stat(k) + pc*dwdz
            pdudz_stat(k) = pdudz_stat(k) + pc*dudz
            pdwdx_stat(k) = pdwdx_stat(k) + pc*dwdx
            ! epsilon_uu_stat(k) = epsilon_uu_stat(k) + dudx*dudx + dudy*dudy + dudz*dudz
            ! epsilon_vv_stat(k) = epsilon_vv_stat(k) + dvdx*dvdx + dvdy*dvdy + dvdz*dvdz
            ! epsilon_ww_stat(k) = epsilon_ww_stat(k) + dwdx*dwdx + dwdy*dwdy + dwdz*dwdz
            ! epsilon_uw_stat(k) = epsilon_uw_stat(k) + dudx*dwdx + dudy*dwdy + dudz*dwdz
        enddo
        enddo
        enddo
        !$OMP END PARALLEL DO

        call calcEnergySpectra_kx(nhalo, u, euu_kx_stat)
        call calcEnergySpectra_ky(nhalo, u, euu_ky_stat)
        call calcEnergySpectra_kx(nhalo, v, evv_kx_stat)
        call calcEnergySpectra_ky(nhalo, v, evv_ky_stat)
        call calcEnergySpectra_kx(nhalo, w, eww_kx_stat)
        call calcEnergySpectra_ky(nhalo, w, eww_ky_stat)
        call calcEnergySpectra_kx(nhalo, u, w, euw_kx_stat)
        call calcEnergySpectra_ky(nhalo, u, w, euw_ky_stat)

        return
    end subroutine calcStatScheme1

    subroutine calcStatScheme2(nt, nhalo, u, v, w, nhalo_one, p)
        implicit none
        integer, intent(in) :: nt
        integer, dimension(6), intent(in) :: nhalo, nhalo_one
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: u, v, w
        real(fp), dimension(1-nhalo_one(1):,1-nhalo_one(3):,1-nhalo_one(5):), intent(in) :: p

        real(fp) :: uc, vc, wc, pc, uf, vf, wf, wfm
        real(fp) :: r1 =  9.0_fp/8.0_fp
        real(fp) :: r2 = -1.0_fp/8.0_fp
        real(fp) :: r1interp, r2interp, r1dxi, r2dxi, r1dyi, r2dyi
        real(fp) :: dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
        real(fp) :: dudzm, dvdzm, dwdxm, dwdym
        real(fp) :: omega_x, omega_y, omega_z, omega_xm, omega_ym
        integer :: i, j, k

        r1interp = r1 * 0.5_fp
        r2interp = r2 * 0.5_fp
        r1dxi = r1 * dx_inv
        r2dxi = r2 * dx_inv/3.0_fp
        r1dyi = r1 * dy_inv
        r2dyi = r2 * dy_inv/3.0_fp

        stat_info%nte  = nt
        stat_info%nspl = stat_info%nspl + 1

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) &
        !$OMP PRIVATE(i, j, k, uc, vc, wc, pc, uf, vf, wf, wfm, &
        !$OMP         dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, &
        !$OMP         dudzm, dvdzm, dwdxm, dwdym, &
        !$OMP         omega_x, omega_y, omega_z, omega_xm, omega_ym)
        do k = 1, ke
        do j = 1, je
        do i = 1, ie
            uf  = u(i,j,k)
            vf  = v(i,j,k)
            wf  = w(i,j,k)               ! zf
            wfm = w(i,j,k-1)
            uc = r1interp*(u(i-1, j, k) + u(i, j, k)) + r2interp*(u(i-2, j, k) + u(i+1, j, k))
            vc = r1interp*(v(i, j-1, k) + v(i, j, k)) + r2interp*(v(i, j-2, k) + v(i, j+1, k))
            wc = (wf + wfm)*0.5_fp
            pc = p(i,j,k)

            ! low order statistics
            u_stat (k) = u_stat (k) + uf
            v_stat (k) = v_stat (k) + vf
            w_stat (k) = w_stat (k) + wc
            u2_stat(k) = u2_stat(k) + uf*uf
            v2_stat(k) = v2_stat(k) + vf*vf
            w2_stat(k) = w2_stat(k) + (wfm*wfm + wf*wf)*0.5_fp
            uv_stat(k) = uv_stat(k) + uc*vc
            uw_stat(k) = uw_stat(k) + uc*wc
            vw_stat(k) = vw_stat(k) + vc*wc
            p_stat (k) = p_stat (k) + pc
            p2_stat(k) = p2_stat(k) + pc*pc

            ! high order statistics for calculation of skewness and flatness
            u3_stat(k) = u3_stat(k) + uf*uf*uf
            v3_stat(k) = v3_stat(k) + vf*vf*vf
            w3_stat(k) = w3_stat(k) + (wfm*wfm*wfm + wf*wf*wf)*0.5_fp
            p3_stat(k) = p3_stat(k) + pc*pc*pc
            u4_stat(k) = u4_stat(k) + uf*uf*uf*uf
            v4_stat(k) = v4_stat(k) + vf*vf*vf*vf
            w4_stat(k) = w4_stat(k) + (wfm*wfm*wfm*wfm + wf*wf*wf*wf)*0.5_fp
            p4_stat(k) = p4_stat(k) + pc*pc*pc*pc

            ! higher order statistics of velocity and pressure
            u5_stat(k) = u5_stat(k) + uf**5
            u6_stat(k) = u6_stat(k) + uf**6
            u7_stat(k) = u7_stat(k) + uf**7
            u8_stat(k) = u8_stat(k) + uf**8
            u9_stat(k) = u9_stat(k) + uf**9
            u10_stat(k) = u10_stat(k) + uf**10
            v5_stat(k) = v5_stat(k) + vf**5
            v6_stat(k) = v6_stat(k) + vf**6
            v7_stat(k) = v7_stat(k) + vf**7
            v8_stat(k) = v8_stat(k) + vf**8
            v9_stat(k) = v9_stat(k) + vf**9
            v10_stat(k) = v10_stat(k) + vf**10
            w5_stat(k) = w5_stat(k) + (wfm**5 + wf**5)*0.5_fp
            w6_stat(k) = w6_stat(k) + (wfm**6 + wf**6)*0.5_fp
            w7_stat(k) = w7_stat(k) + (wfm**7 + wf**7)*0.5_fp
            w8_stat(k) = w8_stat(k) + (wfm**8 + wf**8)*0.5_fp
            w9_stat(k) = w9_stat(k) + (wfm**9 + wf**9)*0.5_fp
            w10_stat(k) = w10_stat(k) + (wfm**10 + wf**10)*0.5_fp
            p5_stat(k) = p5_stat(k) + pc**5
            p6_stat(k) = p6_stat(k) + pc**6
            p7_stat(k) = p7_stat(k) + pc**7
            p8_stat(k) = p8_stat(k) + pc**8
            p9_stat(k) = p9_stat(k) + pc**9
            p10_stat(k) = p10_stat(k) + pc**10
            
            ! RSTE related statistics
            up_stat(k) = up_stat(k) + uc*pc
            vp_stat(k) = vp_stat(k) + vc*pc
            wp_stat(k) = wp_stat(k) + wc*pc
            uuw_stat(k) = uuw_stat(k) + uc*uc*wc
            vvw_stat(k) = vvw_stat(k) + vc*vc*wc
            www_stat(k) = www_stat(k) + (wfm*wfm*wfm + wf*wf*wf)*0.5_fp
            uww_stat(k) = uww_stat(k) + uc*wc*wc
            
            dudx = r1dxi*(u(i, j, k) - u(i-1, j, k)) + r2dxi*(u(i+1, j, k) - u(i-2, j, k))
            dudy = r1dyi*(u(i, j+1, k) - u(i, j, k)) + r2dyi*(u(i, j+2, k) - u(i, j-1, k))
            dudz = (u(i, j, k+1) - u(i, j, k  ))*dzc_inv(k  )               ! zf
            dudzm= (u(i, j, k  ) - u(i, j, k-1))*dzc_inv(k-1)
            dvdx = r1dxi*(v(i+1, j, k) - v(i, j, k)) + r2dxi*(v(i+2, j, k) - v(i-1, j, k))
            dvdy = r1dyi*(v(i, j, k) - v(i, j-1, k)) + r2dyi*(v(i, j+1, k) - v(i, j-2, k))
            dvdz = (v(i, j, k+1) - v(i, j, k  ))*dzc_inv(k  )               ! zf
            dvdzm= (v(i, j, k  ) - v(i, j, k-1))*dzc_inv(k-1)
            dwdx = r1dxi*(w(i+1, j, k) - w(i, j, k)) + r2dxi*(w(i+2, j, k) - w(i-1, j, k))
            dwdxm= r1dxi*(w(i+1, j, k-1) - w(i, j, k-1)) + r2dxi*(w(i+2, j, k-1) - w(i-1, j, k-1))
            dwdy = r1dyi*(w(i, j+1, k) - w(i, j, k)) + r2dyi*(w(i, j+2, k) - w(i, j-1, k))
            dwdym= r1dyi*(w(i, j+1, k-1) - w(i, j, k-1)) + r2dyi*(w(i, j+2, k-1) - w(i, j-1, k-1))
            dwdz = (w(i, j, k) - w(i, j, k-1))*dzf_inv(k)
            
            ! velocity derivatives statistics
            dudx_stat(k) = dudx_stat(k) + dudx
            dudy_stat(k) = dudy_stat(k) + dudy
            dudz_stat(k) = dudz_stat(k) + (dudz + dudzm)*0.5_fp
            dvdx_stat(k) = dvdx_stat(k) + dvdx
            dvdy_stat(k) = dvdy_stat(k) + dvdy
            dvdz_stat(k) = dvdz_stat(k) + (dvdz + dvdzm)*0.5_fp
            dwdx_stat(k) = dwdx_stat(k) + dwdx
            dwdy_stat(k) = dwdy_stat(k) + dwdy
            dwdz_stat(k) = dwdz_stat(k) + dwdz
            dudx_pow2_stat(k) = dudx_pow2_stat(k) + dudx*dudx
            dudy_pow2_stat(k) = dudy_pow2_stat(k) + dudy*dudy
            dudz_pow2_stat(k) = dudz_pow2_stat(k) + (dudz*dudz + dudzm*dudzm)*0.5_fp
            dvdx_pow2_stat(k) = dvdx_pow2_stat(k) + dvdx*dvdx
            dvdy_pow2_stat(k) = dvdy_pow2_stat(k) + dvdy*dvdy
            dvdz_pow2_stat(k) = dvdz_pow2_stat(k) + (dvdz*dvdz + dvdzm*dvdzm)*0.5_fp
            dwdx_pow2_stat(k) = dwdx_pow2_stat(k) + dwdx*dwdx
            dwdy_pow2_stat(k) = dwdy_pow2_stat(k) + dwdy*dwdy
            dwdz_pow2_stat(k) = dwdz_pow2_stat(k) + dwdz*dwdz
            dudy_dvdx_stat(k) = dudy_dvdx_stat(k) + dudy*dvdx
            dudz_dwdx_stat(k) = dudz_dwdx_stat(k) + dudz*dwdx
            dvdz_dwdy_stat(k) = dvdz_dwdy_stat(k) + dvdz*dwdy
            
            ! higher order statistics of velocity derivatives
            dudz_pow3_stat(k) = dudz_pow3_stat(k) + (dudz**3 + dudzm**3)*0.5_fp
            dudz_pow4_stat(k) = dudz_pow4_stat(k) + (dudz**4 + dudzm**4)*0.5_fp
            dudz_pow5_stat(k) = dudz_pow5_stat(k) + (dudz**5 + dudzm**5)*0.5_fp
            dudz_pow6_stat(k) = dudz_pow6_stat(k) + (dudz**6 + dudzm**6)*0.5_fp
            dudz_pow7_stat(k) = dudz_pow7_stat(k) + (dudz**7 + dudzm**7)*0.5_fp
            dudz_pow8_stat(k) = dudz_pow8_stat(k) + (dudz**8 + dudzm**8)*0.5_fp
            dudz_pow9_stat(k) = dudz_pow9_stat(k) + (dudz**9 + dudzm**9)*0.5_fp
            dudz_pow10_stat(k) = dudz_pow10_stat(k) + (dudz**10 + dudzm**10)*0.5_fp
            dvdz_pow3_stat(k) = dvdz_pow3_stat(k) + (dvdz**3 + dvdzm**3)*0.5_fp
            dvdz_pow4_stat(k) = dvdz_pow4_stat(k) + (dvdz**4 + dvdzm**4)*0.5_fp
            dvdz_pow5_stat(k) = dvdz_pow5_stat(k) + (dvdz**5 + dvdzm**5)*0.5_fp
            dvdz_pow6_stat(k) = dvdz_pow6_stat(k) + (dvdz**6 + dvdzm**6)*0.5_fp
            dvdz_pow7_stat(k) = dvdz_pow7_stat(k) + (dvdz**7 + dvdzm**7)*0.5_fp
            dvdz_pow8_stat(k) = dvdz_pow8_stat(k) + (dvdz**8 + dvdzm**8)*0.5_fp
            dvdz_pow9_stat(k) = dvdz_pow9_stat(k) + (dvdz**9 + dvdzm**9)*0.5_fp
            dvdz_pow10_stat(k) = dvdz_pow10_stat(k) + (dvdz**10 + dvdzm**10)*0.5_fp

            ! voticity statistics
            omega_x = dwdy - dvdz               ! zf
            omega_xm = dwdym - dvdzm
            omega_y = dudz - dwdx               ! zf
            omega_ym= dudzm - dwdxm
            omega_z = dvdx - dudy
            omega_x_stat(k) = omega_x_stat(k) + (omega_x + omega_xm)*0.5_fp
            omega_y_stat(k) = omega_y_stat(k) + (omega_y + omega_ym)*0.5_fp
            omega_z_stat(k) = omega_z_stat(k) + omega_z
            omega_x2_stat(k) = omega_x2_stat(k) + (omega_x*omega_x + omega_xm*omega_xm)*0.5_fp
            omega_y2_stat(k) = omega_y2_stat(k) + (omega_y*omega_y + omega_ym*omega_ym)*0.5_fp
            omega_z2_stat(k) = omega_z2_stat(k) + omega_z*omega_z
            omega_xy_stat(k) = omega_xy_stat(k) + omega_x*omega_y               ! zf, problematic
            omega_xz_stat(k) = omega_xz_stat(k) + omega_x*omega_z               ! zf, problematic
            omega_yz_stat(k) = omega_yz_stat(k) + omega_y*omega_z               ! zf, problematic
            
            ! RSTE related statistics
            pdudx_stat(k) = pdudx_stat(k) + pc*dudx
            pdvdy_stat(k) = pdvdy_stat(k) + pc*dvdy
            pdwdz_stat(k) = pdwdz_stat(k) + pc*dwdz
            pdudz_stat(k) = pdudz_stat(k) + pc*(dudz + dudzm)*0.5_fp
            pdwdx_stat(k) = pdwdx_stat(k) + pc*(dwdx + dwdxm)*0.5_fp
            ! epsilon_uu_stat(k) = epsilon_uu_stat(k) + dudx*dudx + dudy*dudy + (dudz*dudz + dudzm*dudzm)*0.5_fp
            ! epsilon_vv_stat(k) = epsilon_vv_stat(k) + dvdx*dvdx + dvdy*dvdy + (dvdz*dvdz + dvdzm*dvdzm)*0.5_fp
            ! epsilon_ww_stat(k) = epsilon_ww_stat(k) + dwdx*dwdx + dwdy*dwdy + dwdz*dwdz
            ! epsilon_uw_stat(k) = epsilon_uw_stat(k) + dudx*dwdx + dudy*dwdy + dudz*dwdz
        enddo
        enddo
        enddo
        !$OMP END PARALLEL DO

        call calcEnergySpectra_kx(nhalo, u, euu_kx_stat)
        call calcEnergySpectra_ky(nhalo, u, euu_ky_stat)
        call calcEnergySpectra_kx(nhalo, v, evv_kx_stat)
        call calcEnergySpectra_ky(nhalo, v, evv_ky_stat)
        call calcEnergySpectra_kx(nhalo, w, eww_kx_stat)
        call calcEnergySpectra_ky(nhalo, w, eww_ky_stat)
        call calcEnergySpectra_kx(nhalo, u, w, euw_kx_stat)
        call calcEnergySpectra_ky(nhalo, u, w, euw_ky_stat)

    end subroutine calcStatScheme2

    subroutine stripHalo(nhalo, sz, var_halo, var)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in)  :: var_halo
        real(fp), dimension( :, :, :), intent(out) :: var
        
        integer :: i, j, k

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            var(i, j, k) = var_halo(i, j, k)
        enddo
        enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine stripHalo

    subroutine stripHaloAndAddConstant(nhalo, sz, var_halo, var, const)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in)  :: var_halo
        real(fp), dimension( :, :, :), intent(out) :: var
        real(fp), intent(in) :: const
        
        integer :: i, j, k

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            var(i, j, k) = var_halo(i, j, k) + const
        enddo
        enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine stripHaloAndAddConstant

    subroutine calcSelfEnergySpectra_kx(nhalo, vel, e_kx, vel_crf)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: vel
        real(fp), dimension(:,:), intent(inout) :: e_kx
        real(fp), optional, intent(in) :: vel_crf

        integer :: i, j, k

        if (present(vel_crf)) then
            call stripHaloAndAddConstant(nhalo, xsz, vel, var_xpen, vel_crf)
        else
            call stripHalo(nhalo, xsz, vel, var_xpen)
        endif

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, wrk_xpen_1d_c)
        do k = 1, xsz(3)
            do j = 1, xsz(2)
#ifdef _SINGLE_PREC
                call fftwf_execute_dft_r2c(plan_r2c_xpen_1d, var_xpen(:, j, k), wrk_xpen_1d_c)
#else
                call fftw_execute_dft_r2c(plan_r2c_xpen_1d, var_xpen(:, j, k), wrk_xpen_1d_c)
#endif
                do i = 1, nx/2+1
                    e_kx(i, k) = e_kx(i, k) + (abs(wrk_xpen_1d_c(i))/nx)**2
                enddo
            enddo
            do i = nx/2+2, nx
                e_kx(i, k) = e_kx(nx-i+2, k)
            enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine calcSelfEnergySpectra_kx

    subroutine calcCrossEnergySpectra_kx(nhalo, vel1, vel2, e12_kx, vel1_crf)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: vel1, vel2
        real(fp), dimension(:,:), intent(inout) :: e12_kx
        real(fp), optional, intent(in) :: vel1_crf

        integer :: i, j, k

        if (present(vel1_crf)) then
            call stripHaloAndAddConstant(nhalo, xsz, vel1, var_xpen, vel1_crf)
        else
            call stripHalo(nhalo, xsz, vel1, var_xpen)
        endif
        call stripHalo(nhalo, xsz, vel2, var2_xpen)

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, wrk_xpen_1d_c, wrk2_xpen_1d_c)
        do k = 1, xsz(3)
            do j = 1, xsz(2)
#ifdef _SINGLE_PREC
                call fftwf_execute_dft_r2c(plan_r2c_xpen_1d, var_xpen(:, j, k), wrk_xpen_1d_c)
                call fftwf_execute_dft_r2c(plan_r2c_xpen_1d, var2_xpen(:, j, k), wrk2_xpen_1d_c)
#else
                call fftw_execute_dft_r2c(plan_r2c_xpen_1d, var_xpen(:, j, k), wrk_xpen_1d_c)
                call fftw_execute_dft_r2c(plan_r2c_xpen_1d, var2_xpen(:, j, k), wrk2_xpen_1d_c)
#endif    
                do i = 1, nx/2+1
                    e12_kx(i, k) = e12_kx(i, k) + real( (wrk_xpen_1d_c(i)/nx) * conjg(wrk2_xpen_1d_c(i)/nx) )
                enddo
            enddo
            do i = nx/2+2, nx
                e12_kx(i, k) = e12_kx(nx-i+2, k)
            enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine calcCrossEnergySpectra_kx

    subroutine calcSelfCrossEnergySpectra_kx(nhalo, vel1, vel2, e1_kx, e2_kx, e12_kx, vel1_crf)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: vel1, vel2
        real(fp), dimension(:,:), intent(inout) :: e1_kx, e2_kx, e12_kx
        real(fp), optional, intent(in) :: vel1_crf

        integer :: i, j, k

        if (present(vel1_crf)) then
            call stripHaloAndAddConstant(nhalo, xsz, vel1, var_xpen, vel1_crf)
        else
            call stripHalo(nhalo, xsz, vel1, var_xpen)
        endif
        call stripHalo(nhalo, xsz, vel2, var2_xpen)

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, wrk_xpen_1d_c, wrk2_xpen_1d_c)
        do k = 1, xsz(3)
            do j = 1, xsz(2)
#ifdef _SINGLE_PREC
                call fftwf_execute_dft_r2c(plan_r2c_xpen_1d, var_xpen(:, j, k), wrk_xpen_1d_c)
                call fftwf_execute_dft_r2c(plan_r2c_xpen_1d, var2_xpen(:, j, k), wrk2_xpen_1d_c)
#else
                call fftw_execute_dft_r2c(plan_r2c_xpen_1d, var_xpen(:, j, k), wrk_xpen_1d_c)
                call fftw_execute_dft_r2c(plan_r2c_xpen_1d, var2_xpen(:, j, k), wrk2_xpen_1d_c)
#endif    
                do i = 1, nx/2+1
                    e1_kx (i, k) = e1_kx (i, k) + (abs(wrk_xpen_1d_c (i))/nx)**2
                    e2_kx (i, k) = e2_kx (i, k) + (abs(wrk2_xpen_1d_c(i))/nx)**2
                    e12_kx(i, k) = e12_kx(i, k) + real( (wrk_xpen_1d_c(i)/nx) * conjg(wrk2_xpen_1d_c(i)/nx) )
                enddo
            enddo
            do i = nx/2+2, nx
                e1_kx (i, k) = e1_kx (nx-i+2, k)
                e2_kx (i, k) = e2_kx (nx-i+2, k)
                e12_kx(i, k) = e12_kx(nx-i+2, k)
            enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine calcSelfCrossEnergySpectra_kx

    subroutine calcSelfEnergySpectra_ky(nhalo, vel, e_ky, vel_crf)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: vel
        real(fp), dimension(:,:), intent(inout) :: e_ky
        real(fp), optional, intent(in) :: vel_crf

        integer :: i, j, k

        if (present(vel_crf)) then
            call stripHaloAndAddConstant(nhalo, xsz, vel, var_xpen, vel_crf)
        else
            call stripHalo(nhalo, xsz, vel, var_xpen)
        endif
        call transpose_x_to_y(var_xpen, var_ypen)

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, wrk_ypen_1d_c)
        do k = 1, ysz(3)
            do i = 1, ysz(1)
#ifdef _SINGLE_PREC
                call fftwf_execute_dft_r2c(plan_r2c_ypen_1d, var_ypen(i, :, k), wrk_ypen_1d_c)
#else
                call fftw_execute_dft_r2c(plan_r2c_ypen_1d, var_ypen(i, :, k), wrk_ypen_1d_c)
#endif
                do j = 1, ny/2+1
                    e_ky(j, k) = e_ky(j, k) + (abs(wrk_ypen_1d_c(j))/ny)**2
                enddo
            enddo
            do j = ny/2+2, ny
                e_ky(j, k) = e_ky(ny-j+2, k)
            enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine calcSelfEnergySpectra_ky

    subroutine calcCrossEnergySpectra_ky(nhalo, vel1, vel2, e12_ky, vel1_crf)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: vel1, vel2
        real(fp), dimension(:,:), intent(inout) :: e12_ky
        real(fp), optional, intent(in) :: vel1_crf

        integer :: i, j, k

        if (present(vel1_crf)) then
            call stripHaloAndAddConstant(nhalo, xsz, vel1, var_xpen, vel1_crf)
        else
            call stripHalo(nhalo, xsz, vel1, var_xpen)
        endif
        call transpose_x_to_y(var_xpen, var_ypen)
        call stripHalo(nhalo, xsz, vel2, var_xpen)
        call transpose_x_to_y(var_xpen, var2_ypen)

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, wrk_ypen_1d_c, wrk2_ypen_1d_c)
        do k = 1, ysz(3)
            do i = 1, ysz(1)
#ifdef _SINGLE_PREC
                call fftwf_execute_dft_r2c(plan_r2c_ypen_1d, var_ypen(i, :, k), wrk_ypen_1d_c)
                call fftwf_execute_dft_r2c(plan_r2c_ypen_1d, var2_ypen(i, :, k), wrk2_ypen_1d_c)
#else
                call fftw_execute_dft_r2c(plan_r2c_ypen_1d, var_ypen(i, :, k), wrk_ypen_1d_c)
                call fftw_execute_dft_r2c(plan_r2c_ypen_1d, var2_ypen(i, :, k), wrk2_ypen_1d_c)
#endif
                do j = 1, ny/2+1
                    e12_ky(j, k) = e12_ky(j, k) + real( (wrk_ypen_1d_c(j)/ny) * conjg(wrk2_ypen_1d_c(j)/ny) )
                enddo    
            enddo
            do j = ny/2+2, ny
                e12_ky(j, k) = e12_ky(ny-j+2, k)
            enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine calcCrossEnergySpectra_ky

    subroutine calcSelfCrossEnergySpectra_ky(nhalo, vel1, vel2, e1_ky, e2_ky, e12_ky, vel1_crf)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: vel1, vel2
        real(fp), dimension(:,:), intent(inout) :: e1_ky, e2_ky, e12_ky
        real(fp), optional, intent(in) :: vel1_crf

        integer :: i, j, k

        if (present(vel1_crf)) then
            call stripHaloAndAddConstant(nhalo, xsz, vel1, var_xpen, vel1_crf)
        else
            call stripHalo(nhalo, xsz, vel1, var_xpen)
        endif
        call transpose_x_to_y(var_xpen, var_ypen)
        call stripHalo(nhalo, xsz, vel2, var_xpen)
        call transpose_x_to_y(var_xpen, var2_ypen)

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, wrk_ypen_1d_c, wrk2_ypen_1d_c)
        do k = 1, ysz(3)
            do i = 1, ysz(1)
#ifdef _SINGLE_PREC
                call fftwf_execute_dft_r2c(plan_r2c_ypen_1d, var_ypen(i, :, k), wrk_ypen_1d_c)
                call fftwf_execute_dft_r2c(plan_r2c_ypen_1d, var2_ypen(i, :, k), wrk2_ypen_1d_c)
#else
                call fftw_execute_dft_r2c(plan_r2c_ypen_1d, var_ypen(i, :, k), wrk_ypen_1d_c)
                call fftw_execute_dft_r2c(plan_r2c_ypen_1d, var2_ypen(i, :, k), wrk2_ypen_1d_c)
#endif
                do j = 1, ny/2+1
                    e1_ky (j, k) = e1_ky (j, k) + (abs(wrk_ypen_1d_c (j))/ny)**2
                    e2_ky (j, k) = e2_ky (j, k) + (abs(wrk2_ypen_1d_c(j))/ny)**2
                    e12_ky(j, k) = e12_ky(j, k) + real( (wrk_ypen_1d_c(j)/ny) * conjg(wrk2_ypen_1d_c(j)/ny) )
                enddo    
            enddo
            do j = ny/2+2, ny
                e1_ky (j, k) = e1_ky (ny-j+2, k)
                e2_ky (j, k) = e2_ky (ny-j+2, k)
                e12_ky(j, k) = e12_ky(ny-j+2, k)
            enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine calcSelfCrossEnergySpectra_ky

    subroutine outputStat()
        implicit none
        character(len=:), allocatable :: fn_prefix
        character(19) :: string_io
        integer :: ng1, st1, sz1, nh1(2)
        integer :: ng2(2), st2(2), sz2(2), nh2(4)
        integer :: nt = -1

        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.outputStat: Writing statistics fields ..."
        
        write(string_io,"('_',I8.8,'-',I8.8,'_')") stat_info%nts, stat_info%nte
        ng1 = nz
        st1 = xst(3)
        sz1 = xsz(3)
        nh1 = (/0, 0/)
        fn_prefix = "avg1d-z_stat"//string_io

        call reduceToAvgLine(u_stat, al_z)
        call outputLine(fn_prefix//'u.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u_stat',  al_z%buffer, stat_info)
        call reduceToAvgLine(v_stat, al_z)
        call outputLine(fn_prefix//'v.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v_stat',  al_z%buffer, stat_info)
        call reduceToAvgLine(w_stat, al_z)
        call outputLine(fn_prefix//'w.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w_stat',  al_z%buffer, stat_info)
        call reduceToAvgLine(u2_stat, al_z)
        call outputLine(fn_prefix//'u2.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u2_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v2_stat, al_z)
        call outputLine(fn_prefix//'v2.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v2_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w2_stat, al_z)
        call outputLine(fn_prefix//'w2.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w2_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(uv_stat, al_z)
        call outputLine(fn_prefix//'uv.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'uv_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(uw_stat, al_z)
        call outputLine(fn_prefix//'uw.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'uw_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(vw_stat, al_z)
        call outputLine(fn_prefix//'vw.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'vw_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p_stat, al_z)
        call outputLine(fn_prefix//'p.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p_stat',  al_z%buffer, stat_info)
        call reduceToAvgLine(p2_stat, al_z)
        call outputLine(fn_prefix//'p2.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p2_stat', al_z%buffer, stat_info)
        
        call reduceToAvgLine(u3_stat, al_z)
        call outputLine(fn_prefix//'u3.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u3_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v3_stat, al_z)
        call outputLine(fn_prefix//'v3.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v3_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w3_stat, al_z)
        call outputLine(fn_prefix//'w3.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w3_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(u4_stat, al_z)
        call outputLine(fn_prefix//'u4.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u4_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v4_stat, al_z)
        call outputLine(fn_prefix//'v4.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v4_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w4_stat, al_z)
        call outputLine(fn_prefix//'w4.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w4_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p3_stat, al_z)
        call outputLine(fn_prefix//'p3.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p3_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p4_stat, al_z)
        call outputLine(fn_prefix//'p4.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p4_stat', al_z%buffer, stat_info)
        
        call reduceToAvgLine(dudx_stat, al_z)
        call outputLine(fn_prefix//'dudx.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dudx_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dudy_stat, al_z)
        call outputLine(fn_prefix//'dudy.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dudy_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dudz_stat, al_z)
        call outputLine(fn_prefix//'dudz.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dudz_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dvdx_stat, al_z)
        call outputLine(fn_prefix//'dvdx.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dvdx_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dvdy_stat, al_z)
        call outputLine(fn_prefix//'dvdy.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dvdy_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dvdz_stat, al_z)
        call outputLine(fn_prefix//'dvdz.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dvdz_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dwdx_stat, al_z)
        call outputLine(fn_prefix//'dwdx.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dwdx_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dwdy_stat, al_z)
        call outputLine(fn_prefix//'dwdy.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dwdy_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(dwdz_stat, al_z)
        call outputLine(fn_prefix//'dwdz.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'dwdz_stat', al_z%buffer, stat_info)
        
        call reduceToAvgLine(dudx_pow2_stat, al_z)
        call outputLine(fn_prefix//'dudx_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudx_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudy_pow2_stat, al_z)
        call outputLine(fn_prefix//'dudy_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudy_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow2_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdx_pow2_stat, al_z)
        call outputLine(fn_prefix//'dvdx_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdx_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdy_pow2_stat, al_z)
        call outputLine(fn_prefix//'dvdy_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdy_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow2_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dwdx_pow2_stat, al_z)
        call outputLine(fn_prefix//'dwdx_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dwdx_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dwdy_pow2_stat, al_z)
        call outputLine(fn_prefix//'dwdy_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dwdy_pow2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dwdz_pow2_stat, al_z)
        call outputLine(fn_prefix//'dwdz_pow2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dwdz_pow2_stat',al_z%buffer,stat_info)
        
        call reduceToAvgLine(dudy_dvdx_stat, al_z)
        call outputLine(fn_prefix//'dudy_dvdx.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudy_dvdx_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_dwdx_stat, al_z)
        call outputLine(fn_prefix//'dudz_dwdx.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_dwdx_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_dwdy_stat, al_z)
        call outputLine(fn_prefix//'dvdz_dwdy.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_dwdy_stat',al_z%buffer,stat_info)


        call reduceToAvgLine(u5_stat, al_z)
        call outputLine(fn_prefix//'u5.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u5_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(u6_stat, al_z)
        call outputLine(fn_prefix//'u6.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u6_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(u7_stat, al_z)
        call outputLine(fn_prefix//'u7.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u7_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(u8_stat, al_z)
        call outputLine(fn_prefix//'u8.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u8_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(u9_stat, al_z)
        call outputLine(fn_prefix//'u9.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u9_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(u10_stat, al_z)
        call outputLine(fn_prefix//'u10.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'u10_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v5_stat, al_z)
        call outputLine(fn_prefix//'v5.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v5_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v6_stat, al_z)
        call outputLine(fn_prefix//'v6.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v6_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v7_stat, al_z)
        call outputLine(fn_prefix//'v7.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v7_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v8_stat, al_z)
        call outputLine(fn_prefix//'v8.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v8_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v9_stat, al_z)
        call outputLine(fn_prefix//'v9.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v9_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(v10_stat, al_z)
        call outputLine(fn_prefix//'v10.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'v10_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w5_stat, al_z)
        call outputLine(fn_prefix//'w5.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w5_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w6_stat, al_z)
        call outputLine(fn_prefix//'w6.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w6_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w7_stat, al_z)
        call outputLine(fn_prefix//'w7.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w7_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w8_stat, al_z)
        call outputLine(fn_prefix//'w8.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w8_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w9_stat, al_z)
        call outputLine(fn_prefix//'w9.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w9_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(w10_stat, al_z)
        call outputLine(fn_prefix//'w10.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'w10_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p5_stat, al_z)
        call outputLine(fn_prefix//'p5.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p5_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p6_stat, al_z)
        call outputLine(fn_prefix//'p6.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p6_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p7_stat, al_z)
        call outputLine(fn_prefix//'p7.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p7_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p8_stat, al_z)
        call outputLine(fn_prefix//'p8.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p8_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p9_stat, al_z)
        call outputLine(fn_prefix//'p9.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p9_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(p10_stat, al_z)
        call outputLine(fn_prefix//'p10.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'p10_stat', al_z%buffer, stat_info)

        call reduceToAvgLine(dudz_pow3_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow3.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow3_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow4_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow4.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow4_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow5_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow5.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow5_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow6_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow6.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow6_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow7_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow7.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow7_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow8_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow8.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow8_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow9_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow9.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow9_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dudz_pow10_stat, al_z)
        call outputLine(fn_prefix//'dudz_pow10.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dudz_pow10_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow3_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow3.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow3_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow4_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow4.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow4_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow5_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow5.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow5_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow6_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow6.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow6_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow7_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow7.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow7_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow8_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow8.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow8_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow9_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow9.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow9_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(dvdz_pow10_stat, al_z)
        call outputLine(fn_prefix//'dvdz_pow10.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'dvdz_pow10_stat',al_z%buffer,stat_info)

        
        call reduceToAvgLine(omega_x_stat, al_z)
        call outputLine(fn_prefix//'omega_x.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_x_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_y_stat, al_z)
        call outputLine(fn_prefix//'omega_y.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_y_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_z_stat, al_z)
        call outputLine(fn_prefix//'omega_z.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_z_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_x2_stat, al_z)
        call outputLine(fn_prefix//'omega_x2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_x2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_y2_stat, al_z)
        call outputLine(fn_prefix//'omega_y2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_y2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_z2_stat, al_z)
        call outputLine(fn_prefix//'omega_z2.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_z2_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_xy_stat, al_z)
        call outputLine(fn_prefix//'omega_xy.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_xy_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_xz_stat, al_z)
        call outputLine(fn_prefix//'omega_xz.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_xz_stat',al_z%buffer,stat_info)
        call reduceToAvgLine(omega_yz_stat, al_z)
        call outputLine(fn_prefix//'omega_yz.h5', al_z%have_results,nt,ng1,st1,sz1,nh1,'omega_yz_stat',al_z%buffer,stat_info)
        
        call reduceToAvgLine(pdudx_stat, al_z)
        call outputLine(fn_prefix//'pdudx.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'pdudx_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(pdvdy_stat, al_z)
        call outputLine(fn_prefix//'pdvdy.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'pdvdy_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(pdwdz_stat, al_z)
        call outputLine(fn_prefix//'pdwdz.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'pdwdz_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(pdudz_stat, al_z)
        call outputLine(fn_prefix//'pdudz.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'pdudz_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(pdwdx_stat, al_z)
        call outputLine(fn_prefix//'pdwdx.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'pdwdx_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(up_stat, al_z)
        call outputLine(fn_prefix//'up.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'up_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(vp_stat, al_z)
        call outputLine(fn_prefix//'vp.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'vp_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(wp_stat, al_z)
        call outputLine(fn_prefix//'wp.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'wp_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(uuw_stat, al_z)
        call outputLine(fn_prefix//'uuw.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'uuw_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(vvw_stat, al_z)
        call outputLine(fn_prefix//'vvw.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'vvw_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(www_stat, al_z)
        call outputLine(fn_prefix//'www.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'www_stat', al_z%buffer, stat_info)
        call reduceToAvgLine(uww_stat, al_z)
        call outputLine(fn_prefix//'uww.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'uww_stat', al_z%buffer, stat_info)
        ! call reduceToAvgLine(epsilon_uu_stat, al_z)
        ! call outputLine(fn_prefix//'epsilon_uu.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'epsilon_uu_stat', al_z%buffer, stat_info)
        ! call reduceToAvgLine(epsilon_vv_stat, al_z)
        ! call outputLine(fn_prefix//'epsilon_vv.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'epsilon_vv_stat', al_z%buffer, stat_info)
        ! call reduceToAvgLine(epsilon_ww_stat, al_z)
        ! call outputLine(fn_prefix//'epsilon_ww.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'epsilon_ww_stat', al_z%buffer, stat_info)
        ! call reduceToAvgLine(epsilon_uw_stat, al_z)
        ! call outputLine(fn_prefix//'epsilon_uw.h5', al_z%have_results, nt, ng1, st1, sz1, nh1, 'epsilon_uw_stat', al_z%buffer, stat_info)
        
        ng2 = (/nx, nz/)
        st2 = (/xst(1), xst(3)/)
        sz2 = (/xsz(1), xsz(3)/)
        nh2 = (/0, 0, 0, 0/)
        fn_prefix = "avg2d-xz_stat"//string_io

        call reduceToAvgPlane(euu_kx_stat, ap_xz)
        call outputPlane(fn_prefix//'Euu_kx.h5', ap_xz%have_results, nt, ng2, st2, sz2, nh2, 'Euu_kx_stat', ap_xz%buffer, stat_info)
        call reduceToAvgPlane(evv_kx_stat, ap_xz)
        call outputPlane(fn_prefix//'Evv_kx.h5', ap_xz%have_results, nt, ng2, st2, sz2, nh2, 'Evv_kx_stat', ap_xz%buffer, stat_info)
        call reduceToAvgPlane(eww_kx_stat, ap_xz)
        call outputPlane(fn_prefix//'Eww_kx.h5', ap_xz%have_results, nt, ng2, st2, sz2, nh2, 'Eww_kx_stat', ap_xz%buffer, stat_info)
        call reduceToAvgPlane(euw_kx_stat, ap_xz)
        call outputPlane(fn_prefix//'Euw_kx.h5', ap_xz%have_results, nt, ng2, st2, sz2, nh2, 'Euw_kx_stat', ap_xz%buffer, stat_info)
        
        ng2 = (/ny, nz/)
        st2 = (/yst(2), yst(3)/)
        sz2 = (/ysz(2), ysz(3)/)
        nh2 = (/0, 0, 0, 0/)
        fn_prefix = "avg2d-yz_stat"//string_io

        call reduceToAvgPlane(euu_ky_stat, ap_yz)
        call outputPlane(fn_prefix//'Euu_ky.h5', ap_yz%have_results, nt, ng2, st2, sz2, nh2, 'Euu_ky_stat', ap_yz%buffer, stat_info)
        call reduceToAvgPlane(evv_ky_stat, ap_yz)
        call outputPlane(fn_prefix//'Evv_ky.h5', ap_yz%have_results, nt, ng2, st2, sz2, nh2, 'Evv_ky_stat', ap_yz%buffer, stat_info)
        call reduceToAvgPlane(eww_ky_stat, ap_yz)
        call outputPlane(fn_prefix//'Eww_ky.h5', ap_yz%have_results, nt, ng2, st2, sz2, nh2, 'Eww_ky_stat', ap_yz%buffer, stat_info)
        call reduceToAvgPlane(euw_ky_stat, ap_yz)
        call outputPlane(fn_prefix//'Euw_ky.h5', ap_yz%have_results, nt, ng2, st2, sz2, nh2, 'Euw_ky_stat', ap_yz%buffer, stat_info)

        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.outputStat: Finish writing statistics fields!"

        deallocate(fn_prefix)
        return
    end subroutine outputStat

end module mod_statistics_postproc