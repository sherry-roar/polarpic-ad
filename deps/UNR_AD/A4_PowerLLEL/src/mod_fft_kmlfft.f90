module mod_fft
    use, intrinsic :: iso_c_binding
    use mod_type, only: fp
    
    implicit none

    ! In order to call FFTW (written in C) from Fortran，include the file 
    ! containing declarations of data types and procedures used in FFTW.
    ! This file is generated automatically after the compiling of FFTW, 
    ! and is located at $(kml_fft_install_dir)/include/ by default. 
    ! To ensure the successful compiling of this module, please follow 
    ! one of two ways listed below: 
    ! 1) add compiling flag "-I$(kml_fft_install_dir)/include/";
    ! 2) copy this file to the same directory where the module file at.
    include 'kfft.f03'

    private

#ifdef USE_C
    interface
        subroutine init_fft(xsz, ysz, bctype_x, bctype_y, work_xpen, work_ypen, fft_plan, fft_normfactor) bind(C, name='init_fft')
            import
            integer(C_INT), dimension(3), intent(in) :: xsz, ysz
            character(kind=c_char), intent(in) :: bctype_x(*)
            character(kind=c_char), intent(in) :: bctype_y(*)
            real(C_DOUBLE), dimension(*), intent(inout) :: work_xpen
            real(C_DOUBLE), dimension(*), intent(inout) :: work_ypen
            type(C_PTR), dimension(*), intent(out) :: fft_plan
            real(C_DOUBLE), intent(out) :: fft_normfactor
        end subroutine init_fft

        subroutine get_eigen_values(ist, isz, isz_global, bctype, lambda) bind(C, name='get_eigen_values')
            import
            integer(C_INT), value, intent(in) :: ist
            integer(C_INT), value, intent(in) :: isz
            integer(C_INT), value, intent(in) :: isz_global
            character(C_CHAR), intent(in) :: bctype(*)
            real(C_DOUBLE), dimension(*), intent(out) :: lambda
        end subroutine get_eigen_values

        subroutine execute_fft(plan, work) bind(C, name='execute_fft')
            import
            type(C_PTR), value :: plan
            real(C_DOUBLE), dimension(*), intent(inout) :: work
        end subroutine execute_fft

        subroutine free_fft(fft_plan) bind(C, name='free_fft')
            import
            type(C_PTR), dimension(*), intent(inout) :: fft_plan
        end subroutine free_fft
    end interface

    public :: init_fft, get_eigen_values, execute_fft, free_fft
#endif

    public :: initFFT, getEigenvalues, executeFFT, freeFFT
    
contains
    subroutine initFFT(xsz, ysz, bctype_xy, work_xpen, work_ypen, fft_plan, fft_normfactor)
        implicit none
        integer, dimension(3), intent(in) :: xsz, ysz
        character(2), dimension(2), intent(in) :: bctype_xy
        real(fp), dimension(:,:,:), intent(inout) :: work_xpen
        real(fp), dimension(:,:,:), intent(inout) :: work_ypen
        type(C_PTR), dimension(2,2), intent(out) :: fft_plan
        real(fp), intent(out) :: fft_normfactor

        type(C_PTR) :: plan_fwd_xpen, plan_bwd_xpen, plan_fwd_ypen, plan_bwd_ypen
        
        integer(C_INT) :: nx_xpen, ny_xpen, nz_xpen
        integer(C_INT) :: nx_ypen, ny_ypen, nz_ypen
        integer(C_INT), dimension(1) :: kind_fwd, kind_bwd
#ifdef _SINGLE_PREC
        type(kml_fftf_iodim), dimension(1) :: dims_t
        type(kml_fftf_iodim), dimension(1) :: howmany_dims_t
#else
        type(kml_fft_iodim), dimension(1) :: dims_t
        type(kml_fft_iodim), dimension(1) :: howmany_dims_t
#endif
        real(fp) :: normfactor_x, normfactor_y

        ! init in-place FFT along x direction (x-pencil)
        nx_xpen = xsz(1)
        ny_xpen = xsz(2)
        nz_xpen = xsz(3)
        
        dims_t(1)%n  = nx_xpen
        dims_t(1)%is = 1
        dims_t(1)%os = 1
        howmany_dims_t(1)%n  = 1
        howmany_dims_t(1)%is = nx_xpen  ! unused
        howmany_dims_t(1)%os = nx_xpen  ! unused
        call getFFTWKind(bctype_xy(1), kind_fwd, kind_bwd)
        call getNormFactor(bctype_xy(1), normfactor_x)
#ifdef _SINGLE_PREC
        plan_fwd_xpen = kml_fftf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                            kind_fwd, KML_FFT_MEASURE)
        plan_bwd_xpen = kml_fftf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                            kind_bwd, KML_FFT_MEASURE)
#else
        plan_fwd_xpen = kml_fft_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                           kind_fwd, KML_FFT_MEASURE)
        plan_bwd_xpen = kml_fft_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                           kind_bwd, KML_FFT_MEASURE)
#endif
                                           
        ! init in-place FFT along y direction (y-pencil)
        nx_ypen = ysz(1)
        ny_ypen = ysz(2)
        nz_ypen = ysz(3)
        
        dims_t(1)%n  = ny_ypen
        dims_t(1)%is = 1
        dims_t(1)%os = 1
        howmany_dims_t(1)%n  = 1
        howmany_dims_t(1)%is = 1  ! unused
        howmany_dims_t(1)%os = 1  ! unused
        call getFFTWKind(bctype_xy(2), kind_fwd, kind_bwd)
        call getNormFactor(bctype_xy(2), normfactor_y)
#ifdef _SINGLE_PREC
        plan_fwd_ypen = kml_fftf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                            kind_fwd, KML_FFT_MEASURE)
        plan_bwd_ypen = kml_fftf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                            kind_bwd, KML_FFT_MEASURE)
#else
        plan_fwd_ypen = kml_fft_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                           kind_fwd, KML_FFT_MEASURE)
        plan_bwd_ypen = kml_fft_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                           kind_bwd, KML_FFT_MEASURE)
#endif

        fft_normfactor = 1.0_fp/(normfactor_x*nx_xpen*normfactor_y*ny_ypen)
        fft_plan(1, 1) = plan_fwd_xpen
        fft_plan(2, 1) = plan_bwd_xpen
        fft_plan(1, 2) = plan_fwd_ypen
        fft_plan(2, 2) = plan_bwd_ypen

        return
    end subroutine initFFT

    subroutine getFFTWKind(bctype, kind_fwd, kind_bwd)
        implicit none
        character(2), intent(in) :: bctype
        integer(C_INT), dimension(:), intent(out) :: kind_fwd, kind_bwd

        select case(bctype)
        case('PP')
            kind_fwd = KML_FFT_R2HC ! or KML_FFT_DHT(but slightly slower)
            kind_bwd = KML_FFT_HC2R ! or KML_FFT_DHT(but slightly slower)
        case('NN')
            kind_fwd = KML_FFT_REDFT10
            kind_bwd = KML_FFT_REDFT01
        case('DD')
            kind_fwd = KML_FFT_RODFT10
            kind_bwd = KML_FFT_RODFT01
        case('ND')
            kind_fwd = KML_FFT_REDFT11
            kind_bwd = KML_FFT_REDFT11
        case('DN')
            kind_fwd = KML_FFT_RODFT11
            kind_bwd = KML_FFT_RODFT11
        end select

        return
    end subroutine getFFTWKind

    subroutine getNormFactor(bctype, normfactor)
        implicit none
        character(2), intent(in) :: bctype
        real(fp), intent(out) :: normfactor

        select case(bctype)
        case('PP')
            normfactor = 1.0_fp
        case('NN')
            normfactor = 2.0_fp
        case('DD')
            normfactor = 2.0_fp
        case('ND')
            normfactor = 2.0_fp
        case('DN')
            normfactor = 2.0_fp
        end select

        return
    end subroutine getNormFactor

    subroutine getEigenvalues(ist, isz, isz_global, bctype, lambda)
        implicit none
        integer,      intent(in) :: ist
        integer,      intent(in) :: isz
        integer,      intent(in) :: isz_global
        character(2), intent(in) :: bctype
        real(fp), dimension(isz), intent(out) :: lambda
        !@cuf real(fp), allocatable, dimension(:) :: lambda_glb, lambda_aux
        !@cuf integer :: n

        real(fp) :: pi
        integer :: i, ien

        pi = acos(-1._fp)
        ien = ist+isz-1
        select case(bctype)
        case('PP')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp*( cos(2.0_fp*pi*(i-1.0_fp)/isz_global) - 1.0_fp ); enddo
        case('NN')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp*( cos(pi*(i-1.0_fp)/isz_global) - 1.0_fp ); enddo
        case('DD')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp*( cos(pi*(i-0.0_fp)/isz_global) - 1.0_fp ); enddo
        case('ND','DN')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp*( cos(pi*(2*i-1.0_fp)/(2.*isz_global)) - 1.0_fp ); enddo
        end select

        return
    end subroutine getEigenvalues

    subroutine executeFFT(plan, work)
        implicit none
        type(C_PTR), intent(in) :: plan
        real(fp), dimension(*), intent(inout) :: work

        ! execute in-place FFT
#ifdef _SINGLE_PREC
        call kml_fftf_execute_r2r(plan, work, work)
#else
        call kml_fft_execute_r2r(plan, work, work)
#endif

        return
    end subroutine executeFFT

    subroutine freeFFT(fft_plan)
        implicit none
        type(C_PTR), dimension(2,2), intent(inout) :: fft_plan
        integer :: idir, i
        
        do idir = 1, 2
            do i = 1, 2
#ifdef _SINGLE_PREC
                call kml_fftf_destroy_plan(fft_plan(i, idir))
#else
                call kml_fft_destroy_plan(fft_plan(i, idir))
#endif
            enddo
        enddo

        return
    end subroutine freeFFT
    
end module mod_fft