module mod_fft
    use, intrinsic :: iso_c_binding
    use mod_type, only: fp, fp_pois, i8
#ifdef _CUDA
    use cufft
    use cudafor
#endif
    
    implicit none

    ! In order to call FFTW (written in C) from Fortran，include the file 
    ! containing declarations of data types and procedures used in FFTW.
    ! This file is generated automatically after the compiling of FFTW, 
    ! and is located at $(fftw_install_dir)/include/ by default. 
    ! To ensure the successful compiling of this module, please follow 
    ! one of two ways listed below: 
    ! 1) add compiling flag "-I$(fftw_install_dir)/include/";
    ! 2) copy this file to the same directory where the module file at.
    include 'fftw3.f03'

    private
#ifdef _CUDA
    type cufft_plan_wrapper
        integer(C_INT) :: plan
        integer :: fft_spatial_dir
        integer :: fft_type
        procedure(preproc_interface), pointer, nopass :: preproc => null()
        procedure(posproc_interface), pointer, nopass :: posproc => null()
    end type

    abstract interface
        subroutine preproc_interface(sz, work, cwork)
            import :: fp_pois
            integer,  dimension(3), intent(in) :: sz
            real(fp_pois), dimension(:,:,:), intent(inout) :: work
            real(fp_pois), dimension(:,:,:), intent(out), optional :: cwork
            attributes(device) :: work, cwork
        end subroutine preproc_interface

        subroutine posproc_interface(sz, work, cwork)
            import :: fp_pois
            integer,  dimension(3), intent(in) :: sz
            real(fp_pois), dimension(:,:,:), intent(inout) :: work
            real(fp_pois), dimension(:,:,:), intent(in), optional :: cwork
            attributes(device) :: work, cwork
        end subroutine posproc_interface
    end interface
    
    complex(fp_pois), allocatable, dimension(:), save :: cufft_workarea
    real(fp_pois), allocatable, dimension(:,:,:), save :: cwork, work_tr, cwork_tr
    attributes(device) :: cufft_workarea, cwork, work_tr, cwork_tr
    
    public :: cufft_plan_wrapper, initCUFFT, executeCUFFT, freeCUFFT, getMemoryFootprintCUFFT
#endif

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
        real(fp_pois), dimension(:,:,:), intent(inout) :: work_xpen
        real(fp_pois), dimension(:,:,:), intent(inout) :: work_ypen
        type(C_PTR), dimension(2,2), intent(out) :: fft_plan
        real(fp), intent(out) :: fft_normfactor

        type(C_PTR) :: plan_fwd_xpen, plan_bwd_xpen, plan_fwd_ypen, plan_bwd_ypen
        
        integer(C_INT) :: nx_xpen, ny_xpen, nz_xpen
        integer(C_INT) :: nx_ypen, ny_ypen, nz_ypen
        integer(C_FFTW_R2R_KIND), dimension(1) :: kind_fwd, kind_bwd
#if defined(_SINGLE_PREC) || defined(SP_POIS)
        type(fftwf_iodim), dimension(1) :: dims_t
        type(fftwf_iodim), dimension(1) :: howmany_dims_t
#else
        type(fftw_iodim), dimension(1) :: dims_t
        type(fftw_iodim), dimension(1) :: howmany_dims_t
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
#if defined(_SINGLE_PREC) || defined(SP_POIS)
        plan_fwd_xpen = fftwf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                            kind_fwd, FFTW_MEASURE)
        plan_bwd_xpen = fftwf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                            kind_bwd, FFTW_MEASURE)
#else
        plan_fwd_xpen = fftw_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                           kind_fwd, FFTW_MEASURE)
        plan_bwd_xpen = fftw_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_xpen, work_xpen, &
                                           kind_bwd, FFTW_MEASURE)
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
#if defined(_SINGLE_PREC) || defined(SP_POIS)
        plan_fwd_ypen = fftwf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                            kind_fwd, FFTW_MEASURE)
        plan_bwd_ypen = fftwf_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                            kind_bwd, FFTW_MEASURE)
#else
        plan_fwd_ypen = fftw_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                           kind_fwd, FFTW_MEASURE)
        plan_bwd_ypen = fftw_plan_guru_r2r(1, dims_t, 1, howmany_dims_t, work_ypen, work_ypen, &
                                           kind_bwd, FFTW_MEASURE)
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
        integer(C_FFTW_R2R_KIND), dimension(:), intent(out) :: kind_fwd, kind_bwd

        select case(bctype)
        case('PP')
            kind_fwd = FFTW_R2HC ! or FFTW_DHT(but slightly slower)
            kind_bwd = FFTW_HC2R ! or FFTW_DHT(but slightly slower)
        case('NN')
            kind_fwd = FFTW_REDFT10
            kind_bwd = FFTW_REDFT01
        case('DD')
            kind_fwd = FFTW_RODFT10
            kind_bwd = FFTW_RODFT01
        case('ND')
            kind_fwd = FFTW_REDFT11
            kind_bwd = FFTW_REDFT11
        case('DN')
            kind_fwd = FFTW_RODFT11
            kind_bwd = FFTW_RODFT11
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
        real(fp_pois), dimension(isz), intent(out) :: lambda
        !@cuf real(fp_pois), allocatable, dimension(:) :: lambda_glb, lambda_aux
        !@cuf integer :: n

        real(fp_pois) :: pi
        integer :: i, ien

        pi = acos(-1._fp_pois)
        ien = ist+isz-1
        select case(bctype)
        case('PP')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp_pois*( cos(2.0_fp_pois*pi*(i-1.0_fp_pois)/isz_global) - 1.0_fp_pois ); enddo
#ifdef _CUDA
            !!! NOTE: reorder (assumes n is even)
            ! from FFTW-format (r[0],r[1],r[2],r[3],r[4],i[3],i[2],i[1])
            ! to cufft-format  (r[0],r[4],r[1],i[1],r[2],i[2],r[3],i[3])
            n = isz_global
            allocate(lambda_glb(n))
            allocate(lambda_aux(n))
            do i = 1, n
                lambda_glb(i) = 2.0_fp_pois*( cos(2.0_fp_pois*pi*(i-1.0_fp_pois)/n) - 1.0_fp_pois )
                lambda_aux(i) = lambda_glb(i)
            enddo
            lambda_glb(1) = lambda_aux(1    )
            lambda_glb(2) = lambda_aux(n/2+1)
            do i = 2, n-1
                if (i<=n/2) then  ! real eigenvalue
                    lambda_glb(2*i-1          ) = lambda_aux(i  )
                else              ! imaginary eigenvalue
                    lambda_glb(n-2*(i-(n/2+1))) = lambda_aux(i+1)
                endif
            enddo
            do i = ist, ien; lambda(i-ist+1) = lambda_glb(i); enddo
            deallocate(lambda_glb)
            deallocate(lambda_aux)
#endif
        case('NN')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp_pois*( cos(pi*(i-1.0_fp_pois)/isz_global) - 1.0_fp_pois ); enddo
        case('DD')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp_pois*( cos(pi*(i-0.0_fp_pois)/isz_global) - 1.0_fp_pois ); enddo
        case('ND','DN')
            do i = ist, ien; lambda(i-ist+1) = 2.0_fp_pois*( cos(pi*(2*i-1.0_fp_pois)/(2.*isz_global)) - 1.0_fp_pois ); enddo
        end select

        return
    end subroutine getEigenvalues

    subroutine executeFFT(plan, work)
        implicit none
        type(C_PTR), intent(in) :: plan
        real(fp_pois), dimension(*), intent(inout) :: work

        ! execute in-place FFT
#if defined(_SINGLE_PREC) || defined(SP_POIS)
        call fftwf_execute_r2r(plan, work, work)
#else
        call fftw_execute_r2r(plan, work, work)
#endif

        return
    end subroutine executeFFT

    subroutine freeFFT(fft_plan)
        implicit none
        type(C_PTR), dimension(2,2), intent(inout) :: fft_plan
        integer :: idir, i
        
        do idir = 1, 2
            do i = 1, 2
#if defined(_SINGLE_PREC) || defined(SP_POIS)
                call fftwf_destroy_plan(fft_plan(i, idir))
#else
                call fftw_destroy_plan(fft_plan(i, idir))
#endif
            enddo
        enddo

        return
    end subroutine freeFFT

#ifdef _CUDA

#if defined(_SINGLE_PREC) || defined(SP_POIS)
#define CUFFT_FWD_TYPE CUFFT_R2C
#define CUFFT_BWD_TYPE CUFFT_C2R
#define CUFFT_EXEC_FWD cufftExecR2C
#define CUFFT_EXEC_BWD cufftExecC2R
#else
#define CUFFT_FWD_TYPE CUFFT_D2Z
#define CUFFT_BWD_TYPE CUFFT_Z2D
#define CUFFT_EXEC_FWD cufftExecD2Z
#define CUFFT_EXEC_BWD cufftExecZ2D
#endif

    subroutine initCUFFT(xsz, ysz, bctype_xy, fft_plan, fft_normfactor)
        implicit none
        integer, dimension(3), intent(in) :: xsz, ysz
        character(2), dimension(2), intent(in) :: bctype_xy
        type(cufft_plan_wrapper), dimension(2,2), intent(out) :: fft_plan
        real(fp), intent(out) :: fft_normfactor

        integer(C_INT) :: plan_fwd_xpen, plan_bwd_xpen, plan_fwd_ypen, plan_bwd_ypen
        integer :: rank, n, istride, idist, ostride, odist, batch
        integer(C_SIZE_T) :: worksize, max_worksize
        integer, pointer :: f_null_ptr
        real(fp) :: normfactor_x, normfactor_y
        integer :: istat

        if (any(bctype_xy == 'ND') .or. any(bctype_xy == 'DN')) then
            write(*,*) "Unsupported B.C. for CUFFT!"
            return
        endif

        call c_f_pointer( c_null_ptr, f_null_ptr )
        max_worksize = 0

        ! init FFT along x direction (x-pencil)
        ! forward
        fft_plan(1,1)%fft_spatial_dir = 1
        fft_plan(1,1)%fft_type = CUFFT_FWD_TYPE
        call setProcFuncPtr(bctype_xy(1), 1, fft_plan(1,1))
        call getNormFactor(bctype_xy(1), normfactor_x)
        rank = 1
        n = xsz(1)
        istride = 1
        idist = n
        ostride = 1
        odist = n
        batch = xsz(2)*xsz(3)
        istat = cufftCreate( plan_fwd_xpen )
        istat = cufftSetAutoAllocation( plan_fwd_xpen, 0 )
        istat = cufftMakePlanMany(plan_fwd_xpen, rank, n, &
                                  f_null_ptr, istride, idist, &     ! inembed, onembed = f_null_ptr 
                                  f_null_ptr, ostride, odist, &     ! assumes contiguous data arrays
                                  CUFFT_FWD_TYPE, batch, worksize)
        max_worksize = max(worksize, max_worksize)
        ! backward
        fft_plan(2,1)%fft_spatial_dir = 1
        fft_plan(2,1)%fft_type = CUFFT_BWD_TYPE
        call setProcFuncPtr(bctype_xy(1), -1, fft_plan(2,1))
        istat = cufftCreate( plan_bwd_xpen )
        istat = cufftSetAutoAllocation( plan_bwd_xpen, 0 )
        istat = cufftMakePlanMany(plan_bwd_xpen, rank, n, &
                                  f_null_ptr, istride, idist, &
                                  f_null_ptr, ostride, odist, &
                                  CUFFT_BWD_TYPE, batch, worksize)
        max_worksize = max(worksize, max_worksize)
                                            
        ! init FFT along y direction (y-pencil)
        ! forward
        fft_plan(1,2)%fft_spatial_dir = 2
        fft_plan(1,2)%fft_type = CUFFT_FWD_TYPE
        call setProcFuncPtr(bctype_xy(2), 1, fft_plan(1,2))
        call getNormFactor(bctype_xy(2), normfactor_y)
        rank = 1
        n = ysz(2)
        istride = 1
        idist = n
        ostride = 1
        odist = n
        batch = ysz(1)*ysz(3)
        istat = cufftCreate( plan_fwd_ypen )
        istat = cufftSetAutoAllocation( plan_fwd_ypen, 0 )
        istat = cufftMakePlanMany(plan_fwd_ypen, rank, n, &
                                  f_null_ptr, istride, idist, &     ! inembed, onembed = f_null_ptr 
                                  f_null_ptr, ostride, odist, &     ! assumes contiguous data arrays
                                  CUFFT_FWD_TYPE, batch, worksize)
        max_worksize = max(worksize, max_worksize)
        ! backward
        fft_plan(2,2)%fft_spatial_dir = 2
        fft_plan(2,2)%fft_type = CUFFT_BWD_TYPE
        call setProcFuncPtr(bctype_xy(2), -1, fft_plan(2,2))
        istat = cufftCreate( plan_bwd_ypen )
        istat = cufftSetAutoAllocation( plan_bwd_ypen, 0 )
        istat = cufftMakePlanMany(plan_bwd_ypen, rank, n, &
                                  f_null_ptr, istride, idist, &
                                  f_null_ptr, ostride, odist, &
                                  CUFFT_BWD_TYPE, batch, worksize)
        max_worksize = max(worksize, max_worksize)

        fft_normfactor = 1.0_fp/(normfactor_x*xsz(1)*normfactor_y*ysz(2))

        allocate( cufft_workarea( max_worksize/(2*sizeof(1.0_fp_pois)) ) )
        istat = cufftSetWorkArea( plan_fwd_xpen, cufft_workarea )
        istat = cufftSetWorkArea( plan_bwd_xpen, cufft_workarea )
        istat = cufftSetWorkArea( plan_fwd_ypen, cufft_workarea )
        istat = cufftSetWorkArea( plan_bwd_ypen, cufft_workarea )
        fft_plan(1,1)%plan = plan_fwd_xpen
        fft_plan(2,1)%plan = plan_bwd_xpen
        fft_plan(1,2)%plan = plan_fwd_ypen
        fft_plan(2,2)%plan = plan_bwd_ypen

        allocate(cwork   (xsz(1)+2, xsz(2), xsz(3)))
        allocate( work_tr(ysz(2)  , ysz(1), ysz(3)))
        allocate(cwork_tr(ysz(2)+2, ysz(1), ysz(3)))

        return
    end subroutine initCUFFT

    subroutine setProcFuncPtr(bctype, fwd_or_bwd, cufft_plan)
        implicit none
        character(2), intent(in) :: bctype
        integer, intent(in) :: fwd_or_bwd
        type(cufft_plan_wrapper) :: cufft_plan

        select case(bctype)
        case('PP')
            select case(fwd_or_bwd)
            case(1)
                cufft_plan%preproc => preproc_fft_fwd
                cufft_plan%posproc => posproc_fft_fwd
            case(-1)
                cufft_plan%preproc => preproc_fft_bwd
                cufft_plan%posproc => posproc_fft_bwd
            end select
        case('NN')
            select case(fwd_or_bwd)
            case(1)
                cufft_plan%preproc => preproc_redft10
                cufft_plan%posproc => posproc_redft10
            case(-1)
                cufft_plan%preproc => preproc_redft01
                cufft_plan%posproc => posproc_redft01
            end select
        case('DD')
            select case(fwd_or_bwd)
            case(1)
                cufft_plan%preproc => preproc_rodft10
                cufft_plan%posproc => posproc_rodft10
            case(-1)
                cufft_plan%preproc => preproc_rodft01
                cufft_plan%posproc => posproc_rodft01
            end select
        end select

        return
    end subroutine setProcFuncPtr

    ! pre-processing of a signal preciding a forward FFT
    subroutine preproc_fft_fwd(sz, work, cwork)
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(out), optional :: cwork !unused
        
        return
    end subroutine preproc_fft_fwd
    
    ! post-processing of a signal following a forward FFT
    ! to reorder the data from
    ! (r[0],   0,r[1],i[1],...,r[n-1],i[n-1],r[n],0)
    ! to
    ! (r[0],r[n],r[1],i[1],...,r[n-1],i[n-1],r[n],0)
    subroutine posproc_fft_fwd(sz, work, cwork)
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(in), optional :: cwork !should be passed in
        integer :: idir, isz, i, j, k

        idir = 1
        isz = sz(idir)
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            work(i, j, k) = cwork(i, j, k)
        enddo
        enddo
        enddo
        !$cuf kernel do(2) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
            work(2, j, k) = cwork(isz+1, j, k)
        enddo
        enddo
        
        return
    end subroutine posproc_fft_fwd

    ! pre-processing of a signal preciding a backward FFT
    ! to reorder the data from
    ! (r[0],r[n],r[1],i[1],...,r[n-1],i[n-1],0,0)
    ! to
    ! (r[0],   0,r[1],i[1],...,r[n-1],i[n-1],r[n],0)
    subroutine preproc_fft_bwd(sz, work, cwork)
        integer, dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(out), optional :: cwork !should be passed in
        integer :: idir, isz, i, j, k

        idir = 1
        isz = sz(idir)
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            cwork(i, j, k) = work(i, j, k)
        enddo
        enddo
        enddo
        !$cuf kernel do(2) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
            cwork(isz+2,j,k) = 0.0_fp_pois
            cwork(isz+1,j,k) = cwork(2,j,k)
            cwork(2,j,k) = 0.0_fp_pois
        enddo
        enddo
        
        return
    end subroutine preproc_fft_bwd
    
    ! post-processing of a signal following a backward FFT
    subroutine posproc_fft_bwd(sz, work, cwork)
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(in), optional :: cwork !unused
        
        return
    end subroutine posproc_fft_bwd

    ! pre-processing of a signal to perform a fast forward 
    ! discrete cosine transform (DCT) with FFTs (see Makhoul 1980)
    ! 
    ! the input signal x(n) is pre-processed into a signal v(n)
    ! as follows:
    !
    ! v(n) = x(2n       ),              0 <= n <= floor((N-1)/2)
    !      = x(2N -2n -1), floor((N+1)/2) <= n <= N-1
    ! with n = 0,...,N-1 and N is the total number of elements 
    ! of the signal.
    subroutine preproc_redft10(sz, work, cwork)
        implicit none
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(out), optional :: cwork !unused
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        integer :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)-1,sz(2),sz(3)))
        
        include 'cufft_preproc_redft10.f90'

        return
    end subroutine preproc_redft10

    ! post-processing of a signal to perform a fast forward 
    ! discrete cosine transform (DCT) with FFTs (see Makhoul 1980)
    subroutine posproc_redft10(sz, work, cwork)
        implicit none
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(in), optional :: cwork !should be passed in
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        real(fp_pois), allocatable, dimension(:,:  ), device :: sincos
        real(fp_pois) :: pi, arg, carg, sarg
        integer  :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)+1,sz(2),sz(3)))
        if(.not.allocated(sincos  )) allocate(sincos(2,0:isz/2))
        
        include 'cufft_posproc_redft10.f90'

        return
    end subroutine posproc_redft10
    
    ! pre-processing of a signal to perform a fast backward 
    ! discrete cosine transform (DCT) with FFTs (see Makhoul 1980)
    subroutine preproc_redft01(sz, work, cwork)
        implicit none
        integer , dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(out), optional :: cwork !should be passed in
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        real(fp_pois), allocatable, dimension(:,:  ), device :: sincos
        real(fp_pois) :: pi, arg, carg, sarg
        integer  :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)+1,sz(2),sz(3)))
        if(.not.allocated(sincos  )) allocate(sincos(2,0:isz/2))
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            cwork(i, j, k) = work(i, j, k)
        enddo
        enddo
        enddo

        include 'cufft_preproc_redft01.f90'

        return
    end subroutine preproc_redft01

    ! post-processing of a signal to perform a fast backward
    ! discrete cosine transform (DCT) with FFTs (see Makhoul 1980)
    subroutine posproc_redft01(sz, work, cwork)
        implicit none
        integer,  dimension(3), intent(in   ) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(in), optional :: cwork !unused
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        integer  :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)-1,sz(2),sz(3)))
        
        include 'cufft_posproc_redft01.f90'
        
        return
    end subroutine posproc_redft01

    ! pre-processing of a signal to perform a fast forward 
    ! discrete sine transform (DST) with FFTs (see Makhoul 1980)
    subroutine preproc_rodft10(sz, work, cwork)
        implicit none
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(out), optional :: cwork !unused
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        integer :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)-1,sz(2),sz(3)))
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz
            if(mod(i,2) == 0) then
                work(i,j,k) = - work(i,j,k)
            endif
        enddo
        enddo
        enddo
        
        include 'cufft_preproc_redft10.f90'

        return
    end subroutine preproc_rodft10

    ! post-processing of a signal to perform a fast forward 
    ! discrete sine transform (DST) with FFTs (see Makhoul 1980)
    subroutine posproc_rodft10(sz, work, cwork)
        implicit none
        integer,  dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(in), optional :: cwork !should be passed in
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        real(fp_pois), allocatable, dimension(:,:  ), device :: sincos
        real(fp_pois) :: pi, arg, carg, sarg, tmp
        integer  :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)+1,sz(2),sz(3)))
        if(.not.allocated(sincos  )) allocate(sincos(2,0:isz/2))
        
        include 'cufft_posproc_redft10.f90'

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz/2
            tmp                 = work(i      , j, k)
            work(i      , j, k) = work(isz-i+1, j, k)
            work(isz-i+1, j, k) = tmp
        enddo
        enddo
        enddo

        return
    end subroutine posproc_rodft10

    ! pre-processing of a signal to perform a fast backward 
    ! discrete sine transform (DST) with FFTs (see Makhoul 1980)
    subroutine preproc_rodft01(sz, work, cwork)
        implicit none
        integer , dimension(3), intent(in) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(out), optional :: cwork !should be passed in
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        real(fp_pois), allocatable, dimension(:,:  ), device :: sincos
        real(fp_pois) :: pi, arg, carg, sarg, tmp
        integer  :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)+1,sz(2),sz(3)))
        if(.not.allocated(sincos  )) allocate(sincos(2,0:isz/2))
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            cwork(i, j, k) = work(i, j, k)
        enddo
        enddo
        enddo
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz/2
            tmp                  = cwork(i      , j, k)
            cwork(i      , j, k) = cwork(isz-i+1, j, k)
            cwork(isz-i+1, j, k) = tmp
        enddo
        enddo
        enddo

        include 'cufft_preproc_redft01.f90'

        return
    end subroutine preproc_rodft01

    ! post-processing of a signal to perform a fast backward
    ! discrete sine transform (DST) with FFTs (see Makhoul 1980)
    subroutine posproc_rodft01(sz, work, cwork)
        implicit none
        integer,  dimension(3), intent(in   ) :: sz
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work
        real(fp_pois), dimension(:,:,:), device, intent(in), optional :: cwork !unused
        real(fp_pois), allocatable, dimension(:,:,:), device :: work_tmp
        integer  :: idir, isz, i, j, k, ii
        
        idir = 1
        isz = sz(idir)
        if(.not.allocated(work_tmp)) allocate(work_tmp(0:sz(idir)-1,sz(2),sz(3)))
        
        include 'cufft_posproc_redft01.f90'

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, isz
            if(mod(i,2) == 0) then
                work(i, j, k) = - work(i, j, k)
            endif
        enddo
        enddo
        enddo
        
        return
    end subroutine posproc_rodft01
    
    subroutine transLeading(idir, sz, array, sz_tr, array_tr)
        implicit none
        integer, intent(in) :: idir
        integer,  dimension(3), intent(in)  :: sz
        integer,  dimension(3), intent(out) :: sz_tr
        real(fp_pois), dimension(:,:,:), device, intent(in) :: array
        real(fp_pois), dimension(:,:,:), device, intent(out) :: array_tr
        integer :: i, j, k

        select case(idir)
        case(2)
            sz_tr(1) = sz(2)
            sz_tr(2) = sz(1)
            sz_tr(3) = sz(3)
            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, sz(3)
            do i = 1, sz(1)
            do j = 1, sz(2)
                array_tr(j, i, k) = array(i, j, k)
            enddo
            enddo
            enddo
        case(3)
            sz_tr(1) = sz(3)
            sz_tr(2) = sz(1)
            sz_tr(3) = sz(2)
            !$cuf kernel do(3) <<<*,*>>>
            do j = 1, sz(2)
            do i = 1, sz(1)
            do k = 1, sz(3)
                array_tr(k, i, j) = array(i, j, k)
            enddo
            enddo
            enddo
        end select

        return
    end subroutine transLeading

    subroutine executeCUFFT(cufft_plan, work)
        implicit none
        type(cufft_plan_wrapper), intent(in) :: cufft_plan
        real(fp_pois), dimension(:,:,:), device, intent(inout) :: work

        integer :: istat, idir, sz(3), sz_tr(3)

        sz(1) = size(work,1)
        sz(2) = size(work,2)
        sz(3) = size(work,3)

        ! execute out-of-place FFT
        idir = cufft_plan%fft_spatial_dir
        select case(cufft_plan%fft_type)
        case(CUFFT_FWD_TYPE)
            if (idir == 1) then
                call cufft_plan%preproc(sz, work)
                istat = CUFFT_EXEC_FWD(cufft_plan%plan, work, cwork)
                call cufft_plan%posproc(sz, work, cwork)
            else
                call transLeading(idir, sz, work, sz_tr, work_tr)
                call cufft_plan%preproc(sz_tr, work_tr)
                istat = CUFFT_EXEC_FWD(cufft_plan%plan, work_tr, cwork_tr)
                call cufft_plan%posproc(sz_tr, work_tr, cwork_tr)
                call transLeading(idir, sz_tr, work_tr, sz, work)
            endif
        case(CUFFT_BWD_TYPE)
            if (idir == 1) then
                call cufft_plan%preproc(sz, work, cwork)
                istat = CUFFT_EXEC_BWD(cufft_plan%plan, cwork, work)
                call cufft_plan%posproc(sz, work)
            else
                call transLeading(idir, sz, work, sz_tr, work_tr)
                call cufft_plan%preproc(sz_tr, work_tr, cwork_tr)
                istat = CUFFT_EXEC_BWD(cufft_plan%plan, cwork_tr, work_tr)
                call cufft_plan%posproc(sz_tr, work_tr)
                call transLeading(idir, sz_tr, work_tr, sz, work)
            endif
        end select

        return
    end subroutine executeCUFFT

    subroutine freeCUFFT(cufft_plan)
        implicit none
        type(cufft_plan_wrapper), dimension(2,2), intent(inout) :: cufft_plan
        integer :: idir, i, istat

        do idir = 1, 2
        do i = 1, 2
            istat = cufftDestroy(cufft_plan(i,idir)%plan)
        enddo
        enddo

        if (allocated(cufft_workarea)) deallocate(cufft_workarea)
        if (allocated(cwork   )) deallocate(cwork   )
        if (allocated( work_tr)) deallocate( work_tr)
        if (allocated(cwork_tr)) deallocate(cwork_tr)
        
        return
    end subroutine freeCUFFT

    function getMemoryFootprintCUFFT() result(tot_bytes)
        implicit none
        integer(i8) :: tot_bytes

        tot_bytes = (size(cufft_workarea)*int(2, i8) + size(cwork) + size(work_tr) + size(cwork_tr)) * int(fp_pois, i8)

    end function getMemoryFootprintCUFFT
#endif
    
end module mod_fft