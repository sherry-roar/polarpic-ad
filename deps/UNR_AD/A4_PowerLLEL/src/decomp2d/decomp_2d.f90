!=======================================================================
! This is part of the 2DECOMP&FFT library
! 
! 2DECOMP&FFT is a software framework for general-purpose 2D (pencil) 
! decomposition. It also implements a highly scalable distributed
! three-dimensional Fast Fourier Transform (FFT).
!
! Copyright (C) 2009-2012 Ning Li, the Numerical Algorithms Group (NAG)
!
!=======================================================================

! This is the main 2D pencil decomposition module

module decomp_2d

  !$ use omp_lib
  use gptl

#ifdef _CUDA
  use cudafor
#ifdef NCCL
  use nccl
#endif
#endif

#ifdef NVTX
  use nvtx
#endif

  implicit none

  include 'mpif.h'

  private        ! Make everything private unless declared public

! xiejb
#if defined(_SINGLE_PREC) || defined(SP_POIS)
#define SINGLE_PRECISION
#endif

#ifndef SINGLE_PRECISION
  integer, parameter, public :: mytype = KIND(0.0D0)
  integer, parameter, public :: real_type = MPI_DOUBLE_PRECISION
!   integer, parameter, public :: complex_type = MPI_DOUBLE_COMPLEX
#else
  integer, parameter, public :: mytype = KIND(0.0)
  integer, parameter, public :: real_type = MPI_REAL
!   integer, parameter, public :: complex_type = MPI_COMPLEX
#endif

  ! some key global variables
  integer, save, public :: nx_global, ny_global, nz_global  ! global size

  integer, save, public :: nrank  ! local MPI rank 
  integer, save, public :: nproc  ! total number of processors

  ! parameters for 2D Cartesian topology 
  integer, save, dimension(2) :: dims, coord
  logical, save, dimension(2) :: periodic
  integer, save, public :: DECOMP_2D_COMM_CART_X, &
       DECOMP_2D_COMM_CART_Y, DECOMP_2D_COMM_CART_Z 
  integer, save :: DECOMP_2D_COMM_ROW, DECOMP_2D_COMM_COL
  integer, save :: row_rank, col_rank

  ! flags for periodic condition in three dimensions
  logical, save :: periodic_x, periodic_y, periodic_z

  ! derived type to store decomposition info for a given global data size
  TYPE, public :: DECOMP_INFO
     ! staring/ending index and size of data held by current processor
     integer, dimension(3) :: xst, xen, xsz  ! x-pencil
     integer, dimension(3) :: yst, yen, ysz  ! y-pencil
     integer, dimension(3) :: zst, zen, zsz  ! z-pencil

     ! in addition to local information, processors also need to know 
     ! some global information for global communications to work 

     ! how each dimension is distributed along pencils
     integer, allocatable, dimension(:) :: &
          x1dist, y1dist, y2dist, z2dist, &
          x1st, y1st, y2st, z2st, &
          x1en, y1en, y2en, z2en

     ! send/receive buffer counts and displacements for MPI_ALLTOALLV
     integer, allocatable, dimension(:) :: &
          x1cnts, y1cnts, y2cnts, z2cnts
     integer, allocatable, dimension(:) :: &
          x1disp, y1disp, y2disp, z2disp
#ifdef _CUDA
     integer, allocatable, dimension(:) :: &
          x1idx, y1idx, y2idx, z2idx
#endif
     ! buffer counts for MPI_ALLTOALL: either for evenly distributed data
     ! or for padded-alltoall
     integer :: x1count, y1count, y2count, z2count

     ! evenly distributed data
     logical :: even

  END TYPE DECOMP_INFO

  ! main (default) decomposition information for global size nx*ny*nz
  TYPE(DECOMP_INFO), save :: decomp_main

  ! staring/ending index and size of data held by current processor
  ! duplicate 'decomp_main', needed by apps to define data structure 
  integer, save, dimension(3), public :: xstart, xend, xsize  ! x-pencil
  integer, save, dimension(3), public :: ystart, yend, ysize  ! y-pencil
  integer, save, dimension(3), public :: zstart, zend, zsize  ! z-pencil

  ! These are the buffers used by MPI_ALLTOALL(V) calls
  integer, save :: decomp_buf_size = 0
  real(mytype),    allocatable, dimension(:) :: work1_r, work2_r
!   complex(mytype), allocatable, dimension(:) :: work1_c, work2_c

  ! Communication backends
  integer, save :: backend_type
  integer, parameter :: BACKEND_RDMA = 0
  integer, parameter :: BACKEND_MPI_A2A = 1
  integer, parameter :: BACKEND_MPI_P2P = 2
  integer, parameter :: BACKEND_NCCL    = 3
  integer, allocatable, dimension(:) :: reqs_a2a

#ifdef USE_RDMA
  integer, allocatable, dimension(:) :: disps_rmt_recv_x2y, disps_rmt_recv_y2x

  interface
    subroutine init_rdma_xy_alltoallv_c(decomp_2d_comm_col, decomp_buf_length, work1_r, work2_r, &
      counts_send_x2y, disps_loc_send_x2y, disps_rmt_recv_x2y, &
      counts_send_y2x, disps_loc_send_y2x, disps_rmt_recv_y2x) bind(C, name='init_rdma_xy_alltoallv_c')
      use, intrinsic :: iso_c_binding
      integer(C_INT), value :: decomp_2d_comm_col
      integer(C_INT), value :: decomp_buf_length
#ifdef SINGLE_PRECISION
      real(C_FLOAT), dimension(*), intent(inout) :: work1_r, work2_r
#else
      real(C_DOUBLE), dimension(*), intent(inout) :: work1_r, work2_r
#endif
#ifdef CUDA_AWARE_MPI
      !@cuf attributes(device) :: work1_r, work2_r
#else
      !@cuf attributes(pinned) :: work1_r, work2_r
#endif
      integer(C_INT), dimension(*), intent(in) :: counts_send_x2y, disps_loc_send_x2y, disps_rmt_recv_x2y
      integer(C_INT), dimension(*), intent(in) :: counts_send_y2x, disps_loc_send_y2x, disps_rmt_recv_y2x
    end subroutine

    subroutine rdma_x2y_alltoallv_c() bind(C, name='rdma_x2y_alltoallv_c')
      
    end subroutine

    subroutine rdma_y2x_alltoallv_c() bind(C, name='rdma_y2x_alltoallv_c')

    end subroutine

    subroutine rdma_x2y_alltoallv_wait_c() bind(C, name='rdma_x2y_alltoallv_wait_c')
      
    end subroutine

    subroutine rdma_y2x_alltoallv_wait_c() bind(C, name='rdma_y2x_alltoallv_wait_c')

    end subroutine
  end interface
  
  public :: init_rdma_xy_alltoallv_c
#endif
! #ifdef USE_RDMA

#ifdef _CUDA
#ifdef CUDA_AWARE_MPI
  real(mytype), allocatable, dimension(:), device :: work1_r_d, work2_r_d
#else
  real(mytype), allocatable, dimension(:), pinned :: work1_r_d, work2_r_d
#endif
  type(cudaEvent), allocatable, dimension(:) :: a2a_event
  integer(kind=cuda_stream_kind), public, save :: a2a_comp, a2a_h2d, a2a_d2h
  integer, allocatable, dimension(:) :: a2a_requests

#ifdef NCCL
#ifndef SINGLE_PRECISION
  type(ncclDataType), parameter, public :: nccl_real_type = ncclDouble
#else
  type(ncclDataType), parameter, public :: nccl_real_type = ncclFloat
#endif
  type(ncclResult) :: nccl_stat
  type(ncclUniqueId) :: nccl_uid
  type(ncclComm), save :: nccl_comm
  integer(kind=cuda_stream_kind), save :: nccl_stream
#endif

  public :: getMemoryFootprintDecomp2d
#endif
! #ifdef _CUDA

  ! public user routines
  public :: decomp_2d_init, decomp_2d_finalize, &
       transpose_x_to_y, transpose_y_to_z, &
       transpose_z_to_y, transpose_y_to_x, &
       decomp_info_init, decomp_info_finalize, partition, &
       decomp_2d_abort, get_decomp_info


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! These are routines to perform global data transpositions
  ! 
  !   Four combinations are available, enough to cover all situations
  !    - transpose_x_to_y (X-pencil --> Y-pencil)
  !    - transpose_y_to_z (Y-pencil --> Z-pencil)
  !    - transpose_z_to_y (Z-pencil --> Y-pencil)
  !    - transpose_y_to_x (Y-pencil --> X-pencil)
  !
  !   Generic interface provided here to support multiple data types
  !    - real and complex types supported through generic interface
  !    - single/double precision supported through pre-processing
  !       * see 'mytype' variable at the beginning
  !    - an optional argument can be supplied to transpose data whose 
  !      global size is not the default nx*ny*nz 
  !       * as the case in fft r2c/c2r interface 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  interface transpose_x_to_y
#ifdef _CUDA
     module procedure transpose_x_to_y_real_d
#endif
     module procedure transpose_x_to_y_real
   !   module procedure transpose_x_to_y_complex
  end interface transpose_x_to_y
  
  interface transpose_y_to_z
#ifdef _CUDA
     module procedure transpose_y_to_z_real_d
#endif
     module procedure transpose_y_to_z_real
   !   module procedure transpose_y_to_z_complex
  end interface transpose_y_to_z
  
  interface transpose_z_to_y
#ifdef _CUDA
     module procedure transpose_z_to_y_real_d
#endif
     module procedure transpose_z_to_y_real
   !   module procedure transpose_z_to_y_complex
  end interface transpose_z_to_y

  interface transpose_y_to_x
#ifdef _CUDA
     module procedure transpose_y_to_x_real_d
#endif
     module procedure transpose_y_to_x_real
   !   module procedure transpose_y_to_x_complex
  end interface transpose_y_to_x

contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Routine to be called by applications to initialise this library
  !   INPUT:
  !     nx, ny, nz   - global data dimension
  !     p_row, p_col - 2D processor grid
  !   OUTPUT:
  !     all internal data structures initialised properly
  !     library ready to use
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine decomp_2d_init(nx,ny,nz,p_row,p_col,periodic_bc,backend)

    implicit none

    integer, intent(IN) :: nx,ny,nz,p_row,p_col
    logical, dimension(3), intent(IN), optional :: periodic_bc
    integer, intent(in), optional :: backend
    
    integer :: errorcode, ierror, row, col
    integer :: m, pos
    logical :: passed
#ifdef _CUDA
    integer :: istat
#endif

    nx_global = nx
    ny_global = ny
    nz_global = nz

    if (present(periodic_bc)) then
       periodic_x = periodic_bc(1)
       periodic_y = periodic_bc(2)
       periodic_z = periodic_bc(3)
    else
       periodic_x = .false.
       periodic_y = .false.
       periodic_z = .false.
    end if

    call MPI_COMM_RANK(MPI_COMM_WORLD,nrank,ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierror)

    if (nproc /= p_row*p_col) then
      errorcode = 1
      call decomp_2d_abort(errorcode, &
            'Invalid 2D processor grid - nproc /= p_row*p_col')
    else
      row = p_row
      col = p_col
    end if
    
    ! Create 2D Catersian topology
    ! Note that in order to support periodic B.C. in the halo-cell code,
    ! need to create multiple topology objects: DECOMP_2D_COMM_CART_?,
    ! corresponding to three pencil orientations. They contain almost
    ! identical topological information but allow different combinations
    ! of periodic conditions.
    dims(1) = row
    dims(2) = col
    periodic(1) = periodic_y
    periodic(2) = periodic_z
    call MPI_CART_CREATE(MPI_COMM_WORLD,2,dims,periodic, &
         .false., &  ! do not reorder rank
         DECOMP_2D_COMM_CART_X, ierror)
    periodic(1) = periodic_x
    periodic(2) = periodic_z
    call MPI_CART_CREATE(MPI_COMM_WORLD,2,dims,periodic, &
         .false., DECOMP_2D_COMM_CART_Y, ierror)
    periodic(1) = periodic_x
    periodic(2) = periodic_y
    call MPI_CART_CREATE(MPI_COMM_WORLD,2,dims,periodic, &
         .false., DECOMP_2D_COMM_CART_Z, ierror)

    call MPI_CART_COORDS(DECOMP_2D_COMM_CART_X,nrank,2,coord,ierror)
    
    ! derive communicators defining sub-groups for ALLTOALL(V)
    call MPI_CART_SUB(DECOMP_2D_COMM_CART_X,(/.true.,.false./), &
         DECOMP_2D_COMM_COL,ierror)
    call MPI_CART_SUB(DECOMP_2D_COMM_CART_X,(/.false.,.true./), &
         DECOMP_2D_COMM_ROW,ierror)
    call MPI_COMM_RANK(DECOMP_2D_COMM_COL,col_rank,ierror)
    call MPI_COMM_RANK(DECOMP_2D_COMM_ROW,row_rank,ierror)
    
    ! actually generate all 2D decomposition information
    call decomp_info_init(nx,ny,nz,decomp_main)

    ! make a copy of the decomposition information associated with the
    ! default global size in these global variables so applications can
    ! use them to create data structures 
    xstart = decomp_main%xst
    ystart = decomp_main%yst
    zstart = decomp_main%zst
    xend   = decomp_main%xen
    yend   = decomp_main%yen
    zend   = decomp_main%zen
    xsize  = decomp_main%xsz
    ysize  = decomp_main%ysz
    zsize  = decomp_main%zsz
    
#ifdef EVEN
    if (nrank==0) write(*,*) 'Padded ALLTOALL optimisation on'
#endif 

    ! Set the communication backend
    if (present(backend)) then
      passed = .false.
#ifdef USE_RDMA
      passed = passed .or. (backend == BACKEND_RDMA)
#endif
      passed = passed .or. (backend == BACKEND_MPI_A2A)
      passed = passed .or. (backend == BACKEND_MPI_P2P)
#if defined(_CUDA) && defined(NCCL)
      passed = passed .or. (backend == BACKEND_NCCL)
#endif
      if (.not. passed) then
        call decomp_2d_abort(11, 'Invalid Comm backend!')
      endif
      backend_type = backend
    else
      backend_type = BACKEND_MPI_A2A
    endif

    allocate( reqs_a2a(2*(row-1)) )

    ! Initialize the RDMA backend
#ifdef USE_RDMA
    allocate(disps_rmt_recv_x2y(0:dims(1)-1))
    allocate(disps_rmt_recv_y2x(0:dims(1)-1))
    do m = 0, dims(1)-1
      disps_rmt_recv_x2y(m) = decomp_main%x1dist(m) * (decomp_main%y1st(col_rank) - 1) * decomp_main%ysz(3)
      disps_rmt_recv_y2x(m) = (decomp_main%x1st(col_rank) - 1) * decomp_main%y1dist(m) * decomp_main%xsz(3)
    enddo
#ifndef _CUDA
    call init_rdma_xy_alltoallv_c(DECOMP_2D_COMM_COL, decomp_buf_size, work1_r, work2_r, &
                                  decomp_main%x1cnts, decomp_main%x1disp, disps_rmt_recv_x2y, &
                                  decomp_main%y1cnts, decomp_main%y1disp, disps_rmt_recv_y2x)
#else
    call init_rdma_xy_alltoallv_c(DECOMP_2D_COMM_COL, decomp_buf_size, work1_r_d, work2_r_d, &
                                  decomp_main%x1cnts, decomp_main%x1disp, disps_rmt_recv_x2y, &
                                  decomp_main%y1cnts, decomp_main%y1disp, disps_rmt_recv_y2x)
#endif

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
#endif

#ifdef _CUDA
    allocate( a2a_event(0:nproc*2) )
    allocate( a2a_requests(0:nproc*2) )
    do m=0,nproc*2
      istat = cudaEventCreate( a2a_event(m) )
    enddo
    istat = cudaStreamCreate( a2a_comp )
    istat = cudaStreamCreate( a2a_d2h  )
    istat = cudaStreamCreate( a2a_h2d  )

#ifdef NCCL
    ! initialize NCCL
    if (nrank == 0) then
      nccl_stat = ncclGetUniqueId(nccl_uid)
    end if
    call MPI_BCAST(nccl_uid, int( sizeof(ncclUniqueId), kind = 4 ), MPI_BYTE, 0, MPI_COMM_WORLD, ierror)
    nccl_stat = ncclCommInitRank(nccl_comm, nproc, nccl_uid, nrank)
    istat = cudaStreamCreate(nccl_stream)
#endif
#endif

    return
  end subroutine decomp_2d_init
  

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Routine to be called by applications to clean things up
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine decomp_2d_finalize

    implicit none
    
    call decomp_info_finalize(decomp_main)

    decomp_buf_size = 0
    deallocate(work1_r, work2_r)
   !  deallocate(work1_c, work2_c)

#ifdef USE_RDMA
    deallocate(disps_rmt_recv_x2y)
    deallocate(disps_rmt_recv_y2x)
#endif
#ifdef _CUDA
    deallocate(work1_r_d, work2_r_d)
#endif

    return
  end subroutine decomp_2d_finalize

#ifdef _CUDA
  function getMemoryFootprintDecomp2d() result(tot_bytes)
    implicit none
    integer, parameter :: i8 = selected_int_kind(18)
    integer(i8) :: tot_bytes

    tot_bytes = decomp_buf_size * 2 * int(mytype, i8)

  end function getMemoryFootprintDecomp2d
#endif


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Return the default decomposition object
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine get_decomp_info(decomp)

    implicit none

    TYPE(DECOMP_INFO), intent(OUT) :: decomp

    decomp = decomp_main

    return
  end subroutine get_decomp_info
    

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Advanced Interface allowing applications to define globle domain of
  ! any size, distribute it, and then transpose data among pencils.
  !  - generate 2D decomposition details as defined in DECOMP_INFO
  !  - the default global data size is nx*ny*nz
  !  - a different global size nx/2+1,ny,nz is used in FFT r2c/c2r
  !  - multiple global sizes can co-exist in one application, each
  !    using its own DECOMP_INFO object
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine decomp_info_init(nx,ny,nz,decomp)

    implicit none
    
    integer, intent(IN) :: nx,ny,nz
    TYPE(DECOMP_INFO), intent(INOUT) :: decomp

    integer :: buf_size, status, errorcode
#ifdef _CUDA
    integer :: i
#endif
    ! verify the global size can actually be distributed as pencils
    if (nx<dims(1) .or. ny<dims(1) .or. ny<dims(2) .or. nz<dims(2)) then
       errorcode = 6
       call decomp_2d_abort(errorcode, &
            'Invalid 2D processor grid. ' // &
            'Make sure that min(nx,ny) >= p_row and ' // &
            'min(ny,nz) >= p_col')
    end if
    
    if (mod(nx,dims(1))==0 .and. mod(ny,dims(1))==0 .and. &
         mod(ny,dims(2))==0 .and. mod(nz,dims(2))==0) then
       decomp%even = .true.
    else
       decomp%even = .false.
    end if

    ! distribute mesh points
    !allocate(decomp%x1dist(0:dims(1)-1),decomp%y1dist(0:dims(1)-1), &
    !     decomp%y2dist(0:dims(2)-1),decomp%z2dist(0:dims(2)-1))
    allocate(decomp%x1dist(0:dims(1)-1),decomp%y1dist(0:dims(1)-1), &
         decomp%y2dist(0:dims(2)-1),decomp%z2dist(0:dims(2)-1))
    allocate(decomp%x1st(0:dims(1)-1),decomp%x1en(0:dims(1)-1), &
         decomp%y1st(0:dims(1)-1),decomp%y1en(0:dims(1)-1))
    allocate(decomp%y2st(0:dims(2)-1),decomp%y2en(0:dims(2)-1), &
         decomp%z2st(0:dims(2)-1),decomp%z2en(0:dims(2)-1))
    call get_dist(nx,ny,nz,decomp)
    
#ifdef _CUDA
    allocate(decomp%x1idx(0:dims(1)-1),decomp%y1idx(0:dims(1)-1), &
             decomp%y2idx(0:dims(2)-1),decomp%z2idx(0:dims(2)-1))

    decomp%x1idx(0) = 1
    decomp%y1idx(0) = 1
    do i=1,dims(1)-1
      decomp%x1idx(i) = decomp%x1idx(i-1) + decomp%x1dist(i-1)
      decomp%y1idx(i) = decomp%y1idx(i-1) + decomp%y1dist(i-1)
    end do

    decomp%y2idx(0) = 1
    decomp%z2idx(0) = 1
    do i=1,dims(2)-1
      decomp%y2idx(i) = decomp%y2idx(i-1) + decomp%y2dist(i-1)
      decomp%z2idx(i) = decomp%z2idx(i-1) + decomp%z2dist(i-1)
    end do
#endif

    ! generate partition information - starting/ending index etc.
    call partition(nx, ny, nz, (/ 1,2,3 /), &
         decomp%xst, decomp%xen, decomp%xsz)
    call partition(nx, ny, nz, (/ 2,1,3 /), &
         decomp%yst, decomp%yen, decomp%ysz)
    call partition(nx, ny, nz, (/ 2,3,1 /), &
         decomp%zst, decomp%zen, decomp%zsz)
    
    ! prepare send/receive buffer displacement and count for ALLTOALL(V)
    !allocate(decomp%x1cnts(0:dims(1)-1),decomp%y1cnts(0:dims(1)-1), &
    !     decomp%y2cnts(0:dims(2)-1),decomp%z2cnts(0:dims(2)-1))
    !allocate(decomp%x1disp(0:dims(1)-1),decomp%y1disp(0:dims(1)-1), &
    !     decomp%y2disp(0:dims(2)-1),decomp%z2disp(0:dims(2)-1))
    allocate(decomp%x1cnts(0:dims(1)-1),decomp%y1cnts(0:dims(1)-1), &
         decomp%y2cnts(0:dims(2)-1),decomp%z2cnts(0:dims(2)-1))
    allocate(decomp%x1disp(0:dims(1)-1),decomp%y1disp(0:dims(1)-1), &
         decomp%y2disp(0:dims(2)-1),decomp%z2disp(0:dims(2)-1))
    call prepare_buffer(decomp)

    ! allocate memory for the MPI_ALLTOALL(V) buffers
    ! define the buffers globally for performance reason
    
    buf_size = max(decomp%xsz(1)*decomp%xsz(2)*decomp%xsz(3), &
         max(decomp%ysz(1)*decomp%ysz(2)*decomp%ysz(3), &
         decomp%zsz(1)*decomp%zsz(2)*decomp%zsz(3)) )
#ifdef EVEN
    ! padded alltoall optimisation may need larger buffer space
    buf_size = max(buf_size, &
         max(decomp%x1count*dims(1),decomp%y2count*dims(2)) ) 
#endif

    ! check if additional memory is required
    ! *** TODO: consider how to share the real/complex buffers 
    if (buf_size > decomp_buf_size) then
       decomp_buf_size = buf_size
#ifdef _CUDA
       if (allocated(work1_r_d)) deallocate(work1_r_d)
       if (allocated(work2_r_d)) deallocate(work2_r_d)
       allocate(work1_r_d(buf_size), STAT=status)
       allocate(work2_r_d(buf_size), STAT=status)
#endif
       if (allocated(work1_r)) deallocate(work1_r)
       if (allocated(work2_r)) deallocate(work2_r)
       allocate(work1_r(buf_size), STAT=status)
       allocate(work2_r(buf_size), STAT=status)
       !  if (allocated(work1_c)) deallocate(work1_c)
       !  if (allocated(work2_c)) deallocate(work2_c)
       !  allocate(work1_c(buf_size), STAT=status)
       !  allocate(work2_c(buf_size), STAT=status)
       if (status /= 0) then
          errorcode = 2
          call decomp_2d_abort(errorcode, &
               'Out of memory when allocating 2DECOMP workspace')
       end if
    end if

    return
  end subroutine decomp_info_init


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Release memory associated with a DECOMP_INFO object
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine decomp_info_finalize(decomp)

    implicit none

    integer :: i
    integer :: ierror
    TYPE(DECOMP_INFO), intent(INOUT) :: decomp

    deallocate(decomp%x1dist,decomp%y1dist,decomp%y2dist,decomp%z2dist)
    deallocate(decomp%x1st,decomp%y1st,decomp%y2st,decomp%z2st)
    deallocate(decomp%x1en,decomp%y1en,decomp%y2en,decomp%z2en)
    deallocate(decomp%x1cnts,decomp%y1cnts,decomp%y2cnts,decomp%z2cnts)
    deallocate(decomp%x1disp,decomp%y1disp,decomp%y2disp,decomp%z2disp)
#ifdef _CUDA
    deallocate(decomp%x1idx,decomp%y1idx)
    deallocate(decomp%y2idx,decomp%z2idx)
#endif

    return
  end subroutine decomp_info_finalize


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Find sub-domain information held by current processor
  !   INPUT: 
  !     nx, ny, nz - global data dimension
  !     pdim(3)    - number of processor grid in each dimension, 
  !                  valid values: 1 - distibute locally; 
  !                                2 - distribute across p_row; 
  !                                3 - distribute across p_col
  !   OUTPUT:
  !     lstart(3)  - starting index
  !     lend(3)    - ending index
  !     lsize(3)   - size of the sub-block (redundant) 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine partition(nx, ny, nz, pdim, lstart, lend, lsize)

    implicit none

    integer, intent(IN) :: nx, ny, nz
    integer, dimension(3), intent(IN) :: pdim
    integer, dimension(3), intent(OUT) :: lstart, lend, lsize

    integer, allocatable, dimension(:) :: st,en,sz
    integer :: i, gsize

    do i = 1, 3
 
      if (i==1) then
        gsize = nx
      else if (i==2) then
        gsize = ny
      else if (i==3) then
        gsize = nz
      end if

      if (pdim(i) == 1) then        ! all local
        lstart(i) = 1
        lend(i)   = gsize
        lsize(i)  = gsize
      elseif (pdim(i) == 2) then    ! distribute across dims(1)
        allocate(st(0:dims(1)-1))
        allocate(en(0:dims(1)-1))
        allocate(sz(0:dims(1)-1))
        call distribute(gsize,dims(1),st,en,sz)
        lstart(i) = st(coord(1))
        lend(i)   = en(coord(1))
        lsize(i)  = sz(coord(1))
        deallocate(st,en,sz)
      elseif (pdim(i) == 3) then    ! distribute across dims(2)
        allocate(st(0:dims(2)-1))
        allocate(en(0:dims(2)-1))
        allocate(sz(0:dims(2)-1))
        call distribute(gsize,dims(2),st,en,sz)
        lstart(i) = st(coord(2))
        lend(i)   = en(coord(2))
        lsize(i)  = sz(coord(2))
        deallocate(st,en,sz)
      end if    

    end do
    return   

  end subroutine partition

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   - distibutes grid points in one dimension
  !   - handles uneven distribution properly 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
  subroutine distribute(data1,proc,st,en,sz)
  
    implicit none
    ! data1 -- data size in any dimension to be partitioned
    ! proc  -- number of processors in that dimension
    ! st    -- array of starting index
    ! en    -- array of ending index
    ! sz    -- array of local size  (redundent)
    integer data1,proc,st(0:proc-1),en(0:proc-1),sz(0:proc-1)
    integer i,size1,nl,nu
  
    size1=data1/proc
    nu = data1 - size1 * proc
    nl = proc - nu
    st(0) = 1
    sz(0) = size1
    en(0) = size1
    do i=1,nl-1
      st(i) = st(i-1) + size1
      sz(i) = size1
      en(i) = en(i-1) + size1
    end do
    size1 = size1 + 1
    do i=nl,proc-1
      st(i) = en(i-1) + 1
      sz(i) = size1
      en(i) = en(i-1) + size1
    end do
    en(proc-1)= data1 
    sz(proc-1)= data1-st(proc-1)+1
  
    return
  end subroutine distribute

  subroutine distribute_among_threads(c_or_f, nthreads, tid, num, factor, sz, st, en)
      implicit none
      character, intent(in) :: c_or_f  ! return index(st,en) in c/fortran style
      integer, intent(in) :: nthreads, tid, num, factor
      integer, intent(out) :: sz, st, en

      integer :: nloops, remainder
    
      sz = num/nthreads
      nloops = sz/factor
      sz = nloops*factor
      remainder = num - sz*nthreads

      if (tid < remainder/factor) then
         sz = sz + factor
         st = tid * sz
      else
         st = remainder + tid * sz
      endif
      en = st + sz

      if (c_or_f == 'f') st = st + 1

      ! e.g., distribute 7 among 3 threads (factor = 1)
      ! thread :       0           1          2
      ! c      : [0, 1, 2, 3), [3, 4, 5), [5, 6, 7)
      ! fortran: [1, 2, 3],    [4, 5],    [6, 7] 

      return
  end subroutine distribute_among_threads

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Define how each dimension is distributed across processors
  !    e.g. 17 meshes across 4 processor would be distibuted as (4,4,4,5)
  !    such global information is required locally at MPI_ALLTOALLV time
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine get_dist(nx,ny,nz,decomp)

    integer, intent(IN) :: nx, ny, nz
    TYPE(DECOMP_INFO), intent(INOUT) :: decomp

    call distribute(nx,dims(1),decomp%x1st,decomp%x1en,decomp%x1dist)
    call distribute(ny,dims(1),decomp%y1st,decomp%y1en,decomp%y1dist)
    call distribute(ny,dims(2),decomp%y2st,decomp%y2en,decomp%y2dist)
    call distribute(nz,dims(2),decomp%z2st,decomp%z2en,decomp%z2dist)

    return
  end subroutine get_dist

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Prepare the send / receive buffers for MPI_ALLTOALLV communications
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine prepare_buffer(decomp)
    
    implicit none
    
    TYPE(DECOMP_INFO), intent(INOUT) :: decomp

    integer :: i, k
    integer :: rank_x, rank_z
    integer :: subsize_y, offset_y
    integer :: ierror

    ! MPI_ALLTOALLV buffer information
    do i=0, dims(1)-1
       decomp%x1cnts(i) = decomp%x1dist(i)*decomp%xsz(2)*decomp%xsz(3)
       decomp%y1cnts(i) = decomp%ysz(1)*decomp%y1dist(i)*decomp%ysz(3)
       if (i==0) then
          decomp%x1disp(i) = 0  ! displacement is 0-based index
          decomp%y1disp(i) = 0
       else
          decomp%x1disp(i) = decomp%x1disp(i-1) + decomp%x1cnts(i-1)
          decomp%y1disp(i) = decomp%y1disp(i-1) + decomp%y1cnts(i-1)
       end if
    end do

    do i=0, dims(2)-1
       decomp%y2cnts(i) = decomp%ysz(1)*decomp%y2dist(i)*decomp%ysz(3)
       decomp%z2cnts(i) = decomp%zsz(1)*decomp%zsz(2)*decomp%z2dist(i)
       if (i==0) then
          decomp%y2disp(i) = 0  ! displacement is 0-based index
          decomp%z2disp(i) = 0
       else
          decomp%y2disp(i) = decomp%y2disp(i-1) + decomp%y2cnts(i-1)
          decomp%z2disp(i) = decomp%z2disp(i-1) + decomp%z2cnts(i-1)
       end if
    end do
    
    ! MPI_ALLTOALL buffer information

    ! For evenly distributed data, following is an easier implementation.
    ! But it should be covered by the more general formulation below.
    !decomp%x1count = decomp%xsz(1)*decomp%xsz(2)*decomp%xsz(3)/dims(1)
    !decomp%y1count = decomp%ysz(1)*decomp%ysz(2)*decomp%ysz(3)/dims(1) 
    !decomp%y2count = decomp%ysz(1)*decomp%ysz(2)*decomp%ysz(3)/dims(2)
    !decomp%z2count = decomp%zsz(1)*decomp%zsz(2)*decomp%zsz(3)/dims(2)

    ! For unevenly distributed data, pad smaller messages. Note the 
    ! last blocks along pencils always get assigned more mesh points
    ! for X <=> Y transposes
    decomp%x1count = decomp%x1dist(dims(1)-1) * &
         decomp%y1dist(dims(1)-1) * decomp%xsz(3)
    decomp%y1count = decomp%x1count
    ! for Y <=> Z transposes
    decomp%y2count = decomp%y2dist(dims(2)-1) * &
         decomp%z2dist(dims(2)-1) * decomp%zsz(1)
    decomp%z2count = decomp%y2count
    
    return
  end subroutine prepare_buffer  


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Transposition routines 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#include "transpose_x_to_y.f90"
#include "transpose_y_to_z.f90"
#include "transpose_z_to_y.f90"
#include "transpose_y_to_x.f90"


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Error handling
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine decomp_2d_abort(errorcode, msg)

    implicit none

    integer, intent(IN) :: errorcode
    character(len=*), intent(IN) :: msg

    integer :: ierror
    
    if (nrank==0) then
       write(*,*) '2DECOMP&FFT ERROR - errorcode: ', errorcode
       write(*,*) 'ERROR MESSAGE: ' // msg
    end if
    call MPI_ABORT(MPI_COMM_WORLD,errorcode,ierror)

    return
  end subroutine decomp_2d_abort
    
  
end module decomp_2d

