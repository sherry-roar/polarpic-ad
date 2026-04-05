module mod_mpi
#ifdef USE_C
    use, intrinsic :: iso_c_binding, only: c_bool
    use decomp_2d_c, only: decomp_2d_init, decomp_2d_finalize, decomp_main
    use decomp_2d_c, only: decomp_2d_comm_cart_x, decomp_2d_comm_cart_y, decomp_2d_comm_cart_z
#else
    use decomp_2d
#endif
    
#ifdef _CUDA
    use cudafor
    use mod_device, only: initDevices, istat, mydev
#endif
    
    implicit none

    include 'mpif.h'

    ! make everything public unless declared private
    public

#ifdef _SINGLE_PREC
    integer, parameter :: REAL_FP = kind(1.0)
    integer, parameter :: MPI_REAL_FP = MPI_REAL
#else
    integer, parameter :: REAL_FP = kind(1.0d0)
    integer, parameter :: MPI_REAL_FP = MPI_DOUBLE_PRECISION
#endif

    integer, save :: myrank, ierr
    integer, save :: comm_cart
    integer, save :: comm_cart_xpen, comm_cart_ypen, comm_cart_zpen
    integer, dimension(6),   save :: halotype_vel, halotype_one
    integer, dimension(6,3), save :: neighbor_xyz
    integer, dimension(6),   save :: neighbor
    integer, dimension(2),   save :: coord_xpen, coord_ypen, coord_zpen
    integer, dimension(2),   save :: coord_pen
    integer, dimension(3),   save ::  st,  en,  sz
    integer, dimension(3),   save :: xst, xen, xsz
    integer, dimension(3),   save :: yst, yen, ysz
    integer, dimension(3),   save :: zst, zen, zsz
#ifdef NB_HALO
    ! 1: south
    ! 2: north
    ! 3: bottom
    ! 4: top
    ! 5: south_bottom
    ! 6: south_top
    ! 7: north_bottom
    ! 8: north_top
    integer, dimension(8),   save :: mpitype_nbhalo_vel
    integer, dimension(8),   save :: neighbor_nbhalo

#ifdef USE_NBHALOBUF

#ifdef USE_RDMA
    integer, parameter :: USE_HALOBUF_AUX = 1
#else
    integer, parameter :: USE_HALOBUF_AUX = 0
#endif

    integer, dimension(8), save :: halobuf_offset, halobuf_length
    real(REAL_FP), allocatable, dimension(:, :), target :: halobuf_send, halobuf_recv
    real(REAL_FP), allocatable, dimension(:, :), target :: halobuf_send_aux, halobuf_recv_aux
#ifdef CUDA_AWARE_MPI
    !@cuf attributes(device) :: halobuf_send, halobuf_recv
    !@cuf attributes(device) :: halobuf_send_aux, halobuf_recv_aux
#else
    !@cuf attributes(managed) :: halobuf_send, halobuf_recv
    !@cuf attributes(managed) :: halobuf_send_aux, halobuf_recv_aux
#endif

#endif
#endif

#ifdef CUDA_AWARE_MPI
    !@cuf real(REAL_FP), allocatable, dimension(:,:,:), device :: sendbuf_y0, recvbuf_y0, sendbuf_y1, recvbuf_y1
    !@cuf real(REAL_FP), allocatable, dimension(:,:,:), device :: sendbuf_z0, recvbuf_z0, sendbuf_z1, recvbuf_z1
#else
    !@cuf real(REAL_FP), allocatable, dimension(:,:,:), managed :: sendbuf_y0, recvbuf_y0, sendbuf_y1, recvbuf_y1
    !@cuf real(REAL_FP), allocatable, dimension(:,:,:), managed :: sendbuf_z0, recvbuf_z0, sendbuf_z1, recvbuf_z1
#endif

    contains
    subroutine initMPI(nx, ny, nz, bctype_p, p_row, p_col, nhalo, backend)
        implicit none
        integer, intent(in) :: nx, ny, nz
        integer, intent(in) :: p_row, p_col
        character(2), dimension(3), intent(in) :: bctype_p
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in), optional :: backend

        logical, dimension(3) :: periodic_bc
        integer :: bcount, bsize, bstride, oldtype
        !@cuf integer :: isz, i

        !@cuf call initDevices()

        periodic_bc(1:3) = .false.
        if ( bctype_p(1) == 'PP' ) periodic_bc(1) = .true.
        if ( bctype_p(2) == 'PP' ) periodic_bc(2) = .true.
        if ( bctype_p(3) == 'PP' ) periodic_bc(3) = .true.
        
#ifdef USE_C
        call decomp_2d_init(nx, ny, nz, p_row, p_col, logical(periodic_bc, c_bool))
#else
        if (present(backend)) then
            call decomp_2d_init(nx, ny, nz, p_row, p_col, periodic_bc, backend)
        else
            call decomp_2d_init(nx, ny, nz, p_row, p_col, periodic_bc)
        endif
#endif
        
        ! staring/ending index and size of data held by current processor       
#ifdef USE_C
        xst = decomp_main%xst; xen = decomp_main%xen; xsz = decomp_main%xsz
        yst = decomp_main%yst; yen = decomp_main%yen; ysz = decomp_main%ysz
        zst = decomp_main%zst; zen = decomp_main%zen; zsz = decomp_main%zsz
#else
        xst(:) = xstart(:); xen(:) = xend(:); xsz(:) = xsize(:)  ! x-pencil
        yst(:) = ystart(:); yen(:) = yend(:); ysz(:) = ysize(:)  ! y-pencil
        zst(:) = zstart(:); zen(:) = zend(:); zsz(:) = zsize(:)  ! z-pencil
#endif

        comm_cart_xpen = DECOMP_2D_COMM_CART_X
        comm_cart_ypen = DECOMP_2D_COMM_CART_Y
        comm_cart_zpen = DECOMP_2D_COMM_CART_Z
        
        ! Find the MPI ranks of neighboring pencils
        !  first dimension 1=west, 2=east, 3=south, 4=north, 5=bottom, 6=top
        ! second dimension 1=x-pencil, 2=y-pencil, 3=z-pencil
        ! x-pencil
        neighbor_xyz(1,1) = MPI_PROC_NULL
        neighbor_xyz(2,1) = MPI_PROC_NULL
        call MPI_CART_SHIFT(comm_cart_xpen, 0, 1, neighbor_xyz(3,1), neighbor_xyz(4,1), ierr)
        call MPI_CART_SHIFT(comm_cart_xpen, 1, 1, neighbor_xyz(5,1), neighbor_xyz(6,1), ierr)
        ! y-pencil
        call MPI_CART_SHIFT(comm_cart_ypen, 0, 1, neighbor_xyz(1,2), neighbor_xyz(2,2), ierr)
        neighbor_xyz(3,2) = MPI_PROC_NULL
        neighbor_xyz(4,2) = MPI_PROC_NULL
        call MPI_CART_SHIFT(comm_cart_ypen, 1, 1, neighbor_xyz(5,2), neighbor_xyz(6,2), ierr)
        ! z-pencil
        call MPI_CART_SHIFT(comm_cart_zpen, 0, 1, neighbor_xyz(1,3), neighbor_xyz(2,3), ierr)
        call MPI_CART_SHIFT(comm_cart_zpen, 1, 1, neighbor_xyz(3,3), neighbor_xyz(4,3), ierr)
        neighbor_xyz(5,3) = MPI_PROC_NULL
        neighbor_xyz(6,3) = MPI_PROC_NULL
        
        ! the coordinary of each process in the x-pencil decomposition
        call MPI_CART_COORDS(comm_cart_xpen, myrank, 2, coord_xpen, ierr)
        coord_ypen = coord_xpen
        coord_zpen = coord_xpen

        ! x-pencil
        st(:) = xst(:); en(:) = xen(:); sz(:) = xsz(:)
        comm_cart = comm_cart_xpen
        neighbor(:) = neighbor_xyz(:,1)
        coord_pen(:) = coord_xpen(:)

        call createHaloMPIType(nhalo, sz, MPI_REAL_FP, halotype_vel)
        call createHaloMPIType((/1,1,1,1,1,1/), sz, MPI_REAL_FP, halotype_one)

#ifdef NB_HALO
        call getNeighborRank2DCart(comm_cart, neighbor_nbhalo)
        call createNBHaloMPIType(nhalo, sz, MPI_REAL_FP, mpitype_nbhalo_vel)
        ! write(*,'(A,I3,A,2I3,A,8I3)') '>>> myrank = ', myrank, ', coord = ', coord_pen, ', nb_neighbor = ', neighbor_nbhalo

#ifdef USE_NBHALOBUF
        isz = sz(1) + nhalo(1) + nhalo(2)
        halobuf_length(1) = isz * nhalo(4) * sz(3)      ! y0
        halobuf_length(2) = isz * nhalo(3) * sz(3)      ! y1
        halobuf_length(3) = isz * sz(2) * nhalo(6)      ! z0
        halobuf_length(4) = isz * sz(2) * nhalo(5)      ! z1
        halobuf_length(5) = isz * nhalo(4) * nhalo(6)   ! y0z0
        halobuf_length(6) = isz * nhalo(3) * nhalo(5)   ! y1z1
        halobuf_length(7) = isz * nhalo(4) * nhalo(5)   ! y0z1
        halobuf_length(8) = isz * nhalo(3) * nhalo(6)   ! y1z0
        halobuf_offset(1) = 0
        do i = 2, 8
            halobuf_offset(i) = halobuf_offset(i-1) + halobuf_length(i-1)
        enddo
        allocate(halobuf_send(halobuf_offset(8) + halobuf_length(8), 3))
        allocate(halobuf_recv(halobuf_offset(8) + halobuf_length(8), 3))
        if (USE_HALOBUF_AUX == 1) then
            allocate(halobuf_send_aux(halobuf_offset(8) + halobuf_length(8), 3))
            allocate(halobuf_recv_aux(halobuf_offset(8) + halobuf_length(8), 3))
        endif
#if defined(_CUDA) && !defined(CUDA_AWARE_MPI)
        istat = cudaMemAdvise(        halobuf_send, size(halobuf_send), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        halobuf_send, size(halobuf_send), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( halobuf_send, size(halobuf_send), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        halobuf_recv, size(halobuf_recv), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        halobuf_recv, size(halobuf_recv), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( halobuf_recv, size(halobuf_recv), cudaCpuDeviceId, 0 )
        if (USE_HALOBUF_AUX == 1) then
            istat = cudaMemAdvise(        halobuf_send_aux, size(halobuf_send_aux), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
            istat = cudaMemAdvise(        halobuf_send_aux, size(halobuf_send_aux), cudaMemAdviseSetAccessedBy, mydev )
            istat = cudaMemPrefetchAsync( halobuf_send_aux, size(halobuf_send_aux), cudaCpuDeviceId, 0 )
            istat = cudaMemAdvise(        halobuf_recv_aux, size(halobuf_recv_aux), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
            istat = cudaMemAdvise(        halobuf_recv_aux, size(halobuf_recv_aux), cudaMemAdviseSetAccessedBy, mydev )
            istat = cudaMemPrefetchAsync( halobuf_recv_aux, size(halobuf_recv_aux), cudaCpuDeviceId, 0 )
        endif
#endif
#endif

#endif

#ifdef _CUDA
        allocate(sendbuf_y0( 1-nhalo(1):sz(1)+nhalo(2), nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)) )
        allocate(recvbuf_y0( 1-nhalo(1):sz(1)+nhalo(2), nhalo(3), 1-nhalo(5):sz(3)+nhalo(6)) )
        allocate(sendbuf_y1( 1-nhalo(1):sz(1)+nhalo(2), nhalo(3), 1-nhalo(5):sz(3)+nhalo(6)) )
        allocate(recvbuf_y1( 1-nhalo(1):sz(1)+nhalo(2), nhalo(4), 1-nhalo(5):sz(3)+nhalo(6)) )
        allocate(sendbuf_z0( 1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), nhalo(6)) )
        allocate(recvbuf_z0( 1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), nhalo(5)) )
        allocate(sendbuf_z1( 1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), nhalo(5)) )
        allocate(recvbuf_z1( 1-nhalo(1):sz(1)+nhalo(2), 1-nhalo(3):sz(2)+nhalo(4), nhalo(6)) )
#ifndef CUDA_AWARE_MPI
        istat = cudaMemAdvise(        sendbuf_y0, size(sendbuf_y0), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        sendbuf_y0, size(sendbuf_y0), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( sendbuf_y0, size(sendbuf_y0), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        recvbuf_y0, size(recvbuf_y0), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        recvbuf_y0, size(recvbuf_y0), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( recvbuf_y0, size(recvbuf_y0), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        sendbuf_y1, size(sendbuf_y1), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        sendbuf_y1, size(sendbuf_y1), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( sendbuf_y1, size(sendbuf_y1), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        recvbuf_y1, size(recvbuf_y1), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        recvbuf_y1, size(recvbuf_y1), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( recvbuf_y1, size(recvbuf_y1), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        sendbuf_z0, size(sendbuf_z0), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        sendbuf_z0, size(sendbuf_z0), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( sendbuf_z0, size(sendbuf_z0), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        recvbuf_z0, size(recvbuf_z0), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        recvbuf_z0, size(recvbuf_z0), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( recvbuf_z0, size(recvbuf_z0), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        sendbuf_z1, size(sendbuf_z1), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        sendbuf_z1, size(sendbuf_z1), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( sendbuf_z1, size(sendbuf_z1), cudaCpuDeviceId, 0 )
        istat = cudaMemAdvise(        recvbuf_z1, size(recvbuf_z1), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId )
        istat = cudaMemAdvise(        recvbuf_z1, size(recvbuf_z1), cudaMemAdviseSetAccessedBy, mydev )
        istat = cudaMemPrefetchAsync( recvbuf_z1, size(recvbuf_z1), cudaCpuDeviceId, 0 )
#endif
#endif

        return
    end subroutine initMPI

    subroutine createHaloMPIType(nhalo, sz, oldtype, halotype)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        integer, intent(in) :: oldtype
        integer, dimension(6), intent(out) :: halotype

        integer :: bcount, bsize, bstride, ierr

        ! halo comm in the west/east direction
        bcount  = (sz(2)+nhalo(3)+nhalo(4)) * (sz(3)+nhalo(5)+nhalo(6))
        bsize   = nhalo(1)
        bstride = sz(1)+nhalo(1)+nhalo(2)
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(1), ierr)
        call MPI_TYPE_COMMIT(halotype(1), ierr)
        bsize   = nhalo(2)
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(2), ierr)
        call MPI_TYPE_COMMIT(halotype(2), ierr)
        
        ! halo comm in the south/north direction
        bcount  = sz(3)+nhalo(5)+nhalo(6)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(3)
        bstride = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4))
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(3), ierr)
        call MPI_TYPE_COMMIT(halotype(3), ierr)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(4)
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(4), ierr)
        call MPI_TYPE_COMMIT(halotype(4), ierr)
        
        ! halo comm in the bottom/top direction
        bcount  = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4)) * nhalo(5)
        bsize   = 1
        bstride = 1
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(5), ierr)
        call MPI_TYPE_COMMIT(halotype(5), ierr)
        bcount  = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4)) * nhalo(6)
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(6), ierr)
        call MPI_TYPE_COMMIT(halotype(6), ierr)

        return
    end subroutine createHaloMPIType

    subroutine getNeighborRank2DCart(comm, neighbor)
        implicit none
        integer, intent(in) :: comm
        integer, dimension(8), intent(out) :: neighbor
        
        integer, dimension(2) :: dims, coords, coords_nb
        integer, dimension(-1:1) :: xcoords_nb, ycoords_nb
        logical, dimension(2) :: periods
        integer :: i, j, ierr

        call MPI_CART_GET(comm, 2, dims, periods, coords, ierr)
        
        xcoords_nb(-1) = coords(1)-1
        xcoords_nb( 0) = coords(1)
        xcoords_nb( 1) = coords(1)+1
        if (periods(1)) then
            xcoords_nb(-1) = modulo(xcoords_nb(-1), dims(1))
            xcoords_nb( 1) = modulo(xcoords_nb( 1), dims(1))
        endif
        
        ycoords_nb(-1) = coords(2)-1
        ycoords_nb( 0) = coords(2)
        ycoords_nb( 1) = coords(2)+1
        if (periods(2)) then
            ycoords_nb(-1) = modulo(ycoords_nb(-1), dims(2))
            ycoords_nb( 1) = modulo(ycoords_nb( 1), dims(2))
        endif

        coords_nb(1) = xcoords_nb(-1)
        coords_nb(2) = ycoords_nb( 0)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(1))
        coords_nb(1) = xcoords_nb( 1)
        coords_nb(2) = ycoords_nb( 0)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(2))
        coords_nb(1) = xcoords_nb( 0)
        coords_nb(2) = ycoords_nb(-1)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(3))
        coords_nb(1) = xcoords_nb( 0)
        coords_nb(2) = ycoords_nb( 1)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(4))
        coords_nb(1) = xcoords_nb(-1)
        coords_nb(2) = ycoords_nb(-1)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(5))
        coords_nb(1) = xcoords_nb(-1)
        coords_nb(2) = ycoords_nb( 1)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(6))
        coords_nb(1) = xcoords_nb( 1)
        coords_nb(2) = ycoords_nb(-1)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(7))
        coords_nb(1) = xcoords_nb( 1)
        coords_nb(2) = ycoords_nb( 1)
        call getMpiRankFromCoords2DCart(comm, coords_nb, dims, neighbor(8))

        return
    end subroutine getNeighborRank2DCart

    subroutine getMpiRankFromCoords2DCart(comm, coords, dims, rank)
        implicit none
        integer, intent(in) :: comm
        integer, dimension(2), intent(in) :: coords, dims
        integer, intent(out) :: rank
        
        integer :: ierr

        if( coords(1)<0 .or. coords(1)>=dims(1) .or. coords(2)<0 .or. coords(2)>=dims(2)) then
            rank = MPI_PROC_NULL
        else
            call MPI_CART_RANK(comm, coords, rank, ierr)
        endif

        return
    end subroutine getMpiRankFromCoords2DCart

    subroutine createNBHaloMPIType(nhalo, sz, oldtype, halotype)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        integer, intent(in) :: oldtype
        integer, dimension(8), intent(out) :: halotype

        integer :: bcount, bsize, bstride, ierr
        
        ! halo exchange in the south/north direction
        bcount  = sz(3)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(3)
        bstride = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4))
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(1), ierr)
        call MPI_TYPE_COMMIT(halotype(1), ierr)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(4)
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(2), ierr)
        call MPI_TYPE_COMMIT(halotype(2), ierr)
        
        ! halo exchange in the bottom/top direction
        bcount  = 1
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * sz(2) * nhalo(5)
        bstride = 1
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(3), ierr)
        call MPI_TYPE_COMMIT(halotype(3), ierr)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * sz(2) * nhalo(6)
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(4), ierr)
        call MPI_TYPE_COMMIT(halotype(4), ierr)

        ! halo exchange in the south_bottom direction
        bcount  = nhalo(5)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(3)
        bstride = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4))
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(5), ierr)
        call MPI_TYPE_COMMIT(halotype(5), ierr)

        ! halo exchange in the south_top direction
        bcount  = nhalo(6)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(3)
        bstride = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4))
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(6), ierr)
        call MPI_TYPE_COMMIT(halotype(6), ierr)

        ! halo exchange in the north_bottom direction
        bcount  = nhalo(5)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(4)
        bstride = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4))
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(7), ierr)
        call MPI_TYPE_COMMIT(halotype(7), ierr)

        ! halo exchange in the north_top direction
        bcount  = nhalo(6)
        bsize   = (sz(1)+nhalo(1)+nhalo(2)) * nhalo(4)
        bstride = (sz(1)+nhalo(1)+nhalo(2)) * (sz(2)+nhalo(3)+nhalo(4))
        call MPI_TYPE_VECTOR(bcount, bsize, bstride, oldtype, halotype(8), ierr)
        call MPI_TYPE_COMMIT(halotype(8), ierr)

        return
    end subroutine createNBHaloMPIType

    subroutine freeMPI()
        implicit none
        integer :: i

        call decomp_2d_finalize()

        do i = 1, 6
            call MPI_TYPE_FREE(halotype_vel(i), ierr)
            call MPI_TYPE_FREE(halotype_one(i), ierr)
        enddo
#ifdef NB_HALO
        do i = 1, 8
            call MPI_TYPE_FREE(mpitype_nbhalo_vel(i), ierr)
        enddo
#ifdef USE_NBHALOBUF
        deallocate(halobuf_send)
        deallocate(halobuf_recv)
        if (USE_HALOBUF_AUX == 1) then
            deallocate(halobuf_send_aux)
            deallocate(halobuf_recv_aux)
        endif
#endif
#endif

#ifdef _CUDA
        deallocate(sendbuf_y0)
        deallocate(recvbuf_y0)
        deallocate(sendbuf_y1)
        deallocate(recvbuf_y1)
        deallocate(sendbuf_z0)
        deallocate(recvbuf_z0)
        deallocate(sendbuf_z1)
        deallocate(recvbuf_z1)
#endif

        return
    end subroutine freeMPI

end module mod_mpi