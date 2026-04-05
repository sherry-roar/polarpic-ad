module mod_updateBound
    use, intrinsic :: iso_c_binding
    use mod_type,       only: fp
    use mod_parameters, only: nhalo, nhalo_one
    use mod_mpi,        only: sz, halotype_vel, halotype_one, neighbor, comm_cart, ierr
#ifdef NB_HALO
    use mod_mpi,        only: mpitype_nbhalo_vel, neighbor_nbhalo
#ifdef USE_NBHALOBUF
    use mod_mpi,        only: halobuf_offset
    use mod_mpi,        only: halobuf_send, halobuf_recv
    use mod_mpi,        only: halobuf_send_aux, halobuf_recv_aux
#endif
#endif
    !$ use omp_lib
    use gptl
#ifdef _CUDA
    use mod_mpi, only: sendbuf_y0, recvbuf_y0, sendbuf_y1, recvbuf_y1
    use mod_mpi, only: sendbuf_z0, recvbuf_z0, sendbuf_z1, recvbuf_z1
    use mod_mpi, only: MPI_REAL_FP
    use cudafor
#endif

    implicit none

    include 'mpif.h'

    ! make everything private unless declared public
    private

    interface
        subroutine update_bound_vel(comm, halotype_vel, neighbor, nhalo, sz, u, v, w, u_crf, tag) bind(C, name='update_bound_vel')
            import
            integer(C_INT), value :: comm
            integer(C_INT), dimension(6), intent(in) :: halotype_vel
            integer(C_INT), dimension(6), intent(in) :: neighbor
            integer(C_INT), dimension(6), intent(in) :: nhalo
            integer(C_INT), dimension(3), intent(in) :: sz
            real(C_DOUBLE), dimension(*), intent(inout) :: u, v, w
            real(C_DOUBLE), value :: u_crf
            character(kind=c_char), intent(in) :: tag(*)
        end subroutine update_bound_vel
    end interface

    interface
        subroutine update_bound_p(comm, halotype_p, neighbor, nhalo, sz, p, tag) bind(C, name='update_bound_p')
            import
            integer(C_INT), value :: comm
            integer(C_INT), dimension(6), intent(in) :: halotype_p
            integer(C_INT), dimension(6), intent(in) :: neighbor
            integer(C_INT), dimension(6), intent(in) :: nhalo
            integer(C_INT), dimension(3), intent(in) :: sz
            real(C_DOUBLE), dimension(*), intent(inout) :: p
            character(kind=c_char), intent(in) :: tag(*)
        end subroutine update_bound_p
    end interface

#ifdef NB_HALO
    interface
#if defined(USE_C) || defined(USE_RDMA)
        subroutine get_neighbor_rank_2d_cart_c(comm) bind(C, name='get_neighbor_rank_2d_cart_c')
            import
            integer(C_INT), value :: comm
        end subroutine get_neighbor_rank_2d_cart_c

        subroutine create_nbhalo_mpitype_c(nhalo, sz, oldtype) bind(C, name='create_nbhalo_mpitype_c')
            import
            integer(C_INT), dimension(6), intent(in) :: nhalo
            integer(C_INT), dimension(3), intent(in) :: sz
            integer(C_INT), value, intent(in) :: oldtype
        end subroutine create_nbhalo_mpitype_c
        
        subroutine free_nbhalo_mpitype() bind(C, name='free_nbhalo_mpitype')

        end subroutine free_nbhalo_mpitype
#endif

#ifdef USE_RDMA
#ifdef USE_NBHALOBUF
        subroutine init_rdma_halo_c(nhalo, sz, halobuf_length, halobuf_offset, &
            halobuf_send, halobuf_recv, halobuf_send_aux, halobuf_recv_aux, comm) bind(C, name='init_rdma_halo_c')
            import
            integer(C_INT), dimension(6), intent(in) :: nhalo
            integer(C_INT), dimension(3), intent(in) :: sz
            integer(C_INT), dimension(8), intent(in) :: halobuf_length
            integer(C_INT), dimension(8), intent(in) :: halobuf_offset
            real(C_DOUBLE), dimension(*), intent(in) :: halobuf_send, halobuf_recv
            real(C_DOUBLE), dimension(*), intent(in) :: halobuf_send_aux, halobuf_recv_aux
#ifdef CUDA_AWARE_MPI
            !@cuf attributes(device) :: halobuf_send, halobuf_recv
            !@cuf attributes(device) :: halobuf_send_aux, halobuf_recv_aux
#else
            !@cuf attributes(managed) :: halobuf_send, halobuf_recv
            !@cuf attributes(managed) :: halobuf_send_aux, halobuf_recv_aux
#endif
            integer(C_INT), value :: comm
        end subroutine init_rdma_halo_c
#else
        subroutine init_rdma_halo_c(nhalo, sz, u, v, w, u1, v1, w1, comm) bind(C, name='init_rdma_halo_c')
            import
            integer(C_INT), dimension(6), intent(in) :: nhalo
            integer(C_INT), dimension(3), intent(in) :: sz
            real(C_DOUBLE), dimension(*), intent(in) :: u, v, w
            real(C_DOUBLE), dimension(*), intent(in) :: u1, v1, w1
            !@cuf attributes(managed) :: u, v, w, u1, v1, w1
            integer(C_INT), value :: comm
        end subroutine init_rdma_halo_c
#endif
        subroutine update_halo_rdma_send(rk_num) bind(C, name='update_halo_rdma_send')
            import
            integer(C_INT), value, intent(in) :: rk_num
        end subroutine update_halo_rdma_send

        subroutine update_halo_rdma_wait(rk_num) bind(C, name='update_halo_rdma_wait')
            import
            integer(C_INT), value, intent(in) :: rk_num
        end subroutine update_halo_rdma_wait
#endif
    end interface
#endif

    public :: updateBoundVel, imposeBCVel, updateBoundCenteredVel, updateBoundP, updateHalo, &
#ifdef NB_HALO
              updateHaloISend, updateHaloIRecv, updateHaloWaitall, &
#if defined(USE_C) || defined(USE_RDMA)
              get_neighbor_rank_2d_cart_c, create_nbhalo_mpitype_c, free_nbhalo_mpitype, &
#endif
#ifdef USE_RDMA
              init_rdma_halo_c, update_halo_rdma_send, update_halo_rdma_wait, &
#endif
#ifdef USE_NBHALOBUF
              memcpyToHaloBuf, memcpyFromHaloBuf, updateHaloBufISend, updateHaloBufIRecv, &
#endif
#endif
              update_bound_vel, update_bound_p
    
contains
    subroutine updateBoundVel(u, v, w, u_crf, tag)
        implicit none
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: u, v, w
        real(fp), intent(in) :: u_crf
        character(*), intent(in) :: tag
        !@cuf attributes(managed) :: u, v, w

        integer :: ibound, idir

        call RGPTLSTART('--Update halo vel '//tag)

        call updateHalo(nhalo, halotype_vel, u)
        call updateHalo(nhalo, halotype_vel, v)
        call updateHalo(nhalo, halotype_vel, w)

        call RGPTLSTOP('--Update halo vel '//tag)
        call RGPTLSTART('--Impose BC vel '//tag)

        call imposeBCVel(u, v, w, u_crf)

        call RGPTLSTOP('--Impose BC vel '//tag)
        
        return
    end subroutine updateBoundVel

    subroutine imposeBCVel(u, v, w, u_crf)
        implicit none
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: u, v, w
        real(fp), intent(in) :: u_crf
        !@cuf attributes(managed) :: u, v, w

        integer :: ibound, idir
        
        ! B.C. in x direction
        idir = 1
        if (neighbor(1) == MPI_PROC_NULL) then
            ibound = 0
            call imposePeriodicBC(nhalo, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo, sz, ibound, idir, w)
        endif
        if (neighbor(2) == MPI_PROC_NULL) then
            ibound = 1
            call imposePeriodicBC(nhalo, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo, sz, ibound, idir, w)
        endif
        ! B.C. in y direction
        idir = 2
        if (neighbor(3) == MPI_PROC_NULL) then
            ibound = 0
            call imposePeriodicBC(nhalo, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo, sz, ibound, idir, w)
        endif
        if (neighbor(4) == MPI_PROC_NULL) then
            ibound = 1
            call imposePeriodicBC(nhalo, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo, sz, ibound, idir, w)
        endif
        ! B.C. in z direction
        idir = 3
        if (neighbor(5) == MPI_PROC_NULL) then
            ibound = 0
            call imposeNoSlipBC(nhalo, sz, ibound, .true.,  u, u_crf)
            call imposeNoSlipBC(nhalo, sz, ibound, .true.,  v)
            call imposeNoSlipBC(nhalo, sz, ibound, .false., w)
        endif
        if (neighbor(6) == MPI_PROC_NULL) then
            ibound = 1
            call imposeNoSlipBC(nhalo, sz, ibound, .true.,  u, u_crf)
            call imposeNoSlipBC(nhalo, sz, ibound, .true.,  v)
            call imposeNoSlipBC(nhalo, sz, ibound, .false., w)
        endif

        return
    end subroutine imposeBCVel

    subroutine updateBoundCenteredVel(u, v, w)
        implicit none
        real(fp), dimension(0:,0:,0:), intent(inout) :: u, v, w
        !@cuf attributes(managed) :: u, v, w
        !@cuf integer :: istat

        integer :: ibound, idir
        integer :: i, j, k

        call updateHalo(nhalo_one, halotype_one, u)
        call updateHalo(nhalo_one, halotype_one, v)
        call updateHalo(nhalo_one, halotype_one, w)

        ! B.C. in x direction
        idir = 1
        if (neighbor(1) == MPI_PROC_NULL) then
            ibound = 0
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, w)
        endif
        if (neighbor(2) == MPI_PROC_NULL) then
            ibound = 1
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, w)
        endif
        ! B.C. in y direction
        idir = 2
        if (neighbor(3) == MPI_PROC_NULL) then
            ibound = 0
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, w)
        endif
        if (neighbor(4) == MPI_PROC_NULL) then
            ibound = 1
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, u)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, v)
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, w)
        endif
        ! B.C. in z direction
        idir = 3
        if (neighbor(5) == MPI_PROC_NULL) then
#ifdef _CUDA
            !$cuf kernel do(2) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
            do j = 1-nhalo_one(3), sz(2)+nhalo_one(4)
            do i = 1-nhalo_one(1), sz(1)+nhalo_one(2)
                u(i,j,0) = 0.0_fp
                v(i,j,0) = 0.0_fp
                w(i,j,0) = 0.0_fp
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#else
            istat=cudaDeviceSynchronize()
#endif
            ! ibound = 0
            ! call imposeNoSlipBC(nhalo_one, sz, ibound, .false., u)
            ! call imposeNoSlipBC(nhalo_one, sz, ibound, .false., v)
            ! call imposeNoSlipBC(nhalo_one, sz, ibound, .false., w)
        endif
        if (neighbor(6) == MPI_PROC_NULL) then
            k = sz(3)+1
#ifdef _CUDA
            !$cuf kernel do(2) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
            do j = 1-nhalo_one(3), sz(2)+nhalo_one(4)
            do i = 1-nhalo_one(1), sz(1)+nhalo_one(2)
                u(i,j,k) = 0.0_fp
                v(i,j,k) = 0.0_fp
                w(i,j,k) = 0.0_fp
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#else
            istat=cudaDeviceSynchronize()
#endif
            ! ibound = 1
            ! call imposeNoSlipBC(nhalo_one, sz, ibound, .true., u)
            ! call imposeNoSlipBC(nhalo_one, sz, ibound, .true., v)
            ! call imposeNoSlipBC(nhalo_one, sz, ibound, .true., w)
        endif
        
        return
    end subroutine updateBoundCenteredVel
    
    subroutine updateBoundP(p, tag)
        implicit none
        real(fp), dimension(0:,0:,0:), intent(inout) :: p
        character(*), intent(in) :: tag
        !@cuf attributes(managed) :: p
        !@cuf integer :: istat

        integer :: ibound, idir

        call RGPTLSTART('--Update halo pres '//tag)

        call updateHalo(nhalo_one, halotype_one, p)

        call RGPTLSTOP('--Update halo pres '//tag)
        call RGPTLSTART('--Impose BC pres '//tag)

        ! B.C. in x direction
        idir = 1
        if (neighbor(1) == MPI_PROC_NULL) then
            ibound = 0
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, p)
        endif
        if (neighbor(2) == MPI_PROC_NULL) then
            ibound = 1
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, p)
        endif
        ! B.C. in y direction
        idir = 2
        if (neighbor(3) == MPI_PROC_NULL) then
            ibound = 0
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, p)
        endif
        if (neighbor(4) == MPI_PROC_NULL) then
            ibound = 1
            call imposePeriodicBC(nhalo_one, sz, ibound, idir, p)
        endif
        ! B.C. in z direction
        idir = 3
        if (neighbor(5) == MPI_PROC_NULL) then
            ibound = 0
            call imposeZeroGradBC(nhalo_one, sz, ibound, p)
        endif
        if (neighbor(6) == MPI_PROC_NULL) then
            ibound = 1
            call imposeZeroGradBC(nhalo_one, sz, ibound, p)
        endif

        call RGPTLSTOP('--Impose BC pres '//tag)
        
        return
    end subroutine updateBoundP

    subroutine imposePeriodicBC(nhalo, sz, ibound, idir, var)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        integer, intent(in) :: ibound, idir
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        !@cuf attributes(managed) :: var
        !@cuf integer :: istat, n

        integer :: i, j, k

#ifdef _CUDA
        if (idir == 1) then
            n = sz(1)
            select case(ibound)
            case(0)
                !$cuf kernel do(3) <<<*,*>>>
                do k = 1-nhalo(5), sz(3)+nhalo(6)
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                    do i = 1, nhalo(1)
                        var(1-i, j, k) = var(n+1-i, j, k)
                    enddo
                enddo
                enddo
            case(1)
                !$cuf kernel do(3) <<<*,*>>>
                do k = 1-nhalo(5), sz(3)+nhalo(6)
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                    do i = 1, nhalo(2)
                        var(n+i, j, k) = var(i, j, k)
                    enddo
                enddo
                enddo
            end select
            istat=cudaDeviceSynchronize()
        endif
#else
        if (idir == 1) then
            select case(ibound)
            case(0)
                !$OMP PARALLEL DO SCHEDULE(STATIC)
                do k = 1-nhalo(5), sz(3)+nhalo(6)
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                    var(1-nhalo(1):0, j, k) = var(sz(1)+1-nhalo(1):sz(1), j, k)
                enddo
                enddo
                !$OMP END PARALLEL DO
            case(1)
                !$OMP PARALLEL DO SCHEDULE(STATIC)
                do k = 1-nhalo(5), sz(3)+nhalo(6)
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                    var(sz(1)+1:sz(1)+nhalo(2), j, k) = var(1:nhalo(2), j, k)
                enddo
                enddo
                !$OMP END PARALLEL DO
            end select
        endif
#endif

        return
    end subroutine imposePeriodicBC

    subroutine imposeNoSlipBC(nhalo, sz, ibound, centered, var, vel_crf)
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        integer, intent(in) :: ibound
        logical, intent(in) :: centered
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        real(fp), optional, intent(in) :: vel_crf
        !@cuf attributes(managed) :: var
        !@cuf integer :: istat

        real(fp) :: bcvalue
        integer :: i, j, k, n

        bcvalue = 0.0_fp
        if (present(vel_crf)) bcvalue = 0.0_fp - vel_crf

        select case(ibound)
        case(0)
            if (centered) then
#ifdef _CUDA
                !$cuf kernel do(2) <<<*,*>>>
#else
                !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                do i = 1-nhalo(1), sz(1)+nhalo(2)
                    var(i,j,0  ) = 2.0_fp*bcvalue - var(i,j,1)
                enddo
                enddo
#ifndef _CUDA
                !$OMP END PARALLEL DO
#endif
            else
#ifdef _CUDA
                !$cuf kernel do(2) <<<*,*>>>
#else
                !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                do i = 1-nhalo(1), sz(1)+nhalo(2)
                    var(i,j,0  ) = bcvalue
                enddo
                enddo
#ifndef _CUDA
                !$OMP END PARALLEL DO
#endif
            endif
        case(1)
            n = sz(3)
            if (centered) then
#ifdef _CUDA
                !$cuf kernel do(2) <<<*,*>>>
#else
                !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                do i = 1-nhalo(1), sz(1)+nhalo(2)
                    var(i,j,n+1) = 2.0_fp*bcvalue - var(i,j,n)
                enddo
                enddo
#ifndef _CUDA
                !$OMP END PARALLEL DO
#endif
            else
#ifdef _CUDA
                !$cuf kernel do(2) <<<*,*>>>
#else
                !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
                do j = 1-nhalo(3), sz(2)+nhalo(4)
                do i = 1-nhalo(1), sz(1)+nhalo(2)
                    var(i,j,n  ) = bcvalue
                    var(i,j,n+1) = bcvalue
                enddo
                enddo
#ifndef _CUDA
                !$OMP END PARALLEL DO
#endif
            endif
        end select
        !@cuf istat=cudaDeviceSynchronize()

        return
    end subroutine imposeNoSlipBC

    subroutine imposeZeroGradBC(nhalo, sz, ibound, var)
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        integer, intent(in) :: ibound
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        !@cuf attributes(managed) :: var
        !@cuf integer :: istat

        integer :: i, j, k, n

        select case(ibound)
        case(0)
#ifdef _CUDA
            !$cuf kernel do(2) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
            do j = 1-nhalo(3), sz(2)+nhalo(4)
            do i = 1-nhalo(1), sz(1)+nhalo(2)
                var(i,j,0  ) = var(i,j,1)
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#endif
        case(1)
            n = sz(3)
#ifdef _CUDA
            !$cuf kernel do(2) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC)
#endif
            do j = 1-nhalo(3), sz(2)+nhalo(4)
            do i = 1-nhalo(1), sz(1)+nhalo(2)
                var(i,j,n+1) = var(i,j,n)
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#endif

        end select
        !@cuf istat=cudaDeviceSynchronize()

        return
    end subroutine imposeZeroGradBC
    
    subroutine updateHalo(nhalo, halotype, var)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(6), intent(in) :: halotype
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        !@cuf attributes(managed) :: var
        !@cuf integer :: i, j, k, istat, n, nh

        integer :: tag_halo = 0

        ! update halo cells
#ifdef _CUDA
        ! *** south/north ***
        !$cuf kernel do(3) <<<*,*>>>
        do k = lbound(var,3), ubound(var,3)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            sendbuf_y0(i, j, k) = var(i, j, k)
        enddo
        enddo
        enddo
        n = sz(2)-nhalo(3)
        !$cuf kernel do(3) <<<*,*>>>
        do k = lbound(var,3), ubound(var,3)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            sendbuf_y1(i, j, k) = var(i, n+j, k)
        enddo
        enddo
        enddo
        istat = cudaDeviceSynchronize()

        call RGPTLSTART('----Halo exchange in S/N')
        call MPI_SENDRECV(sendbuf_y0, size(sendbuf_y0), MPI_REAL_FP, neighbor(3), tag_halo, &
                          recvbuf_y1, size(recvbuf_y1), MPI_REAL_FP, neighbor(4) ,tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(sendbuf_y1, size(sendbuf_y1), MPI_REAL_FP, neighbor(4), tag_halo, &
                          recvbuf_y0, size(recvbuf_y0), MPI_REAL_FP, neighbor(3) ,tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call RGPTLSTOP('----Halo exchange in S/N')

        n = sz(2)
        !$cuf kernel do(3) <<<*,*>>>
        do k = lbound(var,3), ubound(var,3)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            var(i, n+j, k) = recvbuf_y1(i, j, k)
        enddo
        enddo
        enddo
        n = nhalo(3)
        !$cuf kernel do(3) <<<*,*>>>
        do k = lbound(var,3), ubound(var,3)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            var(i, j-n, k) = recvbuf_y0(i, j, k)
        enddo
        enddo
        enddo
        istat = cudaDeviceSynchronize()
        
        ! *** bottom/top ***
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nhalo(6)
        do j = lbound(var,2), ubound(var,2)
        do i = lbound(var,1), ubound(var,1)
            sendbuf_z0(i, j, k) = var(i, j, k)
        enddo
        enddo
        enddo
        n = sz(3)-nhalo(5)
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nhalo(5)
        do j = lbound(var,2), ubound(var,2)
        do i = lbound(var,1), ubound(var,1)
            sendbuf_z1(i, j, k) = var(i, j, n+k)
        enddo
        enddo
        enddo
        istat = cudaDeviceSynchronize()

        call RGPTLSTART('----Halo exchange in B/T')
        call MPI_SENDRECV(sendbuf_z0, size(sendbuf_z0), MPI_REAL_FP, neighbor(5), tag_halo, &
                          recvbuf_z1, size(recvbuf_z1), MPI_REAL_FP, neighbor(6) ,tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(sendbuf_z1, size(sendbuf_z1), MPI_REAL_FP, neighbor(6), tag_halo, &
                          recvbuf_z0, size(recvbuf_z0), MPI_REAL_FP, neighbor(5) ,tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call RGPTLSTOP('----Halo exchange in B/T')

        n = sz(3)
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nhalo(6)
        do j = lbound(var,2), ubound(var,2)
        do i = lbound(var,1), ubound(var,1)
            var(i, j, n+k) = recvbuf_z1(i, j, k)
        enddo
        enddo
        enddo
        n = nhalo(5)
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nhalo(5)
        do j = lbound(var,2), ubound(var,2)
        do i = lbound(var,1), ubound(var,1)
            var(i, j, k-n) = recvbuf_z0(i, j, k)
        enddo
        enddo
        enddo
        istat = cudaDeviceSynchronize()
#else
        ! *** west/east ***
        ! call MPI_SENDRECV(var(1      , 1-nhalo(3), 1-nhalo(5)), 1, halotype(2), neighbor(1), tag_halo, &
        !                   var(sz(1)+1, 1-nhalo(3), 1-nhalo(5)), 1, halotype(2), neighbor(2), tag_halo, &
        !                   comm_cart, MPI_STATUS_IGNORE, ierr)
        ! call MPI_SENDRECV(var(sz(1)+1-nhalo(1), 1-nhalo(3), 1-nhalo(5)), 1, halotype(1), neighbor(2), tag_halo, &
        !                   var(1-nhalo(1)      , 1-nhalo(3), 1-nhalo(5)), 1, halotype(1), neighbor(1), tag_halo, &
        !                   comm_cart, MPI_STATUS_IGNORE, ierr)
        ! *** south/north ***
        call MPI_SENDRECV(var(1-nhalo(1), 1      , 1-nhalo(5)), 1, halotype(4), neighbor(3), tag_halo, &
                          var(1-nhalo(1), sz(2)+1, 1-nhalo(5)), 1, halotype(4), neighbor(4), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(var(1-nhalo(1), sz(2)+1-nhalo(3), 1-nhalo(5)), 1, halotype(3), neighbor(4), tag_halo, &
                          var(1-nhalo(1), 1-nhalo(3)      , 1-nhalo(5)), 1, halotype(3), neighbor(3), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        ! *** bottom/top ***
        call MPI_SENDRECV(var(1-nhalo(1), 1-nhalo(3), 1      ), 1, halotype(6), neighbor(5), tag_halo, &
                          var(1-nhalo(1), 1-nhalo(3), sz(3)+1), 1, halotype(6), neighbor(6), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(var(1-nhalo(1), 1-nhalo(3), sz(3)+1-nhalo(5)), 1, halotype(5), neighbor(6), tag_halo, &
                          var(1-nhalo(1), 1-nhalo(3), 1-nhalo(5)      ), 1, halotype(5), neighbor(5), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
#endif

        return
    end subroutine updateHalo

#ifdef NB_HALO

#ifdef USE_NBHALOBUF
    subroutine memcpyToHaloBuf(nhalo, tag, var, use_aux)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in) :: tag
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        !@cuf attributes(managed) :: var
        integer, intent(in), optional :: use_aux
        
        integer :: ist, isz, jsz, m, n, idx
        !@cuf integer :: istat
        real(fp), dimension(:, :), pointer :: halobuf_ptr
#ifdef CUDA_AWARE_MPI
        !@cuf attributes(device) :: halobuf_ptr
#else
        !@cuf attributes(managed) :: halobuf_ptr
#endif

        ist = 1 - nhalo(1)
        isz = sz(1) + nhalo(1) + nhalo(2)
        jsz = sz(2) + nhalo(3) + nhalo(4)
        m = sz(2)-nhalo(3)
        n = sz(3)-nhalo(5)

        if (present(use_aux) .and. use_aux == 1) then
            halobuf_ptr => halobuf_send_aux
        else
            halobuf_ptr => halobuf_send
        endif

#ifndef _CUDA
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, k, idx)
        ! *** south ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, sz(3)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_y0(i, j, k) = var(i, j, k)
            idx = 1 + halobuf_offset(1) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, j, k)
        enddo
        enddo
        enddo
        ! *** north ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, sz(3)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_y1(i, j, k) = var(i, m+j, k)
            idx = 1 + halobuf_offset(2) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, m+j, k)
        enddo
        enddo
        enddo
        ! *** bottom ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(6)
        do j = 1, sz(2)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_z0(i, j, k) = var(i, j, k)
            idx = 1 + halobuf_offset(3) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, j, k)
        enddo
        enddo
        enddo
        ! *** top ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(5)
        do j = 1, sz(2)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_z1(i, j, k) = var(i, j, n+k)
            idx = 1 + halobuf_offset(4) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, j, n+k)
        enddo
        enddo
        enddo
        ! *** south_bottom ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(6)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_y0z0(i, j, k) = var(i, j, k)
            idx = 1 + halobuf_offset(5) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, j, k)
        enddo
        enddo
        enddo
        ! *** north_top ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(5)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_y1z1(i, j, k) = var(i, m+j, n+k)
            idx = 1 + halobuf_offset(6) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, m+j, n+k)
        enddo
        enddo
        enddo
        ! *** south_top ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(5)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_y0z1(i, j, k) = var(i, j, n+k)
            idx = 1 + halobuf_offset(7) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, j, n+k)
        enddo
        enddo
        enddo
        ! *** north_bottom ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(6)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            ! sendbuf_y1z0(i, j, k) = var(i, m+j, k)
            idx = 1 + halobuf_offset(8) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            halobuf_ptr(idx, tag) = var(i, m+j, k)
        enddo
        enddo
        enddo
        !$OMP END PARALLEL
#else
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(1)+1, tag), isz*nhalo(4), var(ist, 1, 1), isz*jsz, isz*nhalo(4), sz(3))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(2)+1, tag), isz*nhalo(3), var(ist, m+1, 1), isz*jsz, isz*nhalo(3), sz(3))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(3)+1, tag), isz*sz(2), var(ist, 1, 1), isz*jsz, isz*sz(2), nhalo(6))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(4)+1, tag), isz*sz(2), var(ist, 1, n+1), isz*jsz, isz*sz(2), nhalo(5))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(5)+1, tag), isz*nhalo(4), var(ist, 1, 1), isz*jsz, isz*nhalo(4), nhalo(6))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(6)+1, tag), isz*nhalo(3), var(ist, m+1, n+1), isz*jsz, isz*nhalo(3), nhalo(5))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(7)+1, tag), isz*nhalo(4), var(ist, 1, n+1), isz*jsz, isz*nhalo(4), nhalo(5))
        istat = cudaMemcpy2D(halobuf_ptr(halobuf_offset(8)+1, tag), isz*nhalo(3), var(ist, m+1, 1), isz*jsz, isz*nhalo(3), nhalo(6))
#endif

        return
    end subroutine memcpyToHaloBuf

    subroutine memcpyFromHaloBuf(nhalo, tag, var, use_aux)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in) :: tag
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        !@cuf attributes(managed) :: var
        integer, intent(in), optional :: use_aux

        integer :: ist, isz, jsz, idx
        integer :: m1, m2, n1, n2
        !@cuf integer :: istat
        real(fp), dimension(:, :), pointer :: halobuf_ptr
#ifdef CUDA_AWARE_MPI
        !@cuf attributes(device) :: halobuf_ptr
#else
        !@cuf attributes(managed) :: halobuf_ptr
#endif

        ist = 1 - nhalo(1)
        isz = sz(1) + nhalo(1) + nhalo(2)
        jsz = sz(2) + nhalo(3) + nhalo(4)
        m1 = nhalo(3)
        m2 = sz(2)
        n1 = nhalo(5)
        n2 = sz(3)

        if (present(use_aux) .and. use_aux == 1) then
            halobuf_ptr => halobuf_recv_aux
        else
            halobuf_ptr => halobuf_recv
        endif

#ifndef _CUDA
        !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(i, j, k, idx)
        ! *** north ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, sz(3)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, m2+j, k) = recvbuf_y1(i, j, k)
            idx = 1 + halobuf_offset(1) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, m2+j, k) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** south ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, sz(3)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, j-m1, k) = recvbuf_y0(i, j, k)
            idx = 1 + halobuf_offset(2) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, j-m1, k) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** top ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(6)
        do j = 1, sz(2)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, j, n2+k) = recvbuf_z1(i, j, k)
            idx = 1 + halobuf_offset(3) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, j, n2+k) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** bottom ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(5)
        do j = 1, sz(2)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, j, k-n1) = recvbuf_z0(i, j, k)
            idx = 1 + halobuf_offset(4) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, j, k-n1) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** north_top ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(6)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, m2+j, n2+k) = recvbuf_y1z1(i, j, k)
            idx = 1 + halobuf_offset(5) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, m2+j, n2+k) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** south_bottom ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(5)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, j-m1, k-n1) = recvbuf_y0z0(i, j, k)
            idx = 1 + halobuf_offset(6) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, j-m1, k-n1) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** north_bottom ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(5)
        do j = 1, nhalo(4)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, m2+j, k-n1) = recvbuf_y1z0(i, j, k)
            idx = 1 + halobuf_offset(7) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, m2+j, k-n1) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        ! *** south_top ***
        !$OMP DO SCHEDULE(STATIC) COLLAPSE(2) &
        do k = 1, nhalo(6)
        do j = 1, nhalo(3)
        do i = lbound(var,1), ubound(var,1)
            ! var(i, j-m1, n2+k) = recvbuf_y0z1(i, j, k)
            idx = 1 + halobuf_offset(8) + (i - ist) + (j - 1) * isz + (k - 1) * isz * jsz
            var(i, j-m1, n2+k) = halobuf_ptr(idx, tag)
        enddo
        enddo
        enddo
        !$OMP END PARALLEL
#else
        istat = cudaMemcpy2D(var(ist, m2+1, 1), isz*jsz, halobuf_ptr(halobuf_offset(1)+1, tag), isz*nhalo(4), isz*nhalo(4), sz(3))
        istat = cudaMemcpy2D(var(ist, 1-m1, 1), isz*jsz, halobuf_ptr(halobuf_offset(2)+1, tag), isz*nhalo(3), isz*nhalo(3), sz(3))
        istat = cudaMemcpy2D(var(ist, 1, n2+1), isz*jsz, halobuf_ptr(halobuf_offset(3)+1, tag), isz*sz(2), isz*sz(2), nhalo(6))
        istat = cudaMemcpy2D(var(ist, 1, 1-n1), isz*jsz, halobuf_ptr(halobuf_offset(4)+1, tag), isz*sz(2), isz*sz(2), nhalo(5))
        istat = cudaMemcpy2D(var(ist, m2+1, n2+1), isz*jsz, halobuf_ptr(halobuf_offset(5)+1, tag), isz*nhalo(4), isz*nhalo(4), nhalo(6))
        istat = cudaMemcpy2D(var(ist, 1-m1, 1-n1), isz*jsz, halobuf_ptr(halobuf_offset(6)+1, tag), isz*nhalo(3), isz*nhalo(3), nhalo(5))
        istat = cudaMemcpy2D(var(ist, m2+1, 1-n1), isz*jsz, halobuf_ptr(halobuf_offset(7)+1, tag), isz*nhalo(4), isz*nhalo(4), nhalo(5))
        istat = cudaMemcpy2D(var(ist, 1-m1, n2+1), isz*jsz, halobuf_ptr(halobuf_offset(8)+1, tag), isz*nhalo(3), isz*nhalo(3), nhalo(6))
#endif

        return
    end subroutine memcpyFromHaloBuf

    subroutine updateHaloBufIRecv(nhalo, tag, var, irecv_req, use_aux)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in) :: tag
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        integer, dimension(8), intent(out) :: irecv_req
        !@cuf attributes(managed) :: var
        integer, intent(in) :: use_aux        

        integer :: isz
        real(fp), dimension(:, :), pointer :: halobuf_ptr
#ifdef CUDA_AWARE_MPI
        !@cuf attributes(device) :: halobuf_ptr
#else
        !@cuf attributes(managed) :: halobuf_ptr
#endif

        if (use_aux == 1) then
            halobuf_ptr => halobuf_recv_aux
        else
            halobuf_ptr => halobuf_recv
        endif

        isz = sz(1) + nhalo(1) + nhalo(2)
        call MPI_IRECV(halobuf_ptr(halobuf_offset(1)+1, tag), isz*nhalo(4)*sz(3), MPI_REAL_FP, neighbor_nbhalo(2), &
                       tag, comm_cart, irecv_req(2), ierr)   ! y1 recv y0 send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(2)+1, tag), isz*nhalo(3)*sz(3), MPI_REAL_FP, neighbor_nbhalo(1), &
                       tag, comm_cart, irecv_req(1), ierr)   ! y0 recv y1 send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(3)+1, tag), isz*sz(2)*nhalo(6), MPI_REAL_FP, neighbor_nbhalo(4), &
                       tag, comm_cart, irecv_req(4), ierr)   ! z1 recv z0 send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(4)+1, tag), isz*sz(2)*nhalo(5), MPI_REAL_FP, neighbor_nbhalo(3), &
                       tag, comm_cart, irecv_req(3), ierr)   ! z0 recv z1 send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(5)+1, tag), isz*nhalo(4)*nhalo(6), MPI_REAL_FP, neighbor_nbhalo(8), &
                       tag, comm_cart, irecv_req(8), ierr) ! north_top recv south_bottom send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(6)+1, tag), isz*nhalo(3)*nhalo(5), MPI_REAL_FP, neighbor_nbhalo(5), &
                       tag, comm_cart, irecv_req(5), ierr) ! south_bottom recv north_top send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(7)+1, tag), isz*nhalo(4)*nhalo(5), MPI_REAL_FP, neighbor_nbhalo(7), &
                       tag, comm_cart, irecv_req(7), ierr) ! north_bottom recv south_top send
        call MPI_IRECV(halobuf_ptr(halobuf_offset(8)+1, tag), isz*nhalo(3)*nhalo(6), MPI_REAL_FP, neighbor_nbhalo(6), &
                       tag, comm_cart, irecv_req(6), ierr) ! south_top recv north_bottom send

        return
    end subroutine updateHaloBufIRecv

    subroutine updateHaloBufISend(nhalo, tag, var, isend_req, use_aux)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in) :: tag
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        integer, dimension(8), intent(out) :: isend_req
        !@cuf attributes(managed) :: var
        integer, intent(in) :: use_aux

        integer :: isz
        real(fp), dimension(:, :), pointer :: halobuf_ptr
#ifdef CUDA_AWARE_MPI
        !@cuf attributes(device) :: halobuf_ptr
#else
        !@cuf attributes(managed) :: halobuf_ptr
#endif        

        if (use_aux == 1) then
            halobuf_ptr => halobuf_send_aux
        else
            halobuf_ptr => halobuf_send
        endif
        
        isz = sz(1) + nhalo(1) + nhalo(2)
        call MPI_ISEND(halobuf_ptr(halobuf_offset(1)+1, tag), isz*nhalo(4)*sz(3), MPI_REAL_FP, neighbor_nbhalo(1), &
                       tag, comm_cart, isend_req(1), ierr)   ! y1 recv y0 send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(2)+1, tag), isz*nhalo(3)*sz(3), MPI_REAL_FP, neighbor_nbhalo(2), &
                       tag, comm_cart, isend_req(2), ierr)   ! y0 recv y1 send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(3)+1, tag), isz*sz(2)*nhalo(6), MPI_REAL_FP, neighbor_nbhalo(3), &
                       tag, comm_cart, isend_req(3), ierr)   ! z1 recv z0 send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(4)+1, tag), isz*sz(2)*nhalo(5), MPI_REAL_FP, neighbor_nbhalo(4), &
                       tag, comm_cart, isend_req(4), ierr)   ! z0 recv z1 send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(5)+1, tag), isz*nhalo(4)*nhalo(6), MPI_REAL_FP, neighbor_nbhalo(5), &
                       tag, comm_cart, isend_req(5), ierr) ! north_top recv south_bottom send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(6)+1, tag), isz*nhalo(3)*nhalo(5), MPI_REAL_FP, neighbor_nbhalo(8), &
                       tag, comm_cart, isend_req(8), ierr) ! south_bottom recv north_top send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(7)+1, tag), isz*nhalo(4)*nhalo(5), MPI_REAL_FP, neighbor_nbhalo(6), &
                       tag, comm_cart, isend_req(6), ierr) ! north_bottom recv south_top send
        call MPI_ISEND(halobuf_ptr(halobuf_offset(8)+1, tag), isz*nhalo(3)*nhalo(6), MPI_REAL_FP, neighbor_nbhalo(7), &
                       tag, comm_cart, isend_req(7), ierr) ! south_top recv north_bottom send

        return
    end subroutine updateHaloBufISend
#endif

    subroutine updateHaloIRecv(nhalo, tag, var, irecv_req)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in) :: tag
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        integer, dimension(8), intent(out) :: irecv_req
        !@cuf attributes(managed) :: var

        integer :: ist, jst, kst

        ist = 1-nhalo(1)
        jst = 1-nhalo(3)
        kst = 1-nhalo(5)
        call MPI_IRECV(var(ist, sz(2)+1, 1), 1, mpitype_nbhalo_vel(2), neighbor_nbhalo(2), &
                       tag, comm_cart, irecv_req(2), ierr)   ! y1 recv y0 send
        call MPI_IRECV(var(ist, jst    , 1), 1, mpitype_nbhalo_vel(1), neighbor_nbhalo(1), &
                       tag, comm_cart, irecv_req(1), ierr)   ! y0 recv y1 send
        call MPI_IRECV(var(ist, 1, sz(3)+1), 1, mpitype_nbhalo_vel(4), neighbor_nbhalo(4), &
                       tag, comm_cart, irecv_req(4), ierr)   ! z1 recv z0 send
        call MPI_IRECV(var(ist, 1, kst    ), 1, mpitype_nbhalo_vel(3), neighbor_nbhalo(3), &
                       tag, comm_cart, irecv_req(3), ierr)   ! z0 recv z1 send
        call MPI_IRECV(var(ist, sz(2)+1, sz(3)+1), 1, mpitype_nbhalo_vel(8), neighbor_nbhalo(8), &
                       tag, comm_cart, irecv_req(8), ierr) ! north_top recv south_bottom send
        call MPI_IRECV(var(ist,     jst,     kst), 1, mpitype_nbhalo_vel(5), neighbor_nbhalo(5), &
                       tag, comm_cart, irecv_req(5), ierr) ! south_bottom recv north_top send
        call MPI_IRECV(var(ist, sz(2)+1,     kst), 1, mpitype_nbhalo_vel(7), neighbor_nbhalo(7), &
                       tag, comm_cart, irecv_req(7), ierr) ! north_bottom recv south_top send
        call MPI_IRECV(var(ist,     jst, sz(3)+1), 1, mpitype_nbhalo_vel(6), neighbor_nbhalo(6), &
                       tag, comm_cart, irecv_req(6), ierr) ! south_top recv north_bottom send

        return
    end subroutine updateHaloIRecv

    subroutine updateHaloISend(nhalo, tag, var, isend_req)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, intent(in) :: tag
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var
        integer, dimension(8), intent(out) :: isend_req
        !@cuf attributes(managed) :: var

        integer :: ist, jst, kst
        
        ist = 1-nhalo(1)
        jst = 1-nhalo(3)
        kst = 1-nhalo(5)
        call MPI_ISEND(var(ist,         1, 1), 1, mpitype_nbhalo_vel(2), neighbor_nbhalo(1), &
                       tag, comm_cart, isend_req(1), ierr)   ! y1 recv y0 send
        call MPI_ISEND(var(ist, sz(2)+jst, 1), 1, mpitype_nbhalo_vel(1), neighbor_nbhalo(2), &
                       tag, comm_cart, isend_req(2), ierr)   ! y0 recv y1 send
        call MPI_ISEND(var(ist, 1,         1), 1, mpitype_nbhalo_vel(4), neighbor_nbhalo(3), &
                       tag, comm_cart, isend_req(3), ierr)   ! z1 recv z0 send
        call MPI_ISEND(var(ist, 1, sz(3)+kst), 1, mpitype_nbhalo_vel(3), neighbor_nbhalo(4), &
                       tag, comm_cart, isend_req(4), ierr)   ! z0 recv z1 send
        call MPI_ISEND(var(ist,         1,         1), 1, mpitype_nbhalo_vel(8), neighbor_nbhalo(5), &
                       tag, comm_cart, isend_req(5), ierr) ! north_top recv south_bottom send
        call MPI_ISEND(var(ist, sz(2)+jst, sz(3)+kst), 1, mpitype_nbhalo_vel(5), neighbor_nbhalo(8), &
                       tag, comm_cart, isend_req(8), ierr) ! south_bottom recv north_top send
        call MPI_ISEND(var(ist,         1, sz(3)+kst), 1, mpitype_nbhalo_vel(7), neighbor_nbhalo(6), &
                       tag, comm_cart, isend_req(6), ierr) ! north_bottom recv south_top send
        call MPI_ISEND(var(ist, sz(2)+jst,         1), 1, mpitype_nbhalo_vel(6), neighbor_nbhalo(7), &
                       tag, comm_cart, isend_req(7), ierr) ! south_top recv north_bottom send

        return
    end subroutine updateHaloISend

    subroutine updateHaloWaitall(isend_req, irecv_req)
        implicit none
        integer, dimension(:), intent(inout) :: isend_req, irecv_req

        call MPI_WAITALL(size(isend_req), isend_req, MPI_STATUS_IGNORE, ierr)
        call MPI_WAITALL(size(irecv_req), irecv_req, MPI_STATUS_IGNORE, ierr)

        return
    end subroutine updateHaloWaitall
    
    subroutine updateNBHalo(nhalo, halotype, var)
        implicit none
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(8), intent(in) :: halotype
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var

        integer :: tag_halo = 0

        ! update halo cells
        ! *** south/north ***
        call MPI_SENDRECV(var(1-nhalo(1),       1, 1), 1, halotype(2), neighbor_nbhalo(1), tag_halo, &
                          var(1-nhalo(1), sz(2)+1, 1), 1, halotype(2), neighbor_nbhalo(2), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(var(1-nhalo(1), sz(2)+1-nhalo(3), 1), 1, halotype(1), neighbor_nbhalo(2), tag_halo, &
                          var(1-nhalo(1),       1-nhalo(3), 1), 1, halotype(1), neighbor_nbhalo(1), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        ! *** bottom/top ***
        call MPI_SENDRECV(var(1-nhalo(1), 1,       1), 1, halotype(4), neighbor_nbhalo(3), tag_halo, &
                          var(1-nhalo(1), 1, sz(3)+1), 1, halotype(4), neighbor_nbhalo(4), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(var(1-nhalo(1), 1, sz(3)+1-nhalo(5)), 1, halotype(3), neighbor_nbhalo(4), tag_halo, &
                          var(1-nhalo(1), 1,       1-nhalo(5)), 1, halotype(3), neighbor_nbhalo(3), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        ! *** south_bottom/north_top ***
        call MPI_SENDRECV(var(1-nhalo(1),       1,       1), 1, halotype(8), neighbor_nbhalo(5), tag_halo, &
                          var(1-nhalo(1), sz(2)+1, sz(3)+1), 1, halotype(8), neighbor_nbhalo(8), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(var(1-nhalo(1), sz(2)+1-nhalo(3), sz(3)+1-nhalo(5)), 1, halotype(5), neighbor_nbhalo(8), tag_halo, &
                        var(1-nhalo(1),       1-nhalo(3),       1-nhalo(5)), 1, halotype(5), neighbor_nbhalo(5), tag_halo, &
                        comm_cart, MPI_STATUS_IGNORE, ierr)
        ! *** south_top/north_bottom ***
        call MPI_SENDRECV(var(1-nhalo(1),       1, sz(3)+1-nhalo(5)), 1, halotype(7), neighbor_nbhalo(6), tag_halo, &
                          var(1-nhalo(1), sz(2)+1,       1-nhalo(5)), 1, halotype(7), neighbor_nbhalo(7), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        call MPI_SENDRECV(var(1-nhalo(1), sz(2)+1-nhalo(3),       1), 1, halotype(6), neighbor_nbhalo(7), tag_halo, &
                          var(1-nhalo(1),       1-nhalo(3), sz(3)+1), 1, halotype(6), neighbor_nbhalo(6), tag_halo, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)

        return
    end subroutine updateNBHalo

#endif

end module mod_updateBound