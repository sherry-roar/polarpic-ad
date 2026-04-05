#ifdef _CUDA
#define MEAN Mean_d
#else
#define MEAN Mean_h
#endif

module mod_calcVel
    use mod_type,       only: fp
    use mod_parameters, only: re_inv, dt, is_forced, vel_force, nx, ny, nz, lz, nhalo, u_crf, &
                              smooth_wall_visc
    use mod_mpi,        only: sz, halotype_vel
#ifdef PSIP
    use mod_mpi,        only: comm_cart, neighbor, MPI_REAL_FP
#endif
    use mod_mesh,       only: dx_inv, dy_inv, dzc_inv, dzf, dzf_inv, dzflzi, dzf_global, dzc, visc_dzf_inv
    use mod_utils,      only: MEAN
    use mod_updateBound,only: updateHalo, imposeBCVel
#ifdef NB_HALO
    use mod_updateBound,only: updateHaloISend, updateHaloIRecv, updateHaloWaitall
#ifdef USE_NBHALOBUF
    use mod_mpi,        only: use_halobuf_aux
    use mod_updateBound,only: memcpyToHaloBuf, memcpyFromHaloBuf, updateHaloBufISend, updateHaloBufIRecv
#endif
#ifdef USE_RDMA
    use mod_updateBound,only: update_halo_rdma_send, update_halo_rdma_wait
#endif
#endif
    !$ use omp_lib
#ifdef _CUDA
    use cudafor
#endif
    use, intrinsic :: iso_c_binding
    use gptl

    implicit none

#if defined(NB_HALO) || defined(PSIP)
    include 'mpif.h'
#endif

    ! abstract interface
    !     subroutine rk_kernel_interface(st, en, u, v, w, unew, vnew, wnew, u_crf)
    !         import :: fp, nhalo
    !         integer,  dimension(3), intent(in) :: st, en
    !         real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in), contiguous :: u, v, w
    !         real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout), contiguous :: unew, vnew, wnew
    !         real(fp), intent(in) :: u_crf
    !         !@cuf attributes(managed) :: u, v, w, unew, vnew, wnew
    !     end subroutine
    ! end interface

    interface
        subroutine transform_to_crf(vel_crf, nhalo, sz, vel, vel_force) bind(C, name='transform_to_crf')
            import
            real(C_DOUBLE), value :: vel_crf
            integer(C_INT), dimension(*), intent(in) :: nhalo
            integer(C_INT), dimension(*), intent(in) :: sz
            real(C_DOUBLE), dimension(*), intent(inout) :: vel
            real(C_DOUBLE), intent(inout) :: vel_force
        end subroutine transform_to_crf
    end interface

    interface
        subroutine time_int_vel_rk(rk_num, comm, halotype_vel, neighbor, &
                                   re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, &
                                   nhalo, sz, u, v, w, unew, vnew, wnew, u_crf) bind(C, name='time_int_vel_rk')
            import
            integer(C_INT), value :: rk_num
            integer(C_INT), value :: comm
            integer(C_INT), dimension(*), intent(in) :: halotype_vel
            integer(C_INT), dimension(*), intent(in) :: neighbor
            real(C_DOUBLE), value :: re_inv, dt
            real(C_DOUBLE), value :: dx_inv, dy_inv
            real(C_DOUBLE), dimension(*), intent(in ) :: dzf, dzc_inv, dzf_inv, visc_dzf_inv
            integer(C_INT), dimension(*), intent(in ) :: nhalo
            integer(C_INT), dimension(*), intent(in ) :: sz
            real(C_DOUBLE), dimension(*), intent(in ) :: u, v, w
            real(C_DOUBLE), dimension(*), intent(inout) :: unew, vnew, wnew
            real(C_DOUBLE), value :: u_crf
        end subroutine time_int_vel_rk
    end interface

    interface
        subroutine correct_vel(dt, dx_inv, dy_inv, dzc_inv, nhalo_p, nhalo, sz, p, u, v, w) bind(C, name='correct_vel')
            import
            real(C_DOUBLE), value :: dt
            real(C_DOUBLE), value :: dx_inv, dy_inv
            real(C_DOUBLE), dimension(*), intent(in) :: dzc_inv
            integer(C_INT), dimension(*), intent(in) :: nhalo_p, nhalo
            integer(C_INT), dimension(*), intent(in) :: sz
            real(C_DOUBLE), dimension(*), intent(in) :: p
            real(C_DOUBLE), dimension(*), intent(inout) :: u, v, w
        end subroutine correct_vel
    end interface

    interface
        subroutine force_vel(is_forced, vel_force, nx, ny, dzflzi, nhalo, sz, u, v, w) bind(C, name='force_vel')
            import
            logical(C_BOOL), dimension(3), intent(in) :: is_forced
            real(C_DOUBLE), dimension(3), intent(in) :: vel_force
            integer(C_INT), value :: nx, ny
            real(C_DOUBLE), dimension(*), intent(in) :: dzflzi
            integer(C_INT), dimension(*), intent(in) :: nhalo
            integer(C_INT), dimension(*), intent(in) :: sz
            real(C_DOUBLE), dimension(*), intent(inout) :: u, v, w
        end subroutine force_vel
    end interface

#ifdef PSIP
    real(fp), dimension(0:1), save, private :: rhs_u_cor!, rhs_v_cor, rhs_w_cor
    real(fp), allocatable, dimension(:), save, private :: a_c, b_c, c_c
    real(fp), allocatable, dimension(:), save, private :: a_f, b_f, c_f
    real(fp), allocatable, dimension(:,:,:), save, private :: rko_u, rko_v, rko_w
    real(fp), allocatable, dimension(:,:), save, private :: y1_pdd, y2_pdd, y3_pdd, sendbuf_pdd, recvbuf_pdd
    real(fp), allocatable, dimension(:), save, private :: v_pdd_c, w_pdd_c
    real(fp), allocatable, dimension(:), save, private :: v_pdd_f, w_pdd_f
    real(fp), save, private :: tmp_v_pdd_c, tmp_v_pdd_f
#endif

contains

#ifdef PSIP
    subroutine initImpTimeMarcher()
        implicit none
        real(fp) :: alpdt, a_b, c_b
        real(fp) :: dzc0, dzc1

        allocate(rko_u(sz(1), sz(2), sz(3)))
        allocate(rko_v(sz(1), sz(2), sz(3)))
        allocate(rko_w(sz(1), sz(2), sz(3)))
        allocate(a_c(sz(3)), b_c(sz(3)), c_c(sz(3)))
        allocate(a_f(sz(3)), b_f(sz(3)), c_f(sz(3)))
        allocate(v_pdd_c  (sz(3))); v_pdd_c = 0.0_fp
        allocate(w_pdd_c  (sz(3))); w_pdd_c = 0.0_fp
        allocate(v_pdd_f(sz(3))); v_pdd_f = 0.0_fp
        allocate(w_pdd_f(sz(3))); w_pdd_f = 0.0_fp
        ! following BCs only apply to channel flow
        call setTridCoeff(sz(3), dzf, 'DD', 'c', neighbor(5:6), smooth_wall_visc, a_c, b_c, c_c)
        call setTridCoeff(sz(3), dzf, 'DD', 'f', neighbor(5:6), .false., a_f, b_f, c_f)
        alpdt = - 0.5_fp*dt*re_inv
        a_c(:) = a_c(:) * alpdt
        c_c(:) = c_c(:) * alpdt
        b_c(:) = b_c(:) * alpdt + 1.0_fp
        a_f(:) = a_f(:) * alpdt
        c_f(:) = c_f(:) * alpdt
        b_f(:) = b_f(:) * alpdt + 1.0_fp
        call pddFactor(neighbor(5:6), sz(3), a_c, b_c, c_c, v_pdd_c, w_pdd_c, tmp_v_pdd_c)
        if (neighbor(6) == MPI_PROC_NULL) then
            call pddFactor(neighbor(5:6), sz(3)-1, a_f, b_f, c_f, v_pdd_f, w_pdd_f, tmp_v_pdd_f)
        else
            call pddFactor(neighbor(5:6), sz(3), a_f, b_f, c_f, v_pdd_f, w_pdd_f, tmp_v_pdd_f)
        endif
        
        ! calculate the correction of the right-hand side according to BC type in z direction
        a_b = 2.0_fp/( dzf_global( 1)*(dzf_global( 0)+dzf_global(   1)) ) * alpdt
        c_b = 2.0_fp/( dzf_global(nz)*(dzf_global(nz)+dzf_global(nz+1)) ) * alpdt
        if (smooth_wall_visc == .true.) then
            dzc0 = 0.5_fp * (dzf_global(0) + dzf_global(1))
            dzc1 = 0.5_fp * (dzf_global(1) + dzf_global(2))
            a_b = 2.0_fp / dzc0 / (dzc0 + dzc1) * alpdt
            dzc0 = 0.5_fp * (dzf_global(nz-1) + dzf_global(nz))
            dzc1 = 0.5_fp * (dzf_global(nz+1) + dzf_global(nz))
            c_b = 2.0_fp / dzc1 / (dzc0 + dzc1) * alpdt
        endif
        call preprocessRHS((/-u_crf, -u_crf/), 'c', a_b, c_b, rhs_u_cor)
        ! call preprocessRHS((/   0.0,    0.0/), 'c', a_b, c_b, rhs_v_cor)
        ! a_b = 2.0_fp/( dzf_global( 1)*(dzf_global(   1)+dzf_global( 2)) ) * alpdt
        ! c_b = 2.0_fp/( dzf_global(nz)*(dzf_global(nz-1)+dzf_global(nz)) ) * alpdt
        ! call preprocessRHS((/   0.0,    0.0/), 'f', a_b, c_b, rhs_w_cor)
        
        allocate(y1_pdd(sz(1), sz(2))); y1_pdd = 0.0_fp
        allocate(y2_pdd(sz(1), sz(2))); y2_pdd = 0.0_fp
        allocate(y3_pdd(sz(1), sz(2))); y3_pdd = 0.0_fp
        allocate(sendbuf_pdd(sz(1), sz(2)))
        allocate(recvbuf_pdd(sz(1), sz(2)))

        return
    end subroutine initImpTimeMarcher

    subroutine freeImpTimeMarcher()
        implicit none

        if (allocated(rko_u)) deallocate(rko_u)
        if (allocated(rko_v)) deallocate(rko_v)
        if (allocated(rko_w)) deallocate(rko_w)
        if (allocated(a_c)) deallocate(a_c)
        if (allocated(b_c)) deallocate(b_c)
        if (allocated(c_c)) deallocate(c_c)
        if (allocated(a_f)) deallocate(a_f)
        if (allocated(b_f)) deallocate(b_f)
        if (allocated(c_f)) deallocate(c_f)
        if (allocated(y1_pdd)) deallocate(y1_pdd)
        if (allocated(y2_pdd)) deallocate(y2_pdd)
        if (allocated(y3_pdd)) deallocate(y3_pdd)
        if (allocated(recvbuf_pdd)) deallocate(recvbuf_pdd)
        if (allocated(sendbuf_pdd)) deallocate(sendbuf_pdd)
        if (allocated(v_pdd_c)) deallocate(v_pdd_c)
        if (allocated(w_pdd_c)) deallocate(w_pdd_c)
        if (allocated(v_pdd_f)) deallocate(v_pdd_f)
        if (allocated(w_pdd_f)) deallocate(w_pdd_f)

        return
    end subroutine freeImpTimeMarcher

    subroutine setTridCoeff(n, dzf, bctype, c_or_f, neighbor, bound_cor, a, b, c)
        implicit none
        integer,  intent(in) :: n
        real(fp), dimension(0:), intent(in) :: dzf
        character(2),            intent(in) :: bctype
        character(1),            intent(in) :: c_or_f
        integer,  dimension(2),  intent(in) :: neighbor
        logical,                 intent(in) :: bound_cor
        real(fp), dimension(n),  intent(out) :: a, b, c

        real(fp) :: factor
        real(fp) :: dzc0, dzc1
        integer :: k
        
        select case(c_or_f)
        case('c')
            do k = 1, n
                a(k) = 2.0_fp/( dzf(k)*(dzf(k-1)+dzf(k)) )
                c(k) = 2.0_fp/( dzf(k)*(dzf(k+1)+dzf(k)) )
            enddo
            if (bound_cor == .true.) then
                if (neighbor(1) == MPI_PROC_NULL) then
                    dzc0 = 0.5_fp * (dzf(0) + dzf(1))
                    dzc1 = 0.5_fp * (dzf(1) + dzf(2))
                    a(1) = 2.0_fp / dzc0 / (dzc0 + dzc1)
                    c(1) = 2.0_fp / dzc1 / (dzc0 + dzc1)
                endif
                if (neighbor(2) == MPI_PROC_NULL) then
                    dzc0 = 0.5_fp * (dzf(n-1) + dzf(n))
                    dzc1 = 0.5_fp * (dzf(n+1) + dzf(n))
                    a(n) = 2.0_fp / dzc0 / (dzc0 + dzc1)
                    c(n) = 2.0_fp / dzc1 / (dzc0 + dzc1)
                endif
            endif
        case('f')
            do k = 1, n
                a(k) = 2.0_fp/( dzf(k)*(dzf(k+1)+dzf(k)) )
                c(k) = 2.0_fp/( dzf(k+1)*(dzf(k+1)+dzf(k)) )
            enddo
        end select

        b(:) = - a(:) - c(:)

        ! coefficients correction according to BC types
        if (neighbor(1) == MPI_PROC_NULL) then
            select case(bctype(1:1))
            case('P')
                factor = 0.0_fp
            case('D')
                factor = -1.0_fp
            case('N')
                factor = 1.0_fp
            end select

            select case(c_or_f)
            case('c')
                b(1) = b(1) + factor*a(1)
                a(1) = (abs(factor)-1.0_fp)*a(1)
            case('f')
                if(bctype(1:1) == 'N') then
                    b(1) = b(1) + factor*a(1)
                    a(1) = (abs(factor)-1.0_fp)*a(1)
                endif
            end select
        endif
        if (neighbor(2) == MPI_PROC_NULL) then
            select case(bctype(2:2))
            case('P')
                factor = 0.0_fp
            case('D')
                factor = -1.0_fp
            case('N')
                factor = 1.0_fp
            end select

            select case(c_or_f)
            case('c')
                b(n) = b(n) + factor*c(n)
                c(n) = (abs(factor)-1.0_fp)*c(n)
            case('f')
                if(bctype(2:2) == 'N') then
                    b(n) = b(n) + factor*c(n)
                    c(n) = (abs(factor)-1.0_fp)*c(n)
                endif
            end select
        endif

        return
    end subroutine setTridCoeff

    subroutine pddFactor(nb, n, a, b, c, v_pdd, w_pdd, tmp_v_pdd)
        implicit none
        integer, dimension(2), intent(in) :: nb
        integer, intent(in) :: n
        real(fp), dimension(:), intent(inout) :: a, b, c, v_pdd, w_pdd
        real(fp), intent(out) :: tmp_v_pdd

        integer :: ierr, k, ke
        real(fp) :: a_tmp
        
        ke = n
        v_pdd(1) = a(1)
        w_pdd(ke) = c(ke)

        do k = 2, ke
            a_tmp = a(k) / b(k - 1)
            b(k) = b(k) - a_tmp * c(k - 1)
            v_pdd(k) = v_pdd(k) - a_tmp * v_pdd(k - 1)
            w_pdd(k) = w_pdd(k) - a_tmp * w_pdd(k - 1)
        enddo

        if (b(ke) /= 0.0) then
            v_pdd(ke) = v_pdd(ke) / b(ke)
            w_pdd(ke) = w_pdd(ke) / b(ke)
        else
            v_pdd(ke) = 0.0_fp
            w_pdd(ke) = 0.0_fp
        endif

        do k = ke - 1, 1
            v_pdd(k) = (v_pdd(k) - c(k) * v_pdd(k + 1)) / b(k)
            w_pdd(k) = (w_pdd(k) - c(k) * w_pdd(k + 1)) / b(k)
        enddo

        call MPI_SENDRECV( v_pdd(1), 1, MPI_REAL_FP, nb(1), 0, &
                          tmp_v_pdd, 1, MPI_REAL_FP, nb(2), 0, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)
        
        return
    end subroutine pddFactor

    subroutine preprocessRHS(bcvalue, c_or_f, a_b, c_b, rhs_cor)
        implicit none
        real(fp), dimension(0:1), intent(in) :: bcvalue
        character(1), intent(in) :: c_or_f
        real(fp), intent(in) :: a_b, c_b
        real(fp), dimension(0:1), intent(out):: rhs_cor

        if (c_or_f == 'c') then
            rhs_cor(0) = -2.0_fp*bcvalue(0)*a_b
            rhs_cor(1) = -2.0_fp*bcvalue(1)*c_b
        else
            rhs_cor(0) = -bcvalue(0)*a_b
            rhs_cor(1) = -bcvalue(1)*c_b
        endif

        return
    end subroutine preprocessRHS

    subroutine correctRHS(nb, nhalo, sz, rhs_cor, c_or_f, var)
        implicit none
        integer,  dimension(2), intent(in) :: nb
        integer,  dimension(6), intent(in) :: nhalo
        integer,  dimension(3), intent(in) :: sz
        real(fp), dimension(0:1), intent(in) :: rhs_cor
        character(1), intent(in) :: c_or_f
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var

        integer :: i, j, ke

        ! z direction
        if (nb(1) == MPI_PROC_NULL) then
            do j = 1, sz(2)
            do i = 1, sz(1)
                var(i, j, 1 ) = var(i, j, 1 ) + rhs_cor(0)
            enddo
            enddo
        endif
        if (nb(2) == MPI_PROC_NULL) then
            ke = sz(3)
            if (c_or_f == 'f') ke = sz(3) - 1

            do j = 1, sz(2)
            do i = 1, sz(1)
                var(i, j, ke) = var(i, j, ke) + rhs_cor(1)
            enddo
            enddo
        endif

        return
    end subroutine correctRHS

    subroutine solveTrid_PDD(nb, nhalo, sz, v_pdd, w_pdd, tmp_v_pdd, a, b, c, var)
        implicit none
        integer, dimension(2), intent(in) :: nb
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        real(fp), intent(in) :: tmp_v_pdd
        real(fp), dimension(sz(3)), intent(in) :: v_pdd, w_pdd
        real(fp), dimension(sz(3)), intent(in) :: a, c
        real(fp), dimension(sz(3)), intent(inout) :: b
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: var

        integer :: i, j, k, ie, je, ke, ierr
        real(fp) :: a_tmp, det_pdd

        ie = sz(1); je = sz(2); ke = sz(3)

        if (b(ke) /= 0.0) then
            !$OMP PARALLEL DO SCHEDULE(STATIC) &
            !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, a_tmp)
            do j = 1, je
            do i = 1, ie
                do k = 2, ke
                    a_tmp = a(k)/b(k-1)
                    var(i, j, k) = var(i, j, k) - a_tmp*var(i, j, k-1)
                enddo
                var(i, j, ke) = var(i, j, ke)/b(ke)
                do k = ke-1, 1, -1
                    var(i, j, k) = (var(i, j, k) - c(k)*var(i, j, k+1))/b(k)
                enddo

                sendbuf_pdd(i, j) = var(i, j, 1)
            enddo
            enddo
            !$OMP END PARALLEL DO
        else
            !$OMP PARALLEL DO SCHEDULE(STATIC) &
            !$OMP DEFAULT(SHARED) PRIVATE(i, j, k, a_tmp)
            do j = 1, je
            do i = 1, ie
                do k = 2, ke
                    a_tmp = a(k)/b(k-1)
                    var(i, j, k) = var(i, j, k) - a_tmp*var(i, j, k-1)
                enddo
                var(i, j, ke) = 0.0_fp
                do k = ke-1, 1, -1
                    var(i, j, k) = (var(i, j, k) - c(k)*var(i, j, k+1))/b(k)
                enddo

                sendbuf_pdd(i, j) = var(i, j, 1)
            enddo
            enddo
            !$OMP END PARALLEL DO
        endif

        call MPI_SENDRECV(sendbuf_pdd(1,1), sz(1)*sz(2), MPI_REAL_FP, nb(1), 1, &
                          recvbuf_pdd(1,1), sz(1)*sz(2), MPI_REAL_FP, nb(2), 1, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)

        if (nb(2) /= MPI_PROC_NULL) then
            !$OMP PARALLEL DO SCHEDULE(STATIC) &
            !$OMP DEFAULT(SHARED) PRIVATE(i, j, det_pdd)
            do j = 1, je
            do i = 1, ie
                det_pdd = w_pdd(ke)*tmp_v_pdd - 1.0_fp
                y2_pdd(i, j) = (var(i, j, ke)*tmp_v_pdd - recvbuf_pdd(i, j))/det_pdd
                y3_pdd(i, j) = (recvbuf_pdd(i, j)*w_pdd(ke) - var(i, j, ke))/det_pdd
            enddo
            enddo
            !$OMP END PARALLEL DO
        endif

        call MPI_SENDRECV(y3_pdd(1,1), sz(1)*sz(2), MPI_REAL_FP, nb(2), 2, &
                          y1_pdd(1,1), sz(1)*sz(2), MPI_REAL_FP, nb(1), 2, &
                          comm_cart, MPI_STATUS_IGNORE, ierr)

        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
        do k = 1, ke
        do j = 1, je
        do i = 1, ie
            var(i, j, k) = var(i, j, k) - v_pdd(k)*y1_pdd(i, j) - w_pdd(k)*y2_pdd(i, j)
        enddo
        enddo
        enddo
        !$OMP END PARALLEL DO

        return
    end subroutine solveTrid_PDD
#endif
    subroutine transform2CRF(vel_crf, nhalo, sz, vel, vel_force)
        implicit none
        real(fp), intent(in) :: vel_crf
        integer, dimension(6), intent(in) :: nhalo
        integer, dimension(3), intent(in) :: sz
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: vel
        real(fp), intent(inout) :: vel_force
        !@cuf attributes(managed) :: vel
        !@cuf integer :: istat

        integer :: i, j, k

#ifdef _CUDA
        !$cuf kernel do(3) <<<*,*>>>
#else
        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
#endif
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            vel(i, j, k) = vel(i, j, k) - vel_crf
        enddo
        enddo
        enddo
#ifndef _CUDA
        !$OMP END PARALLEL DO
#endif
        !@cuf istat=cudaDeviceSynchronize()

        vel_force = vel_force - vel_crf

        return
    end subroutine transform2CRF

    subroutine timeIntVelRK1_kernel(st, en, u, v, w, u1, v1, w1, u_crf)
        implicit none
        integer,  dimension(3), intent(in) :: st, en
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in), contiguous :: u, v, w
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout), contiguous :: u1, v1, w1
        real(fp), intent(in) :: u_crf
        !@cuf attributes(managed) :: u, v, w, u1, v1, w1

        real(fp) :: r1, r2, r22, rdxidxi, rdyidyi
        real(fp) :: q1m3, q1m1, q1p1, q1p3
        real(fp) :: q2m3, q2m1, q2p1, q2p3
        real(fp) :: duudx, dvudy, dwudz
        real(fp) :: duvdx, dvvdy, dwvdz
        real(fp) :: duwdx, dvwdy, dwwdz
        real(fp) :: conv, visc, visc_imp
        integer  :: i, j, k

        r1 = 9.0_fp/8.0_fp
        r2 = -1.0_fp/8.0_fp
        r22= r2/3.0_fp
        rdxidxi = dx_inv*dx_inv/12.0_fp
        rdyidyi = dy_inv*dy_inv/12.0_fp

#ifdef _CUDA
        !$cuf kernel do(3) <<<*,*>>>
#else
        !$OMP PARALLEL DO SCHEDULE(STATIC) COLLAPSE(2) &
        !$OMP DEFAULT(SHARED) &
        !$OMP FIRSTPRIVATE(r1,r2,r22,rdxidxi,rdyidyi) &
        !$OMP PRIVATE(i,j,k,q1m3,q1m1,q1p1,q1p3,q2m3,q2m1,q2p1,q2p3,duudx,dvudy,dwudz, &
        !$OMP         duvdx,dvvdy,dwvdz,duwdx,dvwdy,dwwdz,conv,visc,visc_imp)
#endif
        do k = st(3), en(3)
        do j = st(2), en(2)
        do i = st(1), en(1)

            q1m3 = r1*(u(i-2,j,k)+u(i-1,j,k)) + r2*(u(i-3,j,k)+u(i  ,j,k))
            q1m1 = r1*(u(i-1,j,k)+u(i  ,j,k)) + r2*(u(i-2,j,k)+u(i+1,j,k))
            q1p1 = r1*(u(i  ,j,k)+u(i+1,j,k)) + r2*(u(i-1,j,k)+u(i+2,j,k))
            q1p3 = r1*(u(i+1,j,k)+u(i+2,j,k)) + r2*(u(i  ,j,k)+u(i+3,j,k))
            q2m3 = u(i-3,j,k)+u(i  ,j,k)
            q2m1 = u(i-1,j,k)+u(i  ,j,k)
            q2p1 = u(i  ,j,k)+u(i+1,j,k)
            q2p3 = u(i  ,j,k)+u(i+3,j,k)
            duudx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dx_inv
            ! add a term induced by the convecting reference frame
            conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5_fp*dx_inv
            q1m3 = r1*(v(i,j-2,k)+v(i+1,j-2,k)) + r2*(v(i-1,j-2,k)+v(i+2,j-2,k))
            q1m1 = r1*(v(i,j-1,k)+v(i+1,j-1,k)) + r2*(v(i-1,j-1,k)+v(i+2,j-1,k))
            q1p1 = r1*(v(i,j  ,k)+v(i+1,j  ,k)) + r2*(v(i-1,j  ,k)+v(i+2,j  ,k))
            q1p3 = r1*(v(i,j+1,k)+v(i+1,j+1,k)) + r2*(v(i-1,j+1,k)+v(i+2,j+1,k))
            q2m3 = u(i,j-3,k)+u(i,j  ,k)
            q2m1 = u(i,j-1,k)+u(i,j  ,k)
            q2p1 = u(i,j  ,k)+u(i,j+1,k)
            q2p3 = u(i,j  ,k)+u(i,j+3,k)
            dvudy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dy_inv
            q1m1  = r1*(w(i,j,k-1)+w(i+1,j,k-1)) + r2*(w(i-1,j,k-1)+w(i+2,j,k-1))
            q1p1  = r1*(w(i,j,k  )+w(i+1,j,k  )) + r2*(w(i-1,j,k  )+w(i+2,j,k  ))
            q2m1  = (u(i,j,k)*dzf(k-1) + u(i,j,k-1)*dzf(k))*dzc_inv(k-1)
            q2p1  = (u(i,j,k)*dzf(k+1) + u(i,j,k+1)*dzf(k))*dzc_inv(k  )
            dwudz = (q1p1*q2p1-q1m1*q2m1)*0.25_fp*dzf_inv(k)
            conv = conv + duudx + dvudy + dwudz
#ifndef PSIP
            visc = (- u(i-2,j,k) + 16.0_fp*u(i-1,j,k) - 30.0_fp*u(i,j,k) &
                    - u(i+2,j,k) + 16.0_fp*u(i+1,j,k))* rdxidxi + &
                   (- u(i,j-2,k) + 16.0_fp*u(i,j-1,k) - 30.0_fp*u(i,j,k) &
                    - u(i,j+2,k) + 16.0_fp*u(i,j+1,k))* rdyidyi + &
                   ((u(i,j,k+1)-u(i,j,k))*dzc_inv(k)-(u(i,j,k)-u(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            u1(i,j,k) = u(i,j,k) + dt*(visc - conv)
#else
            visc = (- u(i-2,j,k) + 16.0_fp*u(i-1,j,k) - 30.0_fp*u(i,j,k) &
                    - u(i+2,j,k) + 16.0_fp*u(i+1,j,k))* rdxidxi + &
                   (- u(i,j-2,k) + 16.0_fp*u(i,j-1,k) - 30.0_fp*u(i,j,k) &
                    - u(i,j+2,k) + 16.0_fp*u(i,j+1,k))* rdyidyi
            visc_imp = ((u(i,j,k+1)-u(i,j,k))*dzc_inv(k)-(u(i,j,k)-u(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            visc_imp = visc_imp*re_inv
            u1(i,j,k) = u(i,j,k) + dt*(visc + 0.5_fp*visc_imp - conv)
            rko_u(i,j,k) = visc - conv
#endif

            q1m3 = r1*(u(i-2,j,k)+u(i-2,j+1,k)) + r2*(u(i-2,j-1,k)+u(i-2,j+2,k))
            q1m1 = r1*(u(i-1,j,k)+u(i-1,j+1,k)) + r2*(u(i-1,j-1,k)+u(i-1,j+2,k))
            q1p1 = r1*(u(i  ,j,k)+u(i  ,j+1,k)) + r2*(u(i  ,j-1,k)+u(i  ,j+2,k))
            q1p3 = r1*(u(i+1,j,k)+u(i+1,j+1,k)) + r2*(u(i+1,j-1,k)+u(i+1,j+2,k))
            q2m3 = v(i-3,j,k)+v(i  ,j,k)
            q2m1 = v(i-1,j,k)+v(i  ,j,k)
            q2p1 = v(i  ,j,k)+v(i+1,j,k)
            q2p3 = v(i  ,j,k)+v(i+3,j,k)
            duvdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dx_inv
            ! add a term induced by the convecting reference frame
            conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5_fp*dx_inv
            q1m3 = r1*(v(i,j-2,k)+v(i,j-1,k)) + r2*(v(i,j-3,k)+v(i,j  ,k))
            q1m1 = r1*(v(i,j-1,k)+v(i,j  ,k)) + r2*(v(i,j-2,k)+v(i,j+1,k))
            q1p1 = r1*(v(i,j  ,k)+v(i,j+1,k)) + r2*(v(i,j-1,k)+v(i,j+2,k))
            q1p3 = r1*(v(i,j+1,k)+v(i,j+2,k)) + r2*(v(i,j  ,k)+v(i,j+3,k))
            q2m3 = v(i,j-3,k)+v(i,j  ,k)
            q2m1 = v(i,j-1,k)+v(i,j  ,k)
            q2p1 = v(i,j  ,k)+v(i,j+1,k)
            q2p3 = v(i,j  ,k)+v(i,j+3,k)
            dvvdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dy_inv
            q1m1  = r1*(w(i,j,k-1)+w(i,j+1,k-1)) + r2*(w(i,j-1,k-1)+w(i,j+2,k-1))
            q1p1  = r1*(w(i,j,k  )+w(i,j+1,k  )) + r2*(w(i,j-1,k  )+w(i,j+2,k  ))
            q2m1  = (v(i,j,k  )*dzf(k-1) + v(i,j,k-1)*dzf(k))*dzc_inv(k-1)
            q2p1  = (v(i,j,k  )*dzf(k+1) + v(i,j,k+1)*dzf(k))*dzc_inv(k  )
            dwvdz = (q1p1*q2p1-q1m1*q2m1)*0.25_fp*dzf_inv(k)
            conv = conv + duvdx + dvvdy + dwvdz
#ifndef PSIP
            visc = (- v(i-2,j,k) + 16.0_fp*v(i-1,j,k) - 30.0_fp*v(i,j,k) &
                    - v(i+2,j,k) + 16.0_fp*v(i+1,j,k))* rdxidxi + &
                   (- v(i,j-2,k) + 16.0_fp*v(i,j-1,k) - 30.0_fp*v(i,j,k) &
                    - v(i,j+2,k) + 16.0_fp*v(i,j+1,k))* rdyidyi + &
                   ((v(i,j,k+1)-v(i,j,k))*dzc_inv(k)-(v(i,j,k)-v(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            v1(i,j,k) = v(i,j,k) + dt*(visc - conv)
#else
            visc = (- v(i-2,j,k) + 16.0_fp*v(i-1,j,k) - 30.0_fp*v(i,j,k) &
                    - v(i+2,j,k) + 16.0_fp*v(i+1,j,k))* rdxidxi + &
                   (- v(i,j-2,k) + 16.0_fp*v(i,j-1,k) - 30.0_fp*v(i,j,k) &
                    - v(i,j+2,k) + 16.0_fp*v(i,j+1,k))* rdyidyi
            visc_imp = ((v(i,j,k+1)-v(i,j,k))*dzc_inv(k)-(v(i,j,k)-v(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            visc_imp = visc_imp*re_inv
            v1(i,j,k) = v(i,j,k) + dt*(visc + 0.5_fp*visc_imp - conv)
            rko_v(i,j,k) = visc - conv
#endif

            q1m3 = (u(i-2,j,k)*dzf(k+1) + u(i-2,j,k+1)*dzf(k))*dzc_inv(k)
            q1m1 = (u(i-1,j,k)*dzf(k+1) + u(i-1,j,k+1)*dzf(k))*dzc_inv(k)
            q1p1 = (u(i  ,j,k)*dzf(k+1) + u(i  ,j,k+1)*dzf(k))*dzc_inv(k)
            q1p3 = (u(i+1,j,k)*dzf(k+1) + u(i+1,j,k+1)*dzf(k))*dzc_inv(k)
            q2m3 = w(i-3,j,k)+w(i  ,j,k)
            q2m1 = w(i-1,j,k)+w(i  ,j,k)
            q2p1 = w(i  ,j,k)+w(i+1,j,k)
            q2p3 = w(i  ,j,k)+w(i+3,j,k)
            duwdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dx_inv
            ! add a term induced by the convecting reference frame
            conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5_fp*dx_inv
            q1m3 = (v(i,j-2,k)*dzf(k+1) + v(i,j-2,k+1)*dzf(k))*dzc_inv(k)
            q1m1 = (v(i,j-1,k)*dzf(k+1) + v(i,j-1,k+1)*dzf(k))*dzc_inv(k)
            q1p1 = (v(i,j  ,k)*dzf(k+1) + v(i,j  ,k+1)*dzf(k))*dzc_inv(k)
            q1p3 = (v(i,j+1,k)*dzf(k+1) + v(i,j+1,k+1)*dzf(k))*dzc_inv(k)
            q2m3 = w(i,j-3,k)+w(i,j  ,k)
            q2m1 = w(i,j-1,k)+w(i,j  ,k)
            q2p1 = w(i,j  ,k)+w(i,j+1,k)
            q2p3 = w(i,j  ,k)+w(i,j+3,k)
            dvwdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dy_inv
            q1m1 = (w(i,j,k)+w(i,j,k-1))
            q1p1 = (w(i,j,k)+w(i,j,k+1))
            q2m1 = q1m1
            q2p1 = q1p1
            dwwdz = (q1p1*q2p1-q1m1*q2m1)*0.25_fp*dzc_inv(k)
            conv = conv + duwdx + dvwdy + dwwdz
#ifndef PSIP
            visc = (- w(i-2,j,k) + 16.0_fp*w(i-1,j,k) - 30.0_fp*w(i,j,k) &
                    - w(i+2,j,k) + 16.0_fp*w(i+1,j,k))* rdxidxi + &
                   (- w(i,j-2,k) + 16.0_fp*w(i,j-1,k) - 30.0_fp*w(i,j,k) &
                    - w(i,j+2,k) + 16.0_fp*w(i,j+1,k))* rdyidyi + &
                   ((w(i,j,k+1)-w(i,j,k))*dzf_inv(k+1)-(w(i,j,k)-w(i,j,k-1))*dzf_inv(k))*dzc_inv(k)
            visc = visc*re_inv
            w1(i,j,k) = w(i,j,k) + dt*(visc - conv)
#else
            visc = (- w(i-2,j,k) + 16.0_fp*w(i-1,j,k) - 30.0_fp*w(i,j,k) &
                    - w(i+2,j,k) + 16.0_fp*w(i+1,j,k))* rdxidxi + &
                   (- w(i,j-2,k) + 16.0_fp*w(i,j-1,k) - 30.0_fp*w(i,j,k) &
                    - w(i,j+2,k) + 16.0_fp*w(i,j+1,k))* rdyidyi
            visc_imp = ((w(i,j,k+1)-w(i,j,k))*dzf_inv(k+1)-(w(i,j,k)-w(i,j,k-1))*dzf_inv(k))*dzc_inv(k)
            visc = visc*re_inv
            visc_imp = visc_imp*re_inv
            w1(i,j,k) = w(i,j,k) + dt*(visc + 0.5_fp*visc_imp - conv)
            rko_w(i,j,k) = visc - conv
#endif

        enddo
        enddo
        enddo
#ifndef _CUDA
        !$OMP END PARALLEL DO
#endif

        return
    end subroutine timeIntVelRK1_kernel

    subroutine timeIntVelRK2_kernel(st, en, u1, v1, w1, u, v, w, u_crf)
        implicit none
        integer,  dimension(3), intent(in) :: st, en
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in), contiguous :: u1, v1, w1
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout), contiguous :: u, v, w
        real(fp), intent(in) :: u_crf
        !@cuf attributes(managed) :: u, v, w, u1, v1, w1

        real(fp) :: r1, r2, r22, rdxidxi, rdyidyi
        real(fp) :: q1m3, q1m1, q1p1, q1p3
        real(fp) :: q2m3, q2m1, q2p1, q2p3
        real(fp) :: duudx, dvudy, dwudz
        real(fp) :: duvdx, dvvdy, dwvdz
        real(fp) :: duwdx, dvwdy, dwwdz
        real(fp) :: conv, visc, visc_imp
        integer  :: i, j, k

        r1 = 9.0_fp/8.0_fp
        r2 = -1.0_fp/8.0_fp
        r22= r2/3.0_fp
        rdxidxi = dx_inv*dx_inv/12.0_fp
        rdyidyi = dy_inv*dy_inv/12.0_fp

#ifdef _CUDA
        !$cuf kernel do(3) <<<*,*>>>
#else
        !$OMP PARALLEL DO SCHEDULE(STATIC) COLLAPSE(2) &
        !$OMP DEFAULT(SHARED) &
        !$OMP FIRSTPRIVATE(r1,r2,r22,rdxidxi,rdyidyi) &
        !$OMP PRIVATE(i,j,k,q1m3,q1m1,q1p1,q1p3,q2m3,q2m1,q2p1,q2p3,duudx,dvudy,dwudz, &
        !$OMP         duvdx,dvvdy,dwvdz,duwdx,dvwdy,dwwdz,conv,visc,visc_imp)
#endif
        do k = st(3), en(3)
        do j = st(2), en(2)
        do i = st(1), en(1)
        
            q1m3 = r1*(u1(i-2,j,k)+u1(i-1,j,k)) + r2*(u1(i-3,j,k)+u1(i  ,j,k))
            q1m1 = r1*(u1(i-1,j,k)+u1(i  ,j,k)) + r2*(u1(i-2,j,k)+u1(i+1,j,k))
            q1p1 = r1*(u1(i  ,j,k)+u1(i+1,j,k)) + r2*(u1(i-1,j,k)+u1(i+2,j,k))
            q1p3 = r1*(u1(i+1,j,k)+u1(i+2,j,k)) + r2*(u1(i  ,j,k)+u1(i+3,j,k))
            q2m3 = u1(i-3,j,k)+u1(i  ,j,k)
            q2m1 = u1(i-1,j,k)+u1(i  ,j,k)
            q2p1 = u1(i  ,j,k)+u1(i+1,j,k)
            q2p3 = u1(i  ,j,k)+u1(i+3,j,k)
            duudx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dx_inv
            ! add a term induced by the convecting reference frame
            conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5_fp*dx_inv
            q1m3 = r1*(v1(i,j-2,k)+v1(i+1,j-2,k)) + r2*(v1(i-1,j-2,k)+v1(i+2,j-2,k))
            q1m1 = r1*(v1(i,j-1,k)+v1(i+1,j-1,k)) + r2*(v1(i-1,j-1,k)+v1(i+2,j-1,k))
            q1p1 = r1*(v1(i,j  ,k)+v1(i+1,j  ,k)) + r2*(v1(i-1,j  ,k)+v1(i+2,j  ,k))
            q1p3 = r1*(v1(i,j+1,k)+v1(i+1,j+1,k)) + r2*(v1(i-1,j+1,k)+v1(i+2,j+1,k))
            q2m3 = u1(i,j-3,k)+u1(i,j  ,k)
            q2m1 = u1(i,j-1,k)+u1(i,j  ,k)
            q2p1 = u1(i,j  ,k)+u1(i,j+1,k)
            q2p3 = u1(i,j  ,k)+u1(i,j+3,k)
            dvudy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dy_inv
            q1m1  = r1*(w1(i,j,k-1)+w1(i+1,j,k-1)) + r2*(w1(i-1,j,k-1)+w1(i+2,j,k-1))
            q1p1  = r1*(w1(i,j,k  )+w1(i+1,j,k  )) + r2*(w1(i-1,j,k  )+w1(i+2,j,k  ))
            q2m1  = (u1(i,j,k)*dzf(k-1) + u1(i,j,k-1)*dzf(k))*dzc_inv(k-1)
            q2p1  = (u1(i,j,k)*dzf(k+1) + u1(i,j,k+1)*dzf(k))*dzc_inv(k  )
            dwudz = (q1p1*q2p1-q1m1*q2m1)*0.25_fp*dzf_inv(k)
            conv = conv + duudx + dvudy + dwudz
#ifndef PSIP
            visc = (- u1(i-2,j,k) + 16.0_fp*u1(i-1,j,k) - 30.0_fp*u1(i,j,k) &
                    - u1(i+2,j,k) + 16.0_fp*u1(i+1,j,k))* rdxidxi + &
                   (- u1(i,j-2,k) + 16.0_fp*u1(i,j-1,k) - 30.0_fp*u1(i,j,k) &
                    - u1(i,j+2,k) + 16.0_fp*u1(i,j+1,k))* rdyidyi + &
                   ((u1(i,j,k+1)-u1(i,j,k))*dzc_inv(k)-(u1(i,j,k)-u1(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            u(i,j,k) = (u1(i,j,k) + dt*(visc-conv) + u(i,j,k))*0.5_fp
#else
            visc = (- u1(i-2,j,k) + 16.0_fp*u1(i-1,j,k) - 30.0_fp*u1(i,j,k) &
                    - u1(i+2,j,k) + 16.0_fp*u1(i+1,j,k))* rdxidxi + &
                   (- u1(i,j-2,k) + 16.0_fp*u1(i,j-1,k) - 30.0_fp*u1(i,j,k) &
                    - u1(i,j+2,k) + 16.0_fp*u1(i,j+1,k))* rdyidyi
            visc_imp = ((u1(i,j,k+1)-u1(i,j,k))*dzc_inv(k)-(u1(i,j,k)-u1(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            visc_imp = visc_imp*re_inv
            u(i,j,k) = u(i,j,k) + 0.5_fp*dt*((visc-conv) + rko_u(i,j,k) + visc_imp)
            ! u(i,j,k) = u(i,j,k) + dt*(0.5_fp*(visc-conv) + 0.5_fp*rko_u(i,j,k) + 0.5_fp*visc_imp)
#endif

            q1m3 = r1*(u1(i-2,j,k)+u1(i-2,j+1,k)) + r2*(u1(i-2,j-1,k)+u1(i-2,j+2,k))
            q1m1 = r1*(u1(i-1,j,k)+u1(i-1,j+1,k)) + r2*(u1(i-1,j-1,k)+u1(i-1,j+2,k))
            q1p1 = r1*(u1(i  ,j,k)+u1(i  ,j+1,k)) + r2*(u1(i  ,j-1,k)+u1(i  ,j+2,k))
            q1p3 = r1*(u1(i+1,j,k)+u1(i+1,j+1,k)) + r2*(u1(i+1,j-1,k)+u1(i+1,j+2,k))
            q2m3 = v1(i-3,j,k)+v1(i  ,j,k)
            q2m1 = v1(i-1,j,k)+v1(i  ,j,k)
            q2p1 = v1(i  ,j,k)+v1(i+1,j,k)
            q2p3 = v1(i  ,j,k)+v1(i+3,j,k)
            duvdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dx_inv
            ! add a term induced by the convecting reference frame
            conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5_fp*dx_inv
            q1m3 = r1*(v1(i,j-2,k)+v1(i,j-1,k)) + r2*(v1(i,j-3,k)+v1(i,j  ,k))
            q1m1 = r1*(v1(i,j-1,k)+v1(i,j  ,k)) + r2*(v1(i,j-2,k)+v1(i,j+1,k))
            q1p1 = r1*(v1(i,j  ,k)+v1(i,j+1,k)) + r2*(v1(i,j-1,k)+v1(i,j+2,k))
            q1p3 = r1*(v1(i,j+1,k)+v1(i,j+2,k)) + r2*(v1(i,j  ,k)+v1(i,j+3,k))
            q2m3 = v1(i,j-3,k)+v1(i,j  ,k)
            q2m1 = v1(i,j-1,k)+v1(i,j  ,k)
            q2p1 = v1(i,j  ,k)+v1(i,j+1,k)
            q2p3 = v1(i,j  ,k)+v1(i,j+3,k)
            dvvdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dy_inv
            q1m1  = r1*(w1(i,j,k-1)+w1(i,j+1,k-1)) + r2*(w1(i,j-1,k-1)+w1(i,j+2,k-1))
            q1p1  = r1*(w1(i,j,k  )+w1(i,j+1,k  )) + r2*(w1(i,j-1,k  )+w1(i,j+2,k  ))
            q2m1  = (v1(i,j,k  )*dzf(k-1) + v1(i,j,k-1)*dzf(k))*dzc_inv(k-1)
            q2p1  = (v1(i,j,k  )*dzf(k+1) + v1(i,j,k+1)*dzf(k))*dzc_inv(k  )
            dwvdz = (q1p1*q2p1-q1m1*q2m1)*0.25_fp*dzf_inv(k)
            conv = conv + duvdx + dvvdy + dwvdz
#ifndef PSIP
            visc = (- v1(i-2,j,k) + 16.0_fp*v1(i-1,j,k) - 30.0_fp*v1(i,j,k) &
                    - v1(i+2,j,k) + 16.0_fp*v1(i+1,j,k))* rdxidxi + &
                   (- v1(i,j-2,k) + 16.0_fp*v1(i,j-1,k) - 30.0_fp*v1(i,j,k) &
                    - v1(i,j+2,k) + 16.0_fp*v1(i,j+1,k))* rdyidyi + &
                   ((v1(i,j,k+1)-v1(i,j,k))*dzc_inv(k)-(v1(i,j,k)-v1(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            v(i,j,k) = (v1(i,j,k) + dt*(visc-conv) + v(i,j,k))*0.5_fp
#else
            visc = (- v1(i-2,j,k) + 16.0_fp*v1(i-1,j,k) - 30.0_fp*v1(i,j,k) &
                    - v1(i+2,j,k) + 16.0_fp*v1(i+1,j,k))* rdxidxi + &
                   (- v1(i,j-2,k) + 16.0_fp*v1(i,j-1,k) - 30.0_fp*v1(i,j,k) &
                    - v1(i,j+2,k) + 16.0_fp*v1(i,j+1,k))* rdyidyi
            visc_imp = ((v1(i,j,k+1)-v1(i,j,k))*dzc_inv(k)-(v1(i,j,k)-v1(i,j,k-1))*dzc_inv(k-1))*visc_dzf_inv(k)
            visc = visc*re_inv
            visc_imp = visc_imp*re_inv
            v(i,j,k) = v(i,j,k) + 0.5_fp*dt*((visc-conv) + rko_v(i,j,k) + visc_imp)
#endif

            q1m3 = (u1(i-2,j,k)*dzf(k+1) + u1(i-2,j,k+1)*dzf(k))*dzc_inv(k)
            q1m1 = (u1(i-1,j,k)*dzf(k+1) + u1(i-1,j,k+1)*dzf(k))*dzc_inv(k)
            q1p1 = (u1(i  ,j,k)*dzf(k+1) + u1(i  ,j,k+1)*dzf(k))*dzc_inv(k)
            q1p3 = (u1(i+1,j,k)*dzf(k+1) + u1(i+1,j,k+1)*dzf(k))*dzc_inv(k)
            q2m3 = w1(i-3,j,k)+w1(i  ,j,k)
            q2m1 = w1(i-1,j,k)+w1(i  ,j,k)
            q2p1 = w1(i  ,j,k)+w1(i+1,j,k)
            q2p3 = w1(i  ,j,k)+w1(i+3,j,k)
            duwdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dx_inv
            ! add a term induced by the convecting reference frame
            conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5_fp*dx_inv
            q1m3 = (v1(i,j-2,k)*dzf(k+1) + v1(i,j-2,k+1)*dzf(k))*dzc_inv(k)
            q1m1 = (v1(i,j-1,k)*dzf(k+1) + v1(i,j-1,k+1)*dzf(k))*dzc_inv(k)
            q1p1 = (v1(i,j  ,k)*dzf(k+1) + v1(i,j  ,k+1)*dzf(k))*dzc_inv(k)
            q1p3 = (v1(i,j+1,k)*dzf(k+1) + v1(i,j+1,k+1)*dzf(k))*dzc_inv(k)
            q2m3 = w1(i,j-3,k)+w1(i,j  ,k)
            q2m1 = w1(i,j-1,k)+w1(i,j  ,k)
            q2p1 = w1(i,j  ,k)+w1(i,j+1,k)
            q2p3 = w1(i,j  ,k)+w1(i,j+3,k)
            dvwdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25_fp*dy_inv
            q1m1 = (w1(i,j,k)+w1(i,j,k-1))
            q1p1 = (w1(i,j,k)+w1(i,j,k+1))
            q2m1 = q1m1 
            q2p1 = q1p1
            dwwdz = (q1p1*q2p1-q1m1*q2m1)*0.25_fp*dzc_inv(k)
            conv = conv + duwdx + dvwdy + dwwdz
#ifndef PSIP
            visc = (- w1(i-2,j,k) + 16.0_fp*w1(i-1,j,k) - 30.0_fp*w1(i,j,k) &
                    - w1(i+2,j,k) + 16.0_fp*w1(i+1,j,k))* rdxidxi + &
                   (- w1(i,j-2,k) + 16.0_fp*w1(i,j-1,k) - 30.0_fp*w1(i,j,k) &
                    - w1(i,j+2,k) + 16.0_fp*w1(i,j+1,k))* rdyidyi + &
                   ((w1(i,j,k+1)-w1(i,j,k))*dzf_inv(k+1)-(w1(i,j,k)-w1(i,j,k-1))*dzf_inv(k))*dzc_inv(k)
            visc = visc*re_inv
            w(i,j,k) = (w1(i,j,k) + dt*(visc-conv) + w(i,j,k))*0.5_fp
#else
            visc = (- w1(i-2,j,k) + 16.0_fp*w1(i-1,j,k) - 30.0_fp*w1(i,j,k) &
                    - w1(i+2,j,k) + 16.0_fp*w1(i+1,j,k))* rdxidxi + &
                   (- w1(i,j-2,k) + 16.0_fp*w1(i,j-1,k) - 30.0_fp*w1(i,j,k) &
                    - w1(i,j+2,k) + 16.0_fp*w1(i,j+1,k))* rdyidyi
            visc_imp = ((w1(i,j,k+1)-w1(i,j,k))*dzf_inv(k+1)-(w1(i,j,k)-w1(i,j,k-1))*dzf_inv(k))*dzc_inv(k)
            visc = visc*re_inv
            visc_imp = visc_imp*re_inv
            w(i,j,k) = w(i,j,k) + 0.5_fp*dt*((visc-conv) + rko_w(i,j,k) + visc_imp)
#endif

        enddo
        enddo
        enddo
#ifndef _CUDA
        !$OMP END PARALLEL DO
#endif

        return
    end subroutine timeIntVelRK2_kernel

    subroutine timeIntVelRK(rk_num, u, v, w, unew, vnew, wnew, u_crf)
        implicit none
        integer, intent(in) :: rk_num
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in), contiguous :: u, v, w
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout), contiguous :: unew, vnew, wnew
        real(fp), intent(in) :: u_crf
        character(1) :: str_rk
        ! procedure(rk_kernel_interface), pointer :: rk_kernel_funptr => null()
        !@cuf attributes(managed) :: u, v, w, unew, vnew, wnew
        !@cuf integer :: istat

#ifdef NB_HALO
        integer :: tag_u = 1, tag_v = 2, tag_w = 3
        integer, dimension(8) :: isend_req_u, irecv_req_u
        integer, dimension(8) :: isend_req_v, irecv_req_v
        integer, dimension(8) :: isend_req_w, irecv_req_w
        integer :: ist, jst, kst
        integer :: y0_send_st, y0_send_en
        integer :: y1_send_st, y1_send_en
        integer :: z0_send_st, z0_send_en
        integer :: z1_send_st, z1_send_en
#ifdef USE_NBHALOBUF
        integer :: use_aux_thistime = 0
        if (USE_HALOBUF_AUX == 1) then
            use_aux_thistime = rk_num-1
        endif
#endif
#endif

        ! if (rk_num == 1) then
        !     rk_kernel_funptr => timeIntVelRK1_kernel
        ! else if (rk_num == 2) then
        !     rk_kernel_funptr => timeIntVelRK2_kernel
        ! endif
        write(str_rk, '(I1)') rk_num
        
#if defined(PSIP) || (!defined(PSIP) && !defined(NB_HALO))

        call RGPTLSTART('--uvw'//str_rk//' comp')

        if (rk_num == 1) then
            call timeIntVelRK1_kernel((/1,1,1/), sz, u, v, w, unew, vnew, wnew, u_crf)
        else if (rk_num == 2) then
            call timeIntVelRK2_kernel((/1,1,1/), sz, u, v, w, unew, vnew, wnew, u_crf)
        endif
        !@cuf istat = cudaDeviceSynchronize()

        call RGPTLSTOP('--uvw'//str_rk//' comp')

#ifdef PSIP
        call RGPTLSTART('--PSIP'//str_rk)

        call correctRHS(neighbor(5:6), nhalo, sz, rhs_u_cor, 'c', unew)
        call solveTrid_PDD(neighbor(5:6), nhalo, sz, v_pdd_c, w_pdd_c, tmp_v_pdd_c, a_c, b_c, c_c, unew)
        ! call correctRHS(neighbor(5:6), nhalo, sz, rhs_v_cor, 'c', vnew)
        call solveTrid_PDD(neighbor(5:6), nhalo, sz, v_pdd_c, w_pdd_c, tmp_v_pdd_c, a_c, b_c, c_c, vnew)
        ! call correctRHS(neighbor(5:6), nhalo, sz, rhs_w_cor, 'f', wnew)
        if (neighbor(6) == MPI_PROC_NULL) then
            call solveTrid_PDD(neighbor(5:6), nhalo, (/sz(1),sz(2),sz(3)-1/), v_pdd_f, w_pdd_f, tmp_v_pdd_f, a_f, b_f, c_f, wnew)
        else
            call solveTrid_PDD(neighbor(5:6), nhalo, sz, v_pdd_f, w_pdd_f, tmp_v_pdd_f, a_f, b_f, c_f, wnew)
        endif
        
        call RGPTLSTOP('--PSIP'//str_rk)
#endif

        call RGPTLSTART('--Update halo vel '//str_rk)

        call updateHalo(nhalo, halotype_vel, unew)
        call updateHalo(nhalo, halotype_vel, vnew)
        call updateHalo(nhalo, halotype_vel, wnew)

        call RGPTLSTOP('--Update halo vel '//str_rk)

#else
! #if defined(PSIP) || (!defined(PSIP) && !defined(NB_HALO))

        ist = 1-nhalo(1)
        jst = 1-nhalo(3)
        kst = 1-nhalo(5)
        z0_send_st = 1;         z0_send_en = nhalo(6)
        z1_send_st = sz(3)+kst; z1_send_en = sz(3)
        y0_send_st = 1;         y0_send_en = nhalo(4)
        y1_send_st = sz(2)+jst; y1_send_en = sz(2)

        call RGPTLSTART('--Update halo vel '//str_rk)

#ifndef USE_RDMA

#ifdef USE_NBHALOBUF
            call updateHaloBufIRecv(nhalo, tag_u, unew, irecv_req_u, use_aux_thistime)
            call updateHaloBufIRecv(nhalo, tag_v, vnew, irecv_req_v, use_aux_thistime)
            call updateHaloBufIRecv(nhalo, tag_w, wnew, irecv_req_w, use_aux_thistime)
#else
            call updateHaloIRecv(nhalo, tag_u, unew, irecv_req_u)
            call updateHaloIRecv(nhalo, tag_v, vnew, irecv_req_v)
            call updateHaloIRecv(nhalo, tag_w, wnew, irecv_req_w)
#endif

#endif

        call RGPTLSTOP('--Update halo vel '//str_rk)
        call RGPTLSTART('--uvw'//str_rk//' comp')

        ! *** bottom/top ***
        if (rk_num == 1) then
            call timeIntVelRK1_kernel((/1,1,z0_send_st/), (/sz(1),sz(2),z0_send_en/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
            call timeIntVelRK1_kernel((/1,1,max(z1_send_st,z0_send_en+1)/), (/sz(1),sz(2),z1_send_en/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
            ! *** south/north ***
            call timeIntVelRK1_kernel((/1,y0_send_st,z0_send_en+1/), (/sz(1),y0_send_en,z1_send_st-1/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
            call timeIntVelRK1_kernel((/1,max(y1_send_st,y0_send_en+1),z0_send_en+1/), (/sz(1),y1_send_en,z1_send_st-1/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
        else if (rk_num == 2) then
            call timeIntVelRK2_kernel((/1,1,z0_send_st/), (/sz(1),sz(2),z0_send_en/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
            call timeIntVelRK2_kernel((/1,1,max(z1_send_st,z0_send_en+1)/), (/sz(1),sz(2),z1_send_en/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
            ! *** south/north ***
            call timeIntVelRK2_kernel((/1,y0_send_st,z0_send_en+1/), (/sz(1),y0_send_en,z1_send_st-1/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
            call timeIntVelRK2_kernel((/1,max(y1_send_st,y0_send_en+1),z0_send_en+1/), (/sz(1),y1_send_en,z1_send_st-1/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
        endif
        !@cuf istat = cudaDeviceSynchronize()

        call RGPTLSTOP('--uvw'//str_rk//' comp')
        call RGPTLSTART('--Update halo vel '//str_rk)

#ifdef USE_NBHALOBUF
        call RGPTLSTART('----Memcpy to halobuf '//str_rk)

        call memcpyToHaloBuf(nhalo, tag_u, unew, use_aux_thistime)
        call memcpyToHaloBuf(nhalo, tag_v, vnew, use_aux_thistime)
        call memcpyToHaloBuf(nhalo, tag_w, wnew, use_aux_thistime)
        !@cuf istat = cudaDeviceSynchronize()

        call RGPTLSTOP('----Memcpy to halobuf '//str_rk)
#endif

#ifdef USE_RDMA
        call update_halo_rdma_send(rk_num-1)
#else

#ifdef USE_NBHALOBUF
        call updateHaloBufISend(nhalo, tag_u, unew, isend_req_u, use_aux_thistime)
        call updateHaloBufISend(nhalo, tag_v, vnew, isend_req_v, use_aux_thistime)
        call updateHaloBufISend(nhalo, tag_w, wnew, isend_req_w, use_aux_thistime)
#else
        call updateHaloISend(nhalo, tag_u, unew, isend_req_u)
        call updateHaloISend(nhalo, tag_v, vnew, isend_req_v)
        call updateHaloISend(nhalo, tag_w, wnew, isend_req_w)
#endif

#endif

        call RGPTLSTOP('--Update halo vel '//str_rk)
        call RGPTLSTART('--uvw'//str_rk//' comp')

        ! *** inner region ***
        if (rk_num == 1) then
            call timeIntVelRK1_kernel((/1,nhalo(4)+1,nhalo(6)+1/), (/sz(1),sz(2)-nhalo(3),sz(3)-nhalo(5)/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
        else if (rk_num == 2) then
            call timeIntVelRK2_kernel((/1,nhalo(4)+1,nhalo(6)+1/), (/sz(1),sz(2)-nhalo(3),sz(3)-nhalo(5)/), &
                                      u, v, w, unew, vnew, wnew, u_crf)
        endif
        !@cuf istat = cudaDeviceSynchronize()
        
        call RGPTLSTOP('--uvw'//str_rk//' comp')
        call RGPTLSTART('--Update halo vel '//str_rk)

#ifdef USE_RDMA
        call update_halo_rdma_wait(rk_num-1)
#else
        call updateHaloWaitall(isend_req_u, irecv_req_u)
        call updateHaloWaitall(isend_req_v, irecv_req_v)
        call updateHaloWaitall(isend_req_w, irecv_req_w)
#endif

#ifdef USE_NBHALOBUF
        call RGPTLSTART('----Memcpy from halobuf '//str_rk)

        call memcpyFromHaloBuf(nhalo, tag_u, unew, use_aux_thistime)
        call memcpyFromHaloBuf(nhalo, tag_v, vnew, use_aux_thistime)
        call memcpyFromHaloBuf(nhalo, tag_w, wnew, use_aux_thistime)
        !@cuf istat = cudaDeviceSynchronize()

        call RGPTLSTOP('----Memcpy from halobuf '//str_rk)
#endif

        call RGPTLSTOP('--Update halo vel '//str_rk)

#endif
! #if defined(PSIP) || (!defined(PSIP) && !defined(NB_HALO))

        call RGPTLSTART('--Impose BC vel '//str_rk)

        call imposeBCVel(unew, vnew, wnew, u_crf)

        call RGPTLSTOP('--Impose BC vel '//str_rk)

        return
    end subroutine timeIntVelRK

    subroutine correctVel(p, u, v, w)
        implicit none
        real(fp), dimension(0:,0:,0:), intent(in   ) :: p
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: u, v, w
        !@cuf attributes(managed) :: u, v, w, p
        !@cuf integer :: istat

        real(fp) :: dtdxi, dtdyi
        integer :: i, j, k

        dtdxi = dt*dx_inv
        dtdyi = dt*dy_inv

#ifdef _CUDA
        !$cuf kernel do(3) <<<*,*>>>
#else
        !$OMP PARALLEL DO SCHEDULE(STATIC) &
        !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
#endif
        do k = 1, sz(3)
        do j = 1, sz(2)
        do i = 1, sz(1)
            u(i, j, k) = u(i, j, k) - (p(i+1, j, k)-p(i, j, k))*dtdxi
            v(i, j, k) = v(i, j, k) - (p(i, j+1, k)-p(i, j, k))*dtdyi
            w(i, j, k) = w(i, j, k) - (p(i, j, k+1)-p(i, j, k))*dt*dzc_inv(k)
        enddo
        enddo
        enddo
#ifndef _CUDA
        !$OMP END PARALLEL DO
#endif
        !@cuf istat = cudaDeviceSynchronize()

        return
    end subroutine correctVel

    subroutine forceVel(u, v, w)
        implicit none
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(inout) :: u, v, w
        !@cuf attributes(managed) :: u, v, w
        !@cuf integer :: istat

        real(fp) :: vel_mean_x, vel_mean_y, vel_mean_z
        real(fp) :: force_x, force_y, force_z
        integer :: i, j, k

        if (is_forced(1)) then
            vel_mean_x = MEAN(nhalo, sz, nx, ny, dzflzi, u)
            force_x = vel_force(1) - vel_mean_x
#ifdef _CUDA
            !$cuf kernel do(3) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC) &
            !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
#endif
            do k = 1, sz(3)
            do j = 1, sz(2)
            do i = 1, sz(1)
                u(i, j, k) = u(i, j, k) + force_x
            enddo
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#endif
            !@cuf istat = cudaDeviceSynchronize()
        endif

        if (is_forced(2)) then
            vel_mean_y = MEAN(nhalo, sz, nx, ny, dzflzi, v)
            force_y = vel_force(2) - vel_mean_y
#ifdef _CUDA
            !$cuf kernel do(3) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC) &
            !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
#endif
            do k = 1, sz(3)
            do j = 1, sz(2)
            do i = 1, sz(1)
                v(i, j, k) = v(i, j, k) + force_y
            enddo
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#endif
            !@cuf istat = cudaDeviceSynchronize()
        endif

        if (is_forced(3)) then
            vel_mean_z = MEAN(nhalo, sz, nx, ny, dzflzi, w)
            force_z = vel_force(3) - vel_mean_z
#ifdef _CUDA
            !$cuf kernel do(3) <<<*,*>>>
#else
            !$OMP PARALLEL DO SCHEDULE(STATIC) &
            !$OMP DEFAULT(SHARED) PRIVATE(i, j, k)
#endif
            do k = 1, sz(3)
            do j = 1, sz(2)
            do i = 1, sz(1)
                w(i, j, k) = w(i, j, k) + force_z
            enddo
            enddo
            enddo
#ifndef _CUDA
            !$OMP END PARALLEL DO
#endif
            !@cuf istat = cudaDeviceSynchronize()
        endif

        return
    end subroutine forceVel

end module mod_calcVel