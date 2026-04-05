! #define RK_KERNEL_C 1
program PowerLLEL
    use, intrinsic :: iso_c_binding, only: c_null_char
    use mod_type,          only: fp, i8
    use mod_parameters
    use mod_mpi,           only: MPI_REAL_FP, myrank, ierr, sz, initMPI, freeMPI
    use mod_mpi,           only: comm_cart, halotype_vel, halotype_one, neighbor, neighbor_xyz
    use mod_variables
    use mod_mesh
    use mod_poissonSolver, only: initPoissonSolver, executePoissonSolver, freePoissonSolver
#ifdef USE_C
    use mod_poissonSolver, only: init_poisson_solver, execute_poisson_solver, free_poisson_solver
#endif
    use mod_initFlow,      only: initFlow
    use mod_updateBound,   only: updateBoundVel, updateBoundP
    use mod_updateBound,   only: update_bound_vel, update_bound_p

#ifdef NB_HALO
#if defined(USE_C) || defined(USE_RDMA)
    use mod_updateBound,   only: get_neighbor_rank_2d_cart_c, create_nbhalo_mpitype_c, free_nbhalo_mpitype
#endif

#if defined(USE_RDMA)
    use mod_updateBound,   only: init_rdma_halo_c
#ifdef USE_NBHALOBUF
    use mod_mpi,           only: halobuf_length, halobuf_offset, halobuf_send, halobuf_recv, halobuf_send_aux, halobuf_recv_aux
#endif
#endif

#endif

    use mod_calcVel,       only: timeIntVelRK, correctVel, forceVel, transform2CRF
#ifdef PSIP
    use mod_calcVel,       only: initImpTimeMarcher, freeImpTimeMarcher
#endif
    use mod_calcVel,       only: transform_to_crf, time_int_vel_rk, correct_vel, force_vel
    use mod_calcRHS,       only: calcRHS, calc_rhs
    use mod_monitor,       only: initMonitor, freeMonitor
    use mod_hdf5,          only: initIO, freeIO
    use mod_dataIO,        only: inputData, outputData
    use mod_statistics,    only: allocStat, freeStat, initStat, calcStat, inputStat, outputStat
    use mod_utils,         only: value_index_pair_t, initCheckCFLAndDiv, freeCheckCFLAndDiv, &
                                 calcMaxCFL, calcMaxDiv, checkCFL, checkDiv, checkNaN
    !$ use omp_lib
    use gptl
#ifdef _CUDA
    use cudafor
    use mod_device, only: istat, mydev
#endif
#ifdef NVTX
    use nvtx
#endif

    implicit none

    include 'mpif.h'
    
    ! interface
    !     subroutine c_binding_test() bind(C)
    !         use, intrinsic :: iso_c_binding
    !     end subroutine
    ! end interface


    integer :: nt, nt_in, nt_start
    real(fp):: wtime, wtime_avg
    type(value_index_pair_t):: cfl_max, div_max
    logical :: check_passed, subcheck_passed
    character(13) :: remaining_time_c
    integer :: backend = -1
    character(80) :: str
    integer :: ret

#ifdef GPTL
    ! GPTL initialization
    call gptlprocess_namelist('gptlnl', 1, ret)
    if (ret /= 0) call abort(999, "main: GPTL namelist read failure!")
    ret = gptlinitialize()
#endif

    call MPI_INIT_THREAD(MPI_THREAD_MULTIPLE, ret, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    !$ nthreads = OMP_GET_MAX_THREADS()

    if (myrank == 0) then
        write(*,'(A)') "================================================================================"
#ifndef PSIP
        write(*,'(A)') "               PowerLLEL_channel (DNS, x/y-4thFD + z-2ndFD + RK2)               "
#else
        write(*,'(A)') "           PowerLLEL_channel (DNS, x/y-4thFD + z-2ndFD + RK2 + z-CN)            "
#endif
        write(*,'(A)') "================================================================================"
        write(*,'(A)') "PowerLLEL.NOTE: Initialization starts ..."
    endif

#ifdef _CUDA
    block
        integer(kind=cuda_count_kind) :: free, total
        istat = cudaMemGetInfo(free, total)
        if (myrank == 0) write(*,'(A,2F8.3)') "PowerLLEL.NOTE: cudaMemGetInfo(free, total) (GB): ", &
                                free/1024.0**3, total/1024.0**3
    end block
#endif

    ! read input parameters from file
    call readInputParam('param.in')

    ! read comm backend parameter from command line
    if (command_argument_count() > 0) then
        call get_command_argument(1, str)
        if (len_trim(str) > 0) then
            read(str, *) backend
            if (myrank == 0) write(*,'(A,I0)') "PowerLLEL.NOTE: Comm backend is set to ", backend
        endif
    endif

    ! initialize MPI
    if (backend == -1) then
        call initMPI(nx, ny, nz, (/'PP','PP','NN'/), p_row, p_col, nhalo)
    else
        call initMPI(nx, ny, nz, (/'PP','PP','NN'/), p_row, p_col, nhalo, backend)
    endif

    ! initialize parallel IO
    call initIO(MPI_COMM_WORLD)
    
    ! initialize the monitoring point
    call initMonitor()
    
    ! allocate variables
    call allocVariables(nhalo, sz)

    ! initialize mesh
    call initMesh()

    ! initialize Poisson solver
#ifdef USE_C
    call init_poisson_solver(nx, ny, nz, dx, dy, dzf_global, "PP"//c_null_char, "PP"//c_null_char, "NN"//c_null_char, &
                             neighbor_xyz)
#else
    call initPoissonSolver(nx, ny, nz, dx, dy, dzf_global, (/'PP','PP','NN'/), real((/0.0,0.0, 0.0,0.0, 0.0,0.0/),fp))
#endif

#ifdef PSIP
    ! initialize implicit time marcher
    call initImpTimeMarcher()
#endif

    ! initialize the flow field
    if (is_restart) then
        if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE: Initializing flow from checkpoint fields ..."
        call inputData(nt_in, u, v, w)
        nt_start = nt_in + 1
    else
        if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE: Initializing flow according to input parameters ..."
        call initFlow(u, v, w)
        nt_start = 1
    endif

    !@cuf istat = cudaMemPrefetchAsync(  u, size( u), mydev, 0 )
    !@cuf istat = cudaMemPrefetchAsync(  v, size( v), mydev, 0 )
    !@cuf istat = cudaMemPrefetchAsync(  w, size( w), mydev, 0 )
    !@cuf istat = cudaMemPrefetchAsync( u1, size(u1), mydev, 0 )
    !@cuf istat = cudaMemPrefetchAsync( v1, size(v1), mydev, 0 )
    !@cuf istat = cudaMemPrefetchAsync( w1, size(w1), mydev, 0 )
    !@cuf istat = cudaMemPrefetchAsync(  p, size( p), mydev, 0 )
#ifdef USE_OMP_OFFLOAD
    !$omp target enter data map(to: u, v, w, u1, v1, w1, p)
    !$omp target enter data map(to: dzflzi, dzf, dzf_inv, dzc_inv, visc_dzf_inv)
#endif
    
    ! important! subtract the convecting reference frame velocity u_crf from u
#ifdef USE_C
    call transform_to_crf(u_crf, nhalo, sz, u, vel_force(1))
#else
    call transform2CRF(u_crf, nhalo, sz, u, vel_force(1))
#endif

    ! update velocity & pressure boundary conditions
#ifdef USE_C
    
#ifdef NB_HALO
    call get_neighbor_rank_2d_cart_c(comm_cart)
    call create_nbhalo_mpitype_c(nhalo, sz, MPI_REAL_FP)
#endif

    call update_bound_vel(comm_cart, halotype_vel, neighbor, nhalo, sz, u, v, w, u_crf, '0'//c_null_char)
    call update_bound_p(comm_cart, halotype_one, neighbor, nhalo_one, sz, p, '0'//c_null_char)
#else
    call updateBoundVel(u, v, w, u_crf, '0')
    call updateBoundP(p, '0')
#endif

    ! load the statistics data if necessary
    if (is_restart .and. nt_start > nt_init_stat+1) then
        call allocStat((/nx,ny,nz/), sz)
        call inputStat(fn_prefix_input_stat)
    endif

    ! initialize MPI variables related to the calculation of CFL and Divergence
    call initCheckCFLAndDiv(cfl_max, div_max)

    if (myrank == 0) then
        write(*,'(A)') "PowerLLEL.NOTE: Initialization ends successfully!"
#ifdef _CUDA
        block
            integer(i8) :: used
            used = getDeviceMemoryFootprint(sz, nhalo, nhalo_one)
            write(*,'(A,F8.3)') "PowerLLEL.NOTE.getDeviceMemoryFootprint (GB): SUM: ", used/1024.0**3
        end block
        block
            integer(kind=cuda_count_kind) :: free, total
            istat = cudaMemGetInfo(free, total)
            write(*,'(A,2F8.3)') "PowerLLEL.NOTE: cudaMemGetInfo(free, total) (GB): ", &
                                 free/1024.0**3, total/1024.0**3
        end block
#endif
        write(*,'(A,I9,A)') "PowerLLEL.NOTE: Simulation starts at nt = ", nt_start, "!"
        write(*,'(A)') "********************************************************************************"
        write(*,999) 'nt', 'speed(wSteps/Day)', 'remaining time', 'cfl_max', 'div_max'
    999 format(A9,2X,A17,2X,A14,2X,A10,2X,A10)
    endif

#if defined(NB_HALO) && defined(USE_RDMA)
    call get_neighbor_rank_2d_cart_c(comm_cart)
#ifdef USE_NBHALOBUF
    call init_rdma_halo_c(nhalo, sz, halobuf_length, halobuf_offset, &
                          halobuf_send, halobuf_recv, halobuf_send_aux, halobuf_recv_aux, comm_cart)
#else
    call init_rdma_halo_c(nhalo, sz, u, v, w, u1, v1, w1, comm_cart)
#endif
#endif

#ifdef NVTX
    call nvtxStartRange("main loop", 2)
#endif

    !Start timing
    wtime = MPI_WTIME()

    !===========================!
    !  Main time marching loop  !
    !===========================!
    do nt = nt_start, nt_end

        call RGPTLSTART('Main loop')
        call RGPTLSTART('uvw1')

#if defined(USE_C) && defined(RK_KERNEL_C)
        call time_int_vel_rk(1, comm_cart, halotype_vel, neighbor, &
                             re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, &
                             nhalo, sz, u, v, w, u1, v1, w1, u_crf)
#else
        call timeIntVelRK(1, u, v, w, u1, v1, w1, u_crf)
#endif

        call RGPTLSTOP('uvw1')
        call RGPTLSTART('uvw2')

#if defined(USE_C) && defined(RK_KERNEL_C)
        call time_int_vel_rk(2, comm_cart, halotype_vel, neighbor, &
                             re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, &
                             nhalo, sz, u1, v1, w1, u, v, w, u_crf)
#else
        call timeIntVelRK(2, u1, v1, w1, u, v, w, u_crf)
#endif

        call RGPTLSTOP('uvw2')
        call RGPTLSTART('Calculate RHS')

#ifdef USE_C
        call calc_rhs(dt, dx_inv, dy_inv, dzf_inv, nhalo, nhalo_one, sz, u, v, w, p)
#else
        call calcRHS(u, v, w, p)
#endif

        call RGPTLSTOP('Calculate RHS')
        call RGPTLSTART('Poisson solver')

#ifdef USE_OMP_OFFLOAD
        !$omp target update from(p)
#endif

#ifdef USE_C
        call execute_poisson_solver(p)
#else
        call executePoissonSolver(p)
#endif

#ifdef USE_OMP_OFFLOAD
        !$omp target update to(p)
#endif

        call RGPTLSTOP('Poisson solver')
        call RGPTLSTART('Update boundary pres')

#ifdef USE_C
        call update_bound_p(comm_cart, halotype_one, neighbor, nhalo_one, sz, p, '1'//c_null_char)
#else
        call updateBoundP(p, '1')
#endif

        call RGPTLSTOP('Update boundary pres')
        call RGPTLSTART('Correct vel')

#ifdef USE_C
        call correct_vel(dt, dx_inv, dy_inv, dzc_inv, nhalo_one, nhalo, sz, p, u, v, w)
#else
        call correctVel(p, u, v, w)
#endif

        call RGPTLSTOP('Correct vel')
        call RGPTLSTART('Force vel')

#ifdef USE_C
        call force_vel(is_forced, vel_force, nx, ny, dzflzi, nhalo, sz, u, v, w)
#else
        call forceVel(u, v, w)
#endif

        call RGPTLSTOP('Force vel')
        call RGPTLSTART('Update boundary vel')

#ifdef USE_C
        call update_bound_vel(comm_cart, halotype_vel, neighbor, nhalo, sz, u, v, w, u_crf, '3'//c_null_char)
#else
        call updateBoundVel(u, v, w, u_crf, '3')
#endif

        call RGPTLSTOP('Update boundary vel')
        call RGPTLSTOP('Main loop')

#ifdef USE_OMP_OFFLOAD
        !$omp target update from(u, v, w, p)
#endif

        if (mod(nt, nt_check) == 0) then
            !@cuf istat = cudaMemPrefetchAsync( u, size(u), cudaCpuDeviceId, 0 )
            !@cuf istat = cudaMemPrefetchAsync( v, size(v), cudaCpuDeviceId, 0 )
            !@cuf istat = cudaMemPrefetchAsync( w, size(w), cudaCpuDeviceId, 0 )
            call checkNaN(u, 'u', subcheck_passed)
            if (myrank == 0) check_passed = subcheck_passed
            call calcMaxCFL(nhalo, sz, dt, dx_inv, dy_inv, dzf_inv, u, v, w, cfl_max)
            call checkCFL(cfl_limit, cfl_max, subcheck_passed)
            if (myrank == 0) check_passed = check_passed .and. subcheck_passed
            call calcMaxDiv(nhalo, sz, dx_inv, dy_inv, dzf_inv, u, v, w, div_max)
            call checkDiv(div_limit, div_max, subcheck_passed)
            if (myrank == 0) check_passed = check_passed .and. subcheck_passed
            call MPI_BCAST(check_passed, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
            if (.not. check_passed) then
                if (myrank == 0) then
                    write(*,997) 'PowerLLEL.ERROR: nt = ',nt,', cfl_max = ',cfl_max%value, &
                                 ' at (',cfl_max%ig,',',cfl_max%jg,',',cfl_max%kg,') from rank', cfl_max%rank
                    write(*,997) 'PowerLLEL.ERROR: nt = ',nt,', div_max = ',div_max%value, &
                                 ' at (',div_max%ig,',',div_max%jg,',',div_max%kg,') from rank', div_max%rank
                997 format(A,I9,A,E10.3,4(A,I5))
                endif
                call MPI_FINALIZE(ierr)
                stop
            endif
        endif

        if (nt>nt_init_stat) then
            if (nt == nt_init_stat+1) then
                call allocStat((/nx,ny,nz/), sz)
                call initStat(nt_init_stat)
                if (myrank == 0) write(*,'(A,I9,A)') "PowerLLEL.NOTE: Statistical process starts at nt = ", nt, "!"
            endif
            if (mod(nt-nt_init_stat, sample_interval) == 0) then
                call calcStat(stat_scheme, nt, nhalo, u, v, w, nhalo_one, p, u_crf)
            endif
        endif

        if (mod(nt, nt_out_scrn) == 0) then
            wtime = MPI_WTIME() - wtime
            wtime_avg = wtime
            call MPI_REDUCE(wtime, wtime_avg, 1, MPI_REAL_FP, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
            wtime_avg = wtime_avg/p_row/p_col
            if (myrank == 0) then
                call convertTime(int((nt_end-nt)*wtime_avg/nt_out_scrn), remaining_time_c)
                write(*,998) nt, nt_out_scrn*3600.0*24.0/wtime_avg/10000, remaining_time_c, cfl_max%value, div_max%value
            998 format(I9,2X,F17.3,2X,A14,2(2X,E10.3))
            endif
        endif
     
        ! output routines below
        if (nt>nt_init_stat .and. mod(nt-nt_init_stat,nt_out_stat) == 0) then
            call outputStat(fn_prefix_stat)
        endif
        
        call outputData(nt, u, v, w, p, u_crf)

        if (mod(nt, nt_out_scrn) == 0) then
            wtime = MPI_WTIME()
        endif

        !@cuf istat = cudaMemPrefetchAsync(  u, size( u), mydev, 0 )
        !@cuf istat = cudaMemPrefetchAsync(  v, size( v), mydev, 0 )
        !@cuf istat = cudaMemPrefetchAsync(  w, size( w), mydev, 0 )
        !@cuf istat = cudaMemPrefetchAsync(  p, size( p), mydev, 0 )

    enddo

#ifdef USE_OMP_OFFLOAD
    !$omp target exit data map(from: u, v, w, u1, v1, w1, p)
    !$omp target exit data map(from: dzflzi, dzf, dzf_inv, dzc_inv, visc_dzf_inv)
#endif

#ifdef NVTX
    call nvtxEndRange()
#endif

    call freeIO()
#ifdef USE_C

#ifdef NB_HALO
    call free_nbhalo_mpitype()
#endif

    call free_poisson_solver()
#else
    call freePoissonSolver()
#endif
#ifdef PSIP
    call freeImpTimeMarcher()
#endif
    call freeMesh()
    call freeMonitor()
    call freeVariables()
    call freeStat()
    call freeCheckCFLAndDiv()
    call freeMPI()

#ifdef GPTL
    ! Print timer stats to file named 'timing.summary' and 'timing.$(myrank)'
    ret = gptlpr_summary(MPI_COMM_WORLD)
    ! ret = gptlpr(myrank)
    ret = gptlfinalize()
#endif

    if (myrank == 0) then
        write(*,'(A)') "********************************************************************************"
        write(*,'(A)') "PowerLLEL.NOTE: Simulation ends successfully!"
    endif

    call MPI_FINALIZE(ierr)

    contains
    subroutine convertTime(t, tc)
        implicit none
        integer, intent(in) :: t
        character(13), intent(out) :: tc

        integer :: t_d, t_h, t_m, t_s, t_res

        t_d = int(t/86400)
        t_res = t-t_d*86400
        t_h = int(t_res/3600)
        t_res = t_res-t_h*3600
        t_m = int(t_res/60)
        t_s = t_res-t_m*60

        write(tc,'(I3,A,3(I2,A))') t_d,'d',t_h,'h',t_m,'m',t_s,'s'

    end subroutine convertTime

#ifdef _CUDA
    !
    ! estimate GPU memory footprint, assuming one MPI task <-> one GPU
    !
    function getDeviceMemoryFootprint(sz, nhalo_vel, nhalo_p) result(tot_bytes)
        use mod_mesh, only: dzf_global
        use decomp_2d, only: getMemoryFootprintDecomp2d
        use mod_poissonSolver, only: getMemoryFootprintPoisson
        use mod_fft, only: getMemoryFootprintCUFFT
        integer, dimension(3), intent(in) :: sz
        integer, dimension(6), intent(in) :: nhalo_vel, nhalo_p
        integer :: sz_h(3)
        integer(i8) :: tot_bytes, tmp_bytes, fp_bytes
        tot_bytes = 0
        tmp_bytes = 0
        fp_bytes = fp
        !
        ! 1. 'main' arrays
        !    u, v, w, u1, v1, w1, p
        !
        sz_h(1) = sz(1) + nhalo_vel(1) + nhalo_vel(2)
        sz_h(2) = sz(2) + nhalo_vel(3) + nhalo_vel(4)
        sz_h(3) = sz(3) + nhalo_vel(5) + nhalo_vel(6)
        tmp_bytes = product(sz_h(:)) * fp_bytes * 6
        sz_h(1) = sz(1) + nhalo_p(1) + nhalo_p(2)
        sz_h(2) = sz(2) + nhalo_p(3) + nhalo_p(4)
        sz_h(3) = sz(3) + nhalo_p(5) + nhalo_p(6)
        tmp_bytes = tmp_bytes + product(sz_h(:)) * fp_bytes * 1
        write(*,'(A,F8.3)') "PowerLLEL.NOTE.getDeviceMemoryFootprint (GB): Main: ", tmp_bytes/1024.0**3
        tot_bytes = tot_bytes + tmp_bytes
        !
        ! 2. mpi & decomp_2d arrays
        !    work1_r_d, work2_r_d
        !    sendbuf_y0, recvbuf_y0, sendbuf_y1, recvbuf_y1, sendbuf_z0, recvbuf_z0, sendbuf_z1, recvbuf_z1
        !    halobuf_send, halobuf_recv, halobuf_send_aux, halobuf_recv_aux
        !
        tmp_bytes = getMemoryFootprintDecomp2d()
        write(*,'(A,F8.3)') "PowerLLEL.NOTE.getDeviceMemoryFootprint (GB): Decomp2d: ", tmp_bytes/1024.0**3
        tot_bytes = tot_bytes + tmp_bytes

        sz_h(1) = sz(1) + nhalo_vel(1) + nhalo_vel(2)
        sz_h(2) = max(nhalo_vel(3), nhalo_vel(4))
        sz_h(3) = sz(3) + nhalo_vel(5) + nhalo_vel(6)
        tmp_bytes = product(sz_h(:)) * fp_bytes * 4
        sz_h(1) = sz(1) + nhalo_vel(1) + nhalo_vel(2)
        sz_h(2) = sz(2) + nhalo_vel(3) + nhalo_vel(4)
        sz_h(3) = max(nhalo_vel(5), nhalo_vel(6))
        tmp_bytes = tmp_bytes + product(sz_h(:)) * fp_bytes * 4
#if defined(NB_HALO) && defined(USE_NBHALOBUF)
        sz_h(1) = sz(1) + nhalo_vel(1) + nhalo_vel(2)
        sz_h(2) = sz(2) + nhalo_vel(3) + nhalo_vel(4)
        sz_h(3) = sz(3) + nhalo_vel(5) + nhalo_vel(6)
        tmp_bytes = tmp_bytes + (product(sz_h(:)) - product(sz(:))) * fp_bytes * 2
#ifdef USE_RDMA
        tmp_bytes = tmp_bytes + (product(sz_h(:)) - product(sz(:))) * fp_bytes * 2
#endif
#endif
        write(*,'(A,F8.3)') "PowerLLEL.NOTE.getDeviceMemoryFootprint (GB): Halobuf: ", tmp_bytes/1024.0**3
        tot_bytes = tot_bytes + tmp_bytes
        !
        ! 3. mesh arrays
        !    dzc, dzf, dzc_inv, dzf_inv, dzflzi, visc_dzf_inv, dzf_global
        !
        tmp_bytes = (sz(3) + nhalo_vel(5) + nhalo_vel(6)) * fp_bytes * 6 + size(dzf_global) * fp_bytes
        tot_bytes = tot_bytes + tmp_bytes
        !
        ! 4. Poisson solver arrays
        !    var_xpen, var_ypen, var_zpen
        !    a, b, c
        !    v_pdd, w_pdd, y1_pdd, y2_pdd, y3_pdd, tmp_var_pdd, tmp_v_pdd, pdd_sbuf, pdd_rbuf
        !    cufft_workarea, cwork, work_tr, cwork_tr
        !
        tmp_bytes = getMemoryFootprintPoisson()
        write(*,'(A,F8.3)') "PowerLLEL.NOTE.getDeviceMemoryFootprint (GB): Poisson: ", tmp_bytes/1024.0**3
        tot_bytes = tot_bytes + tmp_bytes

        tmp_bytes = getMemoryFootprintCUFFT()
        write(*,'(A,F8.3)') "PowerLLEL.NOTE.getDeviceMemoryFootprint (GB): CUFFT: ", tmp_bytes/1024.0**3
        tot_bytes = tot_bytes + tmp_bytes
        
    end function getDeviceMemoryFootprint
#endif

end program PowerLLEL