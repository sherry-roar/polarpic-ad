program PowerLLEL_Postprocess
    use mod_type, only: fp
    use mod_param_postproc
    use mod_variables_postproc
    use mod_mesh_postproc, only: initMesh, freeMesh
    use mod_dataio_postproc, only: inputVelField, inputPreField
    use mod_bc_postproc, only: updateBoundVel, updateBoundPre
    use mod_statistics_postproc, only: allocStat, initStat, calcStatScheme1, calcStatScheme2, outputStat, freeStat
    use mod_vortex_postproc, only: calcVelGradTensor, outVorticity, outQ, outLambda2
    use mod_hdf5, only: initIO, freeIO
    use mod_mpi, only: st, sz, myrank, ierr, initMPI, freeMPI

    implicit none
    
    include 'mpif.h'

    integer :: i
    integer :: nt = -1
    character(80) :: inst_fn_prefix
    character(10) :: string_io
    
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    if (myrank == 0) then
        write(*,'(A)') "================================================================================"
        write(*,'(A)') "                               PowerLLEL_Postprocess                            "
        write(*,'(A)') "================================================================================"
        write(*,'(A)') "PowerLLEL_Postprocess.NOTE: Start postprocessing ..."
    endif
    
    ! read input parameters from file
    call readInputParam('param_postproc.in')

    if ((.not. out_stat) .and. &
        (.not. out_vortex)) then
        if (myrank == 0) then
            write(*,'(A)') "PowerLLEL_Postprocess.NOTE: Postprocessing ends without doing any work!"
        endif
        call MPI_FINALIZE(ierr)
    endif
    
    call get_command_argument(1, inst_fn_prefix)
    if (len_trim(inst_fn_prefix) /= 0) then
        call get_command_argument(2, string_io)
        if (len_trim(string_io) /= 0) then
            read(string_io, '(I10)') nt
            write(string_io, "('_',I8.8,'_')") nt
            if (myrank == 0) then
                write(*,'(A)') "PowerLLEL_Postprocess.NOTE: Inst. file prefix is '"//trim(inst_fn_prefix)//string_io//"'"
            endif
            batch_inst_fn_prefix = inst_fn_prefix
            batch_inst_nt_start = nt
            batch_inst_nt_end = nt
            batch_inst_nt_interval = 1
        endif
    endif

    ! initialize MPI
    call initMPI(nx, ny, nz, (/'PP','PP','NN'/), p_row, p_col, nhalo)

    ! initialize parallel IO
    call initIO(MPI_COMM_WORLD)

    ! initialize mesh
    call initMesh()

    ! allocate variables
    call allocVariable(nhalo, sz, 'u', u)
    call allocVariable(nhalo, sz, 'v', v)
    call allocVariable(nhalo, sz, 'w', w)
    if (out_stat) then
        call allocVariable(nhalo_one,  sz, 'p', p)
        call allocStat(sz)
        call initStat(batch_inst_nt_start)
    endif
    if (out_vortex) then
        call allocVariable(nhalo_zero, sz, 'dudx', vel_grad%ux)
        call allocVariable(nhalo_zero, sz, 'dudy', vel_grad%uy)
        call allocVariable(nhalo_zero, sz, 'dudz', vel_grad%uz)
        call allocVariable(nhalo_zero, sz, 'dvdx', vel_grad%vx)
        call allocVariable(nhalo_zero, sz, 'dvdy', vel_grad%vy)
        call allocVariable(nhalo_zero, sz, 'dvdz', vel_grad%vz)
        call allocVariable(nhalo_zero, sz, 'dwdx', vel_grad%wx)
        call allocVariable(nhalo_zero, sz, 'dwdy', vel_grad%wy)
        call allocVariable(nhalo_zero, sz, 'dwdz', vel_grad%wz)
    endif

    do nt = batch_inst_nt_start, batch_inst_nt_end, batch_inst_nt_interval

        if (myrank == 0) write(*, '(A,I0)') ">>> nt = ", nt

        write(string_io, "('_',I8.8,'_')") nt
        ! read velocity fields
        call inputVelField(trim(batch_inst_fn_prefix)//string_io, st, sz, nhalo, u, v, w)
        call updateBoundVel(nhalo, u, v, w)
        
        if (out_stat) then
            ! statistics
            call inputPreField(trim(batch_inst_fn_prefix)//string_io, st, sz, nhalo_one, p)
            call updateBoundPre(nhalo_one, p)
            select case(fd_scheme)
            case(1)
                call calcStatScheme1(nt, nhalo, u, v, w, nhalo_one, p)
            case(2)
                call calcStatScheme2(nt, nhalo, u, v, w, nhalo_one, p)
            end select
        endif

        if (out_vortex) then
            ! calculate velocity gradient tensor
            call calcVelGradTensor()
            ! write vortex fields
            if (out_vorticity) call outVorticity(trim(batch_inst_fn_prefix)//string_io, nt)
            if (out_q)         call outQ(trim(batch_inst_fn_prefix)//string_io, nt)
            if (out_lambda2)   call outLambda2(trim(batch_inst_fn_prefix)//string_io, nt)
        endif

    enddo

    deallocate(u)
    deallocate(v)
    deallocate(w)
    if (out_stat) then
        deallocate(p)
        call outputStat()
        call freeStat()
    endif
    if (out_vortex) then
        deallocate(vel_grad%ux)
        deallocate(vel_grad%uy)
        deallocate(vel_grad%uz)
        deallocate(vel_grad%vx)
        deallocate(vel_grad%vy)
        deallocate(vel_grad%vz)
        deallocate(vel_grad%wx)
        deallocate(vel_grad%wy)
        deallocate(vel_grad%wz)
    endif

    call freeMesh()
    call freeIO()
    call freeMPI()

    if (myrank == 0) then
        write(*,'(A)') "********************************************************************************"
        write(*,'(A)') "PowerLLEL_Postprocess.NOTE: Postprocessing ends successfully!"
    endif

    call MPI_FINALIZE(ierr)
    
end program PowerLLEL_Postprocess
