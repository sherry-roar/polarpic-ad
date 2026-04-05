module mod_device
    use cudafor
    implicit none

    include 'mpif.h'

    ! make everything private unless declared public
    private

    integer, save :: local_comm, local_rank, global_rank
    integer, save :: ndevices, mydev
    type(cudaDeviceProp), save :: prop
    character(MPI_MAX_PROCESSOR_NAME), save :: procname
    integer, save :: namelen
    integer :: ierr, istat

    public :: ndevices, mydev, istat
    public :: initDevices

contains

    subroutine initDevices()
        implicit none
        integer :: passed
        integer :: table_width

        call MPI_COMM_RANK(MPI_COMM_WORLD, global_rank, ierr)
        call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
             MPI_INFO_NULL, local_comm, ierr)
        call MPI_COMM_RANK(local_comm, local_rank, ierr)

        passed = 0
        ndevices = 0
        ierr = cudaGetDeviceCount(ndevices)
        call MPI_GET_PROCESSOR_NAME(procname, namelen, ierr)
        if (ndevices == 0) then
            passed = 1
            write(*,'(A)') 'ERROR: No CUDA devices found on Node '//procname(1:namelen)
        endif
        call MPI_ALLREDUCE(MPI_IN_PLACE, passed, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
        if (passed /= 0) then
            if (global_rank == 0) write(*,'(A)') 'Stop for CUDA devices initialization failure!'
            call MPI_FINALIZE(ierr)
        endif

        ! MPI process binding to devices
        mydev = mod(local_rank,ndevices)
        ierr = cudaSetDevice(mydev)

        ! print devices info
        ierr = cudaGetDeviceProperties(prop, mydev)
        if (global_rank == 0) then
            table_width = 55
            write(*,'(A)') 'Devices Info Summary'
            write(*,'(A)') repeat('=',table_width)
            write(*,'(A10,1X,A25,1X,A4,1X,A4,1X,A8)') &
            'Node','Device Name','Num.','CC','Mem (GB)'
            write(*,'(A)') repeat('-',table_width)
        endif
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        if (local_rank == 0) then
            write(*,"(A10,1X,A25,1X,I4,1X,I2,'.',I1,1X,F8.3)") &
            procname(1:namelen), trim(prop%name), ndevices, prop%major, prop%minor, prop%totalGlobalMem/1024.0**3
        endif
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        if (global_rank == 0) write(*,'(A)') repeat('=',table_width)

    end subroutine initDevices

end module mod_device