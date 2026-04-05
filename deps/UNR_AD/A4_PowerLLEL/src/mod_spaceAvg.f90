module mod_spaceAvg
    use mod_type
    use mod_mpi, only: MPI_REAL_FP, ierr, myrank
    use mod_utils, only: abort

    implicit none

    include 'mpif.h'
    ! make everything public unless declared private
    public

    type avg_line_t
        integer :: dir
        integer :: avg_count
        integer :: mpicomm_reduce
        integer :: mpicomm_root
        logical :: have_results
        integer, dimension(3) :: dims
        real(fp), allocatable, dimension(:) :: buffer
    end type

    type avg_plane_t
        integer :: norm_dir
        integer :: avg_count
        integer :: mpicomm_reduce
        integer :: mpicomm_root
        logical :: have_results
        integer, dimension(3) :: dims
        real(fp), allocatable, dimension(:,:) :: buffer
    end type

contains
    subroutine initAvgLine(line_dir, dims_box, avg_count, ipencil, pencil_coord, al)
        implicit none
        integer, intent(in) :: line_dir
        integer, dimension(3), intent(in) :: dims_box
        integer, intent(in) :: avg_count
        character(4), intent(in) :: ipencil
        integer, dimension(2), intent(in) :: pencil_coord
        type(avg_line_t), intent(out) :: al

        integer :: color, key, myrank_in_group
        integer :: istatus
        
        ! set the direction of the line
        al%dir = line_dir
        al%avg_count = avg_count

        ! set the mpi communitors for group reduction
        select case(ipencil)
        case('xpen')
            ! x-pencil
            select case(line_dir)
            case(1)
                al%mpicomm_reduce = MPI_COMM_WORLD
                color = MPI_UNDEFINED
                if (myrank == 0) color = 1
                key = myrank
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            case(2)
                color = pencil_coord(1)
                key   = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(al%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            case(3)
                color = pencil_coord(2)
                key   = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(al%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            end select
        case('ypen')
            ! y-pencil
            select case(line_dir)
            case(1)
                color = pencil_coord(1)
                key   = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(al%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            case(2)
                al%mpicomm_reduce = MPI_COMM_WORLD
                color = MPI_UNDEFINED
                if (myrank == 0) color = 1
                key = myrank
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            case(3)
                color = pencil_coord(2)
                key   = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(al%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            end select
        case('zpen')
            ! z-pencil
            select case(line_dir)
            case(1)
                color = pencil_coord(1)
                key   = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(al%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            case(2)
                color = pencil_coord(2)
                key   = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(al%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            case(3)
                al%mpicomm_reduce = MPI_COMM_WORLD
                color = MPI_UNDEFINED
                if (myrank == 0) color = 1
                key = myrank
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, al%mpicomm_root, ierr)
            end select
        end select

        ! determine whether the current process has the reduction results
        al%have_results = .false.
        if (al%mpicomm_root /= MPI_COMM_NULL) al%have_results = .true.

        ! set the buffer for reduction
        select case(line_dir)
        case(1)
            al%dims(1) = dims_box(1)
            al%dims(2) = dims_box(2)
            al%dims(3) = dims_box(3)
        case(2)
            al%dims(1) = dims_box(2)
            al%dims(2) = dims_box(1)
            al%dims(3) = dims_box(3)
        case(3)
            al%dims(1) = dims_box(3)
            al%dims(2) = dims_box(1)
            al%dims(3) = dims_box(2)
        end select
        allocate(al%buffer(al%dims(1)), stat=istatus)
        if (istatus /= 0) call abort(102, "allocStat: Out of memory when allocating buffer for space average!")

        return
    end subroutine initAvgLine

    subroutine freeAvgLine(al)
        implicit none
        type(avg_line_t), intent(inout) :: al

        if (allocated(al%buffer)) deallocate(al%buffer)

        return
    end subroutine freeAvgLine

    subroutine reduceToAvgLine(data, al)
        implicit none
        real(fp), dimension(:), intent(in) :: data
        type(avg_line_t), intent(inout) :: al

        call MPI_REDUCE(data, al%buffer, al%dims(1), MPI_REAL_FP, &
                        MPI_SUM, 0, al%mpicomm_reduce, ierr)

        al%buffer = al%buffer/real(al%avg_count, fp)

        return
    end subroutine reduceToAvgLine

    subroutine recoverFromAvgLine(data, al)
        implicit none
        real(fp), dimension(:), intent(out) :: data
        type(avg_line_t), intent(in) :: al

        data = al%buffer*real(al%dims(2)*al%dims(3), fp)

        return
    end subroutine recoverFromAvgLine
    
    subroutine initAvgPlane(plane_norm, dims_box, avg_count, ipencil, pencil_coord, ap)
        implicit none
        integer, intent(in) :: plane_norm
        integer, dimension(3), intent(in) :: dims_box
        integer, intent(in) :: avg_count
        character(4), intent(in) :: ipencil
        integer, dimension(2), intent(in) :: pencil_coord
        type(avg_plane_t), intent(out) :: ap

        integer :: color, key, myrank_in_group
        integer :: istatus
        
        ! set the normal direction of the plane
        ap%norm_dir = plane_norm
        ap%avg_count = avg_count

        ! set the mpi communitors for group reduction
        select case(ipencil)
        case('xpen')
            ! x-pencil
            select case(plane_norm)
            case(1)
                ap%mpicomm_reduce = MPI_COMM_NULL
                ap%mpicomm_root = MPI_COMM_WORLD
            case(2)
                color = pencil_coord(2)
                key   = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(ap%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_root, ierr)
            case(3)
                color = pencil_coord(1)
                key   = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(ap%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_root, ierr)
            end select
        case('ypen')
            ! y-pencil
            select case(plane_norm)
            case(1)
                color = pencil_coord(2)
                key   = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(ap%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_root, ierr)
            case(2)
                ap%mpicomm_reduce = MPI_COMM_NULL
                ap%mpicomm_root = MPI_COMM_WORLD
            case(3)
                color = pencil_coord(1)
                key   = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(ap%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_root, ierr)
            end select
        case('zpen')
            ! z-pencil
            select case(plane_norm)
            case(1)
                color = pencil_coord(2)
                key   = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(ap%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_root, ierr)
            case(2)
                color = pencil_coord(1)
                key   = pencil_coord(2)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_reduce, ierr)
                call MPI_COMM_RANK(ap%mpicomm_reduce, myrank_in_group, ierr)
                color = MPI_UNDEFINED
                if (myrank_in_group == 0) color = 1
                key = pencil_coord(1)
                call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, key, ap%mpicomm_root, ierr)
            case(3)
                ap%mpicomm_reduce = MPI_COMM_NULL
                ap%mpicomm_root = MPI_COMM_WORLD
            end select
        end select

        ! determine whether the current process has the reduction results
        ap%have_results = .false.
        if (ap%mpicomm_root /= MPI_COMM_NULL) ap%have_results = .true.

        ! set the buffer for reduction
        select case(plane_norm)
        case(1)
            ap%dims(1) = dims_box(2)
            ap%dims(2) = dims_box(3)
            ap%dims(3) = dims_box(1)
        case(2)
            ap%dims(1) = dims_box(1)
            ap%dims(2) = dims_box(3)
            ap%dims(3) = dims_box(2)
        case(3)
            ap%dims(1) = dims_box(1)
            ap%dims(2) = dims_box(2)
            ap%dims(3) = dims_box(3)
        end select
        allocate(ap%buffer(ap%dims(1), ap%dims(2)), stat=istatus)
        if (istatus /= 0) call abort(102, "allocStat: Out of memory when allocating buffer for space average!")

        return
    end subroutine initAvgPlane

    subroutine freeAvgPlane(ap)
        implicit none
        type(avg_plane_t), intent(inout) :: ap

        if (allocated(ap%buffer)) deallocate(ap%buffer)

        return
    end subroutine freeAvgPlane

    subroutine reduceToAvgPlane(data, ap)
        implicit none
        real(fp), dimension(:,:), intent(in) :: data
        type(avg_plane_t), intent(inout) :: ap

        if (ap%mpicomm_reduce /= MPI_COMM_NULL) then
            call MPI_REDUCE(data, ap%buffer, ap%dims(1)*ap%dims(2), MPI_REAL_FP, &
                            MPI_SUM, 0, ap%mpicomm_reduce, ierr)
        else
            ap%buffer = data
        endif

        ap%buffer = ap%buffer/real(ap%avg_count, fp)

        return
    end subroutine reduceToAvgPlane

    subroutine recoverFromAvgPlane(data, ap)
        implicit none
        real(fp), dimension(:,:), intent(out) :: data
        type(avg_plane_t), intent(in) :: ap

        data = ap%buffer*real(ap%dims(3), fp)

        return
    end subroutine recoverFromAvgPlane
end module mod_spaceAvg