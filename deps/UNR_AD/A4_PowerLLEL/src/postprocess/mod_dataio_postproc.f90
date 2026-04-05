module mod_dataio_postproc
    use mod_type, only: stat_info_t
    use mod_hdf5
    use hdf5,     only: HID_T

    implicit none

    include 'mpif.h'
    
    ! make everything private unless declared public
    private

    ! public user routines
    public :: inputVelField, inputPreField, inputField, outputField, outputPlane, outputLine

    interface inputVelField
        module procedure inputVelField_real4
        module procedure inputVelField_real8
    end interface inputVelField

    interface inputPreField
        module procedure inputPreField_real4
        module procedure inputPreField_real8
    end interface inputPreField

    interface inputField
        module procedure inputField_real4
        module procedure inputField_real8
    end interface inputField

    interface outputField
        module procedure outputField_real4
        module procedure outputField_real8
    end interface outputField

    interface outputPlane
        module procedure outputPlane_real4
        module procedure outputPlane_real8
    end interface outputPlane

    interface outputLine
        module procedure outputLine_real4
        module procedure outputLine_real8
    end interface outputLine

contains
    subroutine inputVelField_real4(fn_prefix_input_inst, st, sz, nhalo, u, v, w)
        implicit none
        character(*), intent(in) :: fn_prefix_input_inst
        integer, dimension(3), intent(in) :: st, sz
        integer, dimension(6), intent(in) :: nhalo
        real(4), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: u, v, w

        integer :: nt_in, nt_last
        integer :: myrank, ierr

        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

        call inputField(fn_prefix_input_inst//'u.h5', nt_last, st, sz, 'u', u(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputVelField: Finish reading velocity field <u>!"
        nt_in = nt_last
        call inputField(fn_prefix_input_inst//'v.h5', nt_last, st, sz, 'v', v(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputVelField: Finish reading velocity field <v>!"
        call checkTimestamp(nt_in, nt_last, 'v')
        call inputField(fn_prefix_input_inst//'w.h5', nt_last, st, sz, 'w', w(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputVelField: Finish reading velocity field <w>!"
        call checkTimestamp(nt_in, nt_last, 'w')

        return
    contains
        subroutine checkTimestamp(base, actual, vartag)
            integer, intent(in) :: base, actual
            character(*), intent(in) :: vartag
            if (actual /= base) then
                if (myrank == 0) write(*,'(A)') &
                    "PowerLLEL_Postprocess.ERROR.inputVelField: The timestamp of <"//vartag//"> does not match with that of <u>!"
                call MPI_FINALIZE(ierr)
                stop
            endif
        end subroutine
    end subroutine inputVelField_real4

    subroutine inputVelField_real8(fn_prefix_input_inst, st, sz, nhalo, u, v, w)
        implicit none
        character(*), intent(in) :: fn_prefix_input_inst
        integer, dimension(3), intent(in) :: st, sz
        integer, dimension(6), intent(in) :: nhalo
        real(8), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: u, v, w

        integer :: nt_in, nt_last
        integer :: myrank, ierr

        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

        call inputField(fn_prefix_input_inst//'u.h5', nt_last, st, sz, 'u', u(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputVelField: Finish reading velocity field <u>!"
        nt_in = nt_last
        call inputField(fn_prefix_input_inst//'v.h5', nt_last, st, sz, 'v', v(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputVelField: Finish reading velocity field <v>!"
        call checkTimestamp(nt_in, nt_last, 'v')
        call inputField(fn_prefix_input_inst//'w.h5', nt_last, st, sz, 'w', w(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputVelField: Finish reading velocity field <w>!"
        call checkTimestamp(nt_in, nt_last, 'w')

        return
    contains
        subroutine checkTimestamp(base, actual, vartag)
            integer, intent(in) :: base, actual
            character(*), intent(in) :: vartag
            if (actual /= base) then
                if (myrank == 0) write(*,'(A)') &
                    "PowerLLEL_Postprocess.ERROR.inputVelField: The timestamp of <"//vartag//"> does not match with that of <u>!"
                call MPI_FINALIZE(ierr)
                stop
            endif
        end subroutine
    end subroutine inputVelField_real8

    subroutine inputPreField_real4(fn_prefix_input_inst, st, sz, nhalo, p)
        implicit none
        character(*), intent(in) :: fn_prefix_input_inst
        integer, dimension(3), intent(in) :: st, sz
        integer, dimension(6), intent(in) :: nhalo
        real(4), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: p
        
        integer :: nt_in
        integer :: myrank, ierr

        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

        call inputField(fn_prefix_input_inst//'p.h5', nt_in, st, sz, 'p', p(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputPreField: Finish reading pressure field <p>!"

    end subroutine inputPreField_real4

    subroutine inputPreField_real8(fn_prefix_input_inst, st, sz, nhalo, p)
        implicit none
        character(*), intent(in) :: fn_prefix_input_inst
        integer, dimension(3), intent(in) :: st, sz
        integer, dimension(6), intent(in) :: nhalo
        real(8), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: p
        
        integer :: nt_in
        integer :: myrank, ierr

        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

        call inputField(fn_prefix_input_inst//'p.h5', nt_in, st, sz, 'p', p(1:sz(1),1:sz(2),1:sz(3)))
        if (myrank == 0) write(*,'(A)') "PowerLLEL_Postprocess.NOTE.inputPreField: Finish reading pressure field <p>!"

    end subroutine inputPreField_real8

    subroutine inputField_real4(fn_field, nt, st, sz, vartag, var)
        implicit none
        character(*), intent(in) :: fn_field
        integer, intent(out) :: nt
        integer, dimension(3), intent(in) :: st, sz
        character(*), intent(in) :: vartag
        real(4), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(out) :: var

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp

        call openFile(fn_field, fh)
        call readAttribute(fh, 'nt', nt)
#ifdef DEBUG
        wtime = MPI_WTIME()
#endif
        call read3d(fh, st, sz, vartag, var)
#ifdef DEBUG
        wtime = MPI_WTIME() - wtime
#endif
        call closeFile(fh)

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        call MPI_REDUCE(1.0d0*sz(1)*sz(2)*sz(3), tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        iospeed = 4.0*tmp/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL_Postprocess.NOTE.inputField: Finish reading inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine inputField_real4

    subroutine inputField_real8(fn_field, nt, st, sz, vartag, var)
        implicit none
        character(*), intent(in) :: fn_field
        integer, intent(out) :: nt
        integer, dimension(3), intent(in) :: st, sz
        character(*), intent(in) :: vartag
        real(8), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(out) :: var

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp

        call openFile(fn_field, fh)
        call readAttribute(fh, 'nt', nt)
#ifdef DEBUG
        wtime = MPI_WTIME()
#endif
        call read3d(fh, st, sz, vartag, var)
#ifdef DEBUG
        wtime = MPI_WTIME() - wtime
#endif
        call closeFile(fh)

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        call MPI_REDUCE(1.0d0*sz(1)*sz(2)*sz(3), tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        iospeed = 8.0*tmp/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL_Postprocess.NOTE.inputField: Finish reading inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine inputField_real8

    subroutine outputField_real4(fn, nt, ng, st, sz, nhalo, vartag, var)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(in) :: nt
        integer, dimension(3), intent(in) :: ng, st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: var

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp
        
        call createFile(fn, fh)
        call writeAttribute(fh, 'nt', nt)
#ifdef DEBUG
        wtime = MPI_WTIME()
#endif
        call write3d(fh, ng, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
#ifdef DEBUG
        wtime = MPI_WTIME() - wtime
#endif
        call closeFile(fh)

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        iospeed = 4.0*ng(1)*ng(2)*ng(3)/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL_Postprocess.NOTE.outputField: Finish writing inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine outputField_real4

    subroutine outputField_real8(fn, nt, ng, st, sz, nhalo, vartag, var)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(in) :: nt
        integer, dimension(3), intent(in) :: ng, st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: var

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp
        
        call createFile(fn, fh)
        call writeAttribute(fh, 'nt', nt)
#ifdef DEBUG
        wtime = MPI_WTIME()
#endif
        call write3d(fh, ng, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
#ifdef DEBUG
        wtime = MPI_WTIME() - wtime
#endif
        call closeFile(fh)

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        iospeed = 8.0*ng(1)*ng(2)*ng(3)/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL_Postprocess.NOTE.outputField: Finish writing inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine outputField_real8

    subroutine outputPlane_real4(fn, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        logical, intent(in) :: is_involved
        integer, intent(in) :: nt
        integer, dimension(2), intent(in) :: ng, st, sz
        integer, dimension(4), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):,1-nhalo(3):), intent(in) :: var
        type(stat_info_t), intent(in), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call createFile(fn, fh)
            call writeAttribute(fh, 'nt', nt)
            call write2d(fh, is_involved, ng, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
        else
            call createFile(fn, fh)
            call writeAttribute(fh, 'nts',  stat_info%nts )
            call writeAttribute(fh, 'nte',  stat_info%nte )
            call writeAttribute(fh, 'nspl', stat_info%nspl)
            call write2d(fh, is_involved, ng, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
        end if

        return
    end subroutine outputPlane_real4

    subroutine outputPlane_real8(fn, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        logical, intent(in) :: is_involved
        integer, intent(in) :: nt
        integer, dimension(2), intent(in) :: ng, st, sz
        integer, dimension(4), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):,1-nhalo(3):), intent(in) :: var
        type(stat_info_t), intent(in), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call createFile(fn, fh)
            call writeAttribute(fh, 'nt', nt)
            call write2d(fh, is_involved, ng, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
        else
            call createFile(fn, fh)
            call writeAttribute(fh, 'nts',  stat_info%nts )
            call writeAttribute(fh, 'nte',  stat_info%nte )
            call writeAttribute(fh, 'nspl', stat_info%nspl)
            call write2d(fh, is_involved, ng, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
        end if

        return
    end subroutine outputPlane_real8

    subroutine outputLine_real4(fn, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        logical, intent(in) :: is_involved
        integer, intent(in) :: nt
        integer, intent(in) :: ng, st, sz
        integer, dimension(2), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):), intent(in) :: var
        type(stat_info_t), intent(in), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call createFile(fn, fh)
            call writeAttribute(fh, 'nt', nt)
            call write1d(fh, is_involved, ng, st, sz, vartag, var(1:sz))
            call closeFile(fh)
        else
            call createFile(fn, fh)
            call writeAttribute(fh, 'nts',  stat_info%nts )
            call writeAttribute(fh, 'nte',  stat_info%nte )
            call writeAttribute(fh, 'nspl', stat_info%nspl)
            call write1d(fh, is_involved, ng, st, sz, vartag, var(1:sz))
            call closeFile(fh)
        end if

        return
    end subroutine outputLine_real4

    subroutine outputLine_real8(fn, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        logical, intent(in) :: is_involved
        integer, intent(in) :: nt
        integer, intent(in) :: ng, st, sz
        integer, dimension(2), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):), intent(in) :: var
        type(stat_info_t), intent(in), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call createFile(fn, fh)
            call writeAttribute(fh, 'nt', nt)
            call write1d(fh, is_involved, ng, st, sz, vartag, var(1:sz))
            call closeFile(fh)
        else
            call createFile(fn, fh)
            call writeAttribute(fh, 'nts',  stat_info%nts )
            call writeAttribute(fh, 'nte',  stat_info%nte )
            call writeAttribute(fh, 'nspl', stat_info%nspl)
            call write1d(fh, is_involved, ng, st, sz, vartag, var(1:sz))
            call closeFile(fh)
        end if

        return
    end subroutine outputLine_real8

end module mod_dataio_postproc