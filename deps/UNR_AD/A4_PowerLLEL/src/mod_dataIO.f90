module mod_dataIO
    use mod_type
    use mod_parameters, only: nx, ny, nz, p_row, p_col, nhalo, nhalo_one, &
                              nt_out_inst, fn_prefix_input_inst, fn_prefix_inst, &
                              nt_out_save, fn_prefix_save, overwrite_save, auto_cleanup, num_retained, &
                              nt_out_moni
    use mod_mpi,        only: st, sz, myrank, coord_pen
    use mod_monitor,    only: outputMonitor
    use mod_hdf5,       only: HID_T, openFile, createFile, closeFile, readAttribute, writeAttribute, &
                              read1d, write1d, read2d, write2d, read3d, write3d

    implicit none

    include 'mpif.h'

    ! make everything private unless declared public
    private

    ! public user routines
    public :: inputData, outputData
    public :: inputField, outputField, outputFieldWithHalo, &
              inputPlane, outputPlane, inputLine, outputLine

    interface inputField
        module procedure inputField_real4
        module procedure inputField_real8
    end interface inputField

    interface outputField
        module procedure outputField_real4
        module procedure outputField_real8
    end interface outputField

    interface outputFieldWithHalo
        module procedure outputFieldWithHalo_real4
        module procedure outputFieldWithHalo_real8
    end interface outputFieldWithHalo

    interface inputPlane
        module procedure inputPlane_real4
        module procedure inputPlane_real8
    end interface inputPlane

    interface outputPlane
        module procedure outputPlane_real4
        module procedure outputPlane_real8
    end interface outputPlane

    interface inputLine
        module procedure inputLine_real4
        module procedure inputLine_real8
    end interface inputLine

    interface outputLine
        module procedure outputLine_real4
        module procedure outputLine_real8
    end interface outputLine

contains
    subroutine inputData(nt_in, u, v, w)
        implicit none
        integer, intent(out) :: nt_in
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: u, v, w

        integer :: nt_last

        if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.inputData: Reading checkpoint fields ..."

        call inputField(trim(adjustl(fn_prefix_input_inst))//'_u.h5', nt_last, st, sz, nhalo, 'u', u)
        if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.inputData: Finish reading checkpoint field <u>!"
        nt_in = nt_last
        call inputField(trim(adjustl(fn_prefix_input_inst))//'_v.h5', nt_last, st, sz, nhalo, 'v', v)
        if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.inputData: Finish reading checkpoint field <v>!"
        call checkTimestamp(nt_in, nt_last, 'v')
        call inputField(trim(adjustl(fn_prefix_input_inst))//'_w.h5', nt_last, st, sz, nhalo, 'w', w)
        if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.inputData: Finish reading checkpoint field <w>!"
        call checkTimestamp(nt_in, nt_last, 'w')

        return
    contains
        subroutine checkTimestamp(base, actual, vartag)
            integer, intent(in) :: base, actual
            character(*), intent(in) :: vartag
            integer :: ierr
            if (actual /= base) then
                if (myrank == 0) write(*,'(A)') "PowerLLEL.ERROR.inputData: The timestamp of <"//vartag// &
                                                "> does not match with that of <u>!"
                call MPI_FINALIZE(ierr)
                stop
            endif
        end subroutine
    end subroutine inputData

    subroutine outputData(nt, u, v, w, p, u_crf)
        implicit none
        integer, intent(in) :: nt
        real(fp), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: u, v, w
        real(fp), dimension(0:,0:,0:), intent(in) :: p
        real(fp), intent(in) :: u_crf

        character(len=:), allocatable :: fn_prefix
        character(10) :: string_dump
        integer :: nt_cleanup, ierr
        integer :: ng(3)

        ng = (/nx, ny, nz/)

        if (mod(nt, nt_out_moni) == 0) then
            call outputMonitor(nt, u, v, w, p, u_crf)
        endif
        
        if (mod(nt, nt_out_inst) == 0) then
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Writing instantaneous fields ..."
            write(string_dump, "('_',I8.8,'_')") nt
            fn_prefix = fn_prefix_inst//string_dump
            call outputField(fn_prefix//'u.h5', nt, ng, st, sz, nhalo, 'u', u+u_crf)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing inst. field <u>!"
            call outputField(fn_prefix//'v.h5', nt, ng, st, sz, nhalo, 'v', v)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing inst. field <v>!"
            call outputField(fn_prefix//'w.h5', nt, ng, st, sz, nhalo, 'w', w)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing inst. field <w>!"
            call outputField(fn_prefix//'p.h5', nt, ng, st, sz, nhalo_one, 'p', p)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing inst. field <p>!"
        endif
        
        if (mod(nt, nt_out_save) == 0) then
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Writing checkpoint fields ..."
            if (overwrite_save) then
                string_dump = '_'
            else
                write(string_dump, "('_',I8.8,'_')") nt
            endif
            fn_prefix = fn_prefix_save//string_dump
            call outputField(fn_prefix//'u.h5', nt, ng, st, sz, nhalo, 'u', u+u_crf)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing checkpoint field <u>!"
            call outputField(fn_prefix//'v.h5', nt, ng, st, sz, nhalo, 'v', v)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing checkpoint field <v>!"
            call outputField(fn_prefix//'w.h5', nt, ng, st, sz, nhalo, 'w', w)
            if (myrank == 0) write(*,'(A)') "PowerLLEL.NOTE.outputData: Finish writing checkpoint field <w>!"
        endif

        if (myrank == 0) then
        if ((.not.overwrite_save) .and. auto_cleanup .and. mod(nt, nt_out_save) == 0) then
            nt_cleanup = nt-(num_retained+1)*nt_out_save
            write(string_dump, "('_',I8.8,'_')") nt_cleanup
            fn_prefix = fn_prefix_save//string_dump
            if (nt_cleanup > 0) then
                write(*,'(A)') "PowerLLEL.NOTE.outputData: Automatically cleanup of inst. checkpoint files ..."
                call system("rm "//fn_prefix//'u.h5')
                call system("rm "//fn_prefix//'v.h5')
                call system("rm "//fn_prefix//'w.h5')
            endif
        endif
        endif

        ! call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        return
    end subroutine outputData

    subroutine inputField_real4(fn, nt, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(out) :: nt
        integer, dimension(3), intent(in) :: st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: var
        type(stat_info_t), intent(out), optional :: stat_info

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp

        if (.not. present(stat_info)) then
            call openFile(fn, fh)
            call readAttribute(fh, 'nt', nt)
            wtime = MPI_WTIME()
            call read3d(fh, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
        else
            call openFile(fn, fh)
            call readAttribute(fh, 'nts',  stat_info%nts )
            call readAttribute(fh, 'nte',  stat_info%nte )
            call readAttribute(fh, 'nspl', stat_info%nspl)
            wtime = MPI_WTIME()
            call read3d(fh, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
            nt = stat_info%nte
        endif

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        call MPI_REDUCE(1.0d0*sz(1)*sz(2)*sz(3), tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        iospeed = 4.0*tmp/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL.NOTE.inputField: Finish reading inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine inputField_real4

    subroutine inputField_real8(fn, nt, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(out) :: nt
        integer, dimension(3), intent(in) :: st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(out) :: var
        type(stat_info_t), intent(out), optional :: stat_info

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp

        if (.not. present(stat_info)) then
            call openFile(fn, fh)
            call readAttribute(fh, 'nt', nt)
            wtime = MPI_WTIME()
            call read3d(fh, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
        else
            call openFile(fn, fh)
            call readAttribute(fh, 'nts',  stat_info%nts )
            call readAttribute(fh, 'nte',  stat_info%nte )
            call readAttribute(fh, 'nspl', stat_info%nspl)
            wtime = MPI_WTIME()
            call read3d(fh, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
            nt = stat_info%nte
        endif

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        call MPI_REDUCE(1.0d0*sz(1)*sz(2)*sz(3), tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        iospeed = 8.0*tmp/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL.NOTE.inputField: Finish reading inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine inputField_real8

    subroutine outputField_real4(fn, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(in) :: nt
        integer, dimension(3), intent(in) :: ng, st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: var
        type(stat_info_t), intent(in), optional :: stat_info

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp

        if (.not. present(stat_info)) then
            call createFile(fn, fh)
            call writeAttribute(fh, 'nt', nt)
            wtime = MPI_WTIME()
            call write3d(fh, ng, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
        else
            call createFile(fn, fh)
            call writeAttribute(fh, 'nts',  stat_info%nts )
            call writeAttribute(fh, 'nte',  stat_info%nte )
            call writeAttribute(fh, 'nspl', stat_info%nspl)
            wtime = MPI_WTIME()
            call write3d(fh, ng, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
        end if

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        iospeed = 4.0*ng(1)*ng(2)*ng(3)/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL.NOTE.outputField: Finish writing inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine outputField_real4

    subroutine outputField_real8(fn, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(in) :: nt
        integer, dimension(3), intent(in) :: ng, st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: var
        type(stat_info_t), intent(in), optional :: stat_info

        integer(HID_T) :: fh
        integer :: myrank, nranks, ierr
        double precision :: wtime, iospeed, tmp

        if (.not. present(stat_info)) then
            call createFile(fn, fh)
            call writeAttribute(fh, 'nt', nt)
            wtime = MPI_WTIME()
            call write3d(fh, ng, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
        else
            call createFile(fn, fh)
            call writeAttribute(fh, 'nts',  stat_info%nts )
            call writeAttribute(fh, 'nte',  stat_info%nte )
            call writeAttribute(fh, 'nspl', stat_info%nspl)
            wtime = MPI_WTIME()
            call write3d(fh, ng, st, sz, vartag, var(1:sz(1),1:sz(2),1:sz(3)))
            wtime = MPI_WTIME() - wtime
            call closeFile(fh)
        end if

#ifdef DEBUG
        call MPI_REDUCE(wtime, tmp, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        wtime = tmp/nranks
        iospeed = 8.0*ng(1)*ng(2)*ng(3)/1024.0/1024.0/wtime
        call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        if (myrank == 0) write(*,'(A,2(1PE10.3,A))') &
        "PowerLLEL.NOTE.outputField: Finish writing inst. field <"//vartag// &
        "> in ", wtime,"s, Avg speed = ", iospeed, " MB/s"
#endif
        return
    end subroutine outputField_real8

    subroutine outputFieldWithHalo_real4(fn, nt, ng, st, sz, nhalo, vartag, var)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(in) :: nt
        integer, dimension(3), intent(in) :: ng, st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: var

        integer(HID_T) :: fh
        integer :: nx_h, ny_h, nz_h
        integer, dimension(3) :: st_h, sz_h

        nx_h = ng(1)+(nhalo(1)+nhalo(2))
        ny_h = ng(2)+(nhalo(3)+nhalo(4))*p_row
        nz_h = ng(3)+(nhalo(5)+nhalo(6))*p_col
        sz_h(1) = sz(1) + nhalo(1)+nhalo(2)
        sz_h(2) = sz(2) + nhalo(3)+nhalo(4)
        sz_h(3) = sz(3) + nhalo(5)+nhalo(6)
        st_h(1) = 1
        st_h(2) = 1+coord_pen(1)*sz_h(2)
        st_h(3) = 1+coord_pen(2)*sz_h(3)

        call createFile(fn, fh)
        call writeAttribute(fh, 'nt', nt)
        call write3d(fh, (/nx_h,ny_h,nz_h/), st_h, sz_h, vartag, var)
        call closeFile(fh)

        return
    end subroutine outputFieldWithHalo_real4

    subroutine outputFieldWithHalo_real8(fn, nt, ng, st, sz, nhalo, vartag, var)
        implicit none
        character(*), intent(in) :: fn
        integer, intent(in) :: nt
        integer, dimension(3), intent(in) :: ng, st, sz
        integer, dimension(6), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):,1-nhalo(3):,1-nhalo(5):), intent(in) :: var

        integer(HID_T) :: fh
        integer :: nx_h, ny_h, nz_h
        integer, dimension(3) :: st_h, sz_h

        nx_h = ng(1)+(nhalo(1)+nhalo(2))
        ny_h = ng(2)+(nhalo(3)+nhalo(4))*p_row
        nz_h = ng(3)+(nhalo(5)+nhalo(6))*p_col
        sz_h(1) = sz(1) + nhalo(1)+nhalo(2)
        sz_h(2) = sz(2) + nhalo(3)+nhalo(4)
        sz_h(3) = sz(3) + nhalo(5)+nhalo(6)
        st_h(1) = 1
        st_h(2) = 1+coord_pen(1)*sz_h(2)
        st_h(3) = 1+coord_pen(2)*sz_h(3)

        call createFile(fn, fh)
        call writeAttribute(fh, 'nt', nt)
        call write3d(fh, (/nx_h,ny_h,nz_h/), st_h, sz_h, vartag, var)
        call closeFile(fh)

        return
    end subroutine outputFieldWithHalo_real8

    subroutine inputPlane_real4(fn_plane, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn_plane
        logical, intent(in)  :: is_involved
        integer, intent(out) :: nt
        integer, dimension(2), intent(in) :: ng, st, sz
        integer, dimension(4), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):,1-nhalo(3):), intent(out) :: var
        type(stat_info_t), intent(out), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call openFile(fn_plane, fh)
            call readAttribute(fh, 'nt', nt)
            call read2d(fh, is_involved, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
        else
            call openFile(fn_plane, fh)
            call readAttribute(fh, 'nts',  stat_info%nts )
            call readAttribute(fh, 'nte',  stat_info%nte )
            call readAttribute(fh, 'nspl', stat_info%nspl)
            call read2d(fh, is_involved, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
            nt = stat_info%nte
        end if

        return
    end subroutine inputPlane_real4

    subroutine inputPlane_real8(fn_plane, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn_plane
        logical, intent(in)  :: is_involved
        integer, intent(out) :: nt
        integer, dimension(2), intent(in) :: ng, st, sz
        integer, dimension(4), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):,1-nhalo(3):), intent(out) :: var
        type(stat_info_t), intent(out), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call openFile(fn_plane, fh)
            call readAttribute(fh, 'nt', nt)
            call read2d(fh, is_involved, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
        else
            call openFile(fn_plane, fh)
            call readAttribute(fh, 'nts',  stat_info%nts )
            call readAttribute(fh, 'nte',  stat_info%nte )
            call readAttribute(fh, 'nspl', stat_info%nspl)
            call read2d(fh, is_involved, st, sz, vartag, var(1:sz(1),1:sz(2)))
            call closeFile(fh)
            nt = stat_info%nte
        end if

        return
    end subroutine inputPlane_real8

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

    subroutine inputLine_real4(fn, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        logical, intent(in)  :: is_involved
        integer, intent(out) :: nt
        integer, intent(in)  :: ng, st, sz
        integer, dimension(2), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(4), dimension(1-nhalo(1):), intent(out) :: var
        type(stat_info_t), intent(out), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call openFile(fn, fh)
            call readAttribute(fh, 'nt', nt)
            call read1d(fh, is_involved, st, sz, vartag, var(1:sz))
            call closeFile(fh)
        else
            call openFile(fn, fh)
            call readAttribute(fh, 'nts',  stat_info%nts )
            call readAttribute(fh, 'nte',  stat_info%nte )
            call readAttribute(fh, 'nspl', stat_info%nspl)
            call read1d(fh, is_involved, st, sz, vartag, var(1:sz))
            call closeFile(fh)
            nt = stat_info%nte
        end if

        return
    end subroutine inputLine_real4

    subroutine inputLine_real8(fn, is_involved, nt, ng, st, sz, nhalo, vartag, var, stat_info)
        implicit none
        character(*), intent(in) :: fn
        logical, intent(in)  :: is_involved
        integer, intent(out) :: nt
        integer, intent(in)  :: ng, st, sz
        integer, dimension(2), intent(in) :: nhalo
        character(*), intent(in) :: vartag
        real(8), dimension(1-nhalo(1):), intent(out) :: var
        type(stat_info_t), intent(out), optional :: stat_info

        integer(HID_T) :: fh

        if (.not. present(stat_info)) then
            call openFile(fn, fh)
            call readAttribute(fh, 'nt', nt)
            call read1d(fh, is_involved, st, sz, vartag, var(1:sz))
            call closeFile(fh)
        else
            call openFile(fn, fh)
            call readAttribute(fh, 'nts',  stat_info%nts )
            call readAttribute(fh, 'nte',  stat_info%nte )
            call readAttribute(fh, 'nspl', stat_info%nspl)
            call read1d(fh, is_involved, st, sz, vartag, var(1:sz))
            call closeFile(fh)
            nt = stat_info%nte
        end if

        return
    end subroutine inputLine_real8

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

end module mod_dataIO