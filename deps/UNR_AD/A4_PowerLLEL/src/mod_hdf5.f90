module mod_hdf5
#ifndef USE_C
    use hdf5
#else
    use iso_c_binding, only: c_null_char, c_bool
#endif
    
    implicit none
    
    include 'mpif.h'

    ! make everything private unless declared public
    private

#ifdef USE_C
    !!! --- from H5fortran_types.F90
    ! INTEGER, PARAMETER :: H5_FORTRAN_NUM_INTEGER_KINDS = 4
    ! INTEGER, PARAMETER :: HADDR_T = 8
    INTEGER, PARAMETER :: HSIZE_T = 8
    INTEGER, PARAMETER :: HSSIZE_T = 8
    ! INTEGER, PARAMETER :: OFF_T = 8
    INTEGER, PARAMETER :: SIZE_T = 8
    ! INTEGER, PARAMETER :: Fortran_INTEGER = 4
    ! INTEGER, DIMENSION(1:4), PARAMETER :: Fortran_INTEGER_AVAIL_KINDS = (/1,2,4,8/)
    ! INTEGER, PARAMETER :: Fortran_REAL_C_FLOAT = 4
    ! INTEGER, PARAMETER :: Fortran_REAL_C_DOUBLE = 8
    INTEGER, PARAMETER :: HID_T = 8
    ! INTEGER, PARAMETER :: Fortran_REAL = C_FLOAT
    ! INTEGER, PARAMETER :: Fortran_DOUBLE = C_DOUBLE
    ! INTEGER, PARAMETER :: H5R_DSET_REG_REF_BUF_SIZE_F = 12
    ! INTEGER(SIZE_T), PARAMETER :: OBJECT_NAMELEN_DEFAULT_F = -1
    !!! --- from H5fortran_types.F90
#endif

    integer, parameter :: xfer_size_limit = 2147483647
    integer, parameter :: xfer_size_batch = 1073741824
    ! Each process reads/writes a certain amount of data from/to a subset of a dataset at a time,
    ! if the total size (in byte) of the transfer data is larger than 2^31-1 Bytes (slightly
    ! smaller than 2 GB), then read/write operations will fail because of the limitation of
    ! MPI-IO. Hence, the too large data should be transfered in batches. "nbatch" controls
    ! the number of batches.

    integer, save :: comm, myrank

    integer :: hdferr
    integer :: ks_xfer, ke_xfer
    integer :: rank
    integer(HID_T) :: memspace      ! Dataspace identifier in memory
    integer(HID_T) :: filespace     ! Dataspace identifier in file
    ! integer(HID_T) :: fileid        ! File identifier
    integer(HID_T) :: dsetid        ! Dataset identifier
    integer(HID_T) :: plistid       ! Property list identifier
    integer(HSIZE_T), dimension(1) :: dims_1d
    integer(HSIZE_T), dimension(2) :: dims_2d
    integer(HSIZE_T), dimension(3) :: dims_3d
    integer(HSIZE_T), dimension(3) :: dims_3d_chunk
    integer(HSIZE_T), dimension(3) :: count
    integer(HSSIZE_T),dimension(3) :: offset
    integer(HSIZE_T), dimension(3) :: stride
    integer(HSIZE_T), dimension(3) :: blocksize

    ! public user routines
    public :: HID_T
    public :: initIO, freeIO, createFile, openFile, closeFile, &
              readAttribute, read1d, read2d, read3d, &
              writeAttribute, write1d, write2d, write3d

    interface readAttribute
        module procedure readAttribute_int
        module procedure readAttribute_real4
        module procedure readAttribute_real8
    end interface readAttribute

    interface writeAttribute
        module procedure writeAttribute_int
        module procedure writeAttribute_real4
        module procedure writeAttribute_real8
    end interface writeAttribute

    interface read1d
        module procedure read1d_real4
        module procedure read1d_real8
    end interface read1d

    interface write1d
        module procedure write1d_singleproc_0_real4
        module procedure write1d_singleproc_0_real8
        module procedure write1d_singleproc_real4
        module procedure write1d_singleproc_real8
        module procedure write1d_multiproc_real4
        module procedure write1d_multiproc_real8
    end interface write1d

    interface read2d
        module procedure read2d_real4
        module procedure read2d_real8
    end interface read2d

    interface write2d
        module procedure write2d_singleproc_real4
        module procedure write2d_singleproc_real8
        module procedure write2d_multiproc_real4
        module procedure write2d_multiproc_real8
    end interface write2d

    interface read3d
        module procedure read3d_real4
        module procedure read3d_real8
    end interface read3d

    interface write3d
        module procedure write3d_col_real4
        module procedure write3d_col_real8
        module procedure write3d_ind_real4
        module procedure write3d_ind_real8
    end interface write3d

    interface
        subroutine initIO_() bind(C, name='initIO_')
            use, intrinsic :: iso_c_binding
        end subroutine initIO_

        subroutine freeIO_() bind(C, name='freeIO_')
            use, intrinsic :: iso_c_binding
        end subroutine freeIO_

        ! omit passing argument of MPI_COMM_WORLD
        subroutine createFile_(filename, fileid) bind(C, name='createFile_')
            use iso_c_binding
            import :: HID_T
            character(kind=c_char), intent(in) :: filename(*)
            integer(HID_T), intent(out) :: fileid
        end subroutine createFile_

        subroutine openFile_(filename, fileid) bind(C, name='openFile_')
            use iso_c_binding
            import :: HID_T
            character(kind=c_char), intent(in) :: filename(*)
            integer(HID_T), intent(out) :: fileid
        end subroutine openFile_

        subroutine closeFile_(fileid) bind(C, name='closeFile_')
            import :: HID_T
            integer(HID_T), value :: fileid
        end subroutine closeFile_

        subroutine readAttribute_int_(fileid, tag, var) bind(C, name='readAttribute_int_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), intent(out) :: var
        end subroutine readAttribute_int_

        subroutine readAttribute_real4_(fileid, tag, var) bind(C, name='readAttribute_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), intent(out) :: var
        end subroutine readAttribute_real4_

        subroutine readAttribute_real8_(fileid, tag, var) bind(C, name='readAttribute_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), intent(out) :: var
        end subroutine readAttribute_real8_

        subroutine writeAttribute_int_(fileid, tag, var, myrank) bind(C, name='writeAttribute_int_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), intent(in) :: var
            integer(c_int), value :: myrank
        end subroutine writeAttribute_int_

        subroutine writeAttribute_real4_(fileid, tag, var, myrank) bind(C, name='writeAttribute_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), intent(in) :: var
            integer(c_int), value :: myrank
        end subroutine writeAttribute_real4_

        subroutine writeAttribute_real8_(fileid, tag, var, myrank) bind(C, name='writeAttribute_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), intent(in) :: var
            integer(c_int), value :: myrank
        end subroutine writeAttribute_real8_

        subroutine read1d_real4_(fileid, is_involved, st, sz, tag, var) bind(C, name='read1d_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), value :: st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), dimension(*), intent(out) :: var
        end subroutine read1d_real4_

        subroutine read1d_real8_(fileid, is_involved, st, sz, tag, var) bind(C, name='read1d_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), value :: st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), dimension(*), intent(out) :: var
        end subroutine read1d_real8_

        subroutine read2d_real4_(fileid, is_involved, st, sz, tag, var) bind(C, name='read2d_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(2), intent(in) :: st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), dimension(*), intent(out) :: var
        end subroutine read2d_real4_

        subroutine read2d_real8_(fileid, is_involved, st, sz, tag, var) bind(C, name='read2d_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(2), intent(in) :: st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), dimension(*), intent(out) :: var
        end subroutine read2d_real8_

        subroutine read3d_real4_(fileid, tag, st, sz, var) bind(C, name='read3d_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), dimension(3), intent(in) :: st, sz
            real(c_float), dimension(*), intent(out) :: var
        end subroutine read3d_real4_

        subroutine read3d_real8_(fileid, tag, st, sz, var) bind(C, name='read3d_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), dimension(3), intent(in) :: st, sz
            real(c_double), dimension(*), intent(out) :: var
        end subroutine read3d_real8_

        subroutine write1d_singleproc_0_real4_(fileid, tag, sz, myrank, var) bind(C, name='write1d_singleproc_0_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), value :: sz, myrank
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write1d_singleproc_0_real4_

        subroutine write1d_singleproc_0_real8_(fileid, tag, sz, myrank, var) bind(C, name='write1d_singleproc_0_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), value :: sz, myrank
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write1d_singleproc_0_real8_

        subroutine write1d_singleproc_real4_(fileid, tag, sz, is_involved, var) bind(C, name='write1d_singleproc_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), value :: sz
            logical(c_bool), value :: is_involved
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write1d_singleproc_real4_

        subroutine write1d_singleproc_real8_(fileid, tag, sz, is_involved, var) bind(C, name='write1d_singleproc_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), value :: sz
            logical(c_bool), value :: is_involved
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write1d_singleproc_real8_

        subroutine write1d_multiproc_real4_(fileid, tag, sz_global, st, sz, is_involved, var) bind(C, name='write1d_multiproc_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), value :: sz_global, st, sz
            logical(c_bool), value :: is_involved
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write1d_multiproc_real4_

        subroutine write1d_multiproc_real8_(fileid, tag, sz_global, st, sz, is_involved, var) bind(C, name='write1d_multiproc_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            character(kind=c_char), intent(in) :: tag(*)
            integer(c_int), value :: sz_global, st, sz
            logical(c_bool), value :: is_involved
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write1d_multiproc_real8_

        subroutine write2d_singleproc_real4_(fileid, is_involved, sz, tag, var) bind(C, name='write2d_singleproc_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(2), intent(in) :: sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write2d_singleproc_real4_

        subroutine write2d_singleproc_real8_(fileid, is_involved, sz, tag, var) bind(C, name='write2d_singleproc_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(2), intent(in) :: sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write2d_singleproc_real8_

        subroutine write2d_multiproc_real4_(fileid, is_involved, sz_global, st, sz, tag, var) bind(C, name='write2d_multiproc_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(2), intent(in) :: sz_global, st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write2d_multiproc_real4_

        subroutine write2d_multiproc_real8_(fileid, is_involved, sz_global, st, sz, tag, var) bind(C, name='write2d_multiproc_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(2), intent(in) :: sz_global, st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write2d_multiproc_real8_

        subroutine write3d_col_real4_(fileid, sz_global, st, sz, tag, var) bind(C, name='write3d_col_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            integer(c_int), dimension(3), intent(in) :: sz_global, st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write3d_col_real4_

        subroutine write3d_col_real8_(fileid, sz_global, st, sz, tag, var) bind(C, name='write3d_col_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            integer(c_int), dimension(3), intent(in) :: sz_global, st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write3d_col_real8_
        
        subroutine write3d_ind_real4_(fileid, is_involved, sz_global, st, sz, tag, var) bind(C, name='write3d_ind_real4_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(3), intent(in) :: sz_global, st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_float), dimension(*), intent(in) :: var
        end subroutine write3d_ind_real4_
        
        subroutine write3d_ind_real8_(fileid, is_involved, sz_global, st, sz, tag, var) bind(C, name='write3d_ind_real8_')
            use iso_c_binding
            import :: HID_T
            integer(HID_T), value :: fileid
            logical(c_bool), value :: is_involved
            integer(c_int), dimension(3), intent(in) :: sz_global, st, sz
            character(kind=c_char), intent(in) :: tag(*)
            real(c_double), dimension(*), intent(in) :: var
        end subroutine write3d_ind_real8_
    end interface

contains
    subroutine initIO(comm_in)
        implicit none
        integer, intent(in) :: comm_in

        integer :: ierr
        
        comm = comm_in
        call MPI_COMM_RANK(comm, myrank, ierr)

        ! Initialize FORTRAN predefined datatypes
#ifdef USE_C
        call initIO_()
#else
        call h5open_f(hdferr)
#endif
        return
    end subroutine initIO
    
    subroutine freeIO
        implicit none

        ! Close FORTRAN predefined datatypes
#ifdef USE_C
        call freeIO_()
#else
        call h5close_f(hdferr)
#endif
        return
    end subroutine freeIO

    subroutine createFile(filename, fileid)
        implicit none
        character(*), intent(in) :: filename
        integer(HID_T), intent(out) :: fileid
        
#ifdef USE_C
        call createFile_(filename//c_null_char, fileid)
#else
        ! Setup file access property list with parallel I/O access
        call h5pcreate_f(H5P_FILE_ACCESS_F, plistid, hdferr)
        call h5pset_fapl_mpio_f(plistid, comm, MPI_INFO_NULL, hdferr)

        ! Create the file collectively
        call h5fcreate_f(filename, H5F_ACC_TRUNC_F, fileid, hdferr, access_prp = plistid)
        call h5pclose_f(plistid, hdferr)
#endif
        return
    end subroutine createFile

    subroutine openFile(filename, fileid)
        implicit none
        character(*), intent(in) :: filename
        integer(HID_T), intent(out) :: fileid

        logical :: alive
        integer :: ierr
        
        inquire(file=filename, exist=alive)
        if (.not. alive) then
            if (myrank == 0) write(*,'(A)') "PowerLLEL.ERROR.openFile: "//filename//" doesn't exist!"
            call MPI_FINALIZE(ierr)
            stop
        endif
        
#ifdef USE_C
        call openFile_(filename//c_null_char, fileid)
#else
        ! Setup file access property list with parallel I/O access
        call h5pcreate_f(H5P_FILE_ACCESS_F, plistid, hdferr)
        call h5pset_fapl_mpio_f(plistid, comm, MPI_INFO_NULL, hdferr)

        ! Open an existing file
        call h5fopen_f(filename, H5F_ACC_RDONLY_F, fileid, hdferr, access_prp = plistid)
        call h5pclose_f(plistid, hdferr)
#endif
        return
    end subroutine openFile

    subroutine closeFile(fileid)
        implicit none
        integer(HID_T), intent(in) :: fileid

#ifdef USE_C
        call closeFile_(fileid)
#else
        ! Close the file
        call h5fclose_f(fileid, hdferr)
#endif   
        return
    end subroutine closeFile

    subroutine readAttribute_int(fileid, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        character(*), intent(in) :: tag
        integer, intent(out) :: var

#ifdef USE_C
        call readAttribute_int_(fileid, tag//c_null_char, var)
#else
        ! Create property list for independent dataset read
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! Open an existing dataset
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        ! Read the dataset independently.
        call h5dread_f(dsetid, H5T_NATIVE_INTEGER, var, dims_1d, hdferr, xfer_prp = plistid)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close the property list
        call h5pclose_f(plistid, hdferr)
#endif
        return
    end subroutine readAttribute_int

    subroutine readAttribute_real4(fileid, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        character(*), intent(in) :: tag
        real(4), intent(out) :: var
        
#ifdef USE_C
        call readAttribute_real4_(fileid, tag//c_null_char, var)
#else
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        call h5dread_f(dsetid, H5T_NATIVE_REAL, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5dclose_f(dsetid, hdferr)
        call h5pclose_f(plistid, hdferr)
#endif
        return
    end subroutine readAttribute_real4

    subroutine readAttribute_real8(fileid, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        character(*), intent(in) :: tag
        real(8), intent(out) :: var
        
#ifdef USE_C
        call readAttribute_real8_(fileid, tag//c_null_char, var)
#else
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        call h5dread_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5dclose_f(dsetid, hdferr)
        call h5pclose_f(plistid, hdferr)
#endif
        return
    end subroutine readAttribute_real8

    subroutine writeAttribute_int(fileid, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        character(*), intent(in) :: tag
        integer, intent(in) :: var

#ifdef USE_C
        call writeAttribute_int_(fileid, tag//c_null_char, var, myrank)
#else
        ! create the data space for the dataset
        rank = 1
        dims_1d = 1
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        ! create the dataset with default properties
        call h5dcreate_f(fileid, tag, H5T_NATIVE_INTEGER, filespace, dsetid, hdferr)
        ! Create property list for independent dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! write the dataset
        if (myrank == 0) call h5dwrite_f(dsetid, H5T_NATIVE_INTEGER, var, dims_1d, hdferr, xfer_prp = plistid)
        ! close the property list
        call h5pclose_f(plistid, hdferr)
        ! close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! terminate access to the dataspace
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine writeAttribute_int

    subroutine writeAttribute_real4(fileid, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        character(*), intent(in) :: tag
        real(4), intent(in) :: var

#ifdef USE_C
        call writeAttribute_real4_(fileid, tag//c_null_char, var, myrank)
#else
        rank = 1
        dims_1d = 1
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (myrank == 0) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine writeAttribute_real4

    subroutine writeAttribute_real8(fileid, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        character(*), intent(in) :: tag
        real(8), intent(in) :: var

#ifdef USE_C
        call writeAttribute_real8_(fileid, tag//c_null_char, var, myrank)
#else
        rank = 1
        dims_1d = 1
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (myrank == 0) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine writeAttribute_real8

    subroutine read1d_real4(fileid, is_involved, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, intent(in) :: st, sz
        character(*), intent(in) :: tag
        real(4), dimension(sz), intent(out) :: var

#ifdef USE_C
        call read1d_real4_(fileid, logical(is_involved, c_bool), st, sz, tag, var)
#else
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        call h5dget_space_f(dsetid, filespace, hdferr)
        rank = 1
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, memspace, hdferr)
        offset(1) = st-1
        count(1) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1), count(1), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)
        if (is_involved) call h5dread_f(dsetid, H5T_NATIVE_REAL, var, dims_1d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine read1d_real4

    subroutine read1d_real8(fileid, is_involved, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, intent(in) :: st, sz
        character(*), intent(in) :: tag
        real(8), dimension(sz), intent(out) :: var

#ifdef USE_C
        call read1d_real8_(fileid, logical(is_involved, c_bool), st, sz, tag, var)
#else
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        call h5dget_space_f(dsetid, filespace, hdferr)
        rank = 1
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, memspace, hdferr)
        offset(1) = st-1
        count(1) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1), count(1), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)
        if (is_involved) call h5dread_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_1d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif

        return
    end subroutine read1d_real8

    subroutine write1d_singleproc_0_real4(fileid, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        integer, intent(in) :: sz
        character(*), intent(in) :: tag
        real(4), dimension(sz) :: var

#ifdef USE_C
        call write1d_singleproc_0_real4_(fileid, tag//c_null_char, sz, myrank, var)
#else
        rank = 1
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (myrank == 0) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine write1d_singleproc_0_real4

    subroutine write1d_singleproc_0_real8(fileid, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        integer, intent(in) :: sz
        character(*), intent(in) :: tag
        real(8), dimension(sz) :: var

#ifdef USE_C
        call write1d_singleproc_0_real8_(fileid, tag//c_null_char, sz, myrank, var)
#else
        rank = 1
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (myrank == 0) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine write1d_singleproc_0_real8

    subroutine write1d_singleproc_real4(fileid, is_involved, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, intent(in) :: sz
        character(*), intent(in) :: tag
        real(4), dimension(sz) :: var

#ifdef USE_C
        call write1d_singleproc_real4_(fileid, tag//c_null_char, sz, logical(is_involved, c_bool), var)
#else
        rank = 1
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine write1d_singleproc_real4

    subroutine write1d_singleproc_real8(fileid, is_involved, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, intent(in) :: sz
        character(*), intent(in) :: tag
        real(8), dimension(sz) :: var

#ifdef USE_C
        call write1d_singleproc_real8_(fileid, tag//c_null_char, sz, logical(is_involved, c_bool), var)
#else
        rank = 1
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_1d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine write1d_singleproc_real8

    subroutine write1d_multiproc_real4(fileid, is_involved, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(4), dimension(sz) :: var

#ifdef USE_C
        call write1d_multiproc_real4_(fileid, tag//c_null_char, sz_global, st, sz, logical(is_involved, c_bool), var)
#else
        rank = 1
        dims_1d = sz_global
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, memspace, hdferr)
        offset(1) = st-1
        count(1) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1), count(1), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var, dims_1d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine write1d_multiproc_real4

    subroutine write1d_multiproc_real8(fileid, is_involved, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(8), dimension(sz) :: var

#ifdef USE_C
        call write1d_multiproc_real8_(fileid, tag//c_null_char, sz_global, st, sz, logical(is_involved, c_bool), var)
#else
        rank = 1
        dims_1d = sz_global
        call h5screate_simple_f(rank, dims_1d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)
        dims_1d = sz
        call h5screate_simple_f(rank, dims_1d, memspace, hdferr)
        offset(1) = st-1
        count(1) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1), count(1), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_1d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine write1d_multiproc_real8

    subroutine read2d_real4(fileid, is_involved, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(2), intent(in) :: st, sz
        character(*), intent(in) :: tag
        real(4), dimension(sz(1),sz(2)), intent(out) :: var

#ifdef USE_C
        call read2d_real4_(fileid, logical(is_involved, c_bool), st, sz, tag//c_null_char, var)
#else
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        call h5dget_space_f(dsetid, filespace, hdferr)
        rank = 2
        dims_2d = sz
        call h5screate_simple_f(rank, dims_2d, memspace, hdferr)
        offset(1:2) = st-1
        count(1:2) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1:2), count(1:2), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)
        if (is_involved) call h5dread_f(dsetid, H5T_NATIVE_REAL, var, dims_2d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine read2d_real4

    subroutine read2d_real8(fileid, is_involved, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(2), intent(in) :: st, sz
        character(*), intent(in) :: tag
        real(8), dimension(sz(1),sz(2)), intent(out) :: var

#ifdef USE_C
        call read2d_real8_(fileid, logical(is_involved, c_bool), st, sz, tag//c_null_char, var)
#else
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        call h5dget_space_f(dsetid, filespace, hdferr)
        rank = 2
        dims_2d = sz
        call h5screate_simple_f(rank, dims_2d, memspace, hdferr)
        offset(1:2) = st-1
        count(1:2) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1:2), count(1:2), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)
        if (is_involved) call h5dread_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_2d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine read2d_real8

    subroutine write2d_singleproc_real4(fileid, is_involved, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(2), intent(in) :: sz
        character(*), intent(in) :: tag
        real(4), dimension(sz(1),sz(2)), intent(in) :: var

#ifdef USE_C
        call write2d_singleproc_real4_(fileid, logical(is_involved, c_bool), sz, tag//c_null_char, var)
#else
        rank = 2
        dims_2d = sz
        call h5screate_simple_f(rank, dims_2d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var, dims_2d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine write2d_singleproc_real4

    subroutine write2d_singleproc_real8(fileid, is_involved, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(2), intent(in) :: sz
        character(*), intent(in) :: tag
        real(8), dimension(sz(1),sz(2)), intent(in) :: var

#ifdef USE_C
        call write2d_singleproc_real8_(fileid, logical(is_involved, c_bool), sz, tag//c_null_char, var)
#else
        rank = 2
        dims_2d = sz
        call h5screate_simple_f(rank, dims_2d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_2d, hdferr, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
#endif
        return
    end subroutine write2d_singleproc_real8

    subroutine write2d_multiproc_real4(fileid, is_involved, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(2), intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(4), dimension(sz(1),sz(2)), intent(in) :: var

#ifdef USE_C
        call write2d_multiproc_real4_(fileid, logical(is_involved, c_bool), sz_global, st, sz, tag//c_null_char, var)
#else
        rank = 2
        dims_2d = sz_global
        call h5screate_simple_f(rank, dims_2d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)
        dims_2d = sz
        call h5screate_simple_f(rank, dims_2d, memspace, hdferr)
        offset(1:2) = st-1
        count(1:2) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1:2), count(1:2), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var, dims_2d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif        
        return
    end subroutine write2d_multiproc_real4

    subroutine write2d_multiproc_real8(fileid, is_involved, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(2), intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(8), dimension(sz(1),sz(2)), intent(in) :: var

#ifdef USE_C
        call write2d_multiproc_real8_(fileid, logical(is_involved, c_bool), sz_global, st, sz, tag//c_null_char, var)
#else
        rank = 2
        dims_2d = sz_global
        call h5screate_simple_f(rank, dims_2d, filespace, hdferr)
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)
        dims_2d = sz
        call h5screate_simple_f(rank, dims_2d, memspace, hdferr)
        offset(1:2) = st-1
        count(1:2) = sz
        call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset(1:2), count(1:2), hdferr)
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var, dims_2d, hdferr, &
        file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        call h5pclose_f(plistid, hdferr)
        call h5dclose_f(dsetid, hdferr)
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine write2d_multiproc_real8

    subroutine read3d_real4(fileid, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        integer, dimension(3), intent(in) :: st, sz
        character(*), intent(in) :: tag
        real(4), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(out) :: var

#ifdef USE_C
        call read3d_real4_(fileid, tag//c_null_char, st, sz, var)
#else
        integer :: nbatch = 1
        integer :: ibatch
        integer(8) :: xfer_size

        ! Open an existing dataset
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        
        ! Get the data space for the whole dataset
        call h5dget_space_f(dsetid, filespace, hdferr)
        
        xfer_size = sz(1)*sz(2)*sz(3)*4
        if (xfer_size > xfer_size_limit) nbatch = ceiling(xfer_size/(1.0*xfer_size_batch))
        dims_3d_chunk = [sz(1), sz(2), ceiling(sz(3)/(1.0*nbatch))]
        rank = 3
        call h5screate_simple_f(rank, dims_3d_chunk, memspace, hdferr)

        ! Create property list for independent/collective dataset read
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)

        ! Select hyperslab in the file
        stride(1:3) = 1
        count (1:3) = 1
        blocksize(1) = dims_3d_chunk(1)
        blocksize(2) = dims_3d_chunk(2)
        blocksize(3) = dims_3d_chunk(3)
        do ibatch = 1, nbatch
            offset(1) = st(1)-1
            offset(2) = st(2)-1
            offset(3) = st(3)-1 + (ibatch-1)*blocksize(3)
            call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            ! Array index range of the 3rd dimension
            ks_xfer = 1 + (ibatch-1)*blocksize(3)
            ke_xfer = ibatch*blocksize(3)
            ! Reset the memspace & filespace size of the last batch
            if (xfer_size > xfer_size_limit .and. ibatch == nbatch) THEN
                ke_xfer = sz(3)
                blocksize(3) = sz(3) - (nbatch-1)*blocksize(3)
                call h5sset_extent_simple_f(memspace, rank, blocksize, blocksize, hdferr)
                call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            endif
            ! Read the dataset 
            call h5dread_f(dsetid, H5T_NATIVE_REAL, var(:,:,ks_xfer:ke_xfer), blocksize, hdferr, &
            file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        enddo

        ! Close the property list
        call h5pclose_f(plistid, hdferr)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close dataspaces
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine read3d_real4

    subroutine read3d_real8(fileid, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        integer, dimension(3), intent(in) :: st, sz
        character(*), intent(in) :: tag
        real(8), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(out) :: var

        integer :: nbatch = 1
        integer :: ibatch
        integer(8) :: xfer_size

#ifdef USE_C
        call read3d_real8_(fileid, tag//c_null_char, st, sz, var)
#else
        ! Open an existing dataset
        call h5dopen_f(fileid, tag, dsetid, hdferr)
        
        ! Get the data space for the whole dataset
        call h5dget_space_f(dsetid, filespace, hdferr)
        
        xfer_size = sz(1)*sz(2)*sz(3)*8
        if (xfer_size > xfer_size_limit) nbatch = ceiling(xfer_size/(1.0*xfer_size_batch))
        dims_3d_chunk = [sz(1), sz(2), ceiling(sz(3)/(1.0*nbatch))]
        rank = 3
        call h5screate_simple_f(rank, dims_3d_chunk, memspace, hdferr)

        ! Create property list for independent/collective dataset read
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)

        ! Select hyperslab in the file
        stride(1:3) = 1
        count (1:3) = 1
        blocksize(1) = dims_3d_chunk(1)
        blocksize(2) = dims_3d_chunk(2)
        blocksize(3) = dims_3d_chunk(3)
        do ibatch = 1, nbatch
            offset(1) = st(1)-1
            offset(2) = st(2)-1
            offset(3) = st(3)-1 + (ibatch-1)*blocksize(3)
            call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            ! Array index range of the 3rd dimension
            ks_xfer = 1 + (ibatch-1)*blocksize(3)
            ke_xfer = ibatch*blocksize(3)
            ! Reset the memspace & filespace size of the last batch
            if (xfer_size > xfer_size_limit .and. ibatch == nbatch) THEN
                ke_xfer = sz(3)
                blocksize(3) = sz(3) - (nbatch-1)*blocksize(3)
                call h5sset_extent_simple_f(memspace, rank, blocksize, blocksize, hdferr)
                call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            endif
            ! Read the dataset 
            call h5dread_f(dsetid, H5T_NATIVE_DOUBLE, var(:,:,ks_xfer:ke_xfer), blocksize, hdferr, &
            file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        enddo

        ! Close the property list
        call h5pclose_f(plistid, hdferr)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close dataspaces
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine read3d_real8

    subroutine write3d_col_real4(fileid, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        integer, dimension(3), intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(4), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(in) :: var

        integer :: nbatch = 1
        integer :: ibatch
        integer(8) :: xfer_size

#ifdef USE_C
        call write3d_col_real4_(fileid, sz_global, st, sz, tag//c_null_char, var)
#else
        ! Create the data space for the whole dataset
        rank = 3
        dims_3d = sz_global
        call h5screate_simple_f(rank, dims_3d, filespace, hdferr)
        
        xfer_size = sz(1)*sz(2)*sz(3)*4
        if (xfer_size > xfer_size_limit) nbatch = ceiling(xfer_size/(1.0*xfer_size_batch))
        dims_3d_chunk = [sz(1), sz(2), ceiling(sz(3)/(1.0*nbatch))]
        call h5screate_simple_f(rank, dims_3d_chunk, memspace, hdferr)

        ! Create datasets with default properties
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)

        ! Create property list for independent/collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)

        ! Select hyperslab in the file
        stride(1:3) = 1
        count (1:3) = 1
        blocksize(1) = dims_3d_chunk(1)
        blocksize(2) = dims_3d_chunk(2)
        blocksize(3) = dims_3d_chunk(3)
        do ibatch = 1, nbatch
            offset(1) = st(1)-1
            offset(2) = st(2)-1
            offset(3) = st(3)-1 + (ibatch-1)*blocksize(3)
            call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            ! Array index range of the 3rd dimension
            ks_xfer = 1 + (ibatch-1)*blocksize(3)
            ke_xfer = ibatch*blocksize(3)
            ! Reset the memspace & filespace size of the last batch
            if (xfer_size > xfer_size_limit .and. ibatch == nbatch) THEN
                ke_xfer = sz(3)
                blocksize(3) = sz(3) - (nbatch-1)*blocksize(3)
                call h5sset_extent_simple_f(memspace, rank, blocksize, blocksize, hdferr)
                call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            endif
            ! Write the dataset 
            call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var(:,:,ks_xfer:ke_xfer), blocksize, hdferr, &
            file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        enddo

        ! Close the property list
        call h5pclose_f(plistid, hdferr)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close dataspaces
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine write3d_col_real4

    subroutine write3d_col_real8(fileid, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        integer, dimension(3), intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(8), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(in) :: var

        integer :: nbatch = 1
        integer :: ibatch
        integer(8) :: xfer_size

#ifdef USE_C
        call write3d_col_real8_(fileid, sz_global, st, sz, tag//c_null_char, var)
#else
        ! Create the data space for the whole dataset
        rank = 3
        dims_3d = sz_global
        call h5screate_simple_f(rank, dims_3d, filespace, hdferr)
        
        xfer_size = sz(1)*sz(2)*sz(3)*8
        if (xfer_size > xfer_size_limit) nbatch = ceiling(xfer_size/(1.0*xfer_size_batch))
        dims_3d_chunk = [sz(1), sz(2), ceiling(sz(3)/(1.0*nbatch))]
        call h5screate_simple_f(rank, dims_3d_chunk, memspace, hdferr)

        ! Create datasets with default properties
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)

        ! Create property list for independent/collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)

        ! Select hyperslab in the file
        stride(1:3) = 1
        count (1:3) = 1
        blocksize(1) = dims_3d_chunk(1)
        blocksize(2) = dims_3d_chunk(2)
        blocksize(3) = dims_3d_chunk(3)
        do ibatch = 1, nbatch
            offset(1) = st(1)-1
            offset(2) = st(2)-1
            offset(3) = st(3)-1 + (ibatch-1)*blocksize(3)
            call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            ! Array index range of the 3rd dimension
            ks_xfer = 1 + (ibatch-1)*blocksize(3)
            ke_xfer = ibatch*blocksize(3)
            ! Reset the memspace & filespace size of the last batch
            if (xfer_size > xfer_size_limit .and. ibatch == nbatch) THEN
                ke_xfer = sz(3)
                blocksize(3) = sz(3) - (nbatch-1)*blocksize(3)
                call h5sset_extent_simple_f(memspace, rank, blocksize, blocksize, hdferr)
                call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            endif
            ! Write the dataset 
            call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var(:,:,ks_xfer:ke_xfer), blocksize, hdferr, &
            file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        enddo

        ! Close the property list
        call h5pclose_f(plistid, hdferr)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close dataspaces
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif

        return
    end subroutine write3d_col_real8

    subroutine write3d_ind_real4(fileid, is_involved, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(3), intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(4), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(in) :: var

        integer :: nbatch = 1
        integer :: ibatch
        integer(8) :: xfer_size

#ifdef USE_C
        call write3d_ind_real4_(fileid, logical(is_involved, c_bool), sz_global, st, sz, tag//c_null_char, var)
#else
        ! Create the data space for the whole dataset
        rank = 3
        dims_3d = sz_global
        call h5screate_simple_f(rank, dims_3d, filespace, hdferr)
        
        xfer_size = sz(1)*sz(2)*sz(3)*4
        if (xfer_size > xfer_size_limit) nbatch = ceiling(xfer_size/(1.0*xfer_size_batch))
        dims_3d_chunk = [sz(1), sz(2), ceiling(sz(3)/(1.0*nbatch))]
        call h5screate_simple_f(rank, dims_3d_chunk, memspace, hdferr)

        ! Create datasets with default properties
        call h5dcreate_f(fileid, tag, H5T_NATIVE_REAL, filespace, dsetid, hdferr)

        ! Create property list for independent/collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)

        ! Select hyperslab in the file
        stride(1:3) = 1
        count (1:3) = 1
        blocksize(1) = dims_3d_chunk(1)
        blocksize(2) = dims_3d_chunk(2)
        blocksize(3) = dims_3d_chunk(3)
        do ibatch = 1, nbatch
            offset(1) = st(1)-1
            offset(2) = st(2)-1
            offset(3) = st(3)-1 + (ibatch-1)*blocksize(3)
            call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            ! Array index range of the 3rd dimension
            ks_xfer = 1 + (ibatch-1)*blocksize(3)
            ke_xfer = ibatch*blocksize(3)
            ! Reset the memspace & filespace size of the last batch
            if (xfer_size > xfer_size_limit .and. ibatch == nbatch) THEN
                ke_xfer = sz(3)
                blocksize(3) = sz(3) - (nbatch-1)*blocksize(3)
                call h5sset_extent_simple_f(memspace, rank, blocksize, blocksize, hdferr)
                call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            endif
            ! Write the dataset 
            if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_REAL, var(:,:,ks_xfer:ke_xfer), blocksize, hdferr, &
            file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        enddo

        ! Close the property list
        call h5pclose_f(plistid, hdferr)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close dataspaces
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif
        return
    end subroutine write3d_ind_real4

    subroutine write3d_ind_real8(fileid, is_involved, sz_global, st, sz, tag, var)
        implicit none
        integer(HID_T), intent(in) :: fileid
        logical, intent(in) :: is_involved
        integer, dimension(3), intent(in) :: sz_global, st, sz
        character(*), intent(in) :: tag
        real(8), dimension(1:sz(1),1:sz(2),1:sz(3)), intent(in) :: var

        integer :: nbatch = 1
        integer :: ibatch
        integer(8) :: xfer_size

#ifdef USE_C
        call write3d_ind_real8_(fileid, logical(is_involved, c_bool), sz_global, st, sz, tag//c_null_char, var)
#else
        ! Create the data space for the whole dataset
        rank = 3
        dims_3d = sz_global
        call h5screate_simple_f(rank, dims_3d, filespace, hdferr)
        
        xfer_size = sz(1)*sz(2)*sz(3)*8
        if (xfer_size > xfer_size_limit) nbatch = ceiling(xfer_size/(1.0*xfer_size_batch))
        dims_3d_chunk = [sz(1), sz(2), ceiling(sz(3)/(1.0*nbatch))]
        call h5screate_simple_f(rank, dims_3d_chunk, memspace, hdferr)

        ! Create datasets with default properties
        call h5dcreate_f(fileid, tag, H5T_NATIVE_DOUBLE, filespace, dsetid, hdferr)

        ! Create property list for independent/collective dataset write
        call h5pcreate_f(H5P_DATASET_XFER_F, plistid, hdferr) 
        call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_INDEPENDENT_F, hdferr)
        ! call h5pset_dxpl_mpio_f(plistid, H5FD_MPIO_COLLECTIVE_F, hdferr)

        ! Select hyperslab in the file
        stride(1:3) = 1
        count (1:3) = 1
        blocksize(1) = dims_3d_chunk(1)
        blocksize(2) = dims_3d_chunk(2)
        blocksize(3) = dims_3d_chunk(3)
        do ibatch = 1, nbatch
            offset(1) = st(1)-1
            offset(2) = st(2)-1
            offset(3) = st(3)-1 + (ibatch-1)*blocksize(3)
            call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            ! Array index range of the 3rd dimension
            ks_xfer = 1 + (ibatch-1)*blocksize(3)
            ke_xfer = ibatch*blocksize(3)
            ! Reset the memspace & filespace size of the last batch
            if (xfer_size > xfer_size_limit .and. ibatch == nbatch) THEN
                ke_xfer = sz(3)
                blocksize(3) = sz(3) - (nbatch-1)*blocksize(3)
                call h5sset_extent_simple_f(memspace, rank, blocksize, blocksize, hdferr)
                call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, hdferr, stride, blocksize)
            endif
            ! Write the dataset 
            if (is_involved) call h5dwrite_f(dsetid, H5T_NATIVE_DOUBLE, var(:,:,ks_xfer:ke_xfer), blocksize, hdferr, &
            file_space_id = filespace, mem_space_id = memspace, xfer_prp = plistid)
        enddo

        ! Close the property list
        call h5pclose_f(plistid, hdferr)
        ! Close the dataset
        call h5dclose_f(dsetid, hdferr)
        ! Close dataspaces
        call h5sclose_f(filespace, hdferr)
        call h5sclose_f(memspace, hdferr)
#endif

        return
    end subroutine write3d_ind_real8
    
end module mod_hdf5