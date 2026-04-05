#include <hdf5.h>
#include <mpi.h>
#include <stdint.h>
#include <string.h>

#include "bind.h"

#define xfer_size_limit 2147483647
#define xfer_size_batch 1073741824

#define div_ceiling(val, upper) ((val + upper - 1) / upper)

void initIO_() {
    H5open();
}

void freeIO_() {
    H5close();
}

void createFile_(const char* filename, hid_t *fileid) {
    // Setup file access property list with parallel I/O access
    hid_t plistid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plistid, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Create the file collectively
    *fileid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plistid);
    H5Pclose(plistid);
}

void openFile_(const char* filename, hid_t *fileid) {
    // Setup file access property list with parallel I/O access
    hid_t plistid = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plistid, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Open an existing file
    *fileid = H5Fopen(filename, H5F_ACC_RDONLY, plistid);
    H5Pclose(plistid);
}

void closeFile_(hid_t fileid) {
    H5Fclose(fileid);
}

void readAttribute_int_(hid_t fileid, const char* tag, int* var) {
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    H5Dread(dsetid, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plistid, var);
    H5Dclose(dsetid);
    H5Pclose(plistid);
}

void readAttribute_real4_(hid_t fileid, const char* tag, float* var) {
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    H5Dread(dsetid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plistid, var);
    H5Dclose(dsetid);
    H5Pclose(plistid);
}

void readAttribute_real8_(hid_t fileid, const char* tag, double* var) {
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    H5Dread(dsetid, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plistid, var);
    H5Dclose(dsetid);
    H5Pclose(plistid);
}

void writeAttribute_int_(hid_t fileid, const char* tag, const int* var, int myrank) {
    hsize_t dims_1d = 1;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (myrank == 0) {
        H5Dwrite(dsetid, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void writeAttribute_real4_(hid_t fileid, const char* tag, const float* var, int myrank) {
    hsize_t dims_1d = 1;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (myrank == 0) {
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void writeAttribute_real8_(hid_t fileid, const char* tag, const double* var, int myrank) {
    hsize_t dims_1d = 1;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (myrank == 0) {
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void read1d_real4_(hid_t fileid, _Bool is_involved, int st, int sz, const char* tag, float* var) {
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsetid);

    hsize_t dims_1d = sz;
    hid_t memspace = H5Screate_simple(1, &dims_1d, NULL);

    hsize_t offset = st - 1;
    hsize_t count = sz;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    if (is_involved) {
        H5Dread(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void read1d_real8_(hid_t fileid, _Bool is_involved, int st, int sz, const char* tag, double* var) {
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsetid);

    hsize_t dims_1d = sz;
    hid_t memspace = H5Screate_simple(1, &dims_1d, NULL);

    hsize_t offset = st - 1;
    hsize_t count = sz;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    if (is_involved) {
        H5Dread(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void read2d_real4_(hid_t fileid, _Bool is_involved, const int* st, const int* sz, const char* tag, float* var) {
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsetid);

    hsize_t dims_2d[2] = {sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(2, dims_2d, NULL);

    hsize_t offset[2] = {st[1] - 1, st[0] - 1};
    hsize_t count[2] = {sz[1], sz[0]};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    if (is_involved) {
        H5Dread(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void read2d_real8_(hid_t fileid, _Bool is_involved, const int* st, const int* sz, const char* tag, double* var) {
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsetid);

    hsize_t dims_2d[2] = {sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(2, dims_2d, NULL);

    hsize_t offset[2] = {st[1] - 1, st[0] - 1};
    hsize_t count[2] = {sz[1], sz[0]};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    if (is_involved) {
        H5Dread(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void read3d_real4_(hid_t fileid, const char* tag, const int* st, const int* sz, float* var) {
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsetid);

    int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(float);
    int nbatch = (xfer_size > xfer_size_limit ? (div_ceiling(xfer_size, xfer_size_batch)) : 1);
    // HDF5 follows C-style dimension, so we need to do proper convention
    hsize_t dims_3d_chunk[3] = {div_ceiling(sz[2], nbatch), sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(3, dims_3d_chunk, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    hsize_t count[3] = {1, 1, 1};
    hsize_t blocksize[3];
    memcpy(blocksize, dims_3d_chunk, sizeof(dims_3d_chunk));
    hsize_t offset[3];
    DO(ibatch, 1, nbatch) {
        offset[0] = st[2] - 1 + (ibatch - 1) * blocksize[2];
        offset[1] = st[1] - 1;
        offset[2] = st[0] - 1;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        // Array index range of the 3rd dimension
        int ks_xfer = (ibatch - 1) * blocksize[2];
        // Reset the memspace & filespace size of the last batch
        if (xfer_size > xfer_size_limit && ibatch == nbatch) {
            blocksize[2] = sz[2] - (nbatch - 1) * blocksize[2];
            H5Sset_extent_simple(memspace, 3, blocksize, blocksize);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        }
        // Read the dataset
        H5Dread(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var + ks_xfer * sz[0] * sz[1]);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void read3d_real8_(hid_t fileid, const char* tag, const int* st, const int* sz, double* var) {
    hid_t dsetid = H5Dopen(fileid, tag, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dsetid);

    int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(double);
    int nbatch = (xfer_size > xfer_size_limit ? (div_ceiling(xfer_size, xfer_size_batch)) : 1);
    // HDF5 follows C-style dimension, so we need to do proper convention
    hsize_t dims_3d_chunk[3] = {div_ceiling(sz[2], nbatch), sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(3, dims_3d_chunk, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    hsize_t count[3] = {1, 1, 1};
    hsize_t blocksize[3];
    memcpy(blocksize, dims_3d_chunk, sizeof(dims_3d_chunk));
    hsize_t offset[3];
    DO(ibatch, 1, nbatch) {
        offset[0] = st[2] - 1 + (ibatch - 1) * blocksize[2];
        offset[1] = st[1] - 1;
        offset[2] = st[0] - 1;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        // Array index range of the 3rd dimension
        int ks_xfer = (ibatch - 1) * blocksize[2];
        // Reset the memspace & filespace size of the last batch
        if (xfer_size > xfer_size_limit && ibatch == nbatch) {
            blocksize[2] = sz[2] - (nbatch - 1) * blocksize[2];
            H5Sset_extent_simple(memspace, 3, blocksize, blocksize);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        }
        // Read the dataset
        H5Dread(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var + ks_xfer * sz[0] * sz[1]);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write1d_singleproc_0_real4_(hid_t fileid, const char* tag, int sz, int myrank, const float* var) {
    hsize_t dims_1d = sz;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (myrank == 0) {
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void write1d_singleproc_0_real8_(hid_t fileid, const char* tag, int sz, int myrank, const double* var) {
    hsize_t dims_1d = sz;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (myrank == 0) {
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void write1d_singleproc_real4_(hid_t fileid, const char* tag, int sz, _Bool is_involved, const float* var) {
    hsize_t dims_1d = sz;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void write1d_singleproc_real8_(hid_t fileid, const char* tag, int sz, _Bool is_involved, const double* var) {
    hsize_t dims_1d = sz;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void write1d_multiproc_real4_(hid_t fileid, const char* tag, int sz_global, int st, int sz, _Bool is_involved, const float* var) {
    hsize_t dims_1d = sz_global;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims_1d = sz;
    hid_t memspace = H5Screate_simple(1, &dims_1d, NULL);
    hsize_t offset = st - 1;
    hsize_t count = sz;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write1d_multiproc_real8_(hid_t fileid, const char* tag, int sz_global, int st, int sz, _Bool is_involved, const double* var) {
    hsize_t dims_1d = sz_global;
    hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims_1d = sz;
    hid_t memspace = H5Screate_simple(1, &dims_1d, NULL);
    hsize_t offset = st - 1;
    hsize_t count = sz;
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write2d_singleproc_real4_(hid_t fileid, _Bool is_involved, const int* sz, const char* tag, const float* var) {
    hsize_t dims_2d[2] = {sz[1], sz[0]};
    hid_t filespace = H5Screate_simple(2, dims_2d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void write2d_singleproc_real8_(hid_t fileid, _Bool is_involved, const int* sz, const char* tag, const double* var) {
    hsize_t dims_2d[2] = {sz[1], sz[0]};
    hid_t filespace = H5Screate_simple(2, dims_2d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
}

void write2d_multiproc_real4_(hid_t fileid, _Bool is_involved, const int *sz_global, const int *st, const int *sz, const char* tag, const float* var) {
    hsize_t dims_2d[2] = {sz_global[1], sz_global[0]};
    hid_t filespace = H5Screate_simple(2, dims_2d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims_2d[0] = sz[1];
    dims_2d[1] = sz[0];
    hid_t memspace = H5Screate_simple(2, dims_2d, NULL);
    hsize_t offset[2] = {st[1] - 1, st[0] - 1};
    hsize_t count[2] = {sz[1], sz[0]};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write2d_multiproc_real8_(hid_t fileid, _Bool is_involved, const int *sz_global, const int *st, const int *sz, const char* tag, const double* var) {
    hsize_t dims_2d[2] = {sz_global[1], sz_global[0]};
    hid_t filespace = H5Screate_simple(2, dims_2d, NULL);
    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dims_2d[0] = sz[1];
    dims_2d[1] = sz[0];
    hid_t memspace = H5Screate_simple(2, dims_2d, NULL);
    hsize_t offset[2] = {st[1] - 1, st[0] - 1};
    hsize_t count[2] = {sz[1], sz[0]};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    
    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
    if (is_involved) {
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var);
    }
    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write3d_col_real4_(hid_t fileid, const int* sz_global, const int* st, const int* sz, const char* tag, float* var) {
    
    hsize_t dims_3d[3] = {sz_global[2], sz_global[1], sz_global[0]};
    hid_t filespace = H5Screate_simple(3, dims_3d, NULL);

    int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(float);
    int nbatch = (xfer_size > xfer_size_limit ? (div_ceiling(xfer_size, xfer_size_batch)) : 1);
    // HDF5 follows C-style dimension, so we need to do proper convention
    hsize_t dims_3d_chunk[3] = {div_ceiling(sz[2], nbatch), sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(3, dims_3d_chunk, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_COLLECTIVE);

    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[3] = {1, 1, 1};
    hsize_t blocksize[3];
    memcpy(blocksize, dims_3d_chunk, sizeof(dims_3d_chunk));
    hsize_t offset[3];
    DO(ibatch, 1, nbatch) {
        offset[0] = st[2] - 1 + (ibatch - 1) * blocksize[2];
        offset[1] = st[1] - 1;
        offset[2] = st[0] - 1;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        // Array index range of the 3rd dimension
        int ks_xfer = (ibatch - 1) * blocksize[2];
        // Reset the memspace & filespace size of the last batch
        if (xfer_size > xfer_size_limit && ibatch == nbatch) {
            blocksize[2] = sz[2] - (nbatch - 1) * blocksize[2];
            H5Sset_extent_simple(memspace, 3, blocksize, blocksize);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        }
        // Write the dataset
        H5Dwrite(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var + ks_xfer * sz[0] * sz[1]);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write3d_col_real8_(hid_t fileid, const int* sz_global, const int* st, const int* sz, const char* tag, double* var) {
    
    hsize_t dims_3d[3] = {sz_global[2], sz_global[1], sz_global[0]};
    hid_t filespace = H5Screate_simple(3, dims_3d, NULL);

    int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(double);
    int nbatch = (xfer_size > xfer_size_limit ? (div_ceiling(xfer_size, xfer_size_batch)) : 1);
    // HDF5 follows C-style dimension, so we need to do proper convention
    hsize_t dims_3d_chunk[3] = {div_ceiling(sz[2], nbatch), sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(3, dims_3d_chunk, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_COLLECTIVE);

    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[3] = {1, 1, 1};
    hsize_t blocksize[3];
    memcpy(blocksize, dims_3d_chunk, sizeof(dims_3d_chunk));
    hsize_t offset[3];
    DO(ibatch, 1, nbatch) {
        offset[0] = st[2] - 1 + (ibatch - 1) * blocksize[2];
        offset[1] = st[1] - 1;
        offset[2] = st[0] - 1;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        // Array index range of the 3rd dimension
        int ks_xfer = (ibatch - 1) * blocksize[2];
        // Reset the memspace & filespace size of the last batch
        if (xfer_size > xfer_size_limit && ibatch == nbatch) {
            blocksize[2] = sz[2] - (nbatch - 1) * blocksize[2];
            H5Sset_extent_simple(memspace, 3, blocksize, blocksize);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        }
        // Write the dataset
        H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var + ks_xfer * sz[0] * sz[1]);
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write3d_ind_real4_(hid_t fileid, _Bool is_involved, const int* sz_global, const int* st, const int* sz, const char* tag, float* var) {
    
    hsize_t dims_3d[3] = {sz_global[2], sz_global[1], sz_global[0]};
    hid_t filespace = H5Screate_simple(3, dims_3d, NULL);

    int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(float);
    int nbatch = (xfer_size > xfer_size_limit ? (div_ceiling(xfer_size, xfer_size_batch)) : 1);
    // HDF5 follows C-style dimension, so we need to do proper convention
    hsize_t dims_3d_chunk[3] = {div_ceiling(sz[2], nbatch), sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(3, dims_3d_chunk, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[3] = {1, 1, 1};
    hsize_t blocksize[3];
    memcpy(blocksize, dims_3d_chunk, sizeof(dims_3d_chunk));
    hsize_t offset[3];
    DO(ibatch, 1, nbatch) {
        offset[0] = st[2] - 1 + (ibatch - 1) * blocksize[2];
        offset[1] = st[1] - 1;
        offset[2] = st[0] - 1;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        // Array index range of the 3rd dimension
        int ks_xfer = (ibatch - 1) * blocksize[2];
        // Reset the memspace & filespace size of the last batch
        if (xfer_size > xfer_size_limit && ibatch == nbatch) {
            blocksize[2] = sz[2] - (nbatch - 1) * blocksize[2];
            H5Sset_extent_simple(memspace, 3, blocksize, blocksize);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        }
        if (is_involved) {
            H5Dwrite(dsetid, H5T_NATIVE_FLOAT, memspace, filespace, plistid, var + ks_xfer * sz[0] * sz[1]);
        }
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}

void write3d_ind_real8_(hid_t fileid, _Bool is_involved, const int* sz_global, const int* st, const int* sz, const char* tag, double* var) {
    
    hsize_t dims_3d[3] = {sz_global[2], sz_global[1], sz_global[0]};
    hid_t filespace = H5Screate_simple(3, dims_3d, NULL);

    int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(double);
    int nbatch = (xfer_size > xfer_size_limit ? (div_ceiling(xfer_size, xfer_size_batch)) : 1);
    // HDF5 follows C-style dimension, so we need to do proper convention
    hsize_t dims_3d_chunk[3] = {div_ceiling(sz[2], nbatch), sz[1], sz[0]};
    hid_t memspace = H5Screate_simple(3, dims_3d_chunk, NULL);

    hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);

    hid_t dsetid = H5Dcreate(fileid, tag, H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[3] = {1, 1, 1};
    hsize_t blocksize[3];
    memcpy(blocksize, dims_3d_chunk, sizeof(dims_3d_chunk));
    hsize_t offset[3];
    DO(ibatch, 1, nbatch) {
        offset[0] = st[2] - 1 + (ibatch - 1) * blocksize[2];
        offset[1] = st[1] - 1;
        offset[2] = st[0] - 1;
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        // Array index range of the 3rd dimension
        int ks_xfer = (ibatch - 1) * blocksize[2];
        // Reset the memspace & filespace size of the last batch
        if (xfer_size > xfer_size_limit && ibatch == nbatch) {
            blocksize[2] = sz[2] - (nbatch - 1) * blocksize[2];
            H5Sset_extent_simple(memspace, 3, blocksize, blocksize);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, blocksize);
        }
        if (is_involved) {
            H5Dwrite(dsetid, H5T_NATIVE_DOUBLE, memspace, filespace, plistid, var + ks_xfer * sz[0] * sz[1]);
        }
    }

    H5Pclose(plistid);
    H5Dclose(dsetid);
    H5Sclose(filespace);
    H5Sclose(memspace);
}
