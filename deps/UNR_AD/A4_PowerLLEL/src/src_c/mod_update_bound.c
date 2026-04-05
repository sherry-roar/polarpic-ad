#include <stdbool.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>
#include "bind.h"
#if defined(USE_RDMA) && defined(NB_HALO)
#include <unr.h>
#endif
#ifdef GPTL
#include <stdio.h>
#include <gptl.h>
#endif

#ifdef NB_HALO

enum neighbor {
  SOU = 0,
  NOR = 1,
  BOT = 2,
  TOP = 3,
  SOU_BOT = 4,
  NOR_TOP = 5,
  SOU_TOP = 6,
  NOR_BOT = 7,
};

MPI_Datatype mpitype_nbhalo_vel[8];
int neighbor_nbhalo[8];

int modulo(const int a, const int p) { return a - p * (int)floor(1.0 * a / p); }

int get_mpi_rank_from_coords_2d_cart(const MPI_Comm comm, const int coords[2], const int dims[2]) {
    int rank;
    if( coords[0]<0 || coords[0]>=dims[0] || coords[1]<0 || coords[1]>=dims[1]) {
        rank = MPI_PROC_NULL;
    } else {
        MPI_Cart_rank(comm, coords, &rank);
    }

    return rank;
}

void get_neighbor_rank_2d_cart(const MPI_Comm comm) {
    int dims[2], coords[2], coords_nb[2], periods[2];
    int xcoords_nb[3], ycoords_nb[3];

    MPI_Cart_get(comm, 2, dims, periods, coords);
    
    xcoords_nb[0] = coords[0]-1;
    xcoords_nb[1] = coords[0];
    xcoords_nb[2] = coords[0]+1;
    if (periods[0]) {
        xcoords_nb[0] = modulo(xcoords_nb[0], dims[0]);
        xcoords_nb[2] = modulo(xcoords_nb[2], dims[0]);
    }
    
    ycoords_nb[0] = coords[1]-1;
    ycoords_nb[1] = coords[1];
    ycoords_nb[2] = coords[1]+1;
    if (periods[1]) {
        ycoords_nb[0] = modulo(ycoords_nb[0], dims[1]);
        ycoords_nb[2] = modulo(ycoords_nb[2], dims[1]);
    }

    coords_nb[0] = xcoords_nb[0];
    coords_nb[1] = ycoords_nb[1];
    neighbor_nbhalo[SOU] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

    coords_nb[0] = xcoords_nb[2];
    coords_nb[1] = ycoords_nb[1];
    neighbor_nbhalo[NOR] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

    coords_nb[0] = xcoords_nb[1];
    coords_nb[1] = ycoords_nb[0];
    neighbor_nbhalo[BOT] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);
    
    coords_nb[0] = xcoords_nb[1];
    coords_nb[1] = ycoords_nb[2];
    neighbor_nbhalo[TOP] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

    coords_nb[0] = xcoords_nb[0];
    coords_nb[1] = ycoords_nb[0];
    neighbor_nbhalo[SOU_BOT] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

    coords_nb[0] = xcoords_nb[0];
    coords_nb[1] = ycoords_nb[2];
    neighbor_nbhalo[SOU_TOP] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

    coords_nb[0] = xcoords_nb[2];
    coords_nb[1] = ycoords_nb[0];
    neighbor_nbhalo[NOR_BOT] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

    coords_nb[0] = xcoords_nb[2];
    coords_nb[1] = ycoords_nb[2];
    neighbor_nbhalo[NOR_TOP] = get_mpi_rank_from_coords_2d_cart(comm, coords_nb, dims);

}

void get_neighbor_rank_2d_cart_c(const MPI_Fint comm_f) {
    get_neighbor_rank_2d_cart(MPI_Comm_f2c(comm_f));
}

void create_nbhalo_mpitype(const int nhalo[6], const int sz[3], const MPI_Datatype oldtype) {
    int bcount, bsize, bstride;
    int isz = sz[0]+nhalo[0]+nhalo[1];
    int jsz = sz[1]+nhalo[2]+nhalo[3];
    
    // halo exchange in the south/north direction
    bcount  = sz[2];
    bsize   = isz * nhalo[2];
    bstride = isz * jsz;
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[SOU]);
    MPI_Type_commit(&mpitype_nbhalo_vel[SOU]);
    bsize   = isz * nhalo[3];
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[NOR]);
    MPI_Type_commit(&mpitype_nbhalo_vel[NOR]);
    
    // halo exchange in the bottom/top direction
    bcount  = 1;
    bsize   = isz * sz[1] * nhalo[4];
    bstride = 1;
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[BOT]);
    MPI_Type_commit(&mpitype_nbhalo_vel[BOT]);
    bsize   = isz * sz[1] * nhalo[5];
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[TOP]);
    MPI_Type_commit(&mpitype_nbhalo_vel[TOP]);

    // halo exchange in the south_bottom direction
    bcount  = nhalo[4];
    bsize   = isz * nhalo[2];
    bstride = isz * jsz;
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[SOU_BOT]);
    MPI_Type_commit(&mpitype_nbhalo_vel[SOU_BOT]);

    // halo exchange in the south_top direction
    bcount  = nhalo[5];
    bsize   = isz * nhalo[2];
    bstride = isz * jsz;
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[SOU_TOP]);
    MPI_Type_commit(&mpitype_nbhalo_vel[SOU_TOP]);

    // halo exchange in the north_bottom direction
    bcount  = nhalo[4];
    bsize   = isz * nhalo[3];
    bstride = isz * jsz;
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[NOR_BOT]);
    MPI_Type_commit(&mpitype_nbhalo_vel[NOR_BOT]);

    // halo exchange in the north_top direction
    bcount  = nhalo[5];
    bsize   = isz * nhalo[3];
    bstride = isz * jsz;
    MPI_Type_vector(bcount, bsize, bstride, oldtype, &mpitype_nbhalo_vel[NOR_TOP]);
    MPI_Type_commit(&mpitype_nbhalo_vel[NOR_TOP]);
}

void create_nbhalo_mpitype_c(const int nhalo[6], const int sz[3], const MPI_Fint oldtype_f) {
    create_nbhalo_mpitype(nhalo, sz, MPI_Type_f2c(oldtype_f));
}

void free_nbhalo_mpitype() {
    for (int i = 0; i < 8; i++)
        MPI_Type_free(&mpitype_nbhalo_vel[i]);
}

#define VAR(i,j,k) var[ ((k)-kst)*isz*jsz + ((j)-jst)*isz + ((i)-ist) ]

void update_halo_irecv(const int nhalo[6], const int sz[3], const int tag, double *var, MPI_Request irecv_req[8]) {
    int ist = 1 - nhalo[0];
    int jst = 1 - nhalo[2];
    int kst = 1 - nhalo[4];
    int isz = sz[0] + nhalo[0] + nhalo[1];
    int jsz = sz[1] + nhalo[2] + nhalo[3];

    // y1 recv y0 send
    MPI_Irecv(&VAR(ist, sz[1]+1, 1), 1, mpitype_nbhalo_vel[NOR], neighbor_nbhalo[NOR], tag, MPI_COMM_WORLD, &irecv_req[NOR]);
    // y0 recv y1 send
    MPI_Irecv(&VAR(ist, jst    , 1), 1, mpitype_nbhalo_vel[SOU], neighbor_nbhalo[SOU], tag, MPI_COMM_WORLD, &irecv_req[SOU]);
    // z1 recv z0 send
    MPI_Irecv(&VAR(ist, 1, sz[2]+1), 1, mpitype_nbhalo_vel[TOP], neighbor_nbhalo[TOP], tag, MPI_COMM_WORLD, &irecv_req[TOP]);
    // z0 recv z1 send
    MPI_Irecv(&VAR(ist, 1, kst    ), 1, mpitype_nbhalo_vel[BOT], neighbor_nbhalo[BOT], tag, MPI_COMM_WORLD, &irecv_req[BOT]);

    // north_top recv south_bottom send
    MPI_Irecv(&VAR(ist, sz[1]+1, sz[2]+1), 1, mpitype_nbhalo_vel[NOR_TOP], neighbor_nbhalo[NOR_TOP], tag, MPI_COMM_WORLD, &irecv_req[NOR_TOP]);
    // south_bottom recv north_top send
    MPI_Irecv(&VAR(ist,     jst,     kst), 1, mpitype_nbhalo_vel[SOU_BOT], neighbor_nbhalo[SOU_BOT], tag, MPI_COMM_WORLD, &irecv_req[SOU_BOT]);
    // north_bottom recv south_top send
    MPI_Irecv(&VAR(ist, sz[1]+1,     kst), 1, mpitype_nbhalo_vel[NOR_BOT], neighbor_nbhalo[NOR_BOT], tag, MPI_COMM_WORLD, &irecv_req[NOR_BOT]);
    // south_top recv north_bottom send
    MPI_Irecv(&VAR(ist,     jst, sz[2]+1), 1, mpitype_nbhalo_vel[SOU_TOP], neighbor_nbhalo[SOU_TOP], tag, MPI_COMM_WORLD, &irecv_req[SOU_TOP]);
}

void update_halo_isend(const int nhalo[6], const int sz[3], const int tag, double *var, MPI_Request isend_req[8]) {
    int ist = 1 - nhalo[0];
    int jst = 1 - nhalo[2];
    int kst = 1 - nhalo[4];
    int isz = sz[0] + nhalo[0] + nhalo[1];
    int jsz = sz[1] + nhalo[2] + nhalo[3];

    // y1 recv y0 send
    MPI_Isend(&VAR(ist,         1, 1), 1, mpitype_nbhalo_vel[NOR], neighbor_nbhalo[SOU], tag, MPI_COMM_WORLD, &isend_req[SOU]);
    // y0 recv y1 send
    MPI_Isend(&VAR(ist, sz[1]+jst, 1), 1, mpitype_nbhalo_vel[SOU], neighbor_nbhalo[NOR], tag, MPI_COMM_WORLD, &isend_req[NOR]);
    // z1 recv z0 send
    MPI_Isend(&VAR(ist, 1,         1), 1, mpitype_nbhalo_vel[TOP], neighbor_nbhalo[BOT], tag, MPI_COMM_WORLD, &isend_req[BOT]);
    // z0 recv z1 send
    MPI_Isend(&VAR(ist, 1, sz[2]+kst), 1, mpitype_nbhalo_vel[BOT], neighbor_nbhalo[TOP], tag, MPI_COMM_WORLD, &isend_req[TOP]);

    // north_top recv south_bottom send
    MPI_Isend(&VAR(ist,         1,         1), 1, mpitype_nbhalo_vel[NOR_TOP], neighbor_nbhalo[SOU_BOT], tag, MPI_COMM_WORLD, &isend_req[SOU_BOT]);
    // south_bottom recv north_top send
    MPI_Isend(&VAR(ist, sz[1]+jst, sz[2]+kst), 1, mpitype_nbhalo_vel[SOU_BOT], neighbor_nbhalo[NOR_TOP], tag, MPI_COMM_WORLD, &isend_req[NOR_TOP]);
    // north_bottom recv south_top send
    MPI_Isend(&VAR(ist,         1, sz[2]+kst), 1, mpitype_nbhalo_vel[NOR_BOT], neighbor_nbhalo[SOU_TOP], tag, MPI_COMM_WORLD, &isend_req[SOU_TOP]);
    // south_top recv north_bottom send
    MPI_Isend(&VAR(ist, sz[1]+jst,         1), 1, mpitype_nbhalo_vel[SOU_TOP], neighbor_nbhalo[NOR_BOT], tag, MPI_COMM_WORLD, &isend_req[NOR_BOT]);
}

void update_halo_waitall(MPI_Request isend_req[8], MPI_Request irecv_req[8]) {
    MPI_Status statuses[8];
    MPI_Waitall(8, isend_req, statuses);
    MPI_Waitall(8, irecv_req, statuses);
}

#ifdef USE_RDMA

volatile double *recv_signal[2][8];
volatile double recv_signal_send[2][8][8];
volatile double recv_signal_recv[2][8][8];

unr_plan_h halo_rdma_plan[2];
unr_sig_h halo_rdma_sig[2];
unr_blk_h *halo_send_blk, *halo_recv_blk, *halo_recv_blk_tmp;
int halo_rdma_num_send, halo_rdma_num_send_max;

#ifndef USE_NBHALOBUF

#define BLK_REG_OFFSET(i,j,k) ( ((k)-kst)*isz*jsz + ((j)-jst)*isz + ((i)-ist) )

void MPI_Irecv_revert(unr_mem_h mem_h, int offset, int count_1, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request, unr_sig_h sig) {

    if (count_1 != 1) {
        printf("MPI_Irecv_revert is used for count == 1\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (source == MPI_PROC_NULL) {
        *request = MPI_REQUEST_NULL;
        return;
    }

    int int_arr[3];
    MPI_Datatype oldtype;
    MPI_Type_get_contents(datatype, 3, 0, 1, int_arr, NULL, &oldtype);
    int count = int_arr[0];
    int blocklength = int_arr[1];
    int stride = int_arr[2];
    int oldtype_size;
    MPI_Type_size(oldtype, &oldtype_size);

    unr_blk_h *recv_blk = halo_recv_blk_tmp + halo_rdma_num_send;
    for (int i = 0; i < count; i++) {
        unr_blk_reg(mem_h, (size_t)(offset + i * stride) * oldtype_size, (size_t)blocklength * oldtype_size, UNR_NO_SIGNAL, sig, recv_blk + i);
    }

    MPI_Isend(recv_blk, count, MPI_UNR_BLK_H, source, tag, comm, request);
}

void MPI_Isend_revert(unr_mem_h mem_h, int offset, int count_1, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {

    if (count_1 != 1) {
        printf("MPI_Isend_revert is used for count == 1\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (dest == MPI_PROC_NULL) {
        *request = MPI_REQUEST_NULL;
        return;
    }

    int int_arr[3];
    MPI_Datatype oldtype;
    MPI_Type_get_contents(datatype, 3, 0, 1, int_arr, NULL, &oldtype);
    int count = int_arr[0];
    int blocklength = int_arr[1];
    int stride = int_arr[2];
    int oldtype_size;
    MPI_Type_size(oldtype, &oldtype_size);

    if (halo_rdma_num_send + count > halo_rdma_num_send_max) {
        printf("halo_rdma_num_send + count > halo_rdma_num_send_max\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    unr_blk_h *send_blk = halo_send_blk + halo_rdma_num_send;
    unr_blk_h *recv_blk = halo_recv_blk + halo_rdma_num_send;
    halo_rdma_num_send += count;
    for (int i = 0; i < count; i++) {
        unr_blk_reg(mem_h, (size_t)(offset + i * stride) * oldtype_size, (size_t)blocklength * oldtype_size, UNR_NO_SIGNAL, UNR_NO_SIGNAL, send_blk + i);
    }
    MPI_Irecv(recv_blk, count, MPI_UNR_BLK_H, dest, tag, comm, request);
}

void update_halo_irecv_revert(const int nhalo[6], const int sz[3], const int tag, unr_mem_h mem_h, MPI_Request irecv_req[8], unr_sig_h sig) {
    int ist = 1 - nhalo[0];
    int jst = 1 - nhalo[2];
    int kst = 1 - nhalo[4];
    int isz = sz[0] + nhalo[0] + nhalo[1];
    int jsz = sz[1] + nhalo[2] + nhalo[3];

    // y1 recv y0 send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist, sz[1]+1, 1), 1, mpitype_nbhalo_vel[NOR], neighbor_nbhalo[NOR], tag, MPI_COMM_WORLD, &irecv_req[NOR], sig);
    // y0 recv y1 send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist, jst    , 1), 1, mpitype_nbhalo_vel[SOU], neighbor_nbhalo[SOU], tag, MPI_COMM_WORLD, &irecv_req[SOU], sig);
    // z1 recv z0 send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist, 1, sz[2]+1), 1, mpitype_nbhalo_vel[TOP], neighbor_nbhalo[TOP], tag, MPI_COMM_WORLD, &irecv_req[TOP], sig);
    // z0 recv z1 send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist, 1, kst    ), 1, mpitype_nbhalo_vel[BOT], neighbor_nbhalo[BOT], tag, MPI_COMM_WORLD, &irecv_req[BOT], sig);

    // north_top recv south_bottom send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist, sz[1]+1, sz[2]+1), 1, mpitype_nbhalo_vel[NOR_TOP], neighbor_nbhalo[NOR_TOP], tag, MPI_COMM_WORLD, &irecv_req[NOR_TOP], sig);
    // south_bottom recv north_top send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist,     jst,     kst), 1, mpitype_nbhalo_vel[SOU_BOT], neighbor_nbhalo[SOU_BOT], tag, MPI_COMM_WORLD, &irecv_req[SOU_BOT], sig);
    // north_bottom recv south_top send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist, sz[1]+1,     kst), 1, mpitype_nbhalo_vel[NOR_BOT], neighbor_nbhalo[NOR_BOT], tag, MPI_COMM_WORLD, &irecv_req[NOR_BOT], sig);
    // south_top recv north_bottom send
    MPI_Irecv_revert(mem_h, BLK_REG_OFFSET(ist,     jst, sz[2]+1), 1, mpitype_nbhalo_vel[SOU_TOP], neighbor_nbhalo[SOU_TOP], tag, MPI_COMM_WORLD, &irecv_req[SOU_TOP], sig);
}

void update_halo_isend_revert(const int nhalo[6], const int sz[3], const int tag, unr_mem_h mem_h, MPI_Request isend_req[8]) {
    int ist = 1 - nhalo[0];
    int jst = 1 - nhalo[2];
    int kst = 1 - nhalo[4];
    int isz = sz[0] + nhalo[0] + nhalo[1];
    int jsz = sz[1] + nhalo[2] + nhalo[3];

    // y1 recv y0 send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist,         1, 1), 1, mpitype_nbhalo_vel[NOR], neighbor_nbhalo[SOU], tag, MPI_COMM_WORLD, &isend_req[SOU]);
    // y0 recv y1 send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist, sz[1]+jst, 1), 1, mpitype_nbhalo_vel[SOU], neighbor_nbhalo[NOR], tag, MPI_COMM_WORLD, &isend_req[NOR]);
    // z1 recv z0 send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist, 1,         1), 1, mpitype_nbhalo_vel[TOP], neighbor_nbhalo[BOT], tag, MPI_COMM_WORLD, &isend_req[BOT]);
    // z0 recv z1 send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist, 1, sz[2]+kst), 1, mpitype_nbhalo_vel[BOT], neighbor_nbhalo[TOP], tag, MPI_COMM_WORLD, &isend_req[TOP]);

    // north_top recv south_bottom send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist,         1,         1), 1, mpitype_nbhalo_vel[NOR_TOP], neighbor_nbhalo[SOU_BOT], tag, MPI_COMM_WORLD, &isend_req[SOU_BOT]);
    // south_bottom recv north_top send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist, sz[1]+jst, sz[2]+kst), 1, mpitype_nbhalo_vel[SOU_BOT], neighbor_nbhalo[NOR_TOP], tag, MPI_COMM_WORLD, &isend_req[NOR_TOP]);
    // north_bottom recv south_top send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist,         1, sz[2]+kst), 1, mpitype_nbhalo_vel[NOR_BOT], neighbor_nbhalo[SOU_TOP], tag, MPI_COMM_WORLD, &isend_req[SOU_TOP]);
    // south_top recv north_bottom send
    MPI_Isend_revert(mem_h, BLK_REG_OFFSET(ist, sz[1]+jst,         1), 1, mpitype_nbhalo_vel[SOU_TOP], neighbor_nbhalo[NOR_BOT], tag, MPI_COMM_WORLD, &isend_req[NOR_BOT]);
}

void init_rdma_halo_c(const int nhalo[6], const int sz[3], 
                      double *u,  double *v,  double *w, 
                      double *u1, double *v1, double *w1, 
                      const MPI_Fint comm_f) {
    unr_init();
    unr_mem_h uvw_mem_h[2][3];
    int ist = 1 - nhalo[0];
    int jst = 1 - nhalo[2];
    int kst = 1 - nhalo[4];
    int isz = sz[0] + nhalo[0] + nhalo[1];
    int jsz = sz[1] + nhalo[2] + nhalo[3];
    int ksz = sz[2] + nhalo[4] + nhalo[5];
    uint64_t mem_len = (uint64_t) sizeof(double) * isz * jsz * ksz;
    unr_mem_reg((void *)u1, mem_len, &uvw_mem_h[0][0]);
    unr_mem_reg((void *)v1, mem_len, &uvw_mem_h[0][1]);
    unr_mem_reg((void *)w1, mem_len, &uvw_mem_h[0][2]);
    unr_mem_reg((void *)u,  mem_len, &uvw_mem_h[1][0]);
    unr_mem_reg((void *)v,  mem_len, &uvw_mem_h[1][1]);
    unr_mem_reg((void *)w,  mem_len, &uvw_mem_h[1][2]);
    unr_mem_reg_sync();
    halo_rdma_num_send_max = (
        ((neighbor_nbhalo[SOU] != MPI_PROC_NULL) ? sz[2] : 0) + 
        ((neighbor_nbhalo[NOR] != MPI_PROC_NULL) ? sz[2] : 0) + 
        ((neighbor_nbhalo[BOT] != MPI_PROC_NULL) ? nhalo[5] : 0) + 
        ((neighbor_nbhalo[TOP] != MPI_PROC_NULL) ? nhalo[4] : 0) + 
        ((neighbor_nbhalo[SOU_BOT] != MPI_PROC_NULL) ? nhalo[5] : 0) + 
        ((neighbor_nbhalo[NOR_TOP] != MPI_PROC_NULL) ? nhalo[4] : 0) + 
        ((neighbor_nbhalo[SOU_TOP] != MPI_PROC_NULL) ? nhalo[4] : 0) + 
        ((neighbor_nbhalo[NOR_BOT] != MPI_PROC_NULL) ? nhalo[5] : 0)
    ) * 3;
    halo_send_blk = (unr_blk_h*)malloc(sizeof(unr_blk_h) * halo_rdma_num_send_max);
    halo_recv_blk = (unr_blk_h*)malloc(sizeof(unr_blk_h) * halo_rdma_num_send_max);
    halo_recv_blk_tmp = (unr_blk_h*)malloc(sizeof(unr_blk_h) * halo_rdma_num_send_max);
    for (int rk_num = 0; rk_num < 2; rk_num++) {
        halo_rdma_num_send = 0;
        unr_sig_create(halo_rdma_sig + rk_num, halo_rdma_num_send_max);
        for (int uvw_i = 0; uvw_i < 3; uvw_i++) {
            MPI_Request mpi_recv_req[8], mpi_send_req[8];
            update_halo_irecv_revert(nhalo, sz, uvw_i, uvw_mem_h[rk_num][uvw_i], mpi_recv_req, halo_rdma_sig[rk_num]);
            update_halo_isend_revert(nhalo, sz, uvw_i, uvw_mem_h[rk_num][uvw_i], mpi_send_req);
            MPI_Waitall(8, mpi_send_req, MPI_STATUSES_IGNORE);
            MPI_Waitall(8, mpi_recv_req, MPI_STATUSES_IGNORE);
        }
        if (halo_rdma_num_send != halo_rdma_num_send_max) {
            printf("halo_rdma_num_send != halo_rdma_num_send_max @ %d\n", __LINE__);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        unr_blk_send_batch_plan(halo_rdma_num_send, halo_send_blk, NULL, halo_recv_blk, NULL, halo_rdma_plan + rk_num);
    }
    free(halo_send_blk);
    free(halo_recv_blk);
    free(halo_recv_blk_tmp);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

#else

void init_rdma_halo_c(const int nhalo[6], const int sz[3], 
                      const int halobuf_length[8], const int halobuf_offset[8], 
                      double *halobuf_send, double *halobuf_recv, 
                      double *halobuf_send_aux, double *halobuf_recv_aux, 
                      const MPI_Fint comm_f) {
    unr_init();
    unr_mem_h uvw_mem_h[2][2];
    unr_blk_h uvw_blk_loc[2][2], uvw_blk_rmt[8][2][2];
    uint64_t mem_len = (uint64_t) sizeof(double) * (halobuf_offset[7] + halobuf_length[7]) * 3;

    for (int idir = 0; idir < 8; idir++) {
        if (neighbor_nbhalo[idir] != MPI_PROC_NULL) {
            halo_rdma_num_send_max += 3;
        }
    }
    unr_sig_create(&(halo_rdma_sig[0]), halo_rdma_num_send_max);
    unr_sig_create(&(halo_rdma_sig[1]), halo_rdma_num_send_max);

    unr_mem_reg((void *)halobuf_send, mem_len, &uvw_mem_h[0][0]);
    unr_mem_reg((void *)halobuf_recv, mem_len, &uvw_mem_h[0][1]);
    unr_mem_reg((void *)halobuf_send_aux, mem_len, &uvw_mem_h[1][0]);
    unr_mem_reg((void *)halobuf_recv_aux, mem_len, &uvw_mem_h[1][1]);
    unr_mem_reg_sync();
    unr_blk_reg(uvw_mem_h[0][0], 0, mem_len, UNR_NO_SIGNAL, UNR_NO_SIGNAL, &(uvw_blk_loc[0][0]));
    unr_blk_reg(uvw_mem_h[0][1], 0, mem_len, UNR_NO_SIGNAL, halo_rdma_sig[0], &(uvw_blk_loc[0][1]));
    unr_blk_reg(uvw_mem_h[1][0], 0, mem_len, UNR_NO_SIGNAL, UNR_NO_SIGNAL, &(uvw_blk_loc[1][0]));
    unr_blk_reg(uvw_mem_h[1][1], 0, mem_len, UNR_NO_SIGNAL, halo_rdma_sig[1], &(uvw_blk_loc[1][1]));

    MPI_Request mpi_recv_req[8], mpi_send_req[8];
    halo_rdma_num_send_max = 0;
    for (int idir = 0; idir < 8; idir++) {
        if (neighbor_nbhalo[idir] != MPI_PROC_NULL) {
            halo_rdma_num_send_max += 3;
            MPI_Isend(&(uvw_blk_loc[0][0]), 4, MPI_UNR_BLK_H, neighbor_nbhalo[idir], 0, MPI_COMM_WORLD, mpi_send_req + idir);
            MPI_Irecv(&(uvw_blk_rmt[idir][0][0]), 4, MPI_UNR_BLK_H, neighbor_nbhalo[idir], 0, MPI_COMM_WORLD, mpi_recv_req + idir);
        }
        else {
            mpi_send_req[idir] = MPI_REQUEST_NULL;
            mpi_recv_req[idir] = MPI_REQUEST_NULL;
        }
    }
    MPI_Waitall(8, mpi_send_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(8, mpi_recv_req, MPI_STATUSES_IGNORE);

    unr_blk_h *loc_blk_h_arr = (unr_blk_h*)malloc(sizeof(unr_blk_h) * halo_rdma_num_send_max);
    unr_blk_h *rmt_blk_h_arr = (unr_blk_h*)malloc(sizeof(unr_blk_h) * halo_rdma_num_send_max);
    size_t *offset = (size_t*)malloc(sizeof(size_t) * halo_rdma_num_send_max);
    size_t *size = (size_t*)malloc(sizeof(size_t) * halo_rdma_num_send_max);
    for (int rk_num = 0; rk_num < 2; rk_num++) {
        halo_rdma_num_send = 0;
        int halobuf_len_sum = halobuf_offset[7] + halobuf_length[7];
        for (int uvw_i = 0; uvw_i < 3; uvw_i++) {
            for (int idir = 0; idir < 8; idir++) {
                if (neighbor_nbhalo[idir] != MPI_PROC_NULL) {
                    loc_blk_h_arr[halo_rdma_num_send] = uvw_blk_loc[rk_num][0];
                    rmt_blk_h_arr[halo_rdma_num_send] = uvw_blk_rmt[idir][rk_num][1];
                    offset[halo_rdma_num_send] = sizeof(double) * (halobuf_offset[idir] + uvw_i * halobuf_len_sum);
                    size[halo_rdma_num_send] = sizeof(double) * halobuf_length[idir];
                    halo_rdma_num_send++;
                    if (halo_rdma_num_send > halo_rdma_num_send_max) {
                        printf("halo_rdma_num_send > halo_rdma_num_send_max\n");
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                }
            }
        }
        if (halo_rdma_num_send != halo_rdma_num_send_max) {
            printf("halo_rdma_num_send != halo_rdma_num_send_max @ %d\n", __LINE__);
            printf("halo_rdma_num_send = %d, halo_rdma_num_send_max = %d\n", halo_rdma_num_send, halo_rdma_num_send_max);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        unr_blk_part_send_batch_plan(halo_rdma_num_send_max, 
            loc_blk_h_arr, NULL, offset, size,
            rmt_blk_h_arr, NULL, offset, 
            &(halo_rdma_plan[rk_num])
        );
    }
}

#endif // #ifndef USE_NBHALOBUF

void update_halo_rdma_send(int rk_num) {
    unr_plan_start(halo_rdma_plan[rk_num]);
}

void update_halo_rdma_wait(int rk_num) {
    unr_sig_wait(halo_rdma_sig[rk_num]);
    unr_sig_reset(halo_rdma_sig[rk_num]);
}

#endif // #ifdef USE_RDMA

#endif // #ifdef NB_HALO

void impose_periodic_bc(int ibound, int idir, const int *nhalo, const int *sz, double *var)
{

    BIND1(nhalo)
    BIND1(sz)
    BIND3_EXT(var, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                   I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                   1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))

    const int ist = 1 - nhalo[0], ien = sz[0] + nhalo[1];
    const int jst = 1 - nhalo[2], jen = sz[1] + nhalo[3];
    const int kst = 1 - nhalo[4], ken = sz[2] + nhalo[4];

    if (idir == 1)
    {
        if (ibound == 0)
        {
            int n  = I1(sz, 1);
            int nh = I1(nhalo, 1);
#ifdef USE_OMP_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2)
#else
            #pragma omp parallel for
#endif
            DO(k, kst, ken)
                DO(j, jst, jen)
                    DO(i, 1, nh)
                        I3(var, 1-i, j, k) = I3(var, n+1-i, j, k);
        }
        else if (ibound == 1)
        {
            int n  = I1(sz, 1);
            int nh = I1(nhalo, 2);
#ifdef USE_OMP_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2)
#else
            #pragma omp parallel for
#endif
            DO(k, kst, ken)
                DO(j, jst, jen)
                    DO(i, 1, nh)
                        I3(var, n+i, j, k) = I3(var, i, j, k);
        }
    }
}

void impose_no_slip_bc(int ibound, _Bool centered, const int *nhalo, const int *sz, double *var, const double vel_crf)
{

    BIND1(nhalo)
    BIND1(sz)
    BIND3_EXT(var, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                   I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                   1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))

    const int ist = 1 - nhalo[0], ien = sz[0] + nhalo[1];
    const int jst = 1 - nhalo[2], jen = sz[1] + nhalo[3];
    const int kst = 1 - nhalo[4], ken = sz[2] + nhalo[4];
    
    double bcvalue = 0.0 - vel_crf;

    if (ibound == 0)
    {
        if (centered)
        {
#ifdef USE_OMP_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2)
#else
            #pragma omp parallel for
#endif
            DO(j, jst, jen)
                DO(i, ist, ien)
                    I3(var, i, j, 0) = 2.0*bcvalue - I3(var, i, j, 1);
        }
        else
        {
#ifdef USE_OMP_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2)
#else
            #pragma omp parallel for
#endif
            DO(j, jst, jen)
                DO(i, ist, ien)
                    I3(var, i, j, 0) = bcvalue;
        }
    }
    else if (ibound == 1)
    {
        int n = I1(sz, 3);
        if (centered)
        {
#ifdef USE_OMP_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2)
#else
            #pragma omp parallel for
#endif
            DO(j, jst, jen)
                DO(i, ist, ien)
                    I3(var, i, j, n+1) = 2.0*bcvalue - I3(var, i, j, n);
        }
        else
        {
#ifdef USE_OMP_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2)
#else
            #pragma omp parallel for
#endif
            DO(j, jst, jen) {
                DO(i, ist, ien) {
                    I3(var, i, j, n  ) = bcvalue;
                    I3(var, i, j, n+1) = bcvalue;
                }
            }
        }
    }
}

void impose_zero_grad_bc(int ibound, const int *nhalo, const int *sz, double *var)
{

    BIND1(nhalo)
    BIND1(sz)
    BIND3_EXT(var, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                   I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                   1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))

    const int ist = 1 - nhalo[0], ien = sz[0] + nhalo[1];
    const int jst = 1 - nhalo[2], jen = sz[1] + nhalo[3];
    const int kst = 1 - nhalo[4], ken = sz[2] + nhalo[4];

    if (ibound == 0)
    {
#ifdef USE_OMP_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        DO(j, jst, jen)
            DO(i, ist, ien)
                I3(var, i, j, 0) = I3(var, i, j, 1);
    }
    else if (ibound == 1)
    {
        int n = I1(sz, 3);
#ifdef USE_OMP_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        DO(j, jst, jen)
            DO(i, ist, ien)
                I3(var, i, j, n+1) = I3(var, i, j, n);
    }
}

void update_halo(const MPI_Comm comm, const MPI_Datatype halotype[6], const int *neighbor, 
                 const int nhalo[6], const int sz[3], double *var)
{

    BIND1(nhalo)
    BIND1(sz)
    BIND3_EXT(var, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                   I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                   1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    int ist = 1 - I1(nhalo, 1);
    int jst = 1 - I1(nhalo, 3);
    int kst = 1 - I1(nhalo, 5);

#ifdef USE_OMP_OFFLOAD
    const int sz0 = sz[0], sz1 = sz[1], sz2 = sz[2];
    const int var_size = (sz0 + nhalo[0] + nhalo[1]) * 
                         (sz1 + nhalo[2] + nhalo[3]) *
                         (sz2 + nhalo[4] + nhalo[5]);
    #pragma omp target update from(var[0:var_size])
#endif

    // halo exchange in the west/east direction
    // MPI_Sendrecv(&I3(var,            1, jst, kst), 1, halotype[1], neighbor[0], 0, 
    //              &I3(var, I1(sz,1)+  1, jst, kst), 1, halotype[1], neighbor[1], 0,
    //              comm, MPI_STATUS_IGNORE);
    // MPI_Sendrecv(&I3(var, I1(sz,1)+ist, jst, kst), 1, halotype[0], neighbor[1], 0, 
    //              &I3(var,          ist, jst, kst), 1, halotype[0], neighbor[0], 0,
    //              comm, MPI_STATUS_IGNORE);
    // halo exchange in the south/north direction
    MPI_Sendrecv(&I3(var, ist,            1, kst), 1, halotype[3], neighbor[2], 0, 
                 &I3(var, ist, I1(sz,2)+  1, kst), 1, halotype[3], neighbor[3], 0,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&I3(var, ist, I1(sz,2)+jst, kst), 1, halotype[2], neighbor[3], 0, 
                 &I3(var, ist,          jst, kst), 1, halotype[2], neighbor[2], 0,
                 comm, MPI_STATUS_IGNORE);
    // halo exchange in the bottom/top direction
    MPI_Sendrecv(&I3(var, ist, jst,            1), 1, halotype[5], neighbor[4], 0, 
                 &I3(var, ist, jst, I1(sz,3)+  1), 1, halotype[5], neighbor[5], 0,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&I3(var, ist, jst, I1(sz,3)+kst), 1, halotype[4], neighbor[5], 0, 
                 &I3(var, ist, jst,          kst), 1, halotype[4], neighbor[4], 0,
                 comm, MPI_STATUS_IGNORE);

#ifdef USE_OMP_OFFLOAD
    #pragma omp target update to(var[0:var_size])
#endif
}

void impose_bc_vel(const int neighbor[6], const int nhalo[6], const int sz[3], 
                   double *u, double *v, double *w, double u_crf) {

    int idir, ibound;
    // B.C. in x direction
    idir = 1;
    if (neighbor[0] == MPI_PROC_NULL) {
        ibound = 0;
        impose_periodic_bc(ibound, idir, nhalo, sz, u);
        impose_periodic_bc(ibound, idir, nhalo, sz, v);
        impose_periodic_bc(ibound, idir, nhalo, sz, w);
    }
    if (neighbor[1] == MPI_PROC_NULL) {
        ibound = 1;
        impose_periodic_bc(ibound, idir, nhalo, sz, u);
        impose_periodic_bc(ibound, idir, nhalo, sz, v);
        impose_periodic_bc(ibound, idir, nhalo, sz, w);
    }
    // B.C. in y direction
    idir = 2;
    if (neighbor[2] == MPI_PROC_NULL) {
        ibound = 0;
        impose_periodic_bc(ibound, idir, nhalo, sz, u);
        impose_periodic_bc(ibound, idir, nhalo, sz, v);
        impose_periodic_bc(ibound, idir, nhalo, sz, w);
    }
    if (neighbor[3] == MPI_PROC_NULL) {
        ibound = 1;
        impose_periodic_bc(ibound, idir, nhalo, sz, u);
        impose_periodic_bc(ibound, idir, nhalo, sz, v);
        impose_periodic_bc(ibound, idir, nhalo, sz, w);
    }
    // B.C. in z direction
    idir = 3;
    if (neighbor[4] == MPI_PROC_NULL) {
        ibound = 0;
        impose_no_slip_bc(ibound, true,  nhalo, sz, u, u_crf);
        impose_no_slip_bc(ibound, true,  nhalo, sz, v, 0.0);
        impose_no_slip_bc(ibound, false, nhalo, sz, w, 0.0);
    }
    if (neighbor[5] == MPI_PROC_NULL) {
        ibound = 1;
        impose_no_slip_bc(ibound, true,  nhalo, sz, u, u_crf);
        impose_no_slip_bc(ibound, true,  nhalo, sz, v, 0.0);
        impose_no_slip_bc(ibound, false, nhalo, sz, w, 0.0);
    }
}

#ifdef GPTL
static char str_gptl[30];
#endif

void update_bound_vel(const MPI_Fint comm_f, const MPI_Fint halotype_f[6], const int neighbor[6], 
                      const int nhalo[6], const int sz[3], double *u, double *v, double *w, 
                      double u_crf, char *tag)
{
    MPI_Comm comm = MPI_Comm_f2c(comm_f);
    MPI_Datatype halotype[6];
    for (int i = 0; i < 6; i++)
        halotype[i] = MPI_Type_f2c(halotype_f[i]);

#ifdef GPTL
    sprintf(str_gptl, "%s%s", "--Update halo vel ", tag);
    GPTLstart(str_gptl);
#endif

    update_halo(comm, halotype, neighbor, nhalo, sz, u);
    update_halo(comm, halotype, neighbor, nhalo, sz, v);
    update_halo(comm, halotype, neighbor, nhalo, sz, w);

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%s", "--Impose BC vel ", tag);
    GPTLstart(str_gptl);
#endif

    impose_bc_vel(neighbor, nhalo, sz, u, v, w, u_crf);

#ifdef GPTL
    GPTLstop(str_gptl);
#endif
}

void update_bound_p(const MPI_Fint comm_f, const MPI_Fint halotype_f[6], const int neighbor[6], 
                    const int nhalo[6], const int sz[3], double *p, char *tag)
{
    MPI_Comm comm = MPI_Comm_f2c(comm_f);
    MPI_Datatype halotype[6];
    for (int i = 0; i < 6; i++)
        halotype[i] = MPI_Type_f2c(halotype_f[i]);

#ifdef GPTL
    sprintf(str_gptl, "%s%s", "--Update halo pres ", tag);
    GPTLstart(str_gptl);
#endif

    update_halo(comm, halotype, neighbor, nhalo, sz, p);

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%s", "--Impose BC pres ", tag);
    GPTLstart(str_gptl);
#endif

    // B.C. in x direction
    if (neighbor[0] == MPI_PROC_NULL)
        impose_periodic_bc(0, 1, nhalo, sz, p);
    if (neighbor[1] == MPI_PROC_NULL)
        impose_periodic_bc(1, 1, nhalo, sz, p);
    // B.C. in y direction
    if (neighbor[2] == MPI_PROC_NULL)
        impose_periodic_bc(0, 2, nhalo, sz, p);
    if (neighbor[3] == MPI_PROC_NULL)
        impose_periodic_bc(1, 2, nhalo, sz, p);
    // B.C. in z direction
    if (neighbor[4] == MPI_PROC_NULL)
        impose_zero_grad_bc(0, nhalo, sz, p);
    if (neighbor[5] == MPI_PROC_NULL)
        impose_zero_grad_bc(1, nhalo, sz, p);

#ifdef GPTL
    GPTLstop(str_gptl);
#endif
}
