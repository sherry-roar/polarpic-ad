#if defined(_PDD) && defined(USE_RDMA)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#ifdef GPTL
#include <gptl.h>
#endif
#include "decomp_2d.h"
#include "mod_fft.h"
#include "memory.h"

#ifdef USE_MKL
    fft_mkl_plan_ptr plan[2][2];
#else
#ifdef USE_KMLFFT
    kml_fft_plan plan[2][2];
#else
    fftw_plan plan[2][2];
#endif
#endif
double fft_normfactor;
int *xsize, *ysize, *zsize;
double *var_xpen, *var_ypen, *var_ypen2, *var_zpen;
DECOMP_2D_REAL *work1_r_ptr, *work2_r_ptr;
double *recvflag_backup_var_ypen, *recvflag_backup_work2_r;
double *a, *b, *c;
int *sz_trid, *st_trid, *neighbor_trid;

#ifdef _PDD
MPI_Comm COMM_CART_PDD;
double *w_pdd, *v_pdd, *tmp_v_pdd;
double *y1_pdd, *y2_pdd, *y3_pdd;
double *tmp_var_pdd;
size_t tmp_var_pdd_len;
#endif

size_t var_offset1;
size_t var_offset2;
size_t var_xpen_offset1;
size_t var_xpen_offset2;
size_t var_ypen_offset1;
size_t var_ypen_offset2;

static void set_trid_coeff(int n, double *dzf, char bctype[], char c_or_f, int neighbor[2], 
                           double a[n], double b[n], double c[n]) {
    int dzf_st = -1;
    if (c_or_f == 'c') {
        for (int k = 0; k < n; k++) {
            a[k] = 2.0 / (dzf[k - dzf_st] * (dzf[k-1 - dzf_st] + dzf[k - dzf_st]));
            c[k] = 2.0 / (dzf[k - dzf_st] * (dzf[k+1 - dzf_st] + dzf[k - dzf_st]));
        }
    } else if (c_or_f == 'f') {
        for (int k = 0; k < n; k++) {
            a[k] = 2.0 / (dzf[k   - dzf_st] * (dzf[k+1 - dzf_st] + dzf[k - dzf_st]));
            c[k] = 2.0 / (dzf[k+1 - dzf_st] * (dzf[k+1 - dzf_st] + dzf[k - dzf_st]));
        }
    }

    for (int i = 0; i < n; i++) {
        b[i] = - a[i] - c[i];
    }

    // coefficients correction according to BC types
    double factor;
    char bc;
    if (neighbor[0] == MPI_PROC_NULL) {
        bc = bctype[0];
        if (bc == 'P') {
            factor = 0.0;
        } else if (bc == 'D') {
            factor = -1.0;
        } else if (bc == 'N') {
            factor = 1.0;
        }

        if (c_or_f == 'c') {
            b[0] += factor * a[0];
            a[0] *= fabs(factor) - 1.0;
        } else if (c_or_f == 'f') {
            if (bc == 'N') {
                b[0] += factor * a[0];
                a[0] *= fabs(factor) - 1.0;
            }
        }
    }
    
    if (neighbor[1] == MPI_PROC_NULL) {
        bc = bctype[1];
        if (bc == 'P') {
            factor = 0.0;
        } else if (bc == 'D') {
            factor = -1.0;
        } else if (bc == 'N') {
            factor = 1.0;
        }

        if (c_or_f == 'c') {
            b[n-1] += factor * c[n-1];
            c[n-1] *= fabs(factor) - 1.0;
        } else if (c_or_f == 'f') {
            if (bc == 'N') {
                b[n-1] += factor * c[n-1];
                c[n-1] *= fabs(factor) - 1.0;
            }
        }
    }
}

#ifdef _PDD
static void init_pdd_array(int sz[3]) {
    v_pdd = (double *)aligned_malloc(sizeof(double) * sz[0] * sz[1] * sz[2] + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    w_pdd = (double *)aligned_malloc(sizeof(double) * sz[0] * sz[1] * sz[2] + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    tmp_var_pdd_len = sz[0]*sz[1];
    y1_pdd = (double *)aligned_malloc(sizeof(double) * tmp_var_pdd_len + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    y2_pdd = (double *)aligned_malloc(sizeof(double) * tmp_var_pdd_len + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    y3_pdd = (double *)aligned_malloc(sizeof(double) * tmp_var_pdd_len + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    tmp_v_pdd = (double *)aligned_malloc(sizeof(double) * tmp_var_pdd_len + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    tmp_var_pdd = (double *)aligned_malloc(sizeof(double) * tmp_var_pdd_len + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);

    memset(v_pdd, 0, sizeof(double) * sz[0] * sz[1] * sz[2]);
    memset(w_pdd, 0, sizeof(double) * sz[0] * sz[1] * sz[2]);
    memset(y1_pdd, 0, sizeof(double) * tmp_var_pdd_len);
    memset(y2_pdd, 0, sizeof(double) * tmp_var_pdd_len);
    memset(y3_pdd, 0, sizeof(double) * tmp_var_pdd_len);

    int ie = sz[0];
    int je = sz[1];
    int ke = sz[2];
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < je; j++) {
        for (int i = 0; i < ie; i++) {
            v_pdd[i + j * ie +      0 * ie * je] = a[0];
            w_pdd[i + j * ie + (ke-1) * ie * je] = c[ke-1];
        }
    }
    a[0] = 0.0;
    c[ke-1] = 0.0;

    #pragma omp parallel
    {
        for (int k = 1; k < ke; k++) {
            #pragma omp for schedule(static)
            for (int j = 0; j < je; j++) {
                for (int i = 0; i < ie; i++) {
                    double a_tmp = a[k] / b[i + j * ie + (k-1) * ie * je];
                    v_pdd[i + j * ie + k * ie * je] -= a_tmp * v_pdd[i + j * ie + (k-1) * ie * je];
                    w_pdd[i + j * ie + k * ie * je] -= a_tmp * w_pdd[i + j * ie + (k-1) * ie * je];
                }
            }
        }

        #pragma omp for schedule(static)
        for (int j = 0; j < je; j++) {
            for (int i = 0; i < ie; i++) {
                if (b[i + j * ie + (ke-1) * ie * je] != 0.0) {
                    v_pdd[i + j * ie + (ke-1) * ie * je] /= b[i + j * ie + (ke-1) * ie * je];
                    w_pdd[i + j * ie + (ke-1) * ie * je] /= b[i + j * ie + (ke-1) * ie * je];
                } else {
                    v_pdd[i + j * ie + (ke-1) * ie * je] = 0.0;
                    w_pdd[i + j * ie + (ke-1) * ie * je] = 0.0;
                }
            }
        }

        for (int k = ke-2; k >= 0; k--) {
            #pragma omp for schedule(static)
            for (int j = 0; j < je; j++) {
                for (int i = 0; i < ie; i++) {
                    v_pdd[i + j * ie + k * ie * je] = (v_pdd[i + j * ie + k * ie * je] - c[k] * v_pdd[i + j * ie + (k+1) * ie * je]) / b[i + j * ie + k * ie * je];
                    w_pdd[i + j * ie + k * ie * je] = (w_pdd[i + j * ie + k * ie * je] - c[k] * w_pdd[i + j * ie + (k+1) * ie * je]) / b[i + j * ie + k * ie * je];
                }
            }
        }
    }

    MPI_Sendrecv(v_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[4], 100,
                 tmp_v_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[5], 100,
                 COMM_CART_PDD, MPI_STATUS_IGNORE);
}
#endif

void borrow_poisson_solver_buffer_xy() {
    decomp_2d_info *d = &decomp_main;
    for (int k = 0; k < d->xsz[2]; k++) {
        for (int m = 0; m < dims[0]; m++) {
            int xi1 = d->x1st[m]-1;
            int xi2 = d->x1en[m];
            int xpos = d->x1disp[m] + (k+1)*d->xsz[1]*(xi2-xi1) - 1;
            int ypos = k*d->ysz[0]*d->ysz[1] + d->y1en[m]*d->ysz[0] - 1;
            recvflag_backup_work2_r[m + k*dims[0]] = work2_r_ptr[xpos];
            recvflag_backup_var_ypen[m + k*dims[0]] = var_ypen[ypos];
        }
    }
}

void return_poisson_solver_buffer_xy() {
    decomp_2d_info *d = &decomp_main;
    for (int k = 0; k < d->xsz[2]; k++) {
        for (int m = 0; m < dims[0]; m++) {
            int xi1 = d->x1st[m]-1;
            int xi2 = d->x1en[m];
            int xpos = d->x1disp[m] + (k+1)*d->xsz[1]*(xi2-xi1) - 1;
            int ypos = k*d->ysz[0]*d->ysz[1] + d->y1en[m]*d->ysz[0] - 1;
            work2_r_ptr[xpos] = recvflag_backup_work2_r[m + k*dims[0]];
            var_ypen[ypos] = recvflag_backup_var_ypen[m + k*dims[0]];
        }
    }
}

double * get_poisson_solver_buffer_xpen() {
    return var_xpen;
}
double * get_poisson_solver_buffer_ypen() {
    return var_ypen;
}

unr_plan_h MPI_Alltoallv_convert(
    unr_mem_h send_mem_h, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, 
    unr_mem_h recv_mem_h, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
    MPI_Comm comm, unr_sig_h send_finish_sig, unr_sig_h recv_finish_sig) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int sendtype_size, recvtype_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);

    unr_blk_h *send_blk_h_arr = (unr_blk_h*)malloc(size * sizeof(unr_blk_h));
    unr_blk_h *recv_blk_h_arr = (unr_blk_h*)malloc(size * sizeof(unr_blk_h));
    unr_blk_h *rmt_recv_blk_h_arr = (unr_blk_h*)malloc(size * sizeof(unr_blk_h));
    for (int i = 0; i < size; i++) {
        unr_blk_reg(send_mem_h, sdispls[i]*sendtype_size, sendcounts[i]*sendtype_size, send_finish_sig, UNR_NO_SIGNAL, &send_blk_h_arr[i]);
        unr_blk_reg(recv_mem_h, rdispls[i]*recvtype_size, recvcounts[i]*recvtype_size, UNR_NO_SIGNAL, recv_finish_sig, &recv_blk_h_arr[i]);
    }
    MPI_Alltoall(recv_blk_h_arr, 1, MPI_UNR_BLK_H, rmt_recv_blk_h_arr, 1, MPI_UNR_BLK_H, comm);

    /* Shift left `rank` elements */
    unr_blk_h *tmp_blk_h_arr = recv_blk_h_arr;
    memcpy(tmp_blk_h_arr, send_blk_h_arr, size * sizeof(unr_blk_h));
    memcpy(send_blk_h_arr, tmp_blk_h_arr + rank, (size-rank) * sizeof(unr_blk_h));
    memcpy(send_blk_h_arr + (size-rank), tmp_blk_h_arr, rank * sizeof(unr_blk_h));
    memcpy(tmp_blk_h_arr, rmt_recv_blk_h_arr, size * sizeof(unr_blk_h));
    memcpy(rmt_recv_blk_h_arr, tmp_blk_h_arr + rank, (size-rank) * sizeof(unr_blk_h));
    memcpy(rmt_recv_blk_h_arr + (size-rank), tmp_blk_h_arr, rank * sizeof(unr_blk_h));

    unr_plan_h plan;
    unr_blk_send_batch_plan(size, send_blk_h_arr, NULL, rmt_recv_blk_h_arr, NULL, &plan);
    free(send_blk_h_arr);
    free(recv_blk_h_arr);
    free(rmt_recv_blk_h_arr);
    return plan;
}

unr_plan_h MPI_Sendrecv_convert(
    unr_mem_h send_mem_h, size_t send_offset, int send_count, MPI_Datatype sendtype, int dst,
    unr_mem_h recv_mem_h, size_t recv_offset, int recv_count, MPI_Datatype recvtype, int src,
    MPI_Comm comm, unr_sig_h send_finish_sig, unr_sig_h recv_finish_sig) {

    int sendtype_size, recvtype_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);
    unr_blk_h send_blk, recv_blk, recv_blk_rmt;
    unr_blk_reg(send_mem_h, send_offset*sendtype_size, send_count*sendtype_size, send_finish_sig, UNR_NO_SIGNAL, &send_blk);
    unr_blk_reg(recv_mem_h, recv_offset*recvtype_size, recv_count*recvtype_size, UNR_NO_SIGNAL, recv_finish_sig, &recv_blk);
    MPI_Sendrecv(&recv_blk, 1, MPI_UNR_BLK_H, src, 0,
                 &recv_blk_rmt, 1, MPI_UNR_BLK_H, dst, 0,
                 comm, MPI_STATUS_IGNORE);
    
    unr_plan_h plan;
    memset(&plan, 0, sizeof(unr_plan_h));
    if (dst != MPI_PROC_NULL) {
        unr_blk_send_plan(send_blk, UNR_NO_SIGNAL, recv_blk_rmt, UNR_NO_SIGNAL, &plan);
    }
    return plan;
}

unr_plan_h *transpose_x_to_y_plan, *transpose_y_to_x_plan, *pdd_stage1_plan, *pdd_stage2_plan;
unr_sig_h *transpose_x_to_y_sig, *transpose_y_to_x_sig, *pdd_stage1_sig, *pdd_stage2_sig;

void init_poisson_solver(int nx_global, int ny_global, int nz_global, 
                         double dx, double dy, double *dzf_global,
                         char bctype_x[], char bctype_y[], char bctype_z[], 
                         int neighbor_xyz[3][6]) {

    xsize = decomp_main.xsz;
    ysize = decomp_main.ysz;
    zsize = decomp_main.zsz;
    size_t max_pen_size = get_decomp_2d_work_size();
    work1_r_ptr = get_decomp_2d_work1_r();
    work2_r_ptr = get_decomp_2d_work2_r();

    var_offset1 = (xsize[0]+2) * (xsize[1]+2);
    var_offset2 = (xsize[0]+2);
    var_xpen_offset1 = xsize[0] * xsize[1];
    var_xpen_offset2 = xsize[0];
    var_ypen_offset1 = ysize[0] * ysize[1];
    var_ypen_offset2 = ysize[0];

    size_t var_xpen_size = sizeof(double)*xsize[0]*xsize[1]*xsize[2];
    size_t var_ypen_size = sizeof(double)*ysize[0]*ysize[1]*ysize[2];
    var_xpen = (double*)aligned_malloc(var_xpen_size + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    var_ypen = (double*)aligned_malloc(var_ypen_size + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    var_ypen2= (double*)aligned_malloc(var_ypen_size + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    if (xsize[0] == ysize[0] && xsize[1] == ysize[1] && xsize[2] == ysize[2]) {
        printf("WARNING: Please run in parallel. This version is not optimized for serial running.\n");
    }

    // determine a decomposition mode used for solving the tridiagonal system
#ifdef _PDD
    sz_trid = decomp_main.ysz;
    st_trid = decomp_main.yst;
    neighbor_trid = &neighbor_xyz[1][0];
#else
    sz_trid = decomp_main.zsz;
    st_trid = decomp_main.zst;
    neighbor_trid = &neighbor_xyz[2][0];
#endif

    // initialize FFT
#ifdef USE_MKL
    init_fft(xsize, ysize, bctype_x, bctype_y, plan, &fft_normfactor);
#else
    init_fft(xsize, ysize, bctype_x, bctype_y, var_xpen, var_ypen, plan, &fft_normfactor);
#endif

    // calculate eigenvalues corresponding to BC types
    double *lambdax = (double *)malloc(sz_trid[0] * sizeof(double));
    double *lambday = (double *)malloc(sz_trid[1] * sizeof(double));
    double *lambdaxy = (double *)malloc(sz_trid[0] * sz_trid[1] * sizeof(double));
    get_eigen_values(st_trid[0], sz_trid[0], nx_global, bctype_x, lambdax);
    for (int i = 0; i < sz_trid[0]; i++) {
        lambdax[i] /= (dx*dx);
    }
    get_eigen_values(st_trid[1], sz_trid[1], ny_global, bctype_y, lambday);
    for (int i = 0; i < sz_trid[1]; i++) {
        lambday[i] /= (dy*dy);
    }
    for (int j = 0; j < sz_trid[1]; j++) {
        for (int i = 0; i < sz_trid[0]; i++) {
            lambdaxy[i + j * sz_trid[0]] = lambdax[i] + lambday[j];
        }
    }

    // calculate coefficients of tridiagonal systems
    a = (double *)aligned_malloc(sizeof(double)*sz_trid[2] + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    c = (double *)aligned_malloc(sizeof(double)*sz_trid[2] + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    b = (double *)aligned_malloc(sizeof(double)*sz_trid[0]*sz_trid[1]*sz_trid[2] + MEM_ALIGN_SIZE, MEM_ALIGN_SIZE);
    double *b_tmp = (double *)malloc(sz_trid[2] * sizeof(double));
    double *dzf = (double *)malloc((sz_trid[2] + 2) * sizeof(double));
    
    for (int i = 0; i < sz_trid[2] + 2; i++) {
        dzf[i] = dzf_global[st_trid[2]-1+i];
    }
    set_trid_coeff(sz_trid[2], dzf, bctype_z, 'c', &neighbor_trid[4], a, b_tmp, c);
    #pragma omp parallel
    {
        for (int k = 0; k < sz_trid[2]; k++) {
            #pragma omp for schedule(static)
            for (int j = 0; j < sz_trid[1]; j++) {
                for (int i = 0; i < sz_trid[0]; i++) {
                    b[i + j * sz_trid[0] + k * sz_trid[0] * sz_trid[1]] = 
                        b_tmp[k] + lambdaxy[i + j * sz_trid[0]];
                }
            }
        }
        // decompose coefficient b
        for (int k = 1; k < sz_trid[2]; k++) {
            #pragma omp for schedule(static)
            for (int j = 0; j < sz_trid[1]; j++) {
                for (int i = 0; i < sz_trid[0]; i++) {
                    double a_tmp = a[k] / b[i + j * sz_trid[0] + (k-1) * sz_trid[0] * sz_trid[1]];
                    b[i + j * sz_trid[0] + k * sz_trid[0] * sz_trid[1]] -= (a_tmp * c[k-1]);
                }
            }
        }
    }

    // determine whether the tridiagonal systems are periodic or not
    // NOTE: not yet implemented

    // calculate the correction of the right-hand side according to BC types in x, y, z direction
    // NOTE: not yet implemented

#ifdef _PDD
    COMM_CART_PDD = decomp_2d_comm_cart_y;
    
    // initialize work arrays for PDD algorithm
    init_pdd_array(sz_trid);
#endif

    free(lambdax);
    free(lambday);
    free(lambdaxy);
    free(b_tmp);
    free(dzf);

    /* Init RDMA */
    unr_init();
    int decomp_2d_comm_col_size;
    MPI_Comm_size(decomp_2d_comm_col, &decomp_2d_comm_col_size);

    /* Init for transpose x to y*/
    transpose_x_to_y_plan = (unr_plan_h*)malloc(sizeof(unr_plan_h)*xsize[2]);
    transpose_y_to_x_plan = (unr_plan_h*)malloc(sizeof(unr_plan_h)*xsize[2]);
    transpose_x_to_y_sig = (unr_sig_h*)malloc(sizeof(unr_sig_h)*xsize[2]);
    transpose_y_to_x_sig = (unr_sig_h*)malloc(sizeof(unr_sig_h)*xsize[2]);
    for (size_t k = 0; k < xsize[2]; k++) {
        unr_sig_create(&transpose_x_to_y_sig[k], dims[0]);
        unr_sig_create(&transpose_y_to_x_sig[k], dims[0]);
    }
    unr_mem_h var_ypen_memh, var_ypen_memh2, work1_r_memh, work2_r_memh;
    unr_blk_h var_ypen_blk, var_ypen_blk2, tmp_var_pdd_blk, y1_pdd_blk, y3_pdd_blk;
    unr_mem_reg(var_ypen,    var_ypen_size, &var_ypen_memh);
    unr_mem_reg(var_ypen2,   var_ypen_size, &var_ypen_memh2);
    unr_mem_reg(work1_r_ptr, max_pen_size,  &work1_r_memh);
    unr_mem_reg(work2_r_ptr, max_pen_size,  &work2_r_memh);
    unr_mem_reg_sync();
    int *counts_send = (int*)malloc(sizeof(int)*decomp_2d_comm_col_size);
    int *counts_recv = (int*)malloc(sizeof(int)*decomp_2d_comm_col_size);
    int *displacements_send = (int*)malloc(sizeof(int)*decomp_2d_comm_col_size);
    int *displacements_recv = (int*)malloc(sizeof(int)*decomp_2d_comm_col_size);
    for (size_t k = 0; k < xsize[2]; k++) {
        for (int m = 0; m < dims[0]; m++) {
            int xi1 = decomp_main.x1st[m]-1;
            int xi2 = decomp_main.x1en[m];
            int xpos = decomp_main.x1disp[m] + k*xsize[1]*(xi2-xi1);
            int yi1 = decomp_main.y1st[m]-1;
            int yi2 = decomp_main.y1en[m];
            int ypos = decomp_main.y1disp[m]+k*ysize[0]*(yi2-yi1);
            counts_send[m] = xsize[1]*(xi2-xi1);
            counts_recv[m] = ysize[0]*(yi2-yi1);
            displacements_send[m] = xpos;
            displacements_recv[m] = k*var_ypen_offset1+yi1*var_ypen_offset2;
        }
        transpose_x_to_y_plan[k] = MPI_Alltoallv_convert(
            work1_r_memh, counts_send, displacements_send, MPI_DOUBLE, 
            var_ypen_memh, counts_recv, displacements_recv, MPI_DOUBLE, 
            decomp_2d_comm_col, UNR_NO_SIGNAL, transpose_x_to_y_sig[k]);
    }

    /* Init for transpose y to x*/
    for (size_t k = 0; k < xsize[2]; k++) {
        for (int m = 0; m < dims[0]; m++) {
            int yi1 = decomp_main.y1st[m]-1;
            int yi2 = decomp_main.y1en[m];
            int ypos = decomp_main.y1disp[m]+k*(yi2-yi1)*ysize[0];
            int xi1 = decomp_main.x1st[m]-1;
            int xi2 = decomp_main.x1en[m];
            int xpos = decomp_main.x1disp[m]+k*xsize[1]*(xi2-xi1);
            counts_send[m] = ysize[0]*(yi2-yi1);
            counts_recv[m] = xsize[1]*(xi2-xi1);
            displacements_send[m] = k*var_ypen_offset1+yi1*var_ypen_offset2;
            displacements_recv[m] = xpos;
        }
        transpose_y_to_x_plan[k] = MPI_Alltoallv_convert(
            var_ypen_memh2, counts_send, displacements_send, MPI_DOUBLE, 
            work2_r_memh, counts_recv, displacements_recv, MPI_DOUBLE, 
            decomp_2d_comm_col, UNR_NO_SIGNAL, transpose_y_to_x_sig[k]);
    }
    free(counts_send);
    free(counts_recv);
    free(displacements_send);
    free(displacements_recv);

    /* Init for solved trid */
    unr_mem_h tmp_var_pdd_memh, y1_pdd_memh, y3_pdd_memh;
    unr_mem_reg(var_ypen,    var_ypen_size,                        &var_ypen_memh);
    unr_mem_reg(tmp_var_pdd, sizeof(double)*sz_trid[0]*sz_trid[1], &tmp_var_pdd_memh);
    unr_mem_reg(y1_pdd,      sizeof(double)*sz_trid[0]*sz_trid[1], &y1_pdd_memh);
    unr_mem_reg(y3_pdd,      sizeof(double)*sz_trid[0]*sz_trid[1], &y3_pdd_memh);
    unr_mem_reg_sync();

    int j_block = 8192 / (ysize[0]*sizeof(double)); // TODO: It seems a BUG
    if (j_block == 0) 
        j_block = 1;

    pdd_stage1_plan = (unr_plan_h*)malloc(sizeof(unr_plan_h)*ysize[1]);
    pdd_stage2_plan = (unr_plan_h*)malloc(sizeof(unr_plan_h)*ysize[1]);
    pdd_stage1_sig = (unr_sig_h*)malloc(sizeof(unr_sig_h)*ysize[1]);
    pdd_stage2_sig = (unr_sig_h*)malloc(sizeof(unr_sig_h)*ysize[1]);
    for (int j_block_start = 0; j_block_start < ysize[1]; j_block_start+= j_block) {
        unr_sig_create(pdd_stage1_sig+j_block_start, 1);
        unr_sig_create(pdd_stage2_sig+j_block_start, 1);
    }

    for (int j_block_start = 0; j_block_start < ysize[1]; j_block_start+= j_block) {
        int j_block_end = (j_block_start+j_block) < ysize[1] ? (j_block_start+j_block) : ysize[1];
        pdd_stage1_plan[j_block_start] = MPI_Sendrecv_convert(
            var_ypen_memh, j_block_start*ysize[0], (j_block_end-j_block_start)*ysize[0], MPI_DOUBLE, neighbor_trid[4],
            tmp_var_pdd_memh, j_block_start*ysize[0], (j_block_end-j_block_start)*ysize[0], MPI_DOUBLE, neighbor_trid[5],
            COMM_CART_PDD, UNR_NO_SIGNAL, pdd_stage1_sig[j_block_start]);
    }

    for (int j_block_start = 0; j_block_start < ysize[1]; j_block_start+= j_block) {
        int j_block_end = (j_block_start+j_block) < ysize[1] ? (j_block_start+j_block) : ysize[1];
        pdd_stage2_plan[j_block_start] = MPI_Sendrecv_convert(
            y3_pdd_memh, j_block_start*ysize[0], (j_block_end-j_block_start)*ysize[0], MPI_DOUBLE, neighbor_trid[5],
            y1_pdd_memh, j_block_start*ysize[0], (j_block_end-j_block_start)*ysize[0], MPI_DOUBLE, neighbor_trid[4],
            COMM_CART_PDD, UNR_NO_SIGNAL, pdd_stage2_sig[j_block_start]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void execute_poisson_solver(double *var) {

#ifdef USE_MKL
    static fft_mkl_plan_ptr plan_fwd_xpen, plan_bwd_xpen, plan_fwd_ypen, plan_bwd_ypen;
#else
#ifdef USE_KMLFFT
    static kml_fft_plan plan_fwd_xpen, plan_bwd_xpen, plan_fwd_ypen, plan_bwd_ypen;
#else
    static fftw_plan plan_fwd_xpen, plan_bwd_xpen, plan_fwd_ypen, plan_bwd_ypen;
#endif
#endif
    plan_fwd_xpen = plan[0][0];
    plan_bwd_xpen = plan[0][1];
    plan_fwd_ypen = plan[1][0];
    plan_bwd_ypen = plan[1][1];

    #ifdef GPTL
        GPTLstart("--Copy & Fwd X-FFT & Transpose x to y & Fwd Y-FFT");
    #endif

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t k = 0; k < xsize[2]; k++) {

            /* COPY & X-FFT */
            double *dst = &var_xpen[k*var_xpen_offset1];
            double *src = &var[(k+1)*var_offset1+var_offset2+1];
            for (size_t j = 0; j < xsize[1]; j++) {
                memcpy(dst, src, xsize[0] * sizeof(double));
                dst += var_xpen_offset2;
                src += var_offset2;
            }
            execute_fft(plan_fwd_xpen, &var_xpen[k*var_xpen_offset1]);

            for (int m = 0; m < dims[0]; m++) {
                int xi1 = decomp_main.x1st[m]-1;
                int xi2 = decomp_main.x1en[m];
                int xpos = decomp_main.x1disp[m] + k*xsize[1]*(xi2-xi1);
                for (int j = 0; j < xsize[1]; j++) {
                    memcpy(&work1_r_ptr[xpos+j*(xi2-xi1)], &var_xpen[k*var_xpen_offset1+j*var_xpen_offset2+xi1], sizeof(double)*(xi2-xi1));
                }
            }

            /* ALLTOALL Send */
            unr_plan_start(transpose_x_to_y_plan[k]);
        }

        #pragma omp for schedule(static) nowait
        for (int k = 0; k < xsize[2]; k++) {

            unr_sig_wait(transpose_x_to_y_sig[k]);
            unr_sig_reset(transpose_x_to_y_sig[k]);

            /* Y-FFT */
            execute_fft(plan_fwd_ypen, &var_ypen[k*var_ypen_offset1]);

            /* The Reset of this recv_signal is after solve trid */
        }

        #ifdef GPTL
            GPTLstart("----Thread wait 1");
            #pragma omp barrier
            GPTLstop("----Thread wait 1");

            GPTLstart("----Barrier cost 1");
            #pragma omp barrier
            GPTLstop("----Barrier cost 1");
        #endif
    }

    #ifdef GPTL
        GPTLstop("--Copy & Fwd X-FFT & Transpose x to y & Fwd Y-FFT");
        GPTLstart("--Solve trid");
    #endif

    // #ifndef _PRECALC_TRID_COEFF
    //     Solve trid with out _PRECALC_TRID_COEFF is NOT implemented!
    // #endif

    #pragma omp parallel
    {
        /* ATTENTION: The schedule method in this block should be `static` and the block size should be the same because dependency*/
        int j_block = 8192 / (ysize[0]*sizeof(double)); // TODO: It seems a BUG
        if (j_block == 0) 
            j_block = 1;
        // j_block = 7; /* For Debug*/
        #pragma omp for schedule(static) nowait
        for (int j_block_start = 0; j_block_start < ysize[1]; j_block_start+= j_block) {
            int j_block_end = (j_block_start+j_block) < ysize[1] ? (j_block_start+j_block) : ysize[1];
            for (int j = j_block_start; j < j_block_end; j++) {
                for (int k = 1; k < ysize[2]; k++) {
                    double *var_ypen_up = &var_ypen[k*ysize[0]*ysize[1]+j*ysize[0]];
                    double *var_ypen_down = &var_ypen[(k-1)*ysize[0]*ysize[1]+j*ysize[0]];
                    double *b_p = &b[(k-1)*sz_trid[0]*sz_trid[1]+j*sz_trid[0]];
                    for (int i = 0; i < ysize[0]; i++) {
                        double a_tmp = a[k]/b_p[i];
                        var_ypen_up[i] -= a_tmp * var_ypen_down[i];
                    }
                }

                double *b_p = &b[(ysize[2]-1)*sz_trid[0]*sz_trid[1]+j*sz_trid[0]];
                double *var_ypen_p = &var_ypen[(ysize[2]-1)*ysize[0]*ysize[1]+j*ysize[0]];
                for (int i = 0; i < ysize[0]; i++) {
                    if (b_p[i] != 0) {
                        var_ypen_p[i] /= b_p[i];
                    }
                    else {
                        var_ypen_p[i] = 0;
                    }
                }

                for (int k = ysize[2]-2; k >=0 ; k--) {
                    double *var_ypen_down = &var_ypen[k*ysize[0]*ysize[1]+j*ysize[0]];
                    double *var_ypen_up = &var_ypen[(k+1)*ysize[0]*ysize[1]+j*ysize[0]];
                    double *b_p = &b[k*sz_trid[0]*sz_trid[1]+j*sz_trid[0]];
                    for (int i = 0; i < ysize[0]; i++) {
                        var_ypen_down[i] = (var_ypen_down[i]-c[k]*var_ypen_up[i])/b_p[i];
                    }
                }
            }

            if (neighbor_trid[4] != MPI_PROC_NULL) {
                unr_plan_start(pdd_stage1_plan[j_block_start]);
            }
        }

        if (neighbor_trid[5] != MPI_PROC_NULL) {
            #pragma omp for schedule(static) nowait
            for (int j_block_start = 0; j_block_start < ysize[1]; j_block_start+= j_block) {
                int j_block_end = (j_block_start+j_block) < ysize[1] ? (j_block_start+j_block) : ysize[1];

                /* Recv data */
                unr_sig_wait(pdd_stage1_sig[j_block_start]);
                unr_sig_reset(pdd_stage1_sig[j_block_start]);

                for (int j = j_block_start; j < j_block_end; j++) {
                    double *w_pdd_p = &w_pdd[(ysize[2]-1)*sz_trid[0]*sz_trid[1]+j*sz_trid[0]];
                    double *tmp_v_pdd_p = &tmp_v_pdd[j*sz_trid[0]];
                    double *var_ypen_p = &var_ypen[(ysize[2]-1)*ysize[0]*ysize[1]+j*ysize[0]];
                    double *y2_pdd_p = &y2_pdd[j*sz_trid[0]];
                    double *y3_pdd_p = &y3_pdd[j*sz_trid[0]];
                    double *tmp_var_pdd_p = &tmp_var_pdd[j*sz_trid[0]];
                    for (int i = 0; i < ysize[0]; i++) {
                        double det_pdd = w_pdd_p[i] * tmp_v_pdd_p[i] - ((double)1.0);
                        y2_pdd_p[i] = (var_ypen_p[i] * tmp_v_pdd_p[i] - tmp_var_pdd_p[i]) / det_pdd;
                        y3_pdd_p[i] = (tmp_var_pdd_p[i] * w_pdd_p[i] - var_ypen_p[i]) / det_pdd;
                    }
                }

                unr_plan_start(pdd_stage2_plan[j_block_start]);
            }
        }

        #pragma omp for schedule(static) nowait
        for (int j_block_start = 0; j_block_start < ysize[1]; j_block_start+= j_block) {
            int j_block_end = (j_block_start+j_block) < ysize[1] ? (j_block_start+j_block) : ysize[1];

            /* Recv data */
            if (neighbor_trid[4] != MPI_PROC_NULL) {
                unr_sig_wait(pdd_stage2_sig[j_block_start]);
                unr_sig_reset(pdd_stage2_sig[j_block_start]);
            }

            for (int j = j_block_start; j < j_block_end; j++) {
                for (int k = 0; k < ysize[2]; k++) {
                    double *var_ypen_p = &var_ypen[k*ysize[0]*ysize[1]+j*ysize[0]];
                    double *var_ypen2_p = &var_ypen2[k*ysize[0]*ysize[1]+j*ysize[0]];
                    double *v_pdd_p = &v_pdd[k*sz_trid[0]*sz_trid[1]+j*sz_trid[0]];
                    double *w_pdd_p = &w_pdd[k*sz_trid[0]*sz_trid[1]+j*sz_trid[0]];
                    double *y1_pdd_p = &y1_pdd[j*sz_trid[0]];
                    double *y2_pdd_p = &y2_pdd[j*sz_trid[0]];
                    for (int i = 0; i < ysize[0]; i++) {
                        var_ypen2_p[i] = var_ypen_p[i] - v_pdd_p[i]*y1_pdd_p[i] - w_pdd_p[i]*y2_pdd_p[i];
                    }
                }
            }
        }

        #ifdef GPTL
            GPTLstart("----Thread wait 2");
            #pragma omp barrier
            GPTLstop("----Thread wait 2");

            GPTLstart("----Barrier cost 2");
            #pragma omp barrier
            GPTLstop("----Barrier cost 2");
        #endif
    }

    #ifdef GPTL
        GPTLstop("--Solve trid");
    #endif

    #ifdef GPTL
        GPTLstart("--Bwd Y-FFT & Transpose y to x & Bwd X-FFT & Copy");
    #endif

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t k = 0; k < ysize[2]; k++) {
            /* Bwd Y-FFT */
            execute_fft(plan_bwd_ypen, &var_ypen2[k*var_ypen_offset1]);
            unr_plan_start(transpose_y_to_x_plan[k]);
        }

        #pragma omp for schedule(static) nowait
        for (int k = 0; k < xsize[2]; k++) {

            unr_sig_wait(transpose_y_to_x_sig[k]);
            unr_sig_reset(transpose_y_to_x_sig[k]);

            /* mem_merge_xy_real */
            for (int m = 0; m < dims[0]; m++) {
                int xi1 = decomp_main.x1st[m]-1;
                int xi2 = decomp_main.x1en[m];
                int xpos = decomp_main.x1disp[m]+k*xsize[1]*(xi2-xi1);
                for (int j = 0; j < xsize[1]; j++) {
                    memcpy(&var_xpen[k*var_xpen_offset1+j*var_xpen_offset2+xi1], &work2_r_ptr[xpos+j*(xi2-xi1)], sizeof(double)*(xi2-xi1));
                }
            }

            /*Bwd X-FFT & Copy*/
            double *var_xpen_p = &var_xpen[k*var_xpen_offset1];
            double *var_p = &var[(k+1)*var_offset1+var_offset2+1];
            execute_fft(plan_bwd_xpen, var_xpen_p);
            for (size_t j = 0; j < xsize[1]; j++) {
                for (size_t i = 0; i < xsize[0]; i++) {
                    var_p[i] = var_xpen_p[i] * fft_normfactor;
                }
                var_xpen_p += var_xpen_offset2;
                var_p += var_offset2;
            }
        }

        #ifdef GPTL
            GPTLstart("----Thread wait 3");
            #pragma omp barrier
            GPTLstop("----Thread wait 3");

            GPTLstart("----Barrier cost 3");
            #pragma omp barrier
            GPTLstop("----Barrier cost 3");
        #endif
    }

    #ifdef GPTL
        GPTLstop("--Bwd Y-FFT & Transpose y to x & Bwd X-FFT & Copy");
    #endif
}

void free_poisson_solver() {
    
    // release work arrays
    if (var_xpen != NULL) aligned_free(var_xpen);
    if (var_xpen != var_ypen && var_ypen != NULL) aligned_free(var_ypen);
    if (var_ypen != var_zpen && var_zpen != NULL) aligned_free(var_zpen);
    if (recvflag_backup_work2_r != NULL) free(recvflag_backup_work2_r);
    if (recvflag_backup_var_ypen != NULL) free(recvflag_backup_var_ypen);

    // release tridiagonal coefficients arrays
    if (a != NULL) aligned_free(a);
    if (b != NULL) aligned_free(b);
    if (c != NULL) aligned_free(c);

#ifdef _PDD
    // release PDD related arrays
    if (v_pdd != NULL) aligned_free(v_pdd);
    if (w_pdd != NULL) aligned_free(w_pdd);
    if (y1_pdd != NULL) aligned_free(y1_pdd);
    if (y2_pdd != NULL) aligned_free(y2_pdd);
    if (y3_pdd != NULL) aligned_free(y3_pdd);
    if (tmp_v_pdd != NULL) aligned_free(tmp_v_pdd);
    if (tmp_var_pdd != NULL) aligned_free(tmp_var_pdd);
#endif

    // release FFT
    free_fft(plan);

    unr_finalize();
}

#endif
