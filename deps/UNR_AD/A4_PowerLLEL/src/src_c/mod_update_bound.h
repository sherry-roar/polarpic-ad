#include <mpi.h>

void impose_bc_vel(const int neighbor[6], const int nhalo[6], const int sz[3],
                   double *u, double *v, double *w, double u_crf);
void update_bound_vel(const MPI_Fint comm_f, const MPI_Fint halotype_f[6],
                      const int neighbor[6], const int nhalo[6],
                      const int sz[3], double *u, double *v, double *w,
                      double u_crf, char *tag);
void update_bound_p(const MPI_Fint comm_f, const MPI_Fint halotype_f[6],
                    const int neighbor[6], const int nhalo[6], const int sz[3],
                    double *p, char *tag);

#ifdef NB_HALO
void get_neighbor_rank_2d_cart(const MPI_Comm comm);
void get_neighbor_rank_2d_cart_c(const MPI_Fint comm_f);
void create_nbhalo_mpitype(const int nhalo[6], const int sz[3],
                           const MPI_Datatype oldtype);
void create_nbhalo_mpitype_c(const int nhalo[6], const int sz[3],
                             const MPI_Fint oldtype_f);
void free_nbhalo_mpitype();
void update_halo_isend(const int nhalo[6], const int sz[3], const int tag,
                       double *var, MPI_Request isend_req[8]);
void update_halo_irecv(const int nhalo[6], const int sz[3], const int tag,
                       double *var, MPI_Request irecv_req[8]);
void update_halo_waitall(MPI_Request isend_req[8], MPI_Request irecv_req[8]);
#ifdef USE_RDMA
#ifndef USE_NBHALOBUF
void init_rdma_halo_c(const int nhalo[6], const int sz[3], double *u, double *v,
                      double *w, double *u1, double *v1, double *w1,
                      const MPI_Fint comm_f);
#else
void init_rdma_halo_c(const int nhalo[6], const int sz[3], 
                      const int halobuf_length[8], const int halobuf_offset[8], 
                      double *halobuf_send, double *halobuf_recv, 
                      double *halobuf_send_aux, double *halobuf_recv_aux, 
                      const MPI_Fint comm_f);
#endif
void update_halo_rdma_send(int rk_num);
void update_halo_rdma_wait(int rk_num);
#endif
#endif
