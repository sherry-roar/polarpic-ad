#include "mod_update_bound.h"
#include <stdio.h>
#ifdef NB_HALO
#include <mpi.h>
#endif
#ifdef GPTL
#include <gptl.h>
#endif
#include <omp.h>
#include "bind.h"
#include "mod_utils.h"

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

void transform_to_crf(double vel_crf, const int *nhalo, const int *sz, double *vel, double *vel_force)
{

    BIND1(nhalo)
    BIND1(sz)
    BIND3_EXT(vel, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                   I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                   1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    
    const int sz0 = sz[0], sz1 = sz[1], sz2 = sz[2];
#ifdef USE_OMP_OFFLOAD
    #pragma omp target teams distribute parallel for collapse(3)
#else
    #pragma omp parallel for
#endif
    DO(k, 1, sz2) {
        DO(j, 1, sz1) {
            DO(i, 1, sz0) {
                I3(vel, i, j, k) -= vel_crf;
            }
        }
    }

    *vel_force -= vel_crf;
    
}

void time_int_vel_rk1_kernel(int st[3], int en[3], double re_inv, double dt, double dx_inv, double dy_inv, 
    const double *dzf, const double *dzc_inv, const double *dzf_inv, const double *visc_dzf_inv,
    const int *nhalo, const int *sz, const double * restrict u, const double * restrict v, const double * restrict w, 
    double * restrict u1, double * restrict v1, double * restrict w1, double u_crf)
{

    double r1 = 9.0/8.0;
    double r2 = -1.0/8.0;
    double r22= r2/3.0;
    double rdxidxi = dx_inv*dx_inv/12.0;
    double rdyidyi = dy_inv*dy_inv/12.0;

    BIND1(nhalo)
    BIND1(sz)
    BIND1_EXT(     dzf,     1 - I1(nhalo,5))
    BIND1_EXT(     dzc_inv, 1 - I1(nhalo,5))
    BIND1_EXT(     dzf_inv, 1 - I1(nhalo,5))
    BIND1_EXT(visc_dzf_inv, 1 - I1(nhalo,5))
    BIND3_EXT(u, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(v, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(w, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(u1, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(v1, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(w1, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    
#ifdef USE_OMP_OFFLOAD
    #pragma omp target teams distribute parallel for collapse(3)
#else
    #pragma omp parallel for default(shared) firstprivate(r1,r2,r22,rdxidxi,rdyidyi)
#endif
    DO(k, st[2], en[2]) {
        DO(j, st[1], en[1]) {
            DO(i, st[0], en[0]) {
                double q1m3, q1m1, q1p1, q1p3;
                double q2m3, q2m1, q2p1, q2p3;
                double duudx, dvudy, dwudz;
                double duvdx, dvvdy, dwvdz;
                double duwdx, dvwdy, dwwdz;
                double conv, visc;
                
                q1m3 = r1*(I3(u, i-2,j,k)+I3(u, i-1,j,k)) + r2*(I3(u, i-3,j,k)+I3(u, i  ,j,k));
                q1m1 = r1*(I3(u, i-1,j,k)+I3(u, i  ,j,k)) + r2*(I3(u, i-2,j,k)+I3(u, i+1,j,k));
                q1p1 = r1*(I3(u, i  ,j,k)+I3(u, i+1,j,k)) + r2*(I3(u, i-1,j,k)+I3(u, i+2,j,k));
                q1p3 = r1*(I3(u, i+1,j,k)+I3(u, i+2,j,k)) + r2*(I3(u, i  ,j,k)+I3(u, i+3,j,k));
                q2m3 = I3(u, i-3,j,k)+I3(u, i  ,j,k);
                q2m1 = I3(u, i-1,j,k)+I3(u, i  ,j,k);
                q2p1 = I3(u, i  ,j,k)+I3(u, i+1,j,k);
                q2p3 = I3(u, i  ,j,k)+I3(u, i+3,j,k);
                duudx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dx_inv;
                // add a term induced by the convecting reference frame
                conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5*dx_inv;
                q1m3 = r1*(I3(v, i,j-2,k)+I3(v, i+1,j-2,k)) + r2*(I3(v, i-1,j-2,k)+I3(v, i+2,j-2,k));
                q1m1 = r1*(I3(v, i,j-1,k)+I3(v, i+1,j-1,k)) + r2*(I3(v, i-1,j-1,k)+I3(v, i+2,j-1,k));
                q1p1 = r1*(I3(v, i,j  ,k)+I3(v, i+1,j  ,k)) + r2*(I3(v, i-1,j  ,k)+I3(v, i+2,j  ,k));
                q1p3 = r1*(I3(v, i,j+1,k)+I3(v, i+1,j+1,k)) + r2*(I3(v, i-1,j+1,k)+I3(v, i+2,j+1,k));
                q2m3 = I3(u, i,j-3,k)+I3(u, i,j  ,k);
                q2m1 = I3(u, i,j-1,k)+I3(u, i,j  ,k);
                q2p1 = I3(u, i,j  ,k)+I3(u, i,j+1,k);
                q2p3 = I3(u, i,j  ,k)+I3(u, i,j+3,k);
                dvudy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dy_inv;
                q1m1  = r1*(I3(w, i,j,k-1)+I3(w, i+1,j,k-1)) + r2*(I3(w, i-1,j,k-1)+I3(w, i+2,j,k-1));
                q1p1  = r1*(I3(w, i,j,k  )+I3(w, i+1,j,k  )) + r2*(I3(w, i-1,j,k  )+I3(w, i+2,j,k  ));
                q2m1  = (I3(u, i,j,k)*I1(dzf, k-1) + I3(u, i,j,k-1)*I1(dzf, k))*I1(dzc_inv, k-1);
                q2p1  = (I3(u, i,j,k)*I1(dzf, k+1) + I3(u, i,j,k+1)*I1(dzf, k))*I1(dzc_inv, k  );
                dwudz = (q1p1*q2p1-q1m1*q2m1)*0.25*I1(dzf_inv, k);
                conv = conv + duudx + dvudy + dwudz;
                visc = (- I3(u, i-2,j,k) + 16.0*I3(u, i-1,j,k) - 30.0*I3(u, i,j,k)
                        - I3(u, i+2,j,k) + 16.0*I3(u, i+1,j,k))* rdxidxi +
                       (- I3(u, i,j-2,k) + 16.0*I3(u, i,j-1,k) - 30.0*I3(u, i,j,k)
                        - I3(u, i,j+2,k) + 16.0*I3(u, i,j+1,k))* rdyidyi +
                       ((I3(u, i,j,k+1)-I3(u, i,j,k))*I1(dzc_inv, k)-(I3(u, i,j,k)-I3(u, i,j,k-1))*I1(dzc_inv, k-1))*I1(visc_dzf_inv, k);
                visc = visc*re_inv;
                I3(u1, i,j,k) = I3(u, i,j,k) + dt*(visc - conv);

                q1m3 = r1*(I3(u, i-2,j,k)+I3(u, i-2,j+1,k)) + r2*(I3(u, i-2,j-1,k)+I3(u, i-2,j+2,k));
                q1m1 = r1*(I3(u, i-1,j,k)+I3(u, i-1,j+1,k)) + r2*(I3(u, i-1,j-1,k)+I3(u, i-1,j+2,k));
                q1p1 = r1*(I3(u, i  ,j,k)+I3(u, i  ,j+1,k)) + r2*(I3(u, i  ,j-1,k)+I3(u, i  ,j+2,k));
                q1p3 = r1*(I3(u, i+1,j,k)+I3(u, i+1,j+1,k)) + r2*(I3(u, i+1,j-1,k)+I3(u, i+1,j+2,k));
                q2m3 = I3(v, i-3,j,k)+I3(v, i  ,j,k);
                q2m1 = I3(v, i-1,j,k)+I3(v, i  ,j,k);
                q2p1 = I3(v, i  ,j,k)+I3(v, i+1,j,k);
                q2p3 = I3(v, i  ,j,k)+I3(v, i+3,j,k);
                duvdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dx_inv;
                // add a term induced by the convecting reference frame
                conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5*dx_inv;
                q1m3 = r1*(I3(v, i,j-2,k)+I3(v, i,j-1,k)) + r2*(I3(v, i,j-3,k)+I3(v, i,j  ,k));
                q1m1 = r1*(I3(v, i,j-1,k)+I3(v, i,j  ,k)) + r2*(I3(v, i,j-2,k)+I3(v, i,j+1,k));
                q1p1 = r1*(I3(v, i,j  ,k)+I3(v, i,j+1,k)) + r2*(I3(v, i,j-1,k)+I3(v, i,j+2,k));
                q1p3 = r1*(I3(v, i,j+1,k)+I3(v, i,j+2,k)) + r2*(I3(v, i,j  ,k)+I3(v, i,j+3,k));
                q2m3 = I3(v, i,j-3,k)+I3(v, i,j  ,k);
                q2m1 = I3(v, i,j-1,k)+I3(v, i,j  ,k);
                q2p1 = I3(v, i,j  ,k)+I3(v, i,j+1,k);
                q2p3 = I3(v, i,j  ,k)+I3(v, i,j+3,k);
                dvvdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dy_inv;
                q1m1  = r1*(I3(w, i,j,k-1)+I3(w, i,j+1,k-1)) + r2*(I3(w, i,j-1,k-1)+I3(w, i,j+2,k-1));
                q1p1  = r1*(I3(w, i,j,k  )+I3(w, i,j+1,k  )) + r2*(I3(w, i,j-1,k  )+I3(w, i,j+2,k  ));
                q2m1  = (I3(v, i,j,k  )*I1(dzf, k-1) + I3(v, i,j,k-1)*I1(dzf, k))*I1(dzc_inv, k-1);
                q2p1  = (I3(v, i,j,k  )*I1(dzf, k+1) + I3(v, i,j,k+1)*I1(dzf, k))*I1(dzc_inv, k  );
                dwvdz = (q1p1*q2p1-q1m1*q2m1)*0.25*I1(dzf_inv, k);
                conv = conv + duvdx + dvvdy + dwvdz;
                visc = (- I3(v, i-2,j,k) + 16.0*I3(v, i-1,j,k) - 30.0*I3(v, i,j,k)
                        - I3(v, i+2,j,k) + 16.0*I3(v, i+1,j,k))* rdxidxi +
                       (- I3(v, i,j-2,k) + 16.0*I3(v, i,j-1,k) - 30.0*I3(v, i,j,k)
                        - I3(v, i,j+2,k) + 16.0*I3(v, i,j+1,k))* rdyidyi +
                       ((I3(v, i,j,k+1)-I3(v, i,j,k))*I1(dzc_inv, k)-(I3(v, i,j,k)-I3(v, i,j,k-1))*I1(dzc_inv, k-1))*I1(visc_dzf_inv, k);
                visc = visc*re_inv;
                I3(v1, i,j,k) = I3(v, i,j,k) + dt*(visc - conv);

                q1m3 = (I3(u, i-2,j,k)*I1(dzf, k+1) + I3(u, i-2,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1m1 = (I3(u, i-1,j,k)*I1(dzf, k+1) + I3(u, i-1,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p1 = (I3(u, i  ,j,k)*I1(dzf, k+1) + I3(u, i  ,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p3 = (I3(u, i+1,j,k)*I1(dzf, k+1) + I3(u, i+1,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q2m3 = I3(w, i-3,j,k)+I3(w, i  ,j,k);
                q2m1 = I3(w, i-1,j,k)+I3(w, i  ,j,k);
                q2p1 = I3(w, i  ,j,k)+I3(w, i+1,j,k);
                q2p3 = I3(w, i  ,j,k)+I3(w, i+3,j,k);
                duwdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dx_inv;
                // add a term induced by the convecting reference frame
                conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5*dx_inv;
                q1m3 = (I3(v, i,j-2,k)*I1(dzf, k+1) + I3(v, i,j-2,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1m1 = (I3(v, i,j-1,k)*I1(dzf, k+1) + I3(v, i,j-1,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p1 = (I3(v, i,j  ,k)*I1(dzf, k+1) + I3(v, i,j  ,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p3 = (I3(v, i,j+1,k)*I1(dzf, k+1) + I3(v, i,j+1,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q2m3 = I3(w, i,j-3,k)+I3(w, i,j  ,k);
                q2m1 = I3(w, i,j-1,k)+I3(w, i,j  ,k);
                q2p1 = I3(w, i,j  ,k)+I3(w, i,j+1,k);
                q2p3 = I3(w, i,j  ,k)+I3(w, i,j+3,k);
                dvwdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dy_inv;
                q1m1 = (I3(w, i,j,k)+I3(w, i,j,k-1));
                q1p1 = (I3(w, i,j,k)+I3(w, i,j,k+1));
                q2m1 = q1m1;
                q2p1 = q1p1;
                dwwdz = (q1p1*q2p1-q1m1*q2m1)*0.25*I1(dzc_inv, k);
                conv = conv + duwdx + dvwdy + dwwdz;
                visc = (- I3(w, i-2,j,k) + 16.0*I3(w, i-1,j,k) - 30.0*I3(w, i,j,k)
                        - I3(w, i+2,j,k) + 16.0*I3(w, i+1,j,k))* rdxidxi +
                       (- I3(w, i,j-2,k) + 16.0*I3(w, i,j-1,k) - 30.0*I3(w, i,j,k)
                        - I3(w, i,j+2,k) + 16.0*I3(w, i,j+1,k))* rdyidyi +
                       ((I3(w, i,j,k+1)-I3(w, i,j,k))*I1(dzf_inv, k+1)-(I3(w, i,j,k)-I3(w, i,j,k-1))*I1(dzf_inv, k))*I1(dzc_inv, k);
                visc = visc*re_inv;
                I3(w1, i,j,k) = I3(w, i,j,k) + dt*(visc - conv);
            }
        }
    }

}

void time_int_vel_rk2_kernel(int st[3], int en[3], double re_inv, double dt, double dx_inv, double dy_inv, 
    const double *dzf, const double *dzc_inv, const double *dzf_inv, const double *visc_dzf_inv,
    const int *nhalo, const int *sz, const double * restrict u1, const double * restrict v1, const double * restrict w1, 
    double * restrict u, double * restrict v, double * restrict w, double u_crf)
{

    double r1 = 9.0/8.0;
    double r2 = -1.0/8.0;
    double r22= r2/3.0;
    double rdxidxi = dx_inv*dx_inv/12.0;
    double rdyidyi = dy_inv*dy_inv/12.0;

    BIND1(nhalo)
    BIND1(sz)
    BIND1_EXT(     dzf,     1 - I1(nhalo,5))
    BIND1_EXT(     dzc_inv, 1 - I1(nhalo,5))
    BIND1_EXT(     dzf_inv, 1 - I1(nhalo,5))
    BIND1_EXT(visc_dzf_inv, 1 - I1(nhalo,5))
    BIND3_EXT(u, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(v, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(w, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(u1, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(v1, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(w1, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))

#ifdef USE_OMP_OFFLOAD
    #pragma omp target teams distribute parallel for collapse(3)
#else
    #pragma omp parallel for default(shared) firstprivate(r1,r2,r22,rdxidxi,rdyidyi)
#endif
    DO(k, st[2], en[2]) {
        DO(j, st[1], en[1]) {
            DO(i, st[0], en[0]) {
                double q1m3, q1m1, q1p1, q1p3;
                double q2m3, q2m1, q2p1, q2p3;
                double duudx, dvudy, dwudz;
                double duvdx, dvvdy, dwvdz;
                double duwdx, dvwdy, dwwdz;
                double conv, visc;
                
                q1m3 = r1*(I3(u1, i-2,j,k)+I3(u1, i-1,j,k)) + r2*(I3(u1, i-3,j,k)+I3(u1, i  ,j,k));
                q1m1 = r1*(I3(u1, i-1,j,k)+I3(u1, i  ,j,k)) + r2*(I3(u1, i-2,j,k)+I3(u1, i+1,j,k));
                q1p1 = r1*(I3(u1, i  ,j,k)+I3(u1, i+1,j,k)) + r2*(I3(u1, i-1,j,k)+I3(u1, i+2,j,k));
                q1p3 = r1*(I3(u1, i+1,j,k)+I3(u1, i+2,j,k)) + r2*(I3(u1, i  ,j,k)+I3(u1, i+3,j,k));
                q2m3 = I3(u1, i-3,j,k)+I3(u1, i  ,j,k);
                q2m1 = I3(u1, i-1,j,k)+I3(u1, i  ,j,k);
                q2p1 = I3(u1, i  ,j,k)+I3(u1, i+1,j,k);
                q2p3 = I3(u1, i  ,j,k)+I3(u1, i+3,j,k);
                duudx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dx_inv;
                // add a term induced by the convecting reference frame
                conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5*dx_inv;
                q1m3 = r1*(I3(v1, i,j-2,k)+I3(v1, i+1,j-2,k)) + r2*(I3(v1, i-1,j-2,k)+I3(v1, i+2,j-2,k));
                q1m1 = r1*(I3(v1, i,j-1,k)+I3(v1, i+1,j-1,k)) + r2*(I3(v1, i-1,j-1,k)+I3(v1, i+2,j-1,k));
                q1p1 = r1*(I3(v1, i,j  ,k)+I3(v1, i+1,j  ,k)) + r2*(I3(v1, i-1,j  ,k)+I3(v1, i+2,j  ,k));
                q1p3 = r1*(I3(v1, i,j+1,k)+I3(v1, i+1,j+1,k)) + r2*(I3(v1, i-1,j+1,k)+I3(v1, i+2,j+1,k));
                q2m3 = I3(u1, i,j-3,k)+I3(u1, i,j  ,k);
                q2m1 = I3(u1, i,j-1,k)+I3(u1, i,j  ,k);
                q2p1 = I3(u1, i,j  ,k)+I3(u1, i,j+1,k);
                q2p3 = I3(u1, i,j  ,k)+I3(u1, i,j+3,k);
                dvudy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dy_inv;
                q1m1  = r1*(I3(w1, i,j,k-1)+I3(w1, i+1,j,k-1)) + r2*(I3(w1, i-1,j,k-1)+I3(w1, i+2,j,k-1));
                q1p1  = r1*(I3(w1, i,j,k  )+I3(w1, i+1,j,k  )) + r2*(I3(w1, i-1,j,k  )+I3(w1, i+2,j,k  ));
                q2m1  = (I3(u1, i,j,k)*I1(dzf, k-1) + I3(u1, i,j,k-1)*I1(dzf, k))*I1(dzc_inv, k-1);
                q2p1  = (I3(u1, i,j,k)*I1(dzf, k+1) + I3(u1, i,j,k+1)*I1(dzf, k))*I1(dzc_inv, k  );
                dwudz = (q1p1*q2p1-q1m1*q2m1)*0.25*I1(dzf_inv, k);
                conv = conv + duudx + dvudy + dwudz;
                visc = (- I3(u1, i-2,j,k) + 16.0*I3(u1, i-1,j,k) - 30.0*I3(u1, i,j,k)
                        - I3(u1, i+2,j,k) + 16.0*I3(u1, i+1,j,k))* rdxidxi +
                       (- I3(u1, i,j-2,k) + 16.0*I3(u1, i,j-1,k) - 30.0*I3(u1, i,j,k)
                        - I3(u1, i,j+2,k) + 16.0*I3(u1, i,j+1,k))* rdyidyi +
                       ((I3(u1, i,j,k+1)-I3(u1, i,j,k))*I1(dzc_inv, k)-(I3(u1, i,j,k)-I3(u1, i,j,k-1))*I1(dzc_inv, k-1))*I1(visc_dzf_inv, k);
                visc = visc*re_inv;
                I3(u, i,j,k) = (I3(u1, i,j,k) + dt*(visc - conv) + I3(u, i,j,k))*0.5;

                q1m3 = r1*(I3(u1, i-2,j,k)+I3(u1, i-2,j+1,k)) + r2*(I3(u1, i-2,j-1,k)+I3(u1, i-2,j+2,k));
                q1m1 = r1*(I3(u1, i-1,j,k)+I3(u1, i-1,j+1,k)) + r2*(I3(u1, i-1,j-1,k)+I3(u1, i-1,j+2,k));
                q1p1 = r1*(I3(u1, i  ,j,k)+I3(u1, i  ,j+1,k)) + r2*(I3(u1, i  ,j-1,k)+I3(u1, i  ,j+2,k));
                q1p3 = r1*(I3(u1, i+1,j,k)+I3(u1, i+1,j+1,k)) + r2*(I3(u1, i+1,j-1,k)+I3(u1, i+1,j+2,k));
                q2m3 = I3(v1, i-3,j,k)+I3(v1, i  ,j,k);
                q2m1 = I3(v1, i-1,j,k)+I3(v1, i  ,j,k);
                q2p1 = I3(v1, i  ,j,k)+I3(v1, i+1,j,k);
                q2p3 = I3(v1, i  ,j,k)+I3(v1, i+3,j,k);
                duvdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dx_inv;
                // add a term induced by the convecting reference frame
                conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5*dx_inv;
                q1m3 = r1*(I3(v1, i,j-2,k)+I3(v1, i,j-1,k)) + r2*(I3(v1, i,j-3,k)+I3(v1, i,j  ,k));
                q1m1 = r1*(I3(v1, i,j-1,k)+I3(v1, i,j  ,k)) + r2*(I3(v1, i,j-2,k)+I3(v1, i,j+1,k));
                q1p1 = r1*(I3(v1, i,j  ,k)+I3(v1, i,j+1,k)) + r2*(I3(v1, i,j-1,k)+I3(v1, i,j+2,k));
                q1p3 = r1*(I3(v1, i,j+1,k)+I3(v1, i,j+2,k)) + r2*(I3(v1, i,j  ,k)+I3(v1, i,j+3,k));
                q2m3 = I3(v1, i,j-3,k)+I3(v1, i,j  ,k);
                q2m1 = I3(v1, i,j-1,k)+I3(v1, i,j  ,k);
                q2p1 = I3(v1, i,j  ,k)+I3(v1, i,j+1,k);
                q2p3 = I3(v1, i,j  ,k)+I3(v1, i,j+3,k);
                dvvdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dy_inv;
                q1m1  = r1*(I3(w1, i,j,k-1)+I3(w1, i,j+1,k-1)) + r2*(I3(w1, i,j-1,k-1)+I3(w1, i,j+2,k-1));
                q1p1  = r1*(I3(w1, i,j,k  )+I3(w1, i,j+1,k  )) + r2*(I3(w1, i,j-1,k  )+I3(w1, i,j+2,k  ));
                q2m1  = (I3(v1, i,j,k  )*I1(dzf, k-1) + I3(v1, i,j,k-1)*I1(dzf, k))*I1(dzc_inv, k-1);
                q2p1  = (I3(v1, i,j,k  )*I1(dzf, k+1) + I3(v1, i,j,k+1)*I1(dzf, k))*I1(dzc_inv, k  );
                dwvdz = (q1p1*q2p1-q1m1*q2m1)*0.25*I1(dzf_inv, k);
                conv = conv + duvdx + dvvdy + dwvdz;
                visc = (- I3(v1, i-2,j,k) + 16.0*I3(v1, i-1,j,k) - 30.0*I3(v1, i,j,k)
                        - I3(v1, i+2,j,k) + 16.0*I3(v1, i+1,j,k))* rdxidxi +
                       (- I3(v1, i,j-2,k) + 16.0*I3(v1, i,j-1,k) - 30.0*I3(v1, i,j,k)
                        - I3(v1, i,j+2,k) + 16.0*I3(v1, i,j+1,k))* rdyidyi +
                       ((I3(v1, i,j,k+1)-I3(v1, i,j,k))*I1(dzc_inv, k)-(I3(v1, i,j,k)-I3(v1, i,j,k-1))*I1(dzc_inv, k-1))*I1(visc_dzf_inv, k);
                visc = visc*re_inv;
                I3(v, i,j,k) = (I3(v1, i,j,k) + dt*(visc - conv) + I3(v, i,j,k))*0.5;

                q1m3 = (I3(u1, i-2,j,k)*I1(dzf, k+1) + I3(u1, i-2,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1m1 = (I3(u1, i-1,j,k)*I1(dzf, k+1) + I3(u1, i-1,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p1 = (I3(u1, i  ,j,k)*I1(dzf, k+1) + I3(u1, i  ,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p3 = (I3(u1, i+1,j,k)*I1(dzf, k+1) + I3(u1, i+1,j,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q2m3 = I3(w1, i-3,j,k)+I3(w1, i  ,j,k);
                q2m1 = I3(w1, i-1,j,k)+I3(w1, i  ,j,k);
                q2p1 = I3(w1, i  ,j,k)+I3(w1, i+1,j,k);
                q2p3 = I3(w1, i  ,j,k)+I3(w1, i+3,j,k);
                duwdx = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dx_inv;
                // add a term induced by the convecting reference frame
                conv = u_crf * (r1*(q2p1-q2m1) + r22*(q2p3-q2m3))*0.5*dx_inv;
                q1m3 = (I3(v1, i,j-2,k)*I1(dzf, k+1) + I3(v1, i,j-2,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1m1 = (I3(v1, i,j-1,k)*I1(dzf, k+1) + I3(v1, i,j-1,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p1 = (I3(v1, i,j  ,k)*I1(dzf, k+1) + I3(v1, i,j  ,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q1p3 = (I3(v1, i,j+1,k)*I1(dzf, k+1) + I3(v1, i,j+1,k+1)*I1(dzf, k))*I1(dzc_inv, k);
                q2m3 = I3(w1, i,j-3,k)+I3(w1, i,j  ,k);
                q2m1 = I3(w1, i,j-1,k)+I3(w1, i,j  ,k);
                q2p1 = I3(w1, i,j  ,k)+I3(w1, i,j+1,k);
                q2p3 = I3(w1, i,j  ,k)+I3(w1, i,j+3,k);
                dvwdy = (r1*(q1p1*q2p1-q1m1*q2m1) + r22*(q1p3*q2p3-q1m3*q2m3))*0.25*dy_inv;
                q1m1 = (I3(w1, i,j,k)+I3(w1, i,j,k-1));
                q1p1 = (I3(w1, i,j,k)+I3(w1, i,j,k+1));
                q2m1 = q1m1;
                q2p1 = q1p1;
                dwwdz = (q1p1*q2p1-q1m1*q2m1)*0.25*I1(dzc_inv, k);
                conv = conv + duwdx + dvwdy + dwwdz;
                visc = (- I3(w1, i-2,j,k) + 16.0*I3(w1, i-1,j,k) - 30.0*I3(w1, i,j,k)
                        - I3(w1, i+2,j,k) + 16.0*I3(w1, i+1,j,k))* rdxidxi +
                       (- I3(w1, i,j-2,k) + 16.0*I3(w1, i,j-1,k) - 30.0*I3(w1, i,j,k)
                        - I3(w1, i,j+2,k) + 16.0*I3(w1, i,j+1,k))* rdyidyi +
                       ((I3(w1, i,j,k+1)-I3(w1, i,j,k))*I1(dzf_inv, k+1)-(I3(w1, i,j,k)-I3(w1, i,j,k-1))*I1(dzf_inv, k))*I1(dzc_inv, k);
                visc = visc*re_inv;
                I3(w, i,j,k) = (I3(w1, i,j,k) + dt*(visc - conv) + I3(w, i,j,k))*0.5;
            }
        }
    }

}

#ifdef GPTL
static char str_gptl[30];
#endif

typedef void (*rk_kernel_funptr)(
    int st[3], int en[3], double re_inv, double dt, double dx_inv, double dy_inv, 
    const double *dzf, const double *dzc_inv, const double *dzf_inv, const double *visc_dzf_inv, 
    const int *nhalo, const int *sz, const double *restrict u, const double *restrict v, const double *restrict w, 
    double *restrict unew, double *restrict vnew, double *restrict wnew, double u_crf);

void time_int_vel_rk(
    const int rk_num, const MPI_Fint comm_f, const MPI_Fint halotype_f[6], const int neighbor[6],
    double re_inv, double dt, double dx_inv, double dy_inv, 
    const double *dzf, const double *dzc_inv, const double *dzf_inv, const double *visc_dzf_inv,
    const int *nhalo, const int *sz, const double * restrict u, const double * restrict v, const double * restrict w, 
    double * restrict unew, double * restrict vnew, double * restrict wnew, double u_crf) {

    rk_kernel_funptr rk_kernel = NULL;
    if (rk_num == 1) {
        rk_kernel = time_int_vel_rk1_kernel;
    } else if (rk_num == 2) {
        rk_kernel = time_int_vel_rk2_kernel;
    }
    
#ifndef NB_HALO

    int st[3] = {1, 1, 1};
    int en[3] = {sz[0], sz[1], sz[2]};

#ifdef GPTL
    sprintf(str_gptl, "%s%d%s", "--uvw", rk_num, " comp");
    GPTLstart(str_gptl);
#endif

    rk_kernel(st, en, re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv,
              nhalo, sz, u, v, w, unew, vnew, wnew, u_crf);

#ifdef GPTL
    GPTLstop(str_gptl);
#endif

    char tag[2];
    sprintf(tag, "%d", rk_num);
    update_bound_vel(comm_f, halotype_f, neighbor, nhalo, sz, unew, vnew, wnew, u_crf, tag);

#else
    const int tag_u = 1, tag_v = 2, tag_w = 3;
    int st[3], en[3];
    MPI_Request isend_req_u[8], irecv_req_u[8];
    MPI_Request isend_req_v[8], irecv_req_v[8];
    MPI_Request isend_req_w[8], irecv_req_w[8];

    int y0_send_st = 1;
    int y0_send_en = nhalo[3];
    int y1_send_st = sz[1]+1-nhalo[2];
    int y1_send_en = sz[1];
    int z0_send_st = 1;
    int z0_send_en = nhalo[5];
    int z1_send_st = sz[2]+1-nhalo[4];
    int z1_send_en = sz[2];

#ifdef GPTL
    sprintf(str_gptl, "%s%d", "--Update halo vel ", rk_num);
    GPTLstart(str_gptl);
#endif

#ifndef USE_RDMA
    /* FGN: MARK */
    update_halo_irecv(nhalo, sz, tag_u, unew, irecv_req_u);
    update_halo_irecv(nhalo, sz, tag_v, vnew, irecv_req_v);
    update_halo_irecv(nhalo, sz, tag_w, wnew, irecv_req_w);
#endif

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%d%s", "--uvw", rk_num, " comp");
    GPTLstart(str_gptl);
#endif
    
    // *** bottom ***
    st[0] = 1; st[1] = 1; st[2] = z0_send_st;
    en[0] = sz[0]; en[1] = sz[1]; en[2] = z0_send_en;
    rk_kernel(st, en, re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, nhalo, sz, u, v, w, unew, vnew, wnew, u_crf);
    
    // *** top ***
    st[0] = 1; st[1] = 1; st[2] = MAX(z1_send_st,z0_send_en+1);
    en[0] = sz[0]; en[1] = sz[1]; en[2] = z1_send_en;
    rk_kernel(st, en, re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, nhalo, sz, u, v, w, unew, vnew, wnew, u_crf);

    // *** south ***
    st[0] = 1; st[1] = y0_send_st; st[2] = z0_send_en+1;
    en[0] = sz[0]; en[1] = y0_send_en; en[2] = z1_send_st-1;
    rk_kernel(st, en, re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, nhalo, sz, u, v, w, unew, vnew, wnew, u_crf);
    
    // *** north ***
    st[0] = 1; st[1] = MAX(y1_send_st,y0_send_en+1); st[2] = z0_send_en+1;
    en[0] = sz[0]; en[1] = y1_send_en; en[2] = z1_send_st-1;
    rk_kernel(st, en, re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, nhalo, sz, u, v, w, unew, vnew, wnew, u_crf);

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%d", "--Update halo vel ", rk_num);
    GPTLstart(str_gptl);
#endif

#ifdef USE_RDMA
    update_halo_rdma_send(rk_num-1);
#else
    update_halo_isend(nhalo, sz, tag_u, unew, isend_req_u);
    update_halo_isend(nhalo, sz, tag_v, vnew, isend_req_v);
    update_halo_isend(nhalo, sz, tag_w, wnew, isend_req_w);
#endif

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%d%s", "--uvw", rk_num, " comp");
    GPTLstart(str_gptl);
#endif
    
    // *** inner region ***
    st[0] = 1; st[1] = nhalo[3]+1; st[2] = nhalo[5]+1;
    en[0] = sz[0]; en[1] = sz[1]-nhalo[2]; en[2] = sz[2]-nhalo[4];
    rk_kernel(st, en, re_inv, dt, dx_inv, dy_inv, dzf, dzc_inv, dzf_inv, visc_dzf_inv, nhalo, sz, u, v, w, unew, vnew, wnew, u_crf);

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%d", "--Update halo vel ", rk_num);
    GPTLstart(str_gptl);
#endif

#ifdef USE_RDMA
    update_halo_rdma_wait(rk_num-1);
#else
    update_halo_waitall(isend_req_u, irecv_req_u);
    update_halo_waitall(isend_req_v, irecv_req_v);
    update_halo_waitall(isend_req_w, irecv_req_w);
#endif

#ifdef GPTL
    GPTLstop(str_gptl);
    sprintf(str_gptl, "%s%d", "--Impose BC vel ", rk_num);
    GPTLstart(str_gptl);
#endif
    
    impose_bc_vel(neighbor, nhalo, sz, unew, vnew, wnew, u_crf);

#ifdef GPTL
    GPTLstop(str_gptl);
#endif

#endif
}

void correct_vel(double dt, double dx_inv, double dy_inv, const double *dzc_inv, 
    const int *nhalo_p, const int *nhalo, const int *sz, const double *p, double *u, double *v, double *w)
{
    BIND1(nhalo_p)
    BIND1(nhalo)
    BIND1(sz)
    BIND1_EXT(dzc_inv, 1 - I1(nhalo,5))
    BIND3_EXT(u, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(v, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(w, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(p, I1(sz,1) + I1(nhalo_p,1) + I1(nhalo_p,2),
                 I1(sz,2) + I1(nhalo_p,3) + I1(nhalo_p,4),
                 1 - I1(nhalo_p,1), 1 - I1(nhalo_p,3), 1 - I1(nhalo_p,5))
    
    double dtdxi = dt*dx_inv;
    double dtdyi = dt*dy_inv;
    
    const int sz0 = sz[0], sz1 = sz[1], sz2 = sz[2];
#ifdef USE_OMP_OFFLOAD
    #pragma omp target teams distribute parallel for collapse(3)
#else
    #pragma omp parallel for
#endif
    DO(k, 1, sz2) {
        DO(j, 1, sz1) {
            DO(i, 1, sz0) {
                I3(u, i, j, k) -= (I3(p, i+1, j, k) - I3(p, i, j, k)) * dtdxi;
                I3(v, i, j, k) -= (I3(p, i, j+1, k) - I3(p, i, j, k)) * dtdyi;
                I3(w, i, j, k) -= (I3(p, i, j, k+1) - I3(p, i, j, k)) * dt * I1(dzc_inv, k);
            }
        }
    }
}

void force_vel(const _Bool *is_forced, const double *vel_force, int nx, int ny, const double *dzflzi,
               const int *nhalo, const int *sz, double *u, double *v, double *w)
{
    
    BIND1(nhalo)
    BIND1(sz)
    BIND1_EXT(dzflzi, 1 - I1(nhalo,5))
    BIND3_EXT(u, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(v, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))
    BIND3_EXT(w, I1(sz,1) + I1(nhalo,1) + I1(nhalo,2),
                 I1(sz,2) + I1(nhalo,3) + I1(nhalo,4),
                 1 - I1(nhalo,1), 1 - I1(nhalo,3), 1 - I1(nhalo,5))

    double vel_mean[3] = {0};
    double force[3] = {0};
    const int sz0 = sz[0], sz1 = sz[1], sz2 = sz[2];
    
    if (is_forced[0])
    {
        vel_mean[0] = mean_h(nhalo, sz, nx, ny, dzflzi, u);
        force[0] = vel_force[0] - vel_mean[0];
#ifdef USE_OMP_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3)
#else
        #pragma omp parallel for
#endif
        DO(k, 1, sz2) {
            DO(j, 1, sz1) {
                DO(i, 1, sz0) {
                    I3(u, i, j, k) += force[0];
                }
            }
        }
    }

    if (is_forced[1])
    {
        vel_mean[1] = mean_h(nhalo, sz, nx, ny, dzflzi, v);
        force[1] = vel_force[1] - vel_mean[1];
#ifdef USE_OMP_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3)
#else
        #pragma omp parallel for
#endif
        DO(k, 1, sz2) {
            DO(j, 1, sz1) {
                DO(i, 1, sz0) {
                    I3(v, i, j, k) += force[1];
                }
            }
        }
    }

    if (is_forced[2])
    {
        vel_mean[2] = mean_h(nhalo, sz, nx, ny, dzflzi, w);
        force[2] = vel_force[2] - vel_mean[2];
#ifdef USE_OMP_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3)
#else
        #pragma omp parallel for
#endif
        DO(k, 1, sz2) {
            DO(j, 1, sz1) {
                DO(i, 1, sz0) {
                    I3(w, i, j, k) += force[2];
                }
            }
        }
    }
    
}