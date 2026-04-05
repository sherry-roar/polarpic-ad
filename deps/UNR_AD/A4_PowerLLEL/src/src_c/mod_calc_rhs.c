#include "bind.h"
#include <omp.h>


void calc_rhs(double dt, double dx_inv, double dy_inv, double *dzf_inv,
              int *nhalo, int *nhalo_rhs, int *sz, double *u, double *v,
              double *w, double *rhs) {

  double dti = 1.0 / dt;
  double dtidxi = dti * dx_inv;
  double dtidyi = dti * dy_inv;

  BIND1(nhalo)
  BIND1(nhalo_rhs)
  BIND1(sz)
  BIND1_EXT(dzf_inv, 1 - I1(nhalo, 5))
  BIND3_EXT(u, I1(sz, 1) + I1(nhalo, 1) + I1(nhalo, 2),
            I1(sz, 2) + I1(nhalo, 3) + I1(nhalo, 4), 1 - I1(nhalo, 1),
            1 - I1(nhalo, 3), 1 - I1(nhalo, 5))
  BIND3_EXT(v, I1(sz, 1) + I1(nhalo, 1) + I1(nhalo, 2),
            I1(sz, 2) + I1(nhalo, 3) + I1(nhalo, 4), 1 - I1(nhalo, 1),
            1 - I1(nhalo, 3), 1 - I1(nhalo, 5))
  BIND3_EXT(w, I1(sz, 1) + I1(nhalo, 1) + I1(nhalo, 2),
            I1(sz, 2) + I1(nhalo, 3) + I1(nhalo, 4), 1 - I1(nhalo, 1),
            1 - I1(nhalo, 3), 1 - I1(nhalo, 5))
  BIND3_EXT(rhs, I1(sz, 1) + I1(nhalo_rhs, 1) + I1(nhalo_rhs, 2),
            I1(sz, 2) + I1(nhalo_rhs, 3) + I1(nhalo_rhs, 4),
            1 - I1(nhalo_rhs, 1), 1 - I1(nhalo_rhs, 3), 1 - I1(nhalo_rhs, 5))

  const int sz0 = sz[0], sz1 = sz[1], sz2 = sz[2];
#ifdef USE_OMP_OFFLOAD
  #pragma omp target teams distribute parallel for collapse(3)
#else
  #pragma omp parallel for
#endif
  DO(k, 1, sz2) {
    DO(j, 1, sz1) {
      DO(i, 1, sz0) {
        I3(rhs, i, j, k) =
          (I3(u, i, j, k) - I3(u, i - 1, j, k)) * dtidxi +
          (I3(v, i, j, k) - I3(v, i, j - 1, k)) * dtidyi +
          (I3(w, i, j, k) - I3(w, i, j, k - 1)) * dti * I1(dzf_inv, k);
      }
    }
  }
}