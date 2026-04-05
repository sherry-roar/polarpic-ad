#include "mod_utils.h"
#include "bind.h"

double mean_h(const int *nhalo, const int *sz, int nx_global, int ny_global,
              const double *dzflzi, const double *var) {
  BIND1(nhalo)
  BIND1(sz)
  BIND1_EXT(dzflzi, 0)
  BIND3_EXT(var, I1(sz, 1) + I1(nhalo, 2) + I1(nhalo, 1),
            I1(sz, 2) + I1(nhalo, 4) + I1(nhalo, 3), 1 - I1(nhalo, 1),
            1 - I1(nhalo, 3), 1 - I1(nhalo, 5))

  double res = 0;
  const int sz0 = sz[0], sz1 = sz[1], sz2 = sz[2];
#ifdef USE_OMP_OFFLOAD
  #pragma omp target teams distribute parallel for collapse(3) reduction(+:res) map(res)
#else
  #pragma omp parallel for reduction(+:res)
#endif
  DO(k, 1, sz2) {
    DO(j, 1, sz1) {
      DO(i, 1, sz0) {
        res += I3(var, i, j, k) * I1(dzflzi, k);
      }
    }
  }

  int ierr =
    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  res /= nx_global * ny_global;

  return res;
}