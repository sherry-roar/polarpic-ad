#pragma once

#include <mpi.h>

double mean_h(const int *nhalo, const int *sz, int nx_global, int ny_global, const double *dzflzi, const double *var);