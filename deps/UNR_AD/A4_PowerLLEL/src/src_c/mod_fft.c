#include "mod_fft.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_MKL
#include "mod_fft_mkl.inc"
#else

#ifdef USE_KMLFFT
#include "mod_fft_kmlfft.inc"
#else
#include "mod_fft_fftw.inc"
#endif

#endif
