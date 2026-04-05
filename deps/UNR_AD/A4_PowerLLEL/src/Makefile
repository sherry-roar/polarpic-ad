#==========================#
#  Makefile for PowerLLEL  #
#==========================#
# Choose compilation options as needed
# Global:
#   -D_SINGLE_PREC          use single-precision
#   -mp                     enable OpenMP parallel
# Poisson solver:
#   -D_PDD                  use PDD algorithm to solve tridiagonal systems in parallel
#   -D_PRECALC_TRID_COEFF   pre-calculate tridiagonal coefficients array,
#                           faster but requires more memory

NVHPC_PATH = /GPUFS/sysu_hpcedu_302/xjb/software/nvhpc-21.9/Linux_x86_64/21.9
FFTW_PATH = /GPUFS/sysu_hpcedu_302/xjb/software/fftw-3.3.8-nvhpc-21.9
ZLIB_PATH = /GPUFS/sysu_hpcedu_302/xjb/software/zlib-1.2.11-nvhpc-21.9
HDF5_PATH = /GPUFS/sysu_hpcedu_302/xjb/software/hdf5-1.10.4-nvhpc-21.9
GPTL_PATH = /GPUFS/sysu_hpcedu_302/xjb/software/gptl-5.5.3-nvhpc-21.9

USE_CUDA = true
USE_CUDA_AWARE_MPI = true
USE_PIPELINE_TRANS = true
USE_PROF_GPTL = true
USE_PROF_NVTX = 
STATIC_LINK = true

FC = mpixfort
#FC = $(NVHPC_PATH)/comm_libs/mpi/bin/mpifort
FFLAGS = -fast -cpp
FFLAGS_CUDA = -cuda -gpu=cc70 -cudalib=cufft
FFLAGS_DECOMP2D = $(FFLAGS)# -DEVEN

###############################################################################

USE_CUDA ?= false
USE_CUDA_AWARE_MPI ?= false
USE_PIPELINE_TRANS ?= false
USE_PROF_GPTL ?= false
USE_PROF_NVTX ?= false
STATIC_LINK ?= false

ifeq ($(USE_CUDA),true)
  FFLAGS+=$(FFLAGS_CUDA)
endif
ifeq ($(USE_CUDA_AWARE_MPI),true)
  FFLAGS+= -DCUDA_AWARE_MPI
endif
ifeq ($(USE_PIPELINE_TRANS),true)
  FFLAGS_DECOMP2D+= -DPIPELINE_TRANS
endif
ifeq ($(USE_PROF_GPTL),true)
  FFLAGS+= -DGPTL
endif
ifeq ($(USE_PROF_NVTX),true)
  FFLAGS+= -DNVTX
endif
FFLAGS+= -I$(HDF5_PATH)/include

ifeq ($(STATIC_LINK),true)
  LIBS = -L$(FFTW_PATH)/lib -Wl,-Bstatic -lfftw3_omp -lfftw3
  LIBS+= -L$(HDF5_PATH)/lib -Wl,-Bstatic -lhdf5_fortran -lhdf5 -L$(ZLIB_PATH)/lib -lz
  ifeq ($(USE_PROF_GPTL),true)
    LIBS+= -L$(GPTL_PATH)/lib -Wl,-Bstatic -lgptl
  endif
else 
  LIBS = -L$(FFTW_PATH)/lib -lfftw3_omp -lfftw3
  LIBS+= -L$(HDF5_PATH)/lib -lhdf5_fortran -lhdf5
  ifeq ($(USE_PROF_GPTL),true)
    LIBS+= -L$(GPTL_PATH)/lib -lgptl
  endif
endif
ifeq ($(USE_PROF_NVTX),true)
  LIBS+= -L$(NVHPC_PATH)/cuda/lib64 -Wl,-Bdynamic -lnvToolsExt
else
  LIBS+= -Wl,-Bdynamic
endif

OBJ = main.o \
      mod_type.o \
      mod_parameters.o \
      mod_mpi.o \
      mod_variables.o \
      mod_mesh.o \
      mod_poissonSolver.o \
      mod_fft.o \
      mod_initFlow.o \
      mod_updateBound.o \
      mod_calcVel.o \
      mod_calcRHS.o \
      mod_monitor.o \
      mod_dataIO.o \
      mod_hdf5.o \
      mod_statistics.o \
      mod_utils.o \
      decomp_2d.o
ifeq ($(USE_CUDA),true)
  OBJ+= mod_device.o
endif
ifeq ($(USE_PROF_NVTX),true)
  OBJ+= nvtx.o
endif

ifeq ($(USE_CUDA),true)
TARGET = PowerLLEL_gpu
else
TARGET = PowerLLEL_cpu
endif

all: $(TARGET)

$(TARGET): $(OBJ)
    ifeq ($(STATIC_LINK),true)
		@echo "Static link!"
    else
		@echo "Dynamic link!"
    endif
	$(FC) $(FFLAGS) $^ -o $@ $(LIBS)

main.o: main.f90 mod_type.o mod_parameters.o mod_mpi.o mod_variables.o mod_mesh.o \
        mod_poissonSolver.o mod_initFlow.o mod_updateBound.o mod_calcVel.o mod_calcRHS.o \
        mod_monitor.o mod_dataIO.o mod_hdf5.o mod_statistics.o mod_utils.o mod_device.o nvtx.o
	$(FC) $(FFLAGS) -I$(GPTL_PATH)/include -c $< 
mod_type.o: mod_type.f90
	$(FC) $(FFLAGS) -c $<
mod_parameters.o: mod_parameters.f90 mod_type.o mod_mpi.o
	$(FC) $(FFLAGS) -c $<
mod_mpi.o: mod_mpi.f90 decomp_2d.o mod_device.o
	$(FC) $(FFLAGS) -c $<
mod_variables.o: mod_variables.f90 mod_type.o mod_utils.o mod_device.o
	$(FC) $(FFLAGS) -c $<
mod_mesh.o: mod_mesh.f90 mod_type.o mod_parameters.o mod_mpi.o mod_hdf5.o mod_device.o
	$(FC) $(FFLAGS) -c $<
mod_poissonSolver.o: mod_poissonSolver.f90 mod_type.o mod_mpi.o decomp_2d.o mod_fft.o mod_utils.o
	$(FC) $(FFLAGS) -I$(GPTL_PATH)/include -c $<
mod_fft.o: mod_fft.f90 mod_type.o
	$(FC) $(FFLAGS) -c $< -I$(FFTW_PATH)/include
mod_initFlow.o: mod_initFlow.f90 mod_type.o mod_parameters.o mod_mpi.o mod_mesh.o mod_utils.o
	$(FC) $(FFLAGS) -c $<
mod_updateBound.o: mod_updateBound.f90 mod_type.o mod_parameters.o mod_mpi.o mod_mesh.o
	$(FC) $(FFLAGS) -I$(GPTL_PATH)/include -c $<
mod_calcVel.o: mod_calcVel.f90 mod_type.o mod_parameters.o mod_mpi.o mod_mesh.o mod_utils.o
	$(FC) $(FFLAGS) -I$(GPTL_PATH)/include -c $<
mod_calcRHS.o: mod_calcRHS.f90 mod_type.o mod_parameters.o mod_mpi.o mod_mesh.o
	$(FC) $(FFLAGS) -c $<
mod_monitor.o: mod_monitor.f90 mod_type.o mod_parameters.o mod_mpi.o mod_mesh.o mod_utils.o mod_hdf5.o
	$(FC) $(FFLAGS) -c $<
mod_dataIO.o: mod_dataIO.f90 mod_type.o mod_parameters.o mod_mpi.o mod_monitor.o mod_statistics.o mod_hdf5.o
	$(FC) $(FFLAGS) -c $< -o $@
mod_hdf5.o: mod_hdf5.f90 mod_type.o
	$(FC) $(FFLAGS) -c $<
mod_statistics.o: mod_statistics.f90 mod_type.o mod_parameters.o mod_mpi.o mod_utils.o
	$(FC) $(FFLAGS) -c $< -o $@ -I$(FFTW_PATH)/include
mod_utils.o: mod_utils.f90 mod_type.o mod_mpi.o
	$(FC) $(FFLAGS) -c $<
decomp_2d.o: decomp2d/decomp_2d.f90
	$(FC) $(FFLAGS_DECOMP2D) -I$(GPTL_PATH)/include -c $< 
mod_device.o: mod_device.f90
	$(FC) $(FFLAGS) -c $<
nvtx.o: nvtx.f90
	$(FC) $(FFLAGS) -c $<
.PHONY: clean
clean:
	rm -rf *.o *.mod $(TARGET)
