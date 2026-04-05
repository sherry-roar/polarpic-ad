#!/bin/bash
#SBATCH --partition=bigdata
#SBATCH --nodes=96
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --job-name="PowerLLEL"

if [[ $CALL_BY_RUN_SH != "1" ]]
then
    echo "This script can only be called by run.sh"
fi

# cd $RUN_DIR/mpi
# bash ./batch.sh

# cd $RUN_DIR/mpiomp
# bash ./batch.sh

# cd $RUN_DIR/mpi.pdd
# bash ./batch.sh

cd $RUN_DIR/mpiomp.pdd
bash ./batch.sh
