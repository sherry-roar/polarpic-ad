#!/bin/bash
#SBATCH --partition=nsccgz_yjsb
#SBATCH --nodes=993
#SBATCH --job-name="fgn_PowerLLEL"
#SBATCH --nodelist=cnode[2305-2457,2486-2533,2535-2627,2755-2809,2811-2995,2997-3247,3316-3379,3381-3437,3462-3506]

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
