#!/bin/bash
#SBATCH --partition=nsccgz_yjsb
#SBATCH --nodes=1728
#SBATCH --job-name="fgn_PowerLLEL"
#SBATCH --nodelist=cnode[1536-1631,1634-1791,1793-1893,1921-2025,2050-2303,2305-2425,2427-2457,2486-2533,2535-2627,2711-2738,2755-2809,2811-2995,2997-3247,3316-3379,3381-3437,3462-3506,3548-3583]

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
