#!/bin/bash

# $1: test path
# $2: executable
# $3: array of case names
# $4: test rounds
# $5: nnodes (-N)
# $6: ntasks (-n)
# $7: cpu per tasks (-c)
run_test_case(){
cases_arr=$3

for icase in ${cases_arr[*]}
do
  cd $1/$icase

  for ((i=1; i<=$4; i++))
  do
    echo ">>> Start  test "$2"-"$icase" Round "$i": -N "$5" -n "$6" -c "$7
    OMP_NUM_THREADS=$7 srun -N $5 -n $6 -c $7 $2 2>&1 | tee run.$i.log
    if [ -f "timing.summary" ];then
      mv timing.summary timing.summary.$i
    fi
    echo ">>> Finish test "$2"-"$icase" Round "$i"!"
  done

done
}

base=$(pwd)
exe=../../bin/PowerLLEL_cpu.mpiomp
nrounds=3

# 4320 Nodes, 103680 Cores
cases=(120x72)
run_test_case $base $exe "${cases[*]}" $nrounds 4320 8640 12

# 3456 Nodes, 82944 Cores
cases=(96x72)
run_test_case $base $exe "${cases[*]}" $nrounds 3456 6912 12

# 1728 Nodes, 41472 Cores
cases=(48x72)
run_test_case $base $exe "${cases[*]}" $nrounds 1728 3456 12

# 864 Nodes, 20736 Cores
cases=(24x72)
run_test_case $base $exe "${cases[*]}" $nrounds 864 1728 12

nrounds=3

# 576 Nodes, 13824 Cores
cases=(16x72)
run_test_case $base $exe "${cases[*]}" $nrounds 576 1152 12

# 432 Nodes, 10368 Cores
cases=(12x72)
run_test_case $base $exe "${cases[*]}" $nrounds 432 864 12

# 288 Nodes, 6912 Cores
cases=(8x72)
run_test_case $base $exe "${cases[*]}" $nrounds 288 576 12
