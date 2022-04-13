#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $PROJPATH/.env.sh
cd $PROJPATH

export OMP_NUM_THREADS=2
NCPU=$(nproc --all)
NCPU=$((NCPU/2))
JOBLOGDIR=$PROJPATH/joblogs
mkdir -p $JOBLOGDIR

echo "* Running ${NCPU} processes in parallel ($(nproc --all) CPUs were detected)"
echo "  -> You can modify the size of the parallel pool of processes "
echo "     by changing the variable 'NCPU' in this script (${BASH_SOURCE[0]})."

################################################################################
######################## Checking if the features exist ########################
################################################################################
MISSINGFEAT=""
for DS in "CUB" "cifar" "tieredImagenet" "miniImagenet"; do
  for DT in "novel" "val"; do
    if [[ ! -f "./features/${DS}/WideResNet28_10_S2M2_R/${DT}.hdf5" ]]; then
      MISSINGFEAT="./features/${DS}_${DT}_${BB}.pth"
    fi
  done
done

if [[ ${MISSINGFEAT} != "" ]]; then
  echo "* Features file ${MISSINGFEAT} is missing. Running ./features/download.sh ..."
  ./features/download.sh
fi

################################################################################
######################### Running the configs in loops #########################
################################################################################

for PARTIDX in $(seq 0 5); do
  echo "================================================================================"
  echo "* Working on ./configs/1_mini_co_part${PARTIDX}.json"
  
  for RANK in $(seq 0 $((NCPU-1))); do
    echo -n "  + python main_firth.py --proc_rank ${RANK} --proc_size ${NCPU} --configid "
    echo "1_mini_co_part${PARTIDX} > ${JOBLOGDIR}/mini_part${PARTIDX}.out 2>&1 &"
    python main_firth.py --proc_rank ${RANK} --proc_size ${NCPU} \
      --configid 1_mini_co_part${PARTIDX} > ${JOBLOGDIR}/mini_part${PARTIDX}.out 2>&1 &
  done
  echo "  + wait"
  sleep 10
  wait
done

for PARTIDX in $(seq 0 5); do
  echo "================================================================================"
  echo "* Working on ./configs/2_cifar_co_part${PARTIDX}.json"
  
  for RANK in $(seq 0 $((NCPU-1))); do
    echo -n "  + python main_firth.py --proc_rank ${RANK} --proc_size ${NCPU} --configid "
    echo "2_cifar_co_part${PARTIDX} > ${JOBLOGDIR}/cifar_part${PARTIDX}.out 2>&1 &"
    python main_firth.py --proc_rank ${RANK} --proc_size ${NCPU} \
      --configid 2_cifar_co_part${PARTIDX} > ${JOBLOGDIR}/cifar_part${PARTIDX}.out 2>&1 &
  done
  echo "  + wait"
  wait
done

for PARTIDX in $(seq 0 10); do
  echo "================================================================================"
  echo "* Working on ./configs/3_tiered_co_part${PARTIDX}.json"
  
  for RANK in $(seq 0 $((NCPU-1))); do
    echo -n "  + python main_firth.py --proc_rank ${RANK} --proc_size ${NCPU} --configid "
    echo "3_tiered_co_part${PARTIDX} > ${JOBLOGDIR}/tiered_part${PARTIDX}_r${RANK}.out 2>&1 &"
    python main_firth.py --proc_rank ${RANK} --proc_size ${NCPU} --configid \
      3_tiered_co_part${PARTIDX} > ${JOBLOGDIR}/tiered_part${PARTIDX}.out 2>&1 &
  done
  echo "  + wait"
  wait
done
