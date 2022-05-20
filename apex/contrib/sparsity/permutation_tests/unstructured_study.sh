#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Please specify both the source directory and a run tag: bash unstructured_study.sh <directory> <tag>"
  exit
fi

dir=$1  # or set to the directory containing .npy files of interest
tag=$2 # or set to an identifier, e.g. "network_name"

resdir="results/unstructured_logs/${tag}"
mkdir -p $resdir

CS=channel_swap,0
OSG2=optimize_stripe_groups,8,0
OSG2_100=optimize_stripe_groups,8,100
OSG2_1000=optimize_stripe_groups,8,1000
OSG3=optimize_stripe_groups,12,0

CS_successes=()
OSG2_successes=()
OSG2_100_successes=()
OSG2_1000_successes=()
OSG3_successes=()

for sparsity in {50..100}; do
    CS_successes+=(0)
    OSG2_successes+=(0)
    OSG2_100_successes+=(0)
    OSG2_1000_successes+=(0)
    OSG3_successes+=(0)
done

update_successes () {
    strategy=$1
    local -n _successes=$2
    logfile=$3

    limit=$(grep "${strategy}," $logfile | awk -F "," '{print $3}')
 
    echo $logfile, $strategy, $limit
    for (( sparsity=$limit; sparsity<=100; sparsity++ )); do
        let "entry = $sparsity - 50"
        let "value = ${_successes[$entry]} + 1"
        _successes[$entry]=$value
    done
}

# Figure 4
for filename in $dir/*.npy; do
    out=$(basename -- "$filename")
    echo "Searching for minimum sparsities for $out"
    out=$resdir/$out.unstructured
    python3 permutation_test.py --infile=$filename --pretty_print=False --unstructured=-1 $CS $OSG2 $OSG2_100 $OSG2_1000 $OSG3 > $out

    update_successes "channel_swap_0" CS_successes "$out"
    update_successes "optimize_stripe_groups_8_0" OSG2_successes "$out"
    update_successes "optimize_stripe_groups_8_100" OSG2_100_successes "$out"
    update_successes "optimize_stripe_groups_8_1000" OSG2_1000_successes "$out"
    update_successes "optimize_stripe_groups_12_0" OSG3_successes "$out"
done

#################### save the table
# log a single strategy in as a row in the table
log_success () {
    strategy=$1
    local -n _successes=$2
    OUTFILE=$3

    printf "$strategy," >> $OUTFILE
    for sparsity in {50..100}; do
        let "entry = $sparsity - 50"
        printf "%d," ${_successes[$entry]} >> $OUTFILE
    done
    printf "\n" >> $OUTFILE
}

# prepare the header
OUTFILE="results/unstructured.csv"
printf "Sparsity," > $OUTFILE
for sparsity in {50..100}; do
    printf "%d," $sparsity >> $OUTFILE
done
printf "\n" >> $OUTFILE

# add data for each strategy
log_success "channel_swap_0" CS_successes "$OUTFILE"
log_success "optimize_stripe_groups_8_0" OSG2_successes "$OUTFILE"
log_success "optimize_stripe_groups_8_100" OSG2_100_successes "$OUTFILE"
log_success "optimize_stripe_groups_8_1000" OSG2_1000_successes "$OUTFILE"
log_success "optimize_stripe_groups_12_0" OSG3_successes "$OUTFILE"

echo "Done! ${OUTFILE}"
