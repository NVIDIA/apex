#!/bin/bash

OUTDIR="results/ablation_logs"
mkdir -p $OUTDIR

R1000=random,1000
CS=channel_swap,0
CS_100=channel_swap,100
CS_1000=channel_swap,1000
OSG2=optimize_stripe_groups,8,0
OSG2_100=optimize_stripe_groups,8,100
OSG2_1000=optimize_stripe_groups,8,1000
OSG3=optimize_stripe_groups,12,0
OSG3_100=optimize_stripe_groups,12,100
OSG3_1000=optimize_stripe_groups,12,1000
optimal=optimize_stripe_groups,16,0

# Table 1
for seed in {0..24}; do
    echo $seed
    python3 permutation_test.py --channels 16 --filters 32 --seed $seed --pretty_print=False $R1000 $CS $CS_100 $CS_1000 $OSG2 $OSG2_100 $OSG2_1000 $OSG3 $OSG3_100 $OSG3_1000 $optimal | tee "${OUTDIR}/ablations_32x16_$seed.log"
    python3 permutation_test.py --channels 128 --filters 64 --seed $seed --pretty_print=False $R1000 $CS $CS_100 $CS_1000 $OSG2 $OSG2_100 $OSG2_1000 $OSG3 $OSG3_100 $OSG3_1000 | tee "${OUTDIR}/ablations_64x128_$seed.log"
done

echo "Gathering results ..."

################# collect results into a .csv file
# get mean and stddev of efficacy from all seeds for one strategy
get_mean_stddev() {
    local strategy=$1
    local OUTFILE=$2

    # get the strategy's line,                           pull out efficacy and time,              use sum-of-squares to compute stddev and mean in a single pass
    grep "$strategy," $OUTDIR/ablations_64x128_*.log | awk -F "," '{print $3,$4}' | awk '{sum += $1; sumsq += ($1)^2; timesum += $2} END {printf "%.1f,%.1f,%.2f,", sum/NR, sqrt((sumsq-sum^2/NR)/NR), timesum/NR}' >> $OUTFILE
}

# get the number of times some strategy matched the optimal solution
get_num_optimal() {
    local strategy=$1
    local OUTFILE=$2

    matches=0
    for seed in {0..24}; do
        # compare floats with epsilon: add one thousandth to the efficacy under test
        this_eff=$(grep "$strategy," "${OUTDIR}/ablations_32x16_${seed}.log" | awk -F "," '{print int($3 * 1000 + 1)}')
        best_eff=$(grep "optimize_stripe_groups_16_0," "${OUTDIR}/ablations_32x16_${seed}.log" | awk -F "," '{print int($3 * 1000)}')
        if [ "$this_eff" -ge "$best_eff" ]; then
            let "matches = $matches + 1"
        fi
    done

    printf "$matches," >> $OUTFILE
}

# populate a row of the ablation study table
populate_row() {
    local greedy=$1
    local escape=$2
    local strategy=$(echo "$3" | sed 's/,/_/g')
    local OUTFILE=$4

    printf "$greedy,$escape," >> $OUTFILE
    get_mean_stddev "$strategy" "$OUTFILE"
    printf "," >> $OUTFILE
    get_num_optimal "$strategy" "$OUTFILE"
    printf "\n" >> $OUTFILE
}

# prepare output file header
OUTFILE="results/ablation_studies.csv"
printf ",,25x 64x128,,,,25x 32x16\n" > $OUTFILE
printf ",,Efficacy,,Runtime,,Optimal\n" >> $OUTFILE
printf "Greedy Phase,Escape Phase,Mean,StdDev,Mean,,# Found\n" >> $OUTFILE

# finally, gather the data for each strategy into a row of the table
populate_row "Random 1000" "-" "$R1000" "$OUTFILE"
populate_row "Channel Swap" "-" "$CS" "$OUTFILE"
populate_row "Channel Swap" "BR(100)" "$CS_100" "$OUTFILE"
populate_row "Channel Swap" "BR(1000)" "$CS_1000" "$OUTFILE"
populate_row "OSG(2)" "-" "$OSG2" "$OUTFILE"
populate_row "OSG(2)" "BR(100)" "$OSG2_100" "$OUTFILE"
populate_row "OSG(2)" "BR(1000)" "$OSG2_1000" "$OUTFILE"
populate_row "OSG(3)" "-" "$OSG3" "$OUTFILE"
populate_row "OSG(3)" "BR(100)" "$OSG3_100" "$OUTFILE"
populate_row "OSG(3)" "BR(1000)" "$OSG3_1000" "$OUTFILE"

echo "Done! $OUTFILE"
