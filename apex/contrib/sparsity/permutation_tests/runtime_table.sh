#!/bin/bash

OUTDIR="results/runtime_logs"
mkdir -p $OUTDIR

R1000=random,1000
CS=channel_swap,0
CS_100=channel_swap,100
OSG2=optimize_stripe_groups,8,0
OSG2_100=optimize_stripe_groups,8,100
OSG2_1000=optimize_stripe_groups,8,1000
OSG3=optimize_stripe_groups,12,0
OSG3_100=optimize_stripe_groups,12,100
OSG3_1000=optimize_stripe_groups,12,1000

for cols in "32" "64" "128" "256"; do
    echo "$cols x $cols"
    python3 permutation_test.py --channels $cols --filters $cols --pretty_print=False $R1000 $CS $CS_100 $OSG2 $OSG2_100 $OSG2_1000 $OSG3 $OSG3_100 $OSG3_1000 | tee "${OUTDIR}/runtime_${cols}x${cols}.log"
    let "rows = $cols * 2"
    echo "$cols x $rows"
    python3 permutation_test.py --channels $cols --filters $rows --pretty_print=False $R1000 $CS $CS_100 $OSG2 $OSG2_100 $OSG2_1000 $OSG3 $OSG3_100 $OSG3_1000 | tee "${OUTDIR}/runtime_${cols}x${rows}.log"
done

# 2048x2048 is too large for OSG3
echo "2048 x 2048"
python3 permutation_test.py --channels 2048 --filters 2048 --pretty_print=False $R1000 $CS $CS_100 $OSG2 $OSG2_100 $OSG2_1000 | tee "${OUTDIR}/runtime_2048x2048.log"


############### collect results into a .csv file
echo "Gathering results ..."

# efficacy and runtime from one strategy and size
get_results() {
    local strategy=$1
    local cols=$2
    local rows=$3
    local OUTFILE=$4

    grep "$strategy," "$OUTDIR/runtime_${cols}x${rows}.log" | awk -F "," '{printf "%s,%s,",$3,$4}' >> $OUTFILE
}

# prepare output file headers
OUTFILE="results/runtimes.csv"
printf "Columns," > $OUTFILE
for cols in "32" "64" "128" "256"; do
    printf "$cols,$cols,$cols,$cols," >> $OUTFILE
done
printf "2048,2048\n" >> $OUTFILE

printf "Rows," >> $OUTFILE
for cols in "32" "64" "128" "256"; do
    let "rows = $cols * 2"
    printf "$cols,$cols,$rows,$rows," >> $OUTFILE
done
printf "2048,2048\n" >> $OUTFILE

printf "Metric," >> $OUTFILE
for cols in "32" "64" "128" "256"; do
    printf "Efficacy,Runtime,Efficay,Runtime," >> $OUTFILE
done
printf "Efficacy,Runtime\n" >> $OUTFILE

# gather data in a reasonable order
for strategy in "$R1000" "$CS" "$CS_100" "$OSG2" "$OSG2_100" "$OSG2_1000" "$OSG3" "$OSG3_100" "$OSG3_1000"; do
    strategy=$(echo "$strategy" | sed 's/,/_/g') # replace commas with underscores, as they'll appear in the results logs
    printf "$strategy," >> $OUTFILE
    for cols in "32" "64" "128" "256"; do
        get_results "$strategy" "$cols" "$cols" "$OUTFILE"
        let "rows = $cols * 2"
        get_results "$strategy" "$cols" "$rows" "$OUTFILE"
    done

    get_results "$strategy" "2048" "2048" "$OUTFILE"

    printf "\n" >> $OUTFILE
done

echo "Done! $OUTFILE"
