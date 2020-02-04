#!/bin/bash

set -e

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
PYPROF="$SCRIPTPATH/../.."

parse="python $PYPROF/parse/parse.py"
prof="python $PYPROF/prof/prof.py"

for net in "resnet50"
do
	for optim in adam sgd
	do
		for batch in 32 64
		do
			base="torchvision".$net.$optim.$batch
			sql=$base.sql
			dict=$base.dict

			#NVprof
			echo "nvprof -fo $sql --profile-from-start off python imagenet.py -m ${net} -o $optim -b $batch"
			sudo env "PATH=$PATH" nvprof -fo $sql --profile-from-start off python imagenet.py -m ${net} -o $optim -b $batch

			#Parse
			echo $parse $sql
			$parse $sql > $dict

			#Prof
			echo $prof $dict
			$prof -w 130 $dict
#			\rm $sql $dict
		done
	done
done
