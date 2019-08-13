#!/bin/bash

set -e

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
PYPROF="$SCRIPTPATH/../.."

parse="python $PYPROF/parse/parse.py"
prof="python $PYPROF/prof/prof.py"

for f in *.py
do
	base=`basename $f .py`
	sql=$base.sql
	dict=$base.dict

	#NVprof
	echo "nvprof -fo --profile-from-start off $sql python $f"
	nvprof -fo $sql --profile-from-start off python $f

	#Parse
	echo $parse $sql
	$parse $sql > $dict

	#Prof
	echo $prof $dict
	#$prof -w 130 $dict
	$prof --csv -c idx,layer,dir,mod,op,kernel,params,sil $dict
	\rm $sql $dict
done
