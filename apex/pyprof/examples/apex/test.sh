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
	echo "nvprof -fo $sql python $f"
	nvprof -fo $sql python $f

	#Parse
	echo $parse $sql
	$parse $sql > $dict

	#Prof
	echo $prof $dict
	$prof -w 130 $dict
	\rm $sql $dict
done
