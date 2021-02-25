# Program : run_parafacnn.sh
# Description : Perform nonnegative PARAFAC tensor factorization on input tensor.

which hadoop > /dev/null
status=$?
if test $status -ne 0 ; then
	echo ""
	echo "Hadoop is not installed in the system."
	echo "Please install Hadoop and make sure the hadoop binary is accessible."
	exit 127
fi

if [ $# -ne 6 ]; then
    echo 1>&2 "Usage: $0 [dim_1:..:dim_N (tensor)] [rank] [# of reducers] [max iteration] [tensor path] [output path]"
    exit 127
fi

cd $(dirname $(readlink -f $0))

hadoop jar ./bigtensor.jar bigtensor.Decompositions.PARAFACNN $1 $2 $3 $4 $5 $6
