./trainBird_Tensor.sh 2>&1 | tee  trainBird_Tensor.log
./trainBird_TensorAll.sh 2>&1 | tee  trainBird_TensorAll.log



mkdir logBackup
cp *.log ./logBackup

rm *solverstate
