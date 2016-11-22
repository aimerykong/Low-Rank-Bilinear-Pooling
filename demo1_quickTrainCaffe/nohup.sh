./trainBird.sh 2>&1 | tee  trainBird.log
./trainBird_All.sh 2>&1 | tee  trainBird_All.log



mkdir logBackup
cp *.log ./logBackup

rm *solverstate
