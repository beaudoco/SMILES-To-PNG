#!/bin/bash

echo "Run QGAN with heterogenous learning rates"




python3 framework_cv2.py -c 0.01 -q 0.01
echo "Finish 12"

#python3 1qnn_1dense.py -c 0.01 -q 0.001
#echo "Finish 13"

#python3 1qnn_1dense.py -c 0.001 -q 0.01

#python3 1qnn_1dense.py -c 0.001 -q 0.1
#echo "Finish 14"




echo "Experiments done"
