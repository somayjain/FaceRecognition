#!/bin/bash
libsvm-3.17/svm-train -t 0 train_data0 > /dev/null
libsvm-3.17/svm-train -t 0 train_data1 > /dev/null
libsvm-3.17/svm-train -t 0 train_data2 > /dev/null
libsvm-3.17/svm-train -t 0 train_data3 > /dev/null
libsvm-3.17/svm-predict test_data0 train_data0.model output0
libsvm-3.17/svm-predict test_data1 train_data1.model output1
libsvm-3.17/svm-predict test_data2 train_data2.model output2
libsvm-3.17/svm-predict test_data3 train_data3.model output3
