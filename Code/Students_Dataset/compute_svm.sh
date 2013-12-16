#!/bin/bash
libsvm-3.17/svm-train -t 2 train_data0 > /dev/null
libsvm-3.17/svm-predict test_data0 train_data0.model output0
