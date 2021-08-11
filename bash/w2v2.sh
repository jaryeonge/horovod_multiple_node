horovodrun -np 4 -H host1:1,host2:1,host3:1,host4:1 python ./deep_learning/models/w2v2.py > ./w2v2_training.log 2>&1
