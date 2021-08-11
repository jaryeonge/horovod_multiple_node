horovodrun -np 4 -H host5:1,host6:1,host7:1,host8:1 python ./deep_learning/models/albert.py > ./albert_training.log 2>&1
