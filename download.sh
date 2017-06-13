#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2
#bunzip2 real-sim.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2
#bunzip2 ijcnn1.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.bz2
bunzip2 SUSY.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
#bunzip2 epsilon_normalized.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
#bunzip2 covtype.libsvm.binary.scale.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
#bunzip2 rcv1_train.binary.bz2
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
D=./data
O=./output
L=./logistic
for dir in "$D" "$O"
do 
   if [ ! -d "$dir" ]; then
      mkdir $dir
   fi
done

mv real-sim data
mv skin_nonskin data
mv phishing data
mv ijcnn1 data
mv SUSY data
mv epsilon_normalized data
mv covtype.libsvm.binary.scale data
mv rcv1_train.binary data
mv a9a
