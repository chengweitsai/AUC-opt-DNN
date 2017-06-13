max=10

SUSY=SUSY
ij=ijcnn1
R=real-sim
e=epsilon_normalized
s=skin_nonskin
H=heart_scale
c=covtype.libsvm.binary.scale
rcv1=rcv1_train.binary
p=phishing
a=a9a

i=1

for ratio in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7"  "0.8" "0.9"
do

for file in  "$SUSY" #"$s" "$p" "$rcv1" "$a" "$ij" "$c" "$e" "R"
do
#    if [ ! -e output/${file}_ratio_${ratio}.txt ]; then 
#        python dataset_auc.py --dataset $file --request_ratio ${ratio} --output_file ${file}_ratio_${ratio}.txt --batch_size 32 --num_epochs $max --num_time_steps 50000
#    fi

    if [ ! -e output/trad_${file}_ratio_${ratio}.txt ]; then
        python2.7 dataset_acc.py --dataset $file --request_ratio ${ratio} --output_file trad_${file}_ratio_${ratio}.txt --batch_size 32 --num_epoch $max --num_time_steps 30000
    fi

done
done
