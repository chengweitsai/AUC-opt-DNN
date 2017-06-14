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

D=./data
O=./output
L=./logistic
for dir in "$D" "$O" "$L"
do 
   if [ ! -d "$dir" ]; then
      mkdir $dir
   fi
done

for ratio in "0.01" "0.02" "0.05" "0.95" "0.98" "0.99"
do

for file in "$SUSY"# "$ij" "$e" "$s" "$H" "$a" "$c" "$p" "$rcv1"
do
    if [ ! -e output/${file}_ratio_${ratio}.txt ]; then 
        python dataset_auc.py --dataset $file --request_ratio ${ratio} --output_file ${file}_ratio_${ratio}.txt --batch_size 32 --num_epochs $max --num_time_steps 20000
    fi

    echo $file
    echo $ratio
     
    if [ ! -e logistic/logistic_${file}_ratio_${ratio}.txt ]; then
        python dataset_acc.py --dataset $file --request_ratio ${ratio} --output_file logistic_${file}_ratio_${ratio}.txt --batch_size 32
    fi

done
done
