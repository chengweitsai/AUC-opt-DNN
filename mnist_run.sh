max=10

M=mnist

i=1

for ratio in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "0.95" "0.99" "0.995"
do

for file in "$M"
do
    if [ ! -e output/${file}_ratio_${ratio}.txt ]; then 
        python2.7 new_feat_AUC.py  --request_ratio ${ratio} --output_file ${file}_ratio_${ratio}.txt --batch_size 32 --num_epochs $max --num_time_steps 20000
    fi

    echo $file
    echo $ratio
     
#    if [ ! -e logistic/logistic_${file}_ratio_${ratio}.txt ]; then
#        python2.7 dataset_acc.py --dataset $file --request_ratio ${ratio} --output_file logistic_${file}_ratio_${ratio}.txt --batch_size 32
#    fi

done
done
