max=10

M=mnist

i=1

for ratio in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
do

for file in "$M"
do
    if [ ! -e output/new_m2_${file}_ratio_${ratio}.txt ]; then 
        python m2_feat_AUC.py  --request_ratio ${ratio} --output_file new_m2_${file}_ratio_${ratio}.txt --batch_size 32  --num_time_steps 20000 --num_epochs 3
    fi

    echo $file
    echo $ratio

done
done
