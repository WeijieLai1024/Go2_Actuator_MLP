# Go2_Actuator_MLP

cd ~/projects/Go2_Actuator_MLP
python -m go2_act_net.train_actuator \
       --csv data/raw/go2data.csv \
       --device cuda \
       --batch 4096 \
       --epochs 50
