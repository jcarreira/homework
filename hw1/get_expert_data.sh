NUM_ROLLOUTS=50

python3 get_data_from_expert.py experts/Humanoid-v1.pkl Humanoid-v1  \
   --num_rollouts ${NUM_ROLLOUTS}
python3 get_data_from_expert.py experts/Ant-v1.pkl Ant-v1 \
   --num_rollouts $(NUM_ROLLOUTS)
python3 get_data_from_expert.py experts/HalfCheetah-v1.pkl HalfCheetah-v1 \
   --num_rollouts $(NUM_ROLLOUTS)
python3 get_data_from_expert.py experts/Hopper-v1.pkl Hopper-v1 \
   --num_rollouts $(NUM_ROLLOUTS)
python3 get_data_from_expert.py experts/Reacher-v1.pkl Reacher-v1 \
   --num_rollouts $(NUM_ROLLOUTS)
python3 get_data_from_expert.py experts/Walker2d-v1.pkl Walker2d-v1 \
   --num_rollouts $(NUM_ROLLOUTS)

