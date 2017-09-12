NUM_ROLLOUTS=20
OUTPUT=experts_results

rm -rf ${OUTPUT}

echo "Humanoid"
python3 run_expert.py experts/Humanoid-v1.pkl Humanoid-v1  \
   --num_rollouts ${NUM_ROLLOUTS} >> ${OUTPUT}
echo "Ant"
python3 run_expert.py experts/Ant-v1.pkl Ant-v1 \
   --num_rollouts ${NUM_ROLLOUTS} >> ${OUTPUT}
echo "HalfCheetah"
python3 run_expert.py experts/HalfCheetah-v1.pkl HalfCheetah-v1 \
   --num_rollouts ${NUM_ROLLOUTS} >> ${OUTPUT}
echo "Hopper"
python3 run_expert.py experts/Hopper-v1.pkl Hopper-v1 \
   --num_rollouts ${NUM_ROLLOUTS} >> ${OUTPUT}
echo "Reacher"
python3 run_expert.py experts/Reacher-v1.pkl Reacher-v1 \
   --num_rollouts ${NUM_ROLLOUTS} >> ${OUTPUT}
echo "Walker"
python3 run_expert.py experts/Walker2d-v1.pkl Walker2d-v1 \
   --num_rollouts ${NUM_ROLLOUTS} >> ${OUTPUT}

