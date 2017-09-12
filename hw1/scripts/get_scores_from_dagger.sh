NUM_ROLLOUTS=20
OUTPUT_FILE="dagger_score_results"

#python3 Dagger_task.py experts/Humanoid-v1.pkl Humanoid-v1  \
#   --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> ${OUTPUT_FILE}
#python3 Dagger_task.py experts/Ant-v1.pkl Ant-v1 \
#   --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> ${OUTPUT_FILE}
#python3 Dagger_task.py experts/HalfCheetah-v1.pkl HalfCheetah-v1 \
#   --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> ${OUTPUT_FILE}
#python3 Dagger_task.py experts/Hopper-v1.pkl Hopper-v1 \
#   --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> ${OUTPUT_FILE}
python3 Dagger_task.py experts/Reacher-v1.pkl Reacher-v1 \
   --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> dagger_scores_reacher
#python3 Dagger_task.py experts/Walker2d-v1.pkl Walker2d-v1 \
#   --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> ${OUTPUT_FILE}

