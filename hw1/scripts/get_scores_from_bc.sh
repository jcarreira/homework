NUM_ROLLOUTS=200

python3 BC_task.py Humanoid-v1  --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> bc_scores_results
python3 BC_task.py Ant-v1 --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> bc_scores_results
python3 BC_task.py HalfCheetah-v1 --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> bc_scores_results
python3 BC_task.py Hopper-v1 --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> bc_scores_results
python3 BC_task.py Reacher-v1 --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> bc_scores_results
python3 BC_task.py Walker2d-v1 --num_rollouts ${NUM_ROLLOUTS} 2>&1 >> bc_scores_results

