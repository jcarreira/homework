#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

def save_expert_data(data, env_name):
    with open("expert_data/expert_data_" + env_name + ".picke", 'wb') as f:
        pickle.dump(data, f)

def main():

    f = open("expert_data/expert_data_Humanoid-v1.picke", "r")
    data = pickle.load(f)

    print("data: ", data)
    #print("data shape: ", data.shape)
    print("actions shape: ", data['actions'].shape)
    print("observations shape: ", data['observations'].shape)


#    for i in range(0, data['actions'].shape[0]):
#        action = data['actions'][i]
#        action_sq = np.squeeze(action)
#        print("action shape: ", action.shape)
#        print("action sq: ", action_sq.shape)
#        print("action sq flat: ", action_sq.flatten().shape)
#        data['actions'][i] = action_sq.flatten()
#        print(data['actions'].shape)
#        #print(data['actions'][i].shape)
    
    
    data['actions'] = np.squeeze(data['actions'], axis=1)
    print(data['actions'].shape)
#    sys.exit(0)

    # some code inspired from
    # https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(376,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(17, activation='linear')) 
    model.summary()
    model.compile(loss='mean_squared_error',
            optimizer=RMSprop(),
            metrics=['accuracy'])

    x_train = data['observations']
    y_train = data['actions']
    model.fit(x_train, y_train,
            batch_size = 20,
            epochs=10,
            verbose=1)

    score = model.evaluate(x_train, y_train, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


#    with tf.Session():
#        tf_util.initialize()
#
#        import gym
#        env = gym.make(args.envname)
#        max_steps = args.max_timesteps or env.spec.timestep_limit
#
#        returns = []
#        observations = []
#        actions = []
#
#        expert_data = {}
#        expert_data['expert_action'] = []
#        expert_data['expert_obs'] = []
#
#        for i in range(args.num_rollouts):
#            print('iter', i)
#            obs = env.reset()
#            done = False
#            totalr = 0.
#            steps = 0
#            while not done:
#                action = policy_fn(obs[None,:])
#                observations.append(obs)
#                actions.append(action)
#
#                # Humanoid task has action with dimension 17
#                print("action shape: ", action.shape)
#                obs, r, done, _ = env.step(action)
#
#                expert_data['expert_action'].append(action)
#                expert_data['expert_obs'].append(obs)
#
#                totalr += r
#                steps += 1
#                if args.render:
#                    env.render()
#                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
#                if steps >= max_steps:
#                    break
#            returns.append(totalr)
#
#        save_expert_data(expert_data, args.envname)
#
#        print('returns', returns)
#        print('mean return', np.mean(returns))
#        print('std of return', np.std(returns))
#
#        expert_data = {'observations': np.array(observations),
#                       'actions': np.array(actions)}

if __name__ == '__main__':
    main()
