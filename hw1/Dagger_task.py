#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import sys
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--dagger_iterations", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    f = open("expert_data/expert_data_" + args.envname + ".picke", "rb")
    data = pickle.load(f)

    print("data: ", data)
    print("actions shape: ", data['actions'].shape)
    print("observations shape: ", data['observations'].shape)
    
    data['actions'] = np.squeeze(data['actions'], axis=1)
    print(data['actions'].shape)

    # some code inspired from
    # https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
    model = Sequential()
    model.add(Dense(50, activation='relu',
        input_shape=data['observations'][0].shape))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(data['actions'][0].shape[0], activation='linear')) 
    model.summary()
    model.compile(loss='mean_squared_error',
            optimizer=RMSprop(),
            metrics=['accuracy'])

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
            
    X = data['observations']
    Y = data['actions']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)

    dagger_iterations = args.dagger_iterations or 200
    with tf.Session():
        while dagger_iterations > 0:
            dagger_iterations = dagger_iterations - 1
            # train BC
            model.fit(X_train, Y_train,
                    batch_size = 20,
                    epochs=4,
                    verbose=0)

            score = model.evaluate(X_train, Y_train, verbose=1)

            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            # run BC and expert in lock step
            tf_util.initialize()

            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []

            for i in range(args.num_rollouts):
                print('Iteration', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    expert_action = policy_fn(obs[None,:])
                    bc_action = model.predict(obs[None,:])

                    observations.append(obs)
                    actions.append(expert_action)

                    # Humanoid task has action with dimension 17
                    obs, r, done, _ = env.step(bc_action)

                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            # and aggregate observations from BC <-> decisions from expert
            print(args.envname + " clone mean return: ", np.mean(returns))
            print(args.envname + " clone std return: ", np.std(returns))

            actions_array = np.array(actions)
            actions_array = np.squeeze(actions_array, axis=1)

            X_train = np.vstack((X_train, np.array(observations)))
            Y_train = np.vstack((Y_train, actions_array))

            # loop

if __name__ == '__main__':
    main()
