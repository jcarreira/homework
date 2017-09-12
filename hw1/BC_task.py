#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import sys
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

def save_expert_data(data, env_name):
    with open("expert_data/expert_data_" + env_name + ".picke", 'wb') as f:
        pickle.dump(data, f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    f = open("expert_data/expert_data_" + args.envname + ".picke", "rb")
    data = pickle.load(f)

    print("data: ", data)
    print("actions shape: ", data['actions'].shape)
    print("observations shape: ", data['observations'].shape)
    
    data['actions'] = np.squeeze(data['actions'], axis=1)
    print("actions shape: ", data['actions'].shape)
    print("actions[0] shape: ", data['actions'][0].shape)

    # some code inspired from
    # https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

    model = Sequential()
    model.add(Dense(100, activation='relu',
        input_shape=data['observations'][0].shape))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(data['actions'][0].shape[0], activation='linear')) 
    model.summary()
    model.compile(loss='mean_squared_error',
            optimizer=RMSprop(),
            metrics=['accuracy'])

    X = data['observations']
    Y = data['actions']

    X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)

    model.fit(X_train, Y_train,
            batch_size = 20,
            epochs=20,
            verbose=1)

    score = model.evaluate(X_test, Y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()

        done = False
        totalr = 0.
        steps = 0
        while not done:
            bc_action = model.predict(obs[None,:])
            obs, r, done, _ = env.step(bc_action)

            totalr += r
            steps += 1
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('Env:', args.envname)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
