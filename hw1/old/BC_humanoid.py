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
from sklearn.model_selection import train_test_split

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
    
    
    data['actions'] = np.squeeze(data['actions'], axis=1)
    print(data['actions'].shape)

    # some code inspired from
    # https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(376,)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(17, activation='linear')) 
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
            epochs=10,
            verbose=1)

    score = model.evaluate(X_test, Y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
