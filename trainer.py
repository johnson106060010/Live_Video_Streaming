import numpy as np
import random
from sklearn import linear_model

def preprocess(raw_data, reward_frame):
    train_data = []
    train_label = []
    #pick 100000 frame to train
    pick_frame_id = random.sample(range(800000), 100000)
    # print(pick_frame_id)
    for i in range(100000):
        #get delta by previous and current frame
        #train_data.append(raw_data[pick_frame_id[i]]-raw_data[pick_frame_id[i]-1])
        train_data.append(raw_data[pick_frame_id[i]])
        train_label.append(reward_frame[pick_frame_id[i]])
    return train_data, train_label

def train(train_data, train_label):
    bitrate = range(3)
    target_buffer = range(1)
    latency_limit = range(1000)
    reg = linear_model.LinearRegression()
    np_train_data = np.zeros((100000,14))
    np_tmp = np.array(train_data)
    np_test = np.array([3,0,500])
    np_train_label = np.array(train_label)
    # concatenate 3key value into training data
    # print(np.append(np_train_data[0], np_test).shape)
    for i in range(100000):
        np_train_data[i] = np.append(np_tmp[i], np_test) 
    # print(np_train_data[0].shape)
    reg.fit(np_train_data[:70000], np_train_label[:70000])
    # for i in range(3):
    #     for j in range(1):
    #         for k in range(4):
    #validation
    mse = 0
    vali_data = np_train_data[70001:]
    vali_label = np_train_label[70001:]
    for i in range(len(vali_label)):
        predict = reg.predict([vali_data[i]])
        mse += np.square(predict-vali_label[i])
    mse /= len(vali_label)
    # print('predict:'+str(reg.predict([train_data[80001]])))
    # print('truth:'+str(train_label[80001]))
    print('MSE: '+str(mse))

raw_data = np.load('Fengtimo_2018_11_3_low_raw_data.npy')
reward_frame = np.load('Fengtimo_2018_11_3_low_reward_frame.npy')
# print(raw_data.shape, reward_frame.shape)
train_data, train_label = preprocess(raw_data, reward_frame)
# print(np.array(train_data).shape, np.array(train_label).shape)
train(train_data, train_label)