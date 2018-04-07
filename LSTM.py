import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from matplotlib import pyplot as plt

def Params(Func,Y,MP = True):
    params = {}
    if Func == 'Full':
        epochs = 600#1000
        reps = 10#15
        N_Max = 100#200
        N_min = 2
        T_Max = 48
        samp_size = 15
        Searches = 10
        params['proc']=3
    elif Func == 'Test':
        epochs = 100
        reps = 3
        N_Max = 100
        N_min = 5
        T_Max = 10
        samp_size = 5
        Searches = 2
        params['proc']=3
    if MP == False:
        params['proc']=1
    N = np.array(np.random.rand(samp_size)*N_Max+N_min,dtype='int32')
    T = np.array(np.random.rand(samp_size)*T_Max,dtype='int32')
    d = {'N':N,'T':T}
    Runs = pd.DataFrame(data=d)
    params['T_Max'] = T_Max
    params['N_Max'] = N_Max
    params['N_Min'] = N_min
    params['reps'] = reps
    params['epochs'] = epochs
    params['Y'] = Y
    params['Searches']=Searches
    return(Runs,params)


def LSTM_Model(Neurons,batch_size,time_steps,inputs,lr=1e-4,Memory=.9):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = Memory
    session = tf.Session(config=config)
    model = Sequential()
    model.add(LSTM(Neurons, input_shape=(time_steps,inputs),stateful = False))
    model.add(Dense(1))
    NUM_GPU = 1 # or the number of GPUs available on your machine
    
    adam = keras.optimizers.Adam(lr = lr)
    gpu_list = []
    for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
    model.compile(loss='mean_squared_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
    return(model)

def Train_Steps(epochs,Neurons,X_train,X_test,X_val,y_train,y_test,y_val,i,X_fill,Memory=None):
    np.random.seed(i)
    from keras import backend as K
    Scorez=[]
    lr = 1e-3
    Mod = LSTM_Model(Neurons,X_train.shape[0],X_train.shape[1],X_train.shape[2],lr=lr,Memory=Memory)
    killscore=0
    killmax = 10
    e = 0
    udate = 3
    while killscore < killmax and e < epochs:
        Mod.fit(X_train,y_train,batch_size=X_train.shape[0], nb_epoch=1,shuffle=True,verbose=0)
        old_weights = Mod.get_weights()
        Y = Mod.predict(X_test,batch_size =X_test.shape[0])
        score = metrics.mean_squared_error(y_test,Y)
        Scorez.append(score)
        if e == 0:
            score_min=score
            min_weights=old_weights
        elif score < score_min:
            score_min = score
            min_weights=old_weights
            killscore = 0
        else:
            killscore +=1
        if killscore == math.floor(killmax/2):
            K.set_value(Mod.optimizer.lr, 0.5 * K.get_value(Mod.optimizer.lr))
        Mod.reset_states()
        e +=1
    Mod.set_weights(min_weights)
    Yval = Mod.predict(X_val,batch_size = X_val.shape[0])
    # plt.figure()
    # plt.scatter(Yval,y_val)
    # yl = plt.ylim()
    # plt.xlim(yl[0],yl[1])
    # plt.show()
    MSE = metrics.mean_squared_error(y_val,Yval)
    Scorez=np.asanyarray(Scorez)
    y_fill = Mod.predict(X_fill,batch_size=X_fill.shape[0])
    return(MSE,y_fill)