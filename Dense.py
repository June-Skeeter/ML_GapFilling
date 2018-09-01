import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from matplotlib import pyplot as plt

def Params(Func,Y,MP = True):
    params = {}
    params['proc']=3
    if Func == 'Full':
        epochs = 200#1000
        K = 30
        # T_Max = 48
    elif Func == 'Test':
        epochs = 100
        K = 5
        # T_Max = 10
    if MP == False:
        params['proc']=1
   # N_Max = 14
    #N_min = 1
    #n = 7
    #N = np.linspace(N_min,N_Max,n,dtype='int32')
    N = np.arange(1,11,1.25,dtype='int32')**2
    # T = np.array(np.random.rand(samp_size)*T_Max,dtype='int32')
    d = {'N':N}
    Runs = pd.DataFrame(data=d)
    Runs['MSE'] = 0.0
    Runs['STD'] = 0.0
    Runs['CI'] = 0.0
    Runs['SE'] = 0.0
    # params['T_Max'] = T_Max
    #params['N_Max'] = N_Max
    #params['N_Min'] = N_min
    params['K'] = K
    params['epochs'] = epochs
    params['Y'] = Y
    #params['Searches']=Searches
    return(Runs,params)

def Dense_Model(Neurons,batch_size,inputs,lr=1e-4,Memory=.9):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = Memory
    session = tf.Session(config=config)
    model = Sequential()
    model.add(Dense(Neurons, input_dim=inputs,activation='relu'))
    model.add(Dense(1))
    NUM_GPU = 1 # or the number of GPUs available on your machine
    
    adam = keras.optimizers.Adam(lr = lr)
    gpu_list = []
    for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
    model.compile(loss='mean_squared_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
    return(model)

def Train_Steps(epochs,Neurons,X_train,X_test,X_val,y_train,y_test,y_val,i,X_fill,Memory=None):
    np.random.seed(i)
    from keras import backend as Kb
    Scorez=[]
    lr = 1e-3
    Mod = Dense_Model(Neurons,X_train.shape[0],X_train.shape[1],lr=lr,Memory=Memory)
    killscore=0
    killmax = 2
    e = 0
    batch_size=100
    while killscore < killmax and e < epochs:
        Mod.fit(X_train,y_train,batch_size=batch_size, epochs=1,shuffle=True,verbose=0)
        old_weights = Mod.get_weights()
        Y = Mod.predict(X_test,batch_size =batch_size)
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
            Kb.set_value(Mod.optimizer.lr, 0.5 * Kb.get_value(Mod.optimizer.lr))
        Mod.reset_states()
        e +=1
    Mod.set_weights(min_weights)
    Yval = Mod.predict(X_val,batch_size = batch_size)
    MSE = metrics.mean_squared_error(y_val,Yval)
    Scorez=np.asanyarray(Scorez)
    y_fill = Mod.predict(X_fill,batch_size=batch_size)
    return(MSE,y_fill,Yval,y_val)