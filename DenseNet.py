import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from matplotlib import pyplot as plt

def Params(Func,Y,MP = True):
    params = {}
    params['proc']=3
    if MP == False:
        params['proc']=1
    if Func == 'Full':
        epochs = 200
        K = 12
        N = np.arange(2,15,1,dtype='int32')**1.5
    elif Func == 'Test':
        epochs = 100
        K = 3
        N = np.arange(2,12,2,dtype='int32')**2
    N = np.repeat(N,K)
    d = {'N':N.astype(int)}
    Runs = pd.DataFrame(data=d)
    Runs['MSE'] = 0.0
    Runs['R2'] = 0.0
    params['K'] = K
    params['epochs'] = epochs
    params['Y'] = Y
    return(Runs,params)

def Dense_Model(iteration,Neurons,batch_size,inputs,lr=1e-4,Memory=.9):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = Memory
    session = tf.Session(config=config)
    initializer = keras.initializers.glorot_uniform(seed=iteration)
    model = Sequential()
    model.add(Dense(Neurons, input_dim=inputs,activation='relu',kernel_initializer=initializer))
    model.add(Dense(1))
    NUM_GPU = 1 # or the number of GPUs available on your machine
    adam = keras.optimizers.Adam(lr = lr)
    gpu_list = []
    for i in range(NUM_GPU): gpu_list.append('gpu(%d)' % i)
    model.compile(loss='mean_squared_error', optimizer='adam')#,context=gpu_list) # - Add if using MXNET
    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model'+str(iteration)+'.h5', monitor='val_loss', save_best_only=True)]
    return(model,callbacks)

def Train_Steps(epochs,Neurons,X_train,X_test,X_val,y_train,y_test,y_val,iteration,X_fill,Memory=None):
    np.random.seed(iteration)
    from keras import backend as K
    Scorez=[]
    lr = 1e-3
    Mod,callbacks = Dense_Model(iteration,Neurons,X_train.shape[0],X_train.shape[1],lr=lr,Memory=Memory)
    killscore=0
    killmax = 5
    e = 0
    udate = 3
    batch_size=50#100
    Mod.fit(X_train, # Features
                      y_train, # Target vector
                      epochs=epochs, # Number of epochs
                      callbacks=callbacks, # Early stopping
                      verbose=0, # Print description after each epoch
                      batch_size=batch_size, # Number of observations per batch
                      validation_data=(X_test, y_test)) # Data for evaluation
    Yval = Mod.predict(X_val,batch_size = batch_size)
    MSE = metrics.mean_squared_error(y_val,Yval)
    Scorez=np.asanyarray(Scorez)
    y_fill = Mod.predict(X_fill,batch_size=batch_size)
    Rsq = metrics.r2_score(y_val,Yval)
    return(MSE,y_fill,Yval,y_val,Rsq)