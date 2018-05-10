
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt


from scipy.optimize import minimize

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Custom Functions
import ReadStandardTimeFill as RSTF
import RepRunner as RR
import Dense

class Network(object):
    """docstring for ClassName"""
    def __init__(self,Path,Func,Y,Model,MP = True):
        self.params = {}
        if Func == 'Full':
            epochs = 500#1000
            reps = 30
            N_Max = 150#200
            N_min = 5
            # T_Max = 48
            samp_size = 4
            Searches = 3
            self.params['proc']=3
        elif Func == 'Test':
            epochs = 50
            reps = 4
            N_Max = 100
            N_min = 50
            # T_Max = 10
            samp_size = 3
            Searches = 2
            self.params['proc']=3
        if MP == False:
            self.params['proc']=1
        N = np.linspace(N_min,N_Max,samp_size,dtype='int32')
        # T = np.array(np.random.rand(samp_size)*T_Max,dtype='int32')
        d = {'N':N}

        self.Runs = pd.DataFrame(data=d)
        self.Runs['MSE'] = 0.0
        self.Runs['STD'] = 0.0
        self.Runs['CI'] = 0.0
        self.Runs['Upper_Bounds'] = 0.0
        self.Runs['Lower_Bounds'] = 0.0
        # self.params['T_Max'] = T_Max
        self.params['N_Max'] = N_Max
        self.params['N_Min'] = N_min
        self.params['reps'] = reps
        self.params['epochs'] = epochs
        self.params['Y'] = Y
        self.params['Searches']=Searches
        self.params['T']=0
        self.params['Model']=Model
        self.params['Path'] = Path


    def TTV_Split(self,i,Memory,X,y,X_fill):
        X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.1, random_state=i)
        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train, test_size=0.11, random_state=i)
        return(Dense.Train_Steps(self.params['epochs'],self.params['N'],X_train,X_test,X_val,y_train,y_test,
                                y_val,i,X_fill = X_fill,Memory=Memory))

    def RunReps(self,pool = None,plot=False,FullReturn=False):
        RST = RSTF.ReadStandardTimeFill(self.params['Path'])
        offset = 5/self.params['proc']
        Memory = (math.floor(100/self.params['proc'])- offset) * .01
        MSE = []
        RST.Scale(self.params['Y'],self.params['Model'])
        if self.params['T'] >0:
            RST.TimeShape(self.params['T'])
        y = RST.y*1.0
        X = RST.X*1.0
        X_fill = RST.X_fill*1.0
        MSE = []
        Y_fill = []
        Yval = []
        y_val= []
        if __name__=='__main__'and self.params['proc'] != 1:
            for i,results in enumerate(pool.imap(partial(self.TTV_Split,Memory=Memory,X=X,y=y,X_fill=X_fill),
                                                 range(self.params['reps']))):

                MSE.append(results[0])
                Y_fill.append(results[1])
                Yval.append(results[2])
                y_val.append(results[3])
        else:
            for i in range(self.params['reps']):
                results = self.TTV_Split(i,Memory,X,y,X_fill)
                MSE.append(results[0])
                Y_fill.append(results[1])
                Yval.append(results[2])
                y_val.append(results[3])
                
        MSE = np.asanyarray(MSE)
        Y_fill = np.asanyarray(Y_fill)
        Y_fill = Y_fill.mean(axis=0)
        
        Yval = np.asanyarray(Yval)
        Yval = Yval.mean(axis=0)
        y_val = np.asanyarray(y_val)
        y_val = y_val.mean(axis=0)
        
        FillVarName = self.params['Y'].replace('f','F')
        RST.Fill(Y_fill,FillVarName)
        if plot == True:
            plt.scatter(RST.Master['TempFill'],RST.Master[self.params['Y']],label=np.round(MSE.mean(),2))
            yl = plt.ylim()
            plt.xlim(yl)
            
        if FullReturn == False:
            return(MSE,RST.Master['TempFill'])
        else:
            return(MSE,RST.Master)

    def GP(self,pool):

        def upper_confidence_bound(mu_x, sigma_x, opt_value, kappa=-1.0):
            return mu_x + kappa * sigma_x

        def query(xi, yi, gp):
            acq = upper_confidence_bound
            best_value = np.inf
            for N in np.linspace(1,self.params['N_Max']):
                if self.params['T']>0:
                    for T in np.linspace(0,self.params['T_Max']):
                        def obj(x):
                            x=x.reshape(1,-1)
                            mu_x, sigma_x = gp.predict(x, return_std=True)
                            return acq(mu_x, sigma_x, np.min(yi))
                        x0 = np.asanyarray([N,T]).reshape(1,2)
                        bounds=((1, self.params['N_Max']),(0,self.params['T_Max']))
                        print(x0,bounds)
                        res = minimize(obj, x0, bounds=bounds)

                        if res.fun < best_value:
                            best_value = res.fun
                            query_point = res.x
                else:
                    def obj(x):
                        x=x.reshape(1,-1)
                        mu_x, sigma_x = gp.predict(x, return_std=True)
                        return acq(mu_x, sigma_x, np.min(yi))
                    x0 = np.asanyarray(N).reshape(1,-1)
                    bounds=[(1, self.params['N_Max'])]
                    res = minimize(obj, x0, bounds=bounds)
                    if res.fun < best_value:
                        best_value = res.fun
                        query_point = res.x
            query_point = query_point
            return query_point

        kernel = Matern(length_scale_bounds="fixed") 
        gp = GaussianProcessRegressor(kernel=kernel, alpha=self.Runs['STD'].values, random_state=1,normalize_y=True)
            
        for i in range(self.params['Searches']):
            if self.params['T']>0:
                gp.fit(self.Runs[['N','T']].values, self.Runs['MSE'].values)
                next_x = query(self.Runs[['N','T']].values, self.Runs['MSE'].values, gp)
                N = int(np.round(next_x[0],0))
                T = int(np.round(next_x[1],0))
                o = 0
                while len(self.Runs.loc[(self.Runs['N']==N) & (self.Runs['T']==T)].index) != 0:
                    print('Adjust!')
                    o +=1
                    N += int(o*np.cos(o*np.pi))
                    if N < self.params['N_Min'] or N > self.params['N_Max']:
                        N -= int(o*np.cos(o*np.pi))
                    if o > 5:
                        T += 1
                print(N,T)
                d = {'N':N,'T':T,'MSE':0,'STD':0}
                idx = self.Runs.index[-1] + 1
                D2 = pd.DataFrame(data=d,index=[idx])
                self.Runs = self.Runs.append(D2)
                self.params['T'] = T
                self.params['N'] = N
                Results = self.RunReps(pool)
                MSE = Results[0]
                self.Runs['MSE'][idx]=MSE.mean()
                self.Runs['STD'][idx]=MSE.std()
                self.Runs = self.Runs.sort_values(by = ['N','T']).reset_index(drop=True)
                self.Runs['CI'] = self.Runs['STD']/Net.params['reps']**.5*stats.t.ppf(1-0.05, Net.params['reps']-2)
                self.Runs['Upper_Bounds'] = self.Runs['MSE']+self.Runs['CI']
            else:
                gp.fit(self.Runs['N'].values.reshape(-1,1), self.Runs['MSE'].values)
                next_x = query(self.Runs['N'].values, self.Runs['MSE'].values, gp)
                N = int(np.round(next_x[0],0))
                o = 0
                while len(self.Runs.loc[self.Runs['N']==N].index) != 0:
                    print('Adjust!')
                    o +=1
                    N += int(o*np.cos(o*np.pi))
                    if N < self.params['N_Min'] or N > self.params['N_Max']:
                        N -= int(o*np.cos(o*np.pi))
                print(N)
                d = {'N':N,'MSE':0,'STD':0}
                idx = self.Runs.index[-1] + 1
                D2 = pd.DataFrame(data=d,index=[idx])
                self.Runs = self.Runs.append(D2)
                self.params['N'] = N
                Results = self.RunReps(pool)
                MSE = Results[0]
                self.Runs['MSE'][idx]=MSE.mean()
                self.Runs['STD'][idx]=MSE.std()
                self.Runs = self.Runs.sort_values(by = ['N']).reset_index(drop=True)
                self.Runs['CI'] = self.Runs['STD']/Net.params['reps']**.5*stats.t.ppf(1-0.05, Net.params['reps']-1)
                self.Runs['Upper_Bounds'] = self.Runs['MSE']+self.Runs['CI']

        self.Runs['Lower_Bounds'] = self.Runs['MSE']-self.Runs['CI']