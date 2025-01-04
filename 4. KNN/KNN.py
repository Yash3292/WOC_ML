import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class k_nearest_neighbours:
    
    def test_train_split(self,X,y,size):
        m_test = int(X.shape[0]*size)
        X_test = X[0:m_test,:]
        y_test = y[0:m_test]
        X_train = X[m_test:,:]
        y_train = y[m_test:]
        return X_train, X_test, y_train, y_test   
    
    def fit(self,X,y,k):
        self.X_train = X
        self.y_train = y
        self.k = k
    
    def predict(self,X_test):
        m_train = self.X_train.shape[0] 
        n = self.X_train.shape[1] 
        m_test = X_test.shape[0] 
        self.y_train = self.y_train.reshape(m_train,1)
        y_pred = np.zeros((m_test,1))
        for i in range(m_test):
            p = X_test[i,:]
            d = np.sqrt(np.sum((p-self.X_train)**2,axis =1))        
            d = d.reshape(m_train,1)
            d = np.hstack((d,self.y_train))
            d = d[np.argsort(d[:,0])]                           
            near_neighbours = d[0:self.k,1]
            values, counts = np.unique(near_neighbours, return_counts=True)
            indices = np.argmax(counts)
            y_pred[i][0] = values[indices]           
        return y_pred.flatten()
    
    def accuracy(self, yhat, y):
        return (np.sum(yhat == y) / len(yhat))           
