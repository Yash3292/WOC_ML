import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Linear_Regression:

    def test_train_split(self,X,y,size):
      m_test = int(X.shape[0]*size)
      X_test = X[0:m_test,:]
      y_test = y[0:m_test]
      X_train = X[m_test:,:]
      y_train = y[m_test:]
      return X_train,X_test,y_train,y_test 

    def fit_normalize(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.sigma != 0, (X - self.mu) / self.sigma, 0)
        return X_result

    def normalize(self, X):
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.sigma != 0, (X - self.mu) / self.sigma, 0)
        return X_result

    def train(self,alpha=0.05,lambda_=0,n_iter=6000):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lambda_ = lambda_ 
        self.J_hist = []
        self.noi = []
   
    def cost(self):
        f_wb = np.matmul(self.X, self.w) + self.b
        err = f_wb - self.y
        cost = (1/(2*self.m))*np.sum((err)**2)
        reg_cost = (self.lambda_/(self.m*2))*np.sum(self.w[:,0]**2)
        total_cost = cost + reg_cost
        return total_cost

    def gradient(self):
        f_wb  = np.matmul(self.X, self.w) + self.b
        err = f_wb - self.y
        dJ_dw = (np.matmul(self.X.T, err))/(self.m) + (self.lambda_/self.m)*self.w 
        dJ_db = np.sum(err) / self.m
        return dJ_db, dJ_dw
  
    def fit(self,X,y):
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.w = np.zeros((self.n,1))
        self.b = 0
        
        self.X = self.fit_normalize(X)
        self.y = y[:,np.newaxis]

        for i in range(self.n_iter):
            dJ_db, dJ_dw = self.gradient()
            self.w = self.w - self.alpha * dJ_dw
            self.b = self.b - self.alpha * dJ_db
            self.J_hist.append(self.cost())
            self.noi.append(i)
            if(i==0):
                print("Initial Cost:", self.cost())
            if(i==self.n_iter-1):
                print("Final Cost:", self.cost())  


    def predict(self,X):
        m = X.shape[0]
        X = self.normalize(X)
        y_pred = np.matmul(X, self.w) + self.b
        return y_pred.flatten()
  
  
    def score(self,X,y):
        X = self.normalize(X)
        y = y[:,np.newaxis]
        y_pre = np.matmul(X, self.w) + self.b
        return  1 - (((y - y_pre)**2).sum() / ((y - y.mean())**2).sum()) 
    
  
    def plot_cost(self):
        plt.plot(self.noi,self.J_hist)
        plt.xlabel("Number Of Iterations")
        plt.ylabel("Cost Function")
        plt.title("Const Function vs Iteration")

    def mse(self, yhat, y):
        return np.sum((yhat - y)**2) / len(yhat)

    def get_parameter(self):
        return self.w, self.b