import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logistic_Regression:

    def test_train_split(self,X,y,size):
      m_test = int(X.shape[0]*size)
      X_test = X[0:m_test,:]
      y_test = y[0:m_test]
      X_train = X[m_test:,:]
      y_train = y[m_test:]
      return X_train,X_test,y_train,y_test 

    def fit_normalize(self, X):
        # Calculating mean and SD along each column(axis 0)
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.sigma != 0, (X - self.mu) / self.sigma, 0)
        return X_result

    def normalize(self, X):
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.sigma != 0, (X - self.mu) / self.sigma, 0)
        return X_result

    def train(self,alpha=0.5,lambda_=2.5,n_iter=1000):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lambda_ = lambda_ 
        self.J_hist = []
        self.noi = []

    def g(self, z):
        f = 1 / (1 + np.exp(-z))
        return f
   
    def cost(self):
        z = np.matmul(self.X, self.w) + self.b
        f_wb = self.g(z)
        loss = -(self.y*np.log(f_wb)) - ((1-self.y)*np.log(1-f_wb))
        cost = np.sum(loss) / self.m
        reg_cost = (self.lambda_/(self.m*2))*np.sum(self.w[:,0]**2)
        total_cost = cost + reg_cost
        return total_cost

    def gradient(self):
        z = np.matmul(self.X, self.w) + self.b
        f_wb = self.g(z)
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
        z = np.matmul(X, self.w) + self.b
        yhat = self.g(z)
        y_pred = np.where(yhat >= 0.5, 1, 0)
        return y_pred.flatten()

    def accuracy(self, yhat, y):
        return (np.sum(yhat == y) / len(yhat)) 

    def plot_cost(self):
        plt.plot(self.noi,self.J_hist)
        plt.xlabel("Number Of Iterations")
        plt.ylabel("Cost Function")
        plt.title("Const Function vs Iteration")

    def get_parameter(self):
        return self.w, self.b

    def f1_score(self, yhat, y):
        cm = np.zeros((2,2))
        for i in range(len(yhat)):
            if(yhat[i] == 1 and y[i] == 1):
                cm[0, 0] += 1
            elif(yhat[i] == 1 and y[i] == 0):
                cm[0, 1] += 1
            elif(yhat[i] == 0 and y[i] == 1):
                cm[1, 0] += 1

        precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

