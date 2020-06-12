import numpy as np
import math
import random
from sklearn.preprocessing import LabelEncoder

class SVM:
  class KernelFunction:
    def linear(self, X):
      return np.dot(X, X.T)

    def polynomial(self, X, d):
      return np.power((np.dot(X, X.T)), d)

    def rbf(self, X, sigma):
      # get the square sum of X, and transform it from a row vector to a col vector
      Xsq = np.sum(np.power(X, 2), axis=1).reshape(-1, 1)
      # ||X_i - X_j||^2 = X_i^2 - 2 * X_i * X_j + X_j^2
      result = Xsq - 2 * (np.dot(X, X.T)) + Xsq.T
      # get the kernel matrix
      result = np.exp(- result / (2 * sigma * sigma))
      return result

    def laplace(self, X, sigma):
      # get the square sum of X, and transform it from a row vector to a col vector
      Xsq = np.sum(np.power(X, 2), axis=1).reshape(-1, 1)
      # ||X_i - X_j||^2 = X_i^2 - 2 * X_i * X_j + X_j^2
      result = Xsq - 2 * (np.dot(X, X.T)) + Xsq.T
      # get the kernel matrix
      result = math.exp(- result / sigma)
      return result

    def sigmoid(self, X, beta, theta):
      return np.tanh(beta * np.dot(X, X.T) + theta)

    def __init__(self, kernel, X, d=3, sigma=5, beta=1, theta=-1):
      if kernel == 'linear':
        self.KernelMatrix = self.linear(X)
      elif kernel == 'polynomial':
        self.KernelMatrix = self.polynomial(X, d)
      elif kernel == 'rbf':
        self.KernelMatrix = self.rbf(X, sigma)
      elif kernel == 'laplace':
        self.KernelMatrix = self.laplace(X, sigma)
      elif kernel == 'sigmoid':
        self.KernelMatrix = self.sigmoid(X, beta, theta)
      else:
        print('no such kernel function')
        exit(3)
  class loss:
    def getLoss(self, x, y):
      pass

    def __init__(self, method='hinge'):
      self.method = method
      
  class SMO:
    def satisfyKKT(self, alpha, label, F, epsilon=0.01):
      if label * F < -epsilon and alpha < self.C:
        return False
      elif label * F > epsilon and alpha > 0:
        return False
      else:
        return True

    def train(self, max_iteration = 3):
      # Initialization of parameters
      DataNumber = self.data.shape[0]
      alphas = np.zeros((DataNumber, 1), dtype=float)
      bias = 0.0
      E = np.zeros((DataNumber, 1), dtype=float)
      # set a very small epsilon, which can accelarate the loop
      epsilon = 0.000001
      # Iteration begins
      times = 0
      while(times < max_iteration):
        # set the flag of stoping iteration
        changed = False;
        for i in range(DataNumber):
          # compute the model value
          E[i] = np.sum(alphas * self.label * self.K.KernelMatrix[:, i].reshape(-1,1)) + bias - self.label[i]
          if self.satisfyKKT(alphas[i], self.label[i], E[i], epsilon) == False:
            # find another alpha to compute
            j = random.randint(0, DataNumber - 1)
            while j == i:
              j = random.randint(0, DataNumber - 1)
            alpha_i = alphas[i][0]
            alpha_j = alphas[j][0]
            # Compute L and H of SMO algorithm
            if self.label[i] == self.label[j]:
              L = max(0, alphas[j] + alphas[i] - self.C)
              H = min(self.C, alphas[j] + alphas[i])
            else:
              L = max(0, alphas[j] - alphas[i])
              H = min(self.C, self.C + alphas[j] - alphas[i])
            if L == H:
              # cannot optimize alpha_i
              continue
            # compute eta of SMO
            eta = 2 * self.K.KernelMatrix[i, j] - self.K.KernelMatrix[i, i] - self.K.KernelMatrix[j, j]
            if eta >= 0:
              # cannot optimize alpha_i
              continue
            # update alpha_j, which is not the final value
            alphas[j] = alphas[j] - self.label[j] * (E[i]- E[j]) / eta
            # clip alpha_j to lie within the range [L, H]
            if alphas[j] > H:
              alphas[j] = H
            elif alphas[j] < L:
              alphas[j] = L
            # if the change is less than epsilon, ignore it, which will accelerate this procedure
            if np.abs(alphas[j] - alpha_j) < epsilon:
              alphas[j] = alpha_j
              continue
            # update alpha_i
            alphas[i] = alphas[i] + self.label[i] * self.label[j] * (alpha_j - alphas[j])
            # update the bias
            b1 = bias - E[i] - self.label[i] * (alphas[i] - alpha_i) * \
                self.K.KernelMatrix[i, j] - self.label[j] * (alphas[j] - alpha_j) * \
                self.K.KernelMatrix[i, j]
            b2 = bias - E[j] - self.label[i] * (alphas[i] - alpha_i) * \
                self.K.KernelMatrix[i, j] - self.label[j] * (alphas[j] - alpha_j) * \
                self.K.KernelMatrix[j, j]
            if 0 < alphas[i] < self.C:
              bias = b1
            elif 0 < alphas[j] < self.C:
              bias = b2
            else:
              bias = (b1 + b2) / 2
            changed = True
        if changed:
          times = 0
        else:
          times += 1
      chosen = alphas > 0
      support_vector = self.data[chosen.reshape(1, -1)[0], :]
      corre_label = self.label[chosen.reshape(1, -1)[0]]
      corre_alphas = alphas[chosen.reshape(1, -1)[0]]
      weight = ((alphas * self.label).T @ self.data).T
      return support_vector, corre_label, corre_alphas, bias, weight


    def __init__(self, data, label, KernelMatrix, C):
      self.data = data
      self.label = label
      self.C = C
      self.K = KernelMatrix

  def preprocess(self, data, label):
    if self.cut:
      # data processing
      for col in range(data.shape[1] - 2):
        try:
          temp = int(data[0][col])
        except:
          le = LabelEncoder()
          data[:, col] = le.fit_transform(data[:, col])
          self.les.append(le)
      data = data.astype('int')
      # label processing
      label = label.astype('int')
      label = np.where(label < 10, -1, 1)
    else:
      # data processing
      for col in range(data.shape[1]):
        try:
          temp = int(data[0][col])
        except:
          le = LabelEncoder()
          data[:, col] = le.fit_transform(data[:, col])
          self.les.append(le)
      data = data.astype('int')
      # label processing
      label = np.where(label < 10, -1, 1)
    return data, label

  def fit(self, data, label):
    data, label = self.preprocess(data, label)
    self.alphas = np.zeros((data.shape[0], 1), dtype=float)
    KernelFunction = self.KernelFunction(self.kerner, data)
    smo = self.SMO(data, label, KernelFunction, self.C)
    self.sv, self.sl, self.alphas, self.bias, self.weight = smo.train(max_iteration=3)

  def predict(self, data, label):
    data, label = self.preprocess(data, label)
    DataNumber = data.shape[0]
    Y = np.array([])
    prediction = np.zeros((DataNumber, 1))
    if self.kerner == 'linear':
      Y = data @ self.weight + self.bias
    elif self.kerner == 'rbf':
      Xsq = np.sum(np.power(data, 2), axis = 1).reshape(-1, 1)
      SVsq = (np.sum(np.power(self.sv, 2), axis = 1)).T
      K = Xsq - 2 * (data @ (self.sv).T) + SVsq.T
      K = np.exp(- K / (2 * self.sigma * self.sigma))
      K = self.sl.T * K
      K = self.alphas.T * K
      Y = np.sum(K, axis=1)
    # get prediction according to Y
    prediction[Y >= 0] = 1
    prediction[Y < 0] = -1
    self.getF1Score(prediction, label)

  def getF1Score(self, predictions, target):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, predict in enumerate(predictions):
      if predict == 1:
        if target[index] == 1:
          TP += 1
        else:
          FP += 1
      else:
        if target[index] == 1:
          FN += 1
        else:
          TN += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    self.result = 2 * P * R / (P + R)


  def __init__(self, C=1.0, kernel='rbf', loss='hinge', cut=True, sigma=1):
    self.data = np.array([[]])
    self.label = np.array([[]])
    self.les = []
    self.C = C
    self.kerner = kernel
    self.loss = loss
    self.cut = cut
    self.sv = np.array([])
    self.sl = np.array([])
    self.alphas = np.array([])
    self.bias = -1
    self.weight = np.array([])
    self.sigma = sigma
    self.result = -1
