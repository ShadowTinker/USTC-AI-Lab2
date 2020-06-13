import numpy as np
from sklearn.preprocessing import LabelEncoder
class LogisticRegression:
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
      label = np.where(label < 10, 0, 1)
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
      label = label.astype('int')
      label = np.where(label < 10, 0, 1)
    return data, label
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def getJw(self):
    result = self.label.T @ (np.log(self.sigmoid(self.data @ self.weight))) + \
    (1 - self.label) @ (np.log(1 - self.sigmoid(self.data @ self.weight)))
    return -result.mean()

  def getGradient(self):
    return (self.data.T @ (self.sigmoid(self.data @ self.weight) - self.label)) / self.data.shape[0]

  def fit(self, data, label):
    self.data, self.label = self.preprocess(data, label)
    self.weight = np.zeros(data.shape[1])
    for i in range(self.iteration):
      gradient = self.getGradient()
      self.weight -= self.alpha * gradient

  def predict(self, data, label):
    data, label = self.preprocess(data, label)
    PredictLabel = self.sigmoid(data @ self.weight)
    predictions = np.zeros(label.shape)
    predictions[PredictLabel > 0.5] = 1
    self.getF1Score(predictions, label)

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
    if TP == 0 and FP == 0:
      P = TN / (TN + FN)
      self.result = 'There is no F1 score, because all the data are predicted to be negative. And the precision rate is ' + str(P)
    else:
      P = TP / (TP + FP)
      R = TP / (TP + FN)
      self.result = 2 * P * R / (P + R)

  def __init__(self, alpha, iteration, cut):
    self.weight = np.array([])
    self.result = -1
    self.data = np.array([[]])
    self.label = np.array([])
    self.alpha = alpha
    self.iteration = iteration
    self.cut = cut
    self.les = []
