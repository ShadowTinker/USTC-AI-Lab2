from sklearn.preprocessing import LabelEncoder
import numpy as np

class KNN:
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
      for i in range(len(label)):
        if int(label[i]) < 10:
          label[i] = 0
        else:
          label[i] = 1
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
      for i in range(len(label)):
        if int(label[i]) < 10:
          label[i] = 0
        else:
          label[i] = 1
    return data, label

  def fit(self, data, label):
    self.data, self.label = self.preprocess(data, label)

  def predict(self, TestData, TestLabel):
    TestData, TestLabel = self.preprocess(TestData, TestLabel)
    Prediction = []
    for test_data in TestData:
      # calculate all the distances
      distances = []
      for index, train_data in enumerate(self.data):
        for title in range(len(train_data)):
          distance = (test_data[title] - train_data[title]) ** 2
        distances.append([distance, self.label[index]])
      distances.sort()
      # make a prediction according to the distances calculated above
      pass_cnt = 0
      for index, element in enumerate(distances):
        if index == self.K:
          break
        if element[1] == 1:
          pass_cnt += 1
      if pass_cnt >= self.K/2:
        Prediction.append(1)
      else:
        Prediction.append(0)
    self.getF1Score(Prediction, TestLabel)

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


  def __init__(self, K=10, cut=False):
    self.data = []
    self.label = []
    self.les = []
    self.K = K
    self.result = -1
    self.cut = cut
  