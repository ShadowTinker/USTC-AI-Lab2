import csv
import numpy as np
import random
from KNN import KNN
from SVM import SVM
TestData = 'supervise/data/student-mat.csv'
DATASET1 = '../data/student-mat.csv'
DATASET2 = '../data/student-por.csv'
class DataHandle:
  def read(self, file):
    with open(file) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter = ';')
      for row in csv_reader:
        if self.total == 0:
          self.init(row)
          self.total += 1
        elif self.total == 1:
          self.DataSet = np.array([row])
          self.total += 1
        else:
          row = [row]
          self.add(row)
          self.total += 1

  def init(self, col_name):
    self.title = np.array([col_name])

  def add(self, data):
    self.DataSet = np.append(self.DataSet, data, axis=0)

  def split(self, ratio):
    for index, data in enumerate(self.DataSet):
      if random.random() < ratio:
        if self.TrainData.size == 0:
          self.TrainData = np.array([data])
        else:
          self.TrainData = np.append(self.TrainData, [data], axis=0)
        if self.TrainLabel.size == 0:
          self.TrainLabel = np.array(data[-1])
        else:
          self.TrainLabel = np.append(self.TrainLabel, data[-1])
      else:
        if self.TestData.size == 0:
          self.TestData = np.array([data])
        else:
          self.TestData = np.append(self.TestData, [data], axis=0)
        if self.TestLabel.size == 0:
          self.TestLabel = np.array(data[-1])
        else:
          self.TestLabel = np.append(self.TestLabel, data[-1])
    self.TestData = np.delete(self.TestData, -1, 1)
    self.TrainData = np.delete(self.TrainData, -1, 1)
    
  def __init__(self):
    self.total = 0
    self.title = []
    self.DataSet = np.array([[]])
    self.TrainData = np.array([[]])
    self.TrainLabel = np.array([])
    self.TestData = np.array([[]])
    self.TestLabel = np.array([])

# Data handling
DATA = DataHandle()
DATA.read(DATASET1)
DATA.split(0.7)

# KNN
# classifier = KNN(K=10, cut=True)
# classifier.fit(DATA.TrainData, DATA.TrainLabel)
# classifier.predict(DATA.TestData, DATA.TestLabel)
# print(classifier.result)
# SVM
classifier = SVM(C=1., kernel='rbf')
classifier.fit(DATA.TrainData, DATA.TrainLabel)
classifier.predict(DATA.TestData, DATA.TestLabel)
print(classifier.result)
