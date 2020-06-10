import csv
import numpy as np
import random
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
        else:
          row = [row]
          self.add(row)
          self.total += 1
  def init(self, col_name):
    self.DataSet = np.array([col_name])
  def add(self, data):
    self.DataSet = np.append(self.DataSet, data, axis=0)
  def split(self, ratio):
    self.TestData = np.array([self.DataSet[0]])
    self.TrainData = np.array([self.DataSet[0]])
    for index, data in enumerate(self.DataSet):
      if index == 0:
        continue
      if random.random() < ratio:
        self.TrainData = np.append(self.TrainData, [data], axis=0)
      else:
        self.TestData = np.append(self.TestData, [data], axis=0)
  def __init__(self):
    self.total = 0
    self.DataSet = np.array([[]])
    self.TrainData = np.array([[]])
    self.TestData = np.array([[]])

DATA = DataHandle()
DATA.read(DATASET1)
DATA.split(0.7)
