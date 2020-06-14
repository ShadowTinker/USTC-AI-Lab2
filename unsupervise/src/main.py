import numpy as np

def getData(Path):
  file = open(Path, 'r')
  TrainData = np.array([[]])
  TrainLabel = np.array([])
  for data in file:
    data = data.split(',')
    if TrainLabel.size == 0:
      TrainLabel = np.array([data[0]])
    else:
      TrainLabel = np.append(TrainLabel, [data[0]])
    if TrainData.size == 0:
      TrainData = np.array([data[1:]])
    else:
      TrainData = np.append(TrainData, [data[1:]], axis=0)
  TrainData = TrainData.astype('float')
  TrainLabel = TrainLabel.astype('int')
  return TrainData, TrainLabel

def PCA(data, threshold):
  # centering data
  data -= np.mean(data.T, axis=1)
  # compute covariance matrix
  Cov = np.cov(data.T)
  # implement eigen-decomposition
  eigen_value, eigen_vector = np.linalg.eig(Cov)
  # get selected number of eigen-vectors
  total = np.sum(eigen_value)
  index = 1
  components = np.array([eigen_vector[0]])
  selected = eigen_value[0] / total
  while(selected < threshold):
    components = np.append(components, [eigen_vector[index]], axis=0)
    selected += eigen_value[index] / total
    index += 1
  # project data to eigen-hyperplain
  projected = data @ components.T
  print(projected)

DATASET = '../input/wine.data'
data, label = getData(DATASET)
PCA(data, 0.95)

