import numpy as np
import matplotlib.pyplot as plt
import csv

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
  data -= np.mean(data, axis=0)
  # compute covariance matrix
  Cov = np.cov(data.T)
  # implement eigen-decomposition
  eigen_value, eigen_vector = np.linalg.eig(Cov)
  # sort
  index = np.argsort(eigen_value)[::-1]
  eigen_value = eigen_value[index]
  eigen_vector = eigen_vector[:,index]
  # get selected number of eigen-vectors
  total = np.sum(eigen_value)
  index = 1
  components = np.array([eigen_vector[:, 0]]).T
  selected = eigen_value[0] / total
  while(selected < threshold):
    components = eigen_vector[:, :index]
    selected += eigen_value[index] / total
    index += 1
  # project data to eigen-hyperplain
  print('Get', components.shape[1], 'principal components')
  projected = data @ components
  return projected

def KMeans(k, data):
  # pick initial means
  means = data[np.random.choice(data.shape[0], size=k, replace=False), :]
  # start iterating
  means_old = -1
  while True:
    # Initialization
    Clusters = [np.array([]) for _ in range(k)]
    Prediction = [-1 for _ in range(data.shape[0])]
    # find the closest mean dot
    for pos, dot in enumerate(data):
      min_distance = float('inf')
      for index, mean_dot in enumerate(means):
        distance = np.sqrt(np.sum((dot - mean_dot) ** 2))
        if distance <= min_distance:
          min_distance = distance
          to_be = index
      if Clusters[to_be].size == 0:
        Clusters[to_be] = np.array([dot])
      else:
        Clusters[to_be] = np.append(Clusters[to_be], [dot], axis=0)
      Prediction[pos] = to_be
    # compute new means
    means_old = means
    means = np.array([[]])
    for cluster in Clusters:
      temp = cluster.mean(axis=0)
      if means.size == 0:
        means = np.array([temp])
      else:
        means = np.append(means, [temp], axis=0)
    # if means doesn't change, then end this loop
    change = np.sum(abs(means - means_old))
    if (change < 0.001):
      # compute Silhouette coefficient
      S = []
      for index, i in enumerate(data):
        label = Prediction[index]
        cluster = Clusters[label]
        # compute a_i
        a_i = (np.sqrt(np.sum((i - cluster) ** 2, axis=1))).mean()
        # compute b_i
        min_distance = 100000000
        for index_j, j in enumerate(means):
          distance = np.sqrt(np.sum((i - j) ** 2))
          if distance < min_distance and index_j != label:
            closest = index_j
        b_i = (np.sqrt(np.sum((i - Clusters[closest]) ** 2, axis=1))).mean()
        # get Silhouette coefficient
        sil_coef = (b_i - a_i) / max(a_i, b_i)
        S.append(sil_coef)
      return (Clusters, np.array(S).mean(), Prediction)

def computeRandCoeffient(data, label, K):
  a = 0
  b = 0
  c = 0
  d = 0
  for i, fixed in enumerate(data):
    for j, others in enumerate(data):
      if i == j:
        continue
      same_kind_real = label[i] == label[j]
      for k, cluster in enumerate(K):
        if fixed in cluster:
          label_fixed = k
        if others in cluster:
          label_others = k
      same_kind_predict = label_fixed == label_others
      if same_kind_real:
        if same_kind_predict:
          a += 0.5
        else:
          b += 0.5
      else:
        if same_kind_predict:
          c += 0.5
        else:
          d += 0.5
  RI = (a + d) / (a + b + c + d)
  return RI

DebugData = 'unsupervise/input/wine.data'
DATASET = '../input/wine.data'
Output = '../output/output.csv'
TrainData, label = getData(DATASET)
# normalization of data
col_min = np.amin(TrainData, axis=0)
col_max = np.amax(TrainData, axis=0)
data = (TrainData - col_min) / (col_max - col_min)
data_cut = PCA(data, 0.6)

# test different k and PCA/non-PCA data
group = 10
sil_coef_pca = []
RI_pca = []
sil_coef_nopca = []
RI_nopca = []
# record the best choice. The choice are value of K, Rand coefficient, whether use PCA, prediction
best = (-1, -1, False, np.array([[]]))
for kind in range(2, group):
  # implement PCA
  print('start k-means with k=', kind, ' and PCA')
  result = KMeans(kind, data_cut)
  K = result[0]
  sil_coef = result[1]
  Prediction = result[2]
  # compute Rand coefficient
  RI = computeRandCoeffient(data_cut, label, K)
  # add result to graph
  sil_coef_pca.append(sil_coef)
  RI_pca.append(RI)
  if RI > best[1]:
    best = (kind, RI, True, Prediction)
  # do not implement PCA
  print('start k-means with k=', kind, ' and without PCA')
  result = KMeans(kind, data)
  K = result[0]
  sil_coef = result[1]
  Prediction = result[2]
  # compute Rand coefficient
  RI = computeRandCoeffient(data, label, K)
  # add result to graph
  sil_coef_nopca.append(sil_coef)
  RI_nopca.append(RI)
  if RI > best[1]:
    best = (kind, RI, False, Prediction)
    
plt.plot(range(2, group), sil_coef_pca, label='Silhouette coefficient with PCA')
plt.plot(range(2, group), RI_pca, label='Rand coefficient with PCA')
plt.plot(range(2, group), sil_coef_nopca, label='Silhouette coefficient without PCA')
plt.plot(range(2, group), RI_nopca, label='Rand coefficient without PCA')
plt.xlabel('K of K-Means')
plt.legend()
plt.show()

print('From the test above, the best choice is using', best[0], 'kinds.', 'The corresponding Rand coefficient is ',best[1])
print('And it', 'uses' if best[2] == True else "doesn't use", 'PCA')

# save result
print('The clustered data are written to output.csv in ../output/. And the first row is the predicted class of each wine')
with open(Output, 'w', newline='') as file:
  writer = csv.writer(file)
  for index, predict in enumerate(best[3]):
    # add my prediction to original data, and output it to a .csv file
    out_row =[predict] + TrainData[index].tolist()
    writer.writerow(out_row)
  