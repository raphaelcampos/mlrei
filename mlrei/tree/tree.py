import numpy as np

def entropy_criterion(data, labels):
  """ Entropy
  Parameters
  ----------
  data: numpy array-like = [n_samples, n_features]
  labels: numpy array-like, shape = [n_samples]
  
  Return
  ------
  entropy: float
  """
  classes = np.unique(labels)
  
  s = 0
  for c in classes:
    p = np.mean(labels == c)
    s -= p * np.log(p)
    
  return s
  

def gini_criterion(data, labels):
  """ Gini Index
  Parameters
  ----------
  data: numpy array-like = [n_samples, n_features]
  labels: numpy array-like, shape = [n_samples]
  
  Return
  ------
  gini: float
  """
  classes = np.unique(labels)
  
  s = 0
  for c in classes:
    p = np.mean(labels == c)
    s += p * (1 - p)
    
  return s


def find_cut_point(data, labels, impurity_criterion = gini_criterion):
  """ find the best cut point 
  
  Parameters
  ----------
  data: numpy array-like = [n_samples, n_features]
  labels: numpy array-like, shape = [n_samples]
  impurity_criterion: callable, default=gini_criterion
  
  Return
  ------
  feature, threshold
  """
  n_samples, n_features = data.shape

  max_info_gain = np.iinfo(np.int32).min
  feat_id = 0
  best_threshold = 0

  # pré-calculando a impureza da região atual
  H_parent = impurity_criterion(data, labels)
  # para cada um dos atributos
  # vamos tentar encontrar o limiar que maximiza o ganho de informação
  for j in range(n_features):
    # só nos interessa os valores ordenados únicos 
    # do atributo j nessa região do espaço
    values = np.unique(data[:, j])
    
    for i in range(values.shape[0] - 1):
      # usamos o ponto médio dos valores possíveis
      # como limiar candidato para o ponto de corte
      threshold = (values[i] + values[i + 1]) / 2.

      mask = data[:, j] <= threshold

      info_gain = H_parent \
                  - (mask.sum() * impurity_criterion(data[mask], labels[mask]) \
                  + (~mask).sum() * impurity_criterion(data[~mask], labels[~mask])) \
                  / float(n_samples)

      if max_info_gain < info_gain:
        best_threshold = threshold
        feat_id = j
        max_info_gain = info_gain
        
  return feat_id, best_threshold


def stopping_criterion(n_classes, depth, max_depth):
  """ Stopping criterion

  Parameters
  ----------
  n_classe: int
            number of classes in the region, one means that the region is pure.
  depth: int,
          current tree depth
  max_depth: int, default=None
          maximal tree depth. None for fully grown tree.

  Return
  ------
  bool
  """
  return (max_depth is not None and max_depth == depth) or (n_classes == 1)

def build_tree(data, labels, tree, depth = 1):
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = classes.shape[0]

    # critério de parada
    if not stopping_criterion(n_classes, depth, tree.max_depth):
        node = Node()

        # encontra melhor ponto de corte dado a região atual do espaço
        # de acordo com critério de impureza escolhido
        feature, threshold = find_cut_point(data, labels, 
                                            tree.impurity_criterion)
        
        # aplicando o limiar para particionar o espaço
        mask = data[:, feature] <= threshold
        
        # contruindo árvore recursivamente para
        # os sub-espaço da direita e da esquerda.
        left = build_tree(data[mask], labels[mask], tree, depth + 1)
        right = build_tree(data[~mask], labels[~mask], tree, depth + 1)
     
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    # calcula a quantidade de exemplos por classe nesse nó folha
    # e instancia um nó folha com essas quantidades, lembre-se que isso
    # será usado para predição. 
    values = np.zeros(tree.n_classes)
    values[classes] = counts
    return Node(is_leaf=True, counts=values)


class Node(object):
  """Node"""
  def __init__(self, feature=None, threshold=None,
                     is_leaf=None, counts=None, left=None, right=None):
    super(Node, self).__init__()
    self.threshold = threshold
    self.is_leaf = is_leaf
    self.counts = counts
    self.left = left
    self.right = right
    self.feature = feature
    

class DecisionTreeClassifier(object):
  """DecisionTreeClassifier

  Parameters
  ----------
  max_depth:

  impurity_criterion:

  """
  def __init__(self, max_depth, impurity_criterion = gini_criterion):
    super(DecisionTreeClassifier, self).__init__()
    self.max_depth = max_depth
    self.impurity_criterion = impurity_criterion

  def recursive_predict(self, node, X):

    if node.is_leaf:
      return np.zeros(X.shape[0]) + np.argmax(node.counts)

    mask = X[:, node.feature] <= node.threshold

    y_pred = np.zeros(X.shape[0])
    if mask.sum() > 0:
      y_pred[mask] = self.recursive_predict(node.left, X[mask])

    if (~mask).sum() > 0:
      y_pred[~mask] = self.recursive_predict(node.right, X[~mask])

    return y_pred

  def fit(self, X, y):
    self.classes = np.unique(y)
    self.n_classes = self.classes.shape[0]

    self.root = build_tree(X, y, self)

    return self

  def predict(self, X):
    return self.recursive_predict(self.root, X)

from sklearn.datasets import load_iris
if __name__ == '__main__':
  X = np.array([[1,1], [1,0], [0,1], [0,0]])
  y = np.array([0, 1, 1, 0])

  dt = DecisionTreeClassifier(max_depth=None, impurity_criterion = entropy_criterion)

  print(dt.fit(X, y).predict(X))
  
  X, y = load_iris(return_X_y=True)

  y_pred = dt.fit(X, y).predict(X)

  print(np.mean(y_pred == y))
