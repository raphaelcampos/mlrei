
import numpy as np
import copy

class Bagging(object):
  """docstring for Bagging"""
  def __init__(self, base_estimator, n_estimators=50, random_state=None):
    super(Bagging, self).__init__()
    self.base_estimator = base_estimator
    self.n_estimators = n_estimators
    self.random_state = random_state

  def fit(self, X, y):
    random_instance = np.random.RandomState(self.random_state)
    
    n_samples, n_features = X.shape

    self.classes = np.unique(y)

    self.estimators_ = []
    for i in range(self.n_estimators):
      # fazemos a amostragem aleatória por reposição dos exemplos
      # de treino (esse é procedimento de bootstrap)
      samples = random_instance.choice(np.arange(0, n_samples), n_samples)

      # a cópia do método base é feita, pois teremos um
      # modelo por amostra do treino.
      estimator = copy.copy(self.base_estimator)

      estimator.fit(X[samples], y[samples])

      # após o treino de cada modelo, nós os armazenamos para
      # utilizá-los no momento de predição
      self.estimators_.append(estimator)

    return self

  def predict(self, X):
    pass


class BaggingClassifier(Bagging):
  """docstring for BaggingClassifier"""

  def predict(self, X):
    if len(self.estimators_) <= 0:
      raise Exception("Model is not fitted yet!")

    n_classes = self.classes.shape[0]
    
    pred_count = np.zeros((X.shape[0], n_classes))
    for e in self.estimators_:
      pred_count[np.arange(X.shape[0]), e.predict(X).astype(int)] += 1


    print(pred_count)

    return np.argmax(pred_count, axis=1)
