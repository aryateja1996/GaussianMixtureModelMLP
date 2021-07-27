import numpy as np
import pandas as panda
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()

X = iris.data[:, :2]

d= panda.DataFrame(X)

plot.scatter(d[0],d[1])

gmm = GaussianMixture(n_components=3)

gmm.fit(d)

labels = gmm.predict(d)

d['labels'] =labels

d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]

plot.scatter(d0[0], d0[1], c ='r')
plot.scatter(d1[0], d1[1], c ='yellow')
plot.scatter(d2[0], d2[1], c ='g')

print(gmm.lower_bound_)

print(gmm.n_iter_)