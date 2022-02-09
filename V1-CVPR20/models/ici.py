import math
import random

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize


class ICI(object):

    def __init__(self, classifier='lr', num_class=None, step=5, max_iter='auto',
                 reduce='pca', d=5, norm='l2'):
        self.step = step
        self.max_iter = max_iter
        self.num_class = num_class
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.initial_classifier(classifier)
        self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True,
                                         normalize=True, warm_start=True, selection='cyclic')

    def fit(self, X, y):
        self.support_X = self.norm(X)
        self.support_y = y

    def predict(self, X, unlabel_X=None, show_detail=False, query_y=None):  #query_y=标签
        support_X, support_y = self.support_X, self.support_y
        way, num_support = self.num_class, len(support_X)
        query_X = self.norm(X)
        if unlabel_X is None:   #如果没有无标签数据，就先拿query_X占个位
            unlabel_X = query_X
        else:
            unlabel_X = self.norm(unlabel_X)
        num_unlabel = unlabel_X.shape[0]
        assert self.support_X is not None
        embeddings = np.concatenate([support_X, unlabel_X]) #对task的支持集和无标签集降维
        X = self.embed(embeddings)
        H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)   #Eq(4), 令w偏导=0解出w后反代回，得系数 X * (X'X)^(-1) * X'
        X_hat = np.eye(H.shape[0]) - H                              #redefine x_hat = (I-H)
        if self.max_iter == 'auto':     #ICI分类器需要迭代。这里是在计算最大迭代次数
            # set a big number
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            self.max_iter = math.ceil(num_unlabel/self.step)
        else:
            assert float(self.max_iter).is_integer()
        support_set = np.arange(num_support).tolist()
        self.classifier.fit(self.support_X, self.support_y)
        if show_detail:
            acc_list = []
        for _ in range(self.max_iter):
            if show_detail:
                predicts = self.classifier.predict(query_X)
                acc_list.append(np.mean(predicts == query_y))   #计算当前时刻，对查询集的acc
            pseudo_y = self.classifier.predict(unlabel_X)
            y = np.concatenate([support_y, pseudo_y])           #揉在一起准备洗牌了
            Y = self.label2onehot(y, way)                       #独热编码，[(支持集+无监督集), way数]
            y_hat = np.dot(X_hat, Y)                            #Eq(4) redefine Y_hat = X_hat*Y
            support_set = self.expand(support_set, X_hat, y_hat, way, num_support, pseudo_y,
                                      embeddings, y)            #关键代码：洗牌！
            y = np.argmax(Y, axis=1)
            self.classifier.fit(embeddings[support_set], y[support_set])
            if len(support_set) == len(embeddings):
                break
        predicts = self.classifier.predict(query_X)
        if show_detail:
            acc_list.append(np.mean(predicts == query_y))
            return acc_list
        return predicts


    def expand(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, targets):
        _, coefs, _ = self.elasticnet.path(X_hat, y_hat, l1_ratio=1.0)  # Compute elastic net path with coordinate descent.
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[
                       ::-1, num_support:, :]), axis=2) #把五个值加起来了
        selected = np.zeros(way)
        for idx, gamma in enumerate(coefs):
            for i, g in enumerate(gamma):       #选择gamma为0的送入支持集
                if (g == 0.0) and (i+num_support not in support_set) and (selected[pseudo_y[i]] < self.step):
                    support_set.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:    #五类都获得足够样本时，即提前跳出。本次是91
                break
        return support_set

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ['isomap', 'ltsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce == 'ltsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense')
        elif reduce == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

    def initial_classifier(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', kernel='linear',probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result
