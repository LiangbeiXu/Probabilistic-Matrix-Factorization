# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt

class BPMFSkill(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)


        self.w_Skill = None  # Item feature vectors
        self.w_User = None  # User feature vectors
        self.w_Skill_inc = None
        self.w_User_inc = None


        self.logloss_train = []
        self.logloss_test = []
        self.auc_train = []
        self.auc_test = []
        self.baseline_auc_test= []

        self.classification_train = []
        self.classification_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec, train_order, test_order, num_user, num_skill, num_prob):
        # columns: userid, prodid, correct,
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数


        incremental = False  # 增量
        if ((not incremental) or (self.w_Skill is None)):
            # initialize
            self.epoch = 0
            self.w_Skill = 0.1 * np.random.rand(num_skill, self.num_feat)  # skill latent matrix
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # User latent matrix

            self.w_Skill_inc = np.zeros((num_skill, self.num_feat))
            self.w_User_inc = np.zeros((num_user, self.num_feat))

        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_SkillIDs = train_vec[shuffled_order[batch_idx], 3]

                # accumulate the questions
                sum_w_Skill = np.zeros((self.batch_size, self.num_feat))

                for i in range(self.batch_size):
                    sum_w_Skill[i,:] = np.sum(self.w_Skill[batch_SkillIDs[i],:],axis=0) / len(batch_SkillIDs[i])


                # Compute Objective Function
                new_w_User = self.w_User[batch_UserID, :]
                x = np.sum(np.multiply(new_w_User, sum_w_Skill), axis=1)
                expnx = np.exp(-x)
                linkx = np.divide(1, (1+expnx))

                y = train_vec[shuffled_order[batch_idx], 2]
                # logloss = - np.sum(np.multiply(y,logx) - np.multiply(1-y, lognx) )
                gradlogloss = -  np.multiply(y,1-linkx) + np.multiply(1-y, linkx)

                # Compute gradients
                Ix_User = np.zeros((self.batch_size, self.num_feat), dtype=float)
                Ix_User = 2 * np.multiply(gradlogloss[:, np.newaxis], sum_w_Skill) \
                       + self._lambda * self.w_User[batch_UserID, :]

                dw_Skill = np.zeros((num_skill, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat), dtype=float)


                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_User[batch_UserID[i], :] = dw_User[batch_UserID[i], :]  = Ix_User[i, :]
                    for idx, skill in enumerate(batch_SkillIDs[i]):
                        dw_Skill[skill, :] = dw_Skill[skill, :] +  2 * gradlogloss[i] * self.w_User[batch_UserID[i], :] / len(batch_SkillIDs[i]) + self._lambda * (self.w_Skill[skill, :])


                # Update with momentum
                self.w_Skill_inc = self.momentum * self.w_Skill_inc + self.epsilon * dw_Skill / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size


                self.w_Skill = self.w_Skill - self.w_Skill_inc
                self.w_User = self.w_User - self.w_User_inc


                # positive constraint by projection
                self.w_Skill = self.w_Skill.clip(min=0)


                # Compute Objective Function after
                if batch == self.num_batches - 1:

                    batch_SkillIDs = train_vec[:, 3]
                    sum_w_Skill = np.zeros((pairs_train, self.num_feat))

                    for i in range(pairs_train):
                        sum_w_Skill[i,:] = np.sum(self.w_Skill[batch_SkillIDs[i],:],axis=0) / len(batch_SkillIDs[i])
                    # Compute Objective Function
                    new_w_User = self.w_User[np.array(train_vec[:, 0], dtype='int32'), :]
                    x = np.sum(np.multiply(new_w_User, sum_w_Skill), axis=1)
                    expnx = np.exp(-x)
                    linkx = np.divide(1, (1+expnx))
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)

                    y = train_vec[:, 2]

                    logloss =  np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx) )
                    auc = sklearn.metrics.roc_auc_score(np.array(y, dtype=bool), linkx)
                    obj = logloss \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Skill) ** 2)

                    self.logloss_train.append((obj / pairs_train))
                    self.auc_train.append(auc)

                # Compute validation error
                if batch == self.num_batches - 1:
                    batch_SkillIDs = test_vec[:, 3]
                    sum_w_Skill = np.zeros((pairs_test, self.num_feat))

                    for i in range(pairs_test):
                        sum_w_Skill[i,:] = np.sum(self.w_Skill[batch_SkillIDs[i],:],axis=0) / len(batch_SkillIDs[i])
                    # Compute Objective Function
                    new_w_User = self.w_User[np.array(test_vec[:, 0], dtype='int32'), :]
                    x = np.sum(np.multiply(new_w_User, sum_w_Skill), axis=1)
                    expnx = np.exp(-x)
                    linkx = np.divide(1, (1+expnx))
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)
                    y = test_vec[:, 2]
                    y2 = self.mean_inv * np.ones(y.shape)
                    logloss = np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx) )

                    auc = sklearn.metrics.roc_auc_score(np.array(y, dtype=bool), linkx)
                    baseline_auc = sklearn.metrics.roc_auc_score(np.array(y, dtype=bool), y2)
                    self.auc_test.append(auc)
                    self.baseline_auc_test.append(baseline_auc)
                    self.logloss_test.append((logloss) / (pairs_test))

                    # Print info
                    # plt.hist(self.alpha_Skill)
                    # plt.show()
                    if batch == self.num_batches - 1:
                        print('Training logloss: %f, Test logloss %f, Train AUC %f, Test AUC %f' \
                              % (self.logloss_train[-1], self.logloss_test[-1], self.auc_train[-1], self.auc_test[-1]) )

    def predict(self, invID):
        x  = np.dot(self.w_Skill, self.w_User[int(invID), :])
        expnx = np.exp(0-x)
        pred = np.divide(1, (1+expnx))
        # return np.dot(self.w_Skill, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot 点乘
        return pred

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)


    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  # numpy.argsort索引排序

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)
