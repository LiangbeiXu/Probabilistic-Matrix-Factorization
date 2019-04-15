# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt

class BPMFSkillEncoded(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000, dynamic=True):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
        # self.alpha = alpha
        self.dynamic = dynamic

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors
        self.alpha_Item = None  # Item's teaching ability
        self.gamma_Item = None
        self.gamma_User = None
        self.alpha_User = None
        self.gamma = None

        self.logloss_train = []
        self.logloss_test = []
        self.auc_train = []
        self.auc_test = []
        self.baseline_auc_test= []

        self.classification_train = []
        self.classification_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec, train_order, test_order, num_user, num_item, prob_skill_map):
        # columns: userid, itemid, correct,
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数


        incremental = False  # 增量
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.rand(num_item, self.num_feat)  # Item latent matrix
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # User latent matrix
            for i in range(num_item):
                self.w_Item[i,prob_skill_map[i]] = 1/ len(prob_skill_map[i])


            #self.alpha_Item = 0.1 * np.random.uniform(0,1,(num_item, 1))
            self.alpha_Item = 0 * np.random.rand(num_item, 1)   # item teaching ability
            self.alpha_User = 0 * np.random.rand(num_user, 1)   # user learning ability
            self.gamma_Item = 0.1 * np.random.rand(num_item, 1)   # item difficulity
            self.gamma_User = 0.1 * np.random.rand(num_user, 1)   # user ability
            self.gamma = 0.1 * np.random.rand(1)

            self.w_Item_inc = np.zeros((num_item, self.num_feat))
            self.w_User_inc = np.zeros((num_user, self.num_feat))
            self.alpha_Item_inc = np.zeros((num_item, 1))
            self.alpha_User_inc = np.zeros((num_user, 1))
            self.gamma_Item_inc = np.zeros((num_item, 1))
            self.gamma_User_inc = np.zeros((num_user, 1))
            self.gamma_inc = np.zeros(1)

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
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # accumulate the questions
                sum_item = np.zeros((self.batch_size, self.num_feat))
                if  self.dynamic:
                    for i in range(self.batch_size):
                        hist = train_order[shuffled_order[i]]
                        user_id = train_vec[shuffled_order[i], 0]
                        if not hist:
                            continue
                        if len(hist) > 10:
                            hist = hist[0:9]
                        temp = np.matmul(self.alpha_Item[hist] + self.alpha_User[user_id], np.ones((1,self.num_feat)))
                        sum_item[i,:] = np.sum(np.multiply(temp, self.w_Item[hist,:]), axis=0)


                # Compute Objective Function
                new_w_User = self.w_User[batch_UserID, :] + sum_item
                x = np.sum(np.multiply(new_w_User, self.w_Item[batch_ItemID, :]), axis=1) + \
                    self.gamma_Item[batch_ItemID,:].flatten()  + self.gamma_User[batch_UserID,:].flatten() + self.gamma
                expnx = np.exp(-x)
                linkx = np.divide(1, (1+expnx))
                logx = np.log(linkx)
                lognx = np.log(1-linkx)
                # rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                y = train_vec[shuffled_order[batch_idx], 2]
                # logloss = - np.sum(np.multiply(y,logx) - np.multiply(1-y, lognx) )
                gradlogloss = -  np.multiply(y,1-linkx) + np.multiply(1-y, linkx)

                # Compute gradients
                Ix_User = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]

                Ix_Item = 2 * np.multiply(gradlogloss[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :]) + 2 * sum_item   # np.newaxis :increase the dimension
                Ix_gamma_Item = 2 * np.reshape(gradlogloss,(self.batch_size,1)) + self._lambda * self.gamma_Item[batch_ItemID,:]
                Ix_gamma_User = 2 * np.reshape(gradlogloss,(self.batch_size,1)) + self._lambda * self.gamma_User[batch_UserID,:]
                Ix_gamma= 2 * np.sum(gradlogloss) + self._lambda * self.gamma

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))
                dw_alpha = np.zeros((num_item, 1))
                dw_alpha_user = np.zeros((num_user, 1))
                dw_gamma_Item = np.zeros((num_item, 1))
                dw_gamma_User = np.zeros((num_user, 1))
                dw_gamma = Ix_gamma
                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] = dw_Item[batch_ItemID[i], :] + Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] = dw_User[batch_UserID[i], :] + Ix_User[i, :]
                    dw_gamma_Item[batch_ItemID[i], :] = dw_gamma_Item[batch_ItemID[i], :] +  Ix_gamma_Item[i,:]
                    dw_gamma_User[batch_UserID[i], :] = dw_gamma_User[batch_UserID[i], :] +  Ix_gamma_User[i,:]
                    if self.dynamic:
                        hist = train_order[shuffled_order[i]]
                        user_id = train_vec[shuffled_order[i], 0]
                        if not hist:
                            continue
                        if len(hist) > 10:
                            hist = hist[0:9]
                        dw_alpha[hist,:] +=  0.2 * gradlogloss[i] * np.matmul(self.w_Item[hist,: ], np.transpose(self.w_Item[[batch_ItemID[i]],:])) + self._lambda * self.alpha_Item[hist,:]
                        dw_alpha_user[user_id,:] +=  0.2 * gradlogloss[i] * \
                            np.sum(np.matmul(self.w_Item[hist,: ], np.transpose(self.w_Item[[batch_ItemID[i]],:])))+ \
                            self._lambda * self.alpha_User[user_id,:]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size
                self.alpha_Item_inc = self.momentum * self.alpha_Item_inc + self.epsilon * dw_alpha / self.batch_size
                self.alpha_User_inc = self.momentum * self.alpha_User_inc + self.epsilon * dw_alpha_user / self.batch_size
                self.gamma_Item_inc = self.momentum * self.gamma_Item_inc + self.epsilon * dw_gamma_Item / self.batch_size
                self.gamma_User_inc = self.momentum * self.gamma_User_inc + self.epsilon * dw_gamma_User / self.batch_size
                self.gamma_inc = self.momentum * self.gamma_inc + self.epsilon * dw_gamma / self.batch_size


                # self.w_Item = self.w_Item - self.w_Item_inc

                self.w_User = self.w_User - self.w_User_inc
                self.alpha_Item = self.alpha_Item - self.alpha_Item_inc
                self.alpha_User = self.alpha_User - self.alpha_User_inc
                self.gamma_Item = self.gamma_Item - self.gamma_Item_inc
                self.gamma_User = self.gamma_User - self.gamma_User_inc
                self.gamma = self.gamma - self.gamma_inc

                # set to zeros
                # self.w_Item.fill(0)
                # self.alpha_Item.fill(0)
                # self.alpha_User.fill(0)
                # self.gamma_Item.fill(0)
                # self.gamma_User.fill(0)
                self.gamma.fill(0)

                # positive constraint by projection
                self.w_Item = self.w_Item.clip(min=0)


                # Compute Objective Function after
                if batch == self.num_batches - 1:

                    sum_item = np.zeros((pairs_train, self.num_feat))
                    if  self.dynamic:
                        for i in range(pairs_train):
                            hist = train_order[i]
                            user_id = train_vec[i, 0]
                            if not hist:
                                continue
                            if len(hist) > 10:
                                hist = hist[0:9]
                            temp = np.matmul(self.alpha_Item[hist] + self.alpha_User[user_id], np.ones((1,self.num_feat)))
                            sum_item[i,:] = np.sum(np.multiply(temp, self.w_Item[hist,:]), axis=0)
                    # Compute Objective Function
                    new_w_User = self.w_User[np.array(train_vec[:, 0], dtype='int32'), :] + sum_item
                    x = np.sum(np.multiply(new_w_User, self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]), axis=1) + \
                        self.gamma_Item[np.array(train_vec[:, 1], dtype='int32'),:].flatten() + self.gamma_User[np.array(train_vec[:, 0], dtype='int32'),:].flatten() + self.gamma
                    expnx = np.exp(-x)
                    linkx = np.divide(1, (1+expnx))
                    logx = np.log(linkx)
                    lognx = np.log(1-linkx)

                    y = train_vec[:, 2]
                    auc = sklearn.metrics.roc_auc_score(np.array(y, dtype=bool), linkx)

                    logloss =  np.sum(- np.multiply(y,logx) - np.multiply(1-y, lognx) )
                    obj = logloss \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)
                    self.auc_train.append(auc)
                    self.logloss_train.append((obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    sum_item = np.zeros((pairs_test, self.num_feat))
                    if  self.dynamic:
                        for i in range(pairs_test):
                            hist = test_order[i]
                            user_id = test_vec[i,0]
                            if not hist:
                                continue
                            if len(hist) > 10:
                                hist = hist[0:9]
                            temp = np.matmul(self.alpha_Item[hist] + self.alpha_User[user_id], np.ones((1,self.num_feat)))
                            sum_item[i,:] = np.sum(np.multiply(temp, self.w_Item[hist,:]), axis=0)
                    # Compute Objective Function
                    new_w_User = self.w_User[np.array(test_vec[:, 0], dtype='int32'), :] + sum_item
                    x = np.sum(np.multiply(new_w_User, self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]), axis=1) + \
                        self.gamma_Item[np.array(test_vec[:, 1], dtype='int32'),:].flatten()  + self.gamma_User[np.array(test_vec[:, 0], dtype='int32'),:].flatten() + self.gamma
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
                    # plt.hist(self.alpha_Item)
                    # plt.show()
                    if batch == self.num_batches - 1:
                        print('Training logloss: %f, Test logloss %f, Train AUC %f, Test AUC %f' \
                              % (self.logloss_train[-1], self.logloss_test[-1], self.auc_train[-1], self.auc_test[-1]) )

    def predict(self, invID):
        x  = np.dot(self.w_Item, self.w_User[int(invID), :])
        expnx = np.exp(0-x)
        pred = np.divide(1, (1+expnx))
        # return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot 点乘
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
            # self.alpha = parameters.get("alpha", 0)
            self.dynamic = parameters.get("dynamic", True)

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
