#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:42:44 2019

@author: lxu
"""

import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF
from BinaryProbabilisticMatrixFactorization import BPMF

if __name__ == "__main__":
    # file_path = "data/ml-100k/u.data"
    file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills_finished.csv"
    pmf = BPMF()
    pmf.set_params({"num_feat": 6, "epsilon": 2, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 100, "num_batches": 300,
                    "batch_size": 1000})
    ratings, order = load_rating_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    
    train, test, order_train, order_test = train_test_split(ratings, order, test_size=0.2)  # spilt_rating_dat(ratings)
    
    pmf.two_step_fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])))

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.logloss_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.logloss_test, marker='v', label='Test Data')
    plt.plot(range(pmf.maxepoch), pmf.auc_test, marker='*', label='AUC: Test Data')
    plt.title('The Truncated Assitment Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))
