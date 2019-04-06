#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:09:58 2019

@author: lxu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PerformanceFactorAnalysis import PFA
from copy import deepcopy
import matplotlib.pyplot as plt

from PreprocessAssistment import PreprocessAssistmentSkillBuilder, PreprocessAssistmentProblemSkill

# not complete yet

if __name__ == "__main__":
    # file_path = "data/ml-100k/u.data"
    file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills.csv"
    item = 'builder'
    model = PFA()
    model.set_params({"num_feat": 6, "epsilon": 5, "_lambda": 0, "alpha": 0, "momentum": 0.8, "maxepoch": 20, "num_batches": 300,
                    "batch_size": 1000})
    data = PreprocessAssistmentSkillBuilder(file_path)   
    
    print(data.shape[0], len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))
    
    trainnp, testnp = train_test_split(data, test_size=0.2)
    
    model.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])))

    # Check performance by plotting train and test errors
    plt.plot(range(model.maxepoch), model.logloss_train, marker='o', label='Training Data')
    plt.plot(range(model.maxepoch), model.logloss_test, marker='v', label='Test Data')
    plt.plot(range(model.maxepoch), model.auc_test, marker='*', label='AUC: Test Data')
    plt.title('The Truncated Assitment Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    print("precision_acc,recall_acc:" + str(model.topK(test)))