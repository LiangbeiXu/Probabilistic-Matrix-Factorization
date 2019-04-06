import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF
from BinaryProbabilisticMatrixFactorization import BPMF
from PreprocessAssistment import PreprocessAssistmentSkillBuilder, PreprocessAssistmentProblemSkill

if __name__ == "__main__":
    # file_path = "data/ml-100k/u.data"
    file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills.csv"
    item = 'problem'
    pmf = BPMF()
    pmf.set_params({"num_feat": 6, "epsilon": 5, "_lambda": 0, "momentum": 0.8, "maxepoch": 40, "num_batches": 300,
                    "batch_size": 1000, "dynamic": True})
    if item == 'builder': 
        data = PreprocessAssistmentSkillBuilder(file_path) 
        ratings = data.loc[:,['user_id', 'problem_id', 'correct']].values
        order = list(data.loc[:]['hist'].values)
    elif item == 'skill' or item == 'problem':
        data = PreprocessAssistmentProblemSkill(file_path, item)
        ratings = data.loc[:,['user_id', item+'_id', 'correct']].values
        order = list(data.loc[:]['hist'].values)
    
    print(ratings.shape[0], len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    
    train, test, order_train, order_test = train_test_split(ratings, order, test_size=0.2)  # spilt_rating_dat(ratings)
    
    pmf.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])))

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
