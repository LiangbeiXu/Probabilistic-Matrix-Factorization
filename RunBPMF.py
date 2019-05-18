import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF
from BinaryProbabilisticMatrixFactorization import BPMF
from BPMFskill import BPMFskill
from BPMFSkill import BPMFSkill
from BPMFSkillEncoded import BPMFSkillEncoded
from PreprocessAssistment import PreprocessAssistmentSkillBuilder, PreprocessAssistmentProblemSkill

def train_test_split():
        

if __name__ == "__main__":
    # file_path = "data/ml-100k/u.data"
    models = ['BPMFSkillEncode']
    file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills_section.csv"
    item = 'builder'
    for model in models:
        print(model)
        if item == 'builder':
            data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(file_path)
        elif item == 'skill' or item == 'problem':
            data = PreprocessAssistmentProblemSkill(file_path, item)
        print('statistics ', data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), num_skills)

        if model == 'BPMF':
            pmf = BPMF()
            pmf.set_params({"num_feat": 5, "epsilon": 10, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 50, "num_batches": 300,
                            "batch_size": 1000, "dynamic": False})
            if item == 'builder':
                ratings = data.loc[:,['user_id', 'problem_id', 'correct']].values
                order = list(data.loc[:]['hist'].values)
            elif item == 'skill' or item == 'problem':
                ratings = data.loc[:,['user_id', item+'_id', 'correct']].values
                order = list(data.loc[:]['hist'].values)
            train, test, order_train, order_test = train_test_split(ratings, order, test_size=0.2)  # spilt_rating_dat(ratings)

            pmf.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])))
        elif model == 'BPMFSkill':
            pmf = BPMFSkill()
            pmf.set_params({"num_feat": 5, "epsilon": 5, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000})
            if item == 'builder':
                ratings = data.loc[:,['user_id', 'problem_id', 'correct', 'skill_ids', 'sCount', 'fCount']].values
                order = list(data.loc[:]['hist'].values)
            elif item == 'skill' or item == 'problem':
                ratings = data.loc[:,['user_id', item+'_id', 'correct']].values
                order = list(data.loc[:]['hist'].values)

            train, test, order_train, order_test = train_test_split(ratings, order, test_size=0.2)  # spilt_rating_dat(ratings)
            pmf.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), num_skills, len(np.unique(ratings[:, 1])))
        elif model == 'BPMFSkillEncode':
            pmf = BPMFSkillEncoded()
            pmf.set_params({"num_feat": num_skills, "epsilon": 5, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 40, "num_batches": 300,
                            "batch_size": 1000, 'dynamic':False})
            if item == 'builder':
                ratings = data.loc[:,['user_id', 'problem_id', 'correct', 'skill_ids', 'sCount', 'fCount']].values
                order = list(data.loc[:]['hist'].values)
            elif item == 'skill' or item == 'problem':
                ratings = data.loc[:,['user_id', item+'_id', 'correct']].values
                order = list(data.loc[:]['hist'].values)

            train, test, order_train, order_test = train_test_split(ratings, order, test_size=0.2)  # spilt_rating_dat(ratings)
            pmf.fit(train, test, order_train, order_test, len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), prob_skill_map)

        # Check performance by plotting train and test errors
        plt.plot(range(pmf.maxepoch), pmf.logloss_train, marker='o', label='log los: Training Data')
        plt.plot(range(pmf.maxepoch), pmf.logloss_test, marker='v', label='log loss: Test Data')
        plt.plot(range(pmf.maxepoch), pmf.auc_train, marker='+', label='AUC: Train Data')
        plt.plot(range(pmf.maxepoch), pmf.auc_test, marker='*', label='AUC: Test Data')
        plt.title('The Truncated Assitment Dataset Learning Curve ' + model)
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()
        print("precision_acc,recall_acc:" + str(pmf.topK(test)))
