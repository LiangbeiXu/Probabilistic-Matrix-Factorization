
from PreprocessAssistment import *

from IRT import IRT

import matplotlib.pyplot as plt


file_path = "/home/lxu/Documents/StudentLearningProcess/2015_100_skill_builders_main_problems.csv"
data = PreprocessAssistment15SkillBuilder(file_path)
print('ASSISTment 14-15 statistics ', data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])))


# prepare data
datanp = data.values
trainnp, testnp = train_test_split(datanp, test_size=0.2)
#
columnNames = list(data.head(0))
train = pd.DataFrame(data=trainnp, columns=columnNames)
test = pd.DataFrame(data=testnp, columns=columnNames)

def PlotLoss(model, model_name):
    plt.plot(range(model.maxepoch), model.logloss_train, marker='o', label='Training Data')
    plt.plot(range(model.maxepoch), model.logloss_test, marker='v', label='Test Data')
    plt.plot(range(model.maxepoch), model.auc_train, marker='*', label='AUC: Train Data')
    plt.plot(range(model.maxepoch), model.auc_test, marker='*', label='AUC: Test Data')
    plt.title('The Truncated Assitment Dataset Learning Curve ' + model_name)
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
# baseline model: global + user  + skill
name = 'baseline'
skill1 = IRT()
skill1.set_params({ "epsilon": 1, "_lambda": 0.2, "momentum": 0.5, "maxepoch": 50, "num_batches": 300,
                            "batch_size": 1000, 'MF_skill':True})
skill1.fit(train, test, len(np.unique(data['user_id'])), len(np.unique(data['skill_id'])), 10) # IRT

PlotLoss(skill1, name)
