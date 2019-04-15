
from PreprocessAssistment import *
file_path = "/home/lxu/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills_section.csv"
data, num_skills, prob_skill_map = PreprocessAssistmentSkillBuilder(file_path)


num_user = len(np.unique(data['user_id']))
num_prob = len(np.unique(data['problem_id']))
prob_cnt_per_user = np.zeros((num_user))
for i in range(data.shape[0]):
    prob_cnt_per_user[data.loc[i, 'user_id']] += 1

plt.hist(prob_cnt_per_user)
plt.show()
