
from PreprocessAssistment import *



    
file_path = "/home/lxu/Documents/StudentLearningProcess/2015_100_skill_builders_main_problems.csv"
output_path = "/home/lxu/Documents/StudentLearningProcess/2015_ASSISTment.pickle"
data = PreprocessAssistment15SkillBuilder(file_path)
data.to_pickle(output_path)
