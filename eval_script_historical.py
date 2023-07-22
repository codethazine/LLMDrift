from src.visualize import Visualize
from src.analysis import Analysis
import pandas

# Specify parameters
MyAnalysis = Analysis()
MyVisual = Visualize(root_path='figure/')
models_GPT4 = ['openaichat/gpt-4-0314',"openaichat/gpt-4-0613"]
models_GPT35 = ['openaichat/gpt-3.5-turbo-0301',"openaichat/gpt-3.5-turbo-0613"]
GPT_4_MAP = {'openaichat/gpt-4-0314':'March 2023',"openaichat/gpt-4-0613":"June 2023"}
GPT_35_MAP = {'openaichat/gpt-3.5-turbo-0301':'March 2023',"openaichat/gpt-3.5-turbo-0613":"June 2023"}

################# SOLVING MATH PROBLEMS #################
# PRIME Dataset
prime_data = pandas.read_csv(open('generation/PRIME_EVAL.csv'))
prime_metric = 'Accuracy'

# Accuracy 
prime_acc_score, prime_acc_scores_std = MyAnalysis.get_score(prime_data, prime_metric)
print(prime_acc_score)

# Verbosity
prime_verb_score, prime_verb_scores_std = MyAnalysis.get_verbosity(prime_data)
print(prime_verb_score)

print("\n")
# Overlap
prime_overlap_score_gpt4, prime_overlap_scores_std_gpt4 = MyAnalysis.get_overlap(data=prime_data,models=models_GPT4,name=prime_metric)
print(prime_overlap_score_gpt4)
prime_overlap_score_gpt35, prime_overlap_scores_std_gpt35 = MyAnalysis.get_overlap(data=prime_data,models=models_GPT35,name=prime_metric)
print(prime_overlap_score_gpt35)

# Normalize data into a single dataframe
# Columns are: model, timestamp, accuracy, verbosity # SKIP OVERLAP AND CALCULATE IT ON CLIENT SIDE
prime_df = pandas.DataFrame(columns=['model','timestamp','accuracy','verbosity'])
prime_df['model'] = ['openaichat/gpt-3.5-turbo-0301','openaichat/gpt-3.5-turbo-0613', 'openaichat/gpt-4-0314','openaichat/gpt-4-0613']
prime_df['timestamp'] = ['2023-03-01','2023-06-13','2023-03-01','2023-06-14']
prime_df['accuracy'] = [prime_acc_score['openaichat/gpt-3.5-turbo-0301'],prime_acc_score['openaichat/gpt-3.5-turbo-0613'],prime_acc_score['openaichat/gpt-4-0314'],prime_acc_score['openaichat/gpt-4-0613']]
prime_df['verbosity'] = [prime_verb_score['openaichat/gpt-3.5-turbo-0301'],prime_verb_score['openaichat/gpt-3.5-turbo-0613'],prime_verb_score['openaichat/gpt-4-0314'],prime_verb_score['openaichat/gpt-4-0613']]

print(prime_df)


'''
################# CODE GENERATION #################
# LEETCODE Dataset
data = pandas.read_csv(open('generation/LEETCODE_EASY_EVAL.csv'))
name = "leetcode"

# Directly Executable
metric = 'Directly Executable'
score, scores_std = MyAnalysis.get_score(data,metric)
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_4_MAP)
MyVisual.save_figure(name+"_GPT4.svg")
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_35_MAP)
MyVisual.save_figure(name+"_GPT35.svg")

# Verbosity
score, scores_std = MyAnalysis.get_verbosity(data)
MyVisual.plot_bars(score,scores_std,yrange=[0,600],fontsize=fontsize,name_map=GPT_4_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT4.svg")
MyVisual.plot_bars(score,scores_std,yrange=[0,800],fontsize=fontsize,name_map=GPT_35_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT35.svg")

# Overlap
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT4,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT4.svg")
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT35,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT35.svg")


################# VISUAL REASONING #################
# ARC Dataset
data = pandas.read_csv(open('generation/ARC_EVAL.csv'))
name = 'arc'

# Directly Executable
metric = 'Exact Match'
score, scores_std = MyAnalysis.get_score(data,metric)
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_4_MAP)
MyVisual.save_figure(name+"_GPT4.svg")
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_35_MAP)
MyVisual.save_figure(name+"_GPT35.svg")

# Verbosity
score, scores_std = MyAnalysis.get_verbosity(data)
MyVisual.plot_bars(score,scores_std,yrange=[0,400],fontsize=fontsize,name_map=GPT_4_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT4.svg")
MyVisual.plot_bars(score,scores_std,yrange=[0,400],fontsize=fontsize,name_map=GPT_35_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT35.svg")

# Overlap
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT4,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT4.svg")
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT35,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT35.svg")

################# ANSWERING SENSITIVE QUESTIONS #################
# SensitiveQ Dataset
data = pandas.read_csv(open('generation/SENSITIVEQ_EVAL.csv'))
name = 'sensitiveq'

# Directly Executable
metric = 'Answer Rate'
score, scores_std = MyAnalysis.get_score(data,metric)
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_4_MAP)
MyVisual.save_figure(name+"_GPT4.svg")
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_35_MAP)
MyVisual.save_figure(name+"_GPT35.svg")

# Verbosity
score, scores_std = MyAnalysis.get_verbosity(data)
MyVisual.plot_bars(score,scores_std,yrange=[0,800],fontsize=fontsize,name_map=GPT_4_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT4.svg")
MyVisual.plot_bars(score,scores_std,yrange=[0,500],fontsize=fontsize,name_map=GPT_35_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT35.svg")

# Overlap
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT4,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT4.svg")
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT35,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT35.svg")
'''