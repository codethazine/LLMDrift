from src.visualize import Visualize
from src.analysis import Analysis
import pandas

# Specify parameters
MyAnalysis = Analysis()
MyVisual = Visualize(root_path='figure/')
fontsize=50
models_GPT4 = ['openaichat/gpt-4-0314',"openaichat/gpt-4-0613"]
models_GPT35 = ['openaichat/gpt-3.5-turbo-0301',"openaichat/gpt-3.5-turbo-0613"]
GPT_4_MAP = {'openaichat/gpt-4-0314':'March 2023',"openaichat/gpt-4-0613":"June 2023"}
GPT_35_MAP = {'openaichat/gpt-3.5-turbo-0301':'March 2023',"openaichat/gpt-3.5-turbo-0613":"June 2023"}

################# SOLVING MATH PROBLEMS #################
# PRIME Dataset
data = pandas.read_csv(open('generation/PRIME_EVAL.csv'))
name = "prime"

# Accuracy 
metric = 'Accuracy'
score, scores_std = MyAnalysis.get_score(data,metric)
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_4_MAP)
MyVisual.save_figure(name+"_GPT4.svg")
MyVisual.plot_bars(score,scores_std,fontsize=fontsize,name_map=GPT_35_MAP)
MyVisual.save_figure(name+"_GPT35.svg")

# Verbosity
score, scores_std = MyAnalysis.get_verbosity(data)
MyVisual.plot_bars(score,scores_std,yrange=[0,1000],fontsize=fontsize,name_map=GPT_4_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT4.svg")
MyVisual.plot_bars(score,scores_std,yrange=[0,1200],fontsize=fontsize,name_map=GPT_35_MAP,no_text=True)
MyVisual.save_figure(name+"_verbose_GPT35.svg")

# Overlap
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT4,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT4.svg")
score, scores_std = MyAnalysis.get_overlap(data=data,models=models_GPT35,name=metric)
MyVisual.plot_bar(score,scores_std,fontsize=fontsize)
MyVisual.save_figure(name+"_overlap_GPT35.svg")

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