# RIDGE_LASSO_REGRESSION.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses ridge, lasso, and elastic net regression to predict body mass index (bmi = weight / height^2), with built-in cross-validation for parameters
# Training data: empirical health-related data for 741 persons (from https://www.kaggle.com/datasets/rukenmissonnier/age-weight-height-bmi-analysis)

import time
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from colorama import Fore, Style
from sklearn import linear_model as lm

time0 = time.time()
ver = ''  # version (empty or integer)

topic = 'Ridge/lasso regression'
topic_underscore = topic.replace(' ','_')
topic_underscore = topic_underscore.replace('/','_')

#--------------#
#  PARAMETERS  #
#--------------#

y_var = 'bmi'                                   # target variable
covars = ['weight', 'height', 'age']                  # feature variables
reg_types = ['ridge', 'lasso', 'elastic net']   # regression types to examine
include_intercept = False                       # whether to also fit an intercept (generally unneeded with these particular regression types)

# alphas - ridge
ridge_alpha_min = 0.01
ridge_alpha_max = 10
ridge_alpha_increment = 0.01

# alpha - elastic net
elastic_net_alpha_min = 0.01
elastic_net_alpha_max = 0.99
elastic_net_alpha_increment = 0.01

var_units = {'bmi':'kg/m^2', 'weight':'kg', 'height':'m', 'age':'yrs'}

decimal_places = 2

#-----------------#
#  DATA CLEANING  #
#-----------------#

total_covars = len(covars)

y_var_zscale = y_var + ' z'
covars_zscale = [c + ' z' for c in covars]

# import training data
bmi = pd.read_csv('bmi.csv')
total_obs = len(bmi)

# column names -> lowercase
for col in bmi.columns:
    bmi[col.lower()] = bmi[col]
    bmi.drop(columns=col, axis=1, inplace=True)

# bmi levels (discrete)
bmi.rename(columns={'bmiclass':'bmi group 6'}, inplace=True) 
bmi['bmi group 6'] = [bmi['bmi group 6'][i].lower() for i in bmi.index]  # converts column values to lowercase
bmi['bmi group 4'] = bmi['bmi group 6'].apply(lambda x: 'obese' if x[0:len('obese')] == 'obese' else x)  # 4-level bmi (collapses all 'obese' into one level)

# z-scores
for var in itertools.chain([y_var], covars):
    bmi[var + ' z'] = zscore(bmi[var])

# correlation matrix
corr_vars = [y_var] + covars
corr_matrix = bmi[corr_vars].corr(method='pearson')

# training data stats
training_stats = pd.DataFrame()
training_stats['feature'] = [y_var] + covars
training_stats['mean'] = training_stats['feature'].apply(lambda x: np.mean(bmi[x]))
training_stats['stdev'] = training_stats['feature'].apply(lambda x: np.std(bmi[x]))
training_stats['units'] = training_stats['feature'].apply(lambda x: var_units[x])
training_stats.drop('feature', axis=1, inplace=True)
training_stats = round(training_stats, decimal_places)
training_stats.index = [y_var] + covars

#-----------------------#
#  REGRESSION FUNCTION  #
#-----------------------#

def ridge_lasso_elasticnet(reg_type):
    # model data
    y = bmi[y_var_zscale]
    x = bmi[covars_zscale]
    
    total_params = total_covars + int(include_intercept) + int(reg_type == 'elastic net') + 1  # plus one is for the 'alpha' parameter
    
    # run regression
    if reg_type == 'ridge':
        eqn = lm.RidgeCV(alphas=np.arange(ridge_alpha_min, ridge_alpha_max, ridge_alpha_increment), fit_intercept=include_intercept)
    elif reg_type == 'lasso':
        eqn = lm.LassoCV(fit_intercept=include_intercept)
    elif reg_type == 'elastic net':
        eqn = lm.ElasticNetCV(l1_ratio=np.arange(elastic_net_alpha_min, elastic_net_alpha_max, elastic_net_alpha_increment), fit_intercept=include_intercept)
    reg = eqn.fit(x, y)

    # intercept
    try:
        intercept_z = reg.intercept_[0]
    except:
        intercept_z = reg.intercept_

    # coefficients
    coefs = pd.DataFrame(np.resize(reg.coef_, (total_covars, 1)))
    coefs.loc[len(coefs)] = intercept_z
    coefs.index = coef_names

    # lasso weight (if applicable)
    if reg_type == 'elastic net':
        lasso_weight_best = round(reg.l1_ratio_, decimal_places)
    else:
        lasso_weight_best = 'n/a'
    
    # predictions (z-scale)
    preds = pd.DataFrame(reg.predict(x))
    
    # model summary
    summary = pd.DataFrame()
    summary.index = summary_index_vals
    alpha_best = round(reg.alpha_, decimal_places)
    incl_intercept_str = 'yes' if (include_intercept == True) else 'no'
    r_sq = reg.score(x, y)
    # adjusted R-squared (both rewards models that have better fits, and penalizes models that have more parameters)
    adj_r_sq = 1 - (1 - r_sq) * (total_obs - 1) / (total_obs - total_params - 1)  # see https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
    adj_r_sq = round(adj_r_sq, decimal_places)
    summary[reg_type] = [total_obs, total_covars, incl_intercept_str, adj_r_sq, alpha_best, lasso_weight_best]
    
    return summary, coefs, preds

#-------------------#
#  RUN REGRESSIONS  #
#-------------------#

# summary dataframe
reg_summary = pd.DataFrame()
summary_index_vals = ['total obs', 'total covariates', 'intercept', 'adjusted R^2', 'alpha (best)', 'lasso weight (best)']
reg_summary.index = summary_index_vals

# coefficient names
coef_names = []
for var in covars_zscale:
    coef_names.append(var)
coef_names.append('intercept')

# coefficients dataframe
reg_coefs = pd.DataFrame()
reg_coefs.index = coef_names

# loop over regression scenarios
reg_preds = pd.DataFrame()
for reg in reg_types:
    summary, coefs, preds = ridge_lasso_elasticnet(reg)  # runs regression
    reg_summary[reg] = summary
    reg_coefs[reg] = round(coefs, decimal_places)
    reg_preds[y_var_zscale + ' pred ' + reg] = preds
    del summary, coefs, preds

# convert all z-score predictions to original scale values
for reg in reg_types:
    reg_preds[y_var + ' pred ' + reg] = np.mean(bmi[y_var]) + (reg_preds[y_var_zscale + ' pred ' + reg] * np.std(bmi[y_var]))  # z = (x - mean) / sigma

#--------#
#  PLOT  #
#--------#

plot_title_y_margin = 1.03  # margin to leave above plot for plot title (one = no buffer)

# generate pairplot
hue_var = 'bmi group 4'
pairplot_vars = [y_var] + covars + [hue_var]
sns.set_style('darkgrid')
bmi.sort_values(by='bmi', inplace=True)  # sort so the order of the bmi groups will be monotonic (for plot legend)
fig1 = sns.pairplot(bmi[pairplot_vars], hue=hue_var)
bmi.sort_index(inplace=True)
fig1.fig.suptitle('Scatter plot matrix - training data', y=plot_title_y_margin, fontsize=25, fontweight='bold')
plt.show(True)

#----------#
#  EXPORT  #
#----------#

# functions
def console_print(subtitle, df, first_line):
    if first_line == 1:
        print(Fore.GREEN + '\033[1m' + '\n' + title_top + Style.RESET_ALL)
    print(Fore.GREEN + '\033[1m' + '\n' + subtitle + Style.RESET_ALL)
    print(df)
def txt_export(subtitle, df, f, first_line):
    if first_line == 1:
        print(title_top, file=f)
    print('\n' + subtitle, file=f)
    print(df, file=f)

# parameters
file_title = topic_underscore
title_top = topic.upper()
dfs = [training_stats, reg_coefs, reg_summary]
df_labels = ['training data', 'coefficients', 'summary']

# export summaries (console, txt)
with open(file_title + '_summary' + ver + '.txt', 'w') as f:
    for out in ['console', 'txt']:
        first_line = 1
        for d in range(len(dfs)):
            subtitle = df_labels[d].upper()
            if out == 'console':
                console_print(subtitle, dfs[d], first_line)  # console print
            elif out == 'txt':
                txt_export(subtitle, dfs[d], f, first_line)  # txt file
            first_line = 0
del f, subtitle, out, dfs, first_line

# export plot (png)
fig1.savefig(file_title + '_plots' + ver + '.png')

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


