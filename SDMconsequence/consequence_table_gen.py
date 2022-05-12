# -*- coding: utf-8 -*-

# import unittest
import numpy as np
import sys
import seaborn as sns
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt
# import openpyxl

sys.path.append("..")

import venpy

# make parameter changes before running
def setChanges(mdl, setup):
    if 'changes' in setup.keys():
        cf = False
        for changeFile in setup['changes']:
            if not cf:
                mdl.cmd("SIMULATE>READCIN|" + changeFile)
                cf = True
            else:
                mdl.cmd("SIMULATE>ADDCIN|" + changeFile)

    if 'setvals' in setup.keys():
        for change in setup['setvals']:
            mdl.cmd("SIMULATE>SETVAL|" + change)

# do an ordinary run
def run(mdl, setup, savelist, parmlist):
    setChanges(mdl, setup)

    model.run(runname=setup['name'])

    r = model.result(names=savelist)  # the time series output of interest
    p = model.result(names=parmlist)  # the parameter vector that generated the run
    return r, p

# do a sensitivity run
def runSensitivity(mdl, setup, savelist, parmlist, ssetup, stimes):
    setChanges(mdl, setup)

    model.run(runname=setup['name'], sensitivity=ssetup, debug=True)

    rtimes = []
    for st in stimes:
        r = model.result(names=savelist, sensitivitytime=st)
        # r['Time'] = pd.Series([st for x in range(len(r.index))])
        rtimes.append(r)
    rres = pd.concat(rtimes, keys=stimes, names=['Time', 'Simulation']) # concatenate the multiple dataframes, adding a time index at the outermost level
    p = model.result(names=parmlist)  # the parameter vector that generated the run
    return rres, p

# remove subscripts from a variable name, if any
# this is rather simple and won't handle quoted variable names containing brackets

def unsubbed(varname):
    lBracket = varname.find("[")
    if lBracket > 0:
        return varname[:lBracket]
    else:
        return varname

# create a list that maps unsubscripted names to lists of matching subscripted elements
def unsubbedList(indexList):
    unsub = {}
    for n in indexList:
        u = unsubbed(n)
        if u not in unsub.keys():
            unsub[u] = [n]
        else:
            unsub[u].append(n)
    return unsub

model = venpy.load(
    "C:/Users/Public/Vensim/venpy/SDMconsequence/community virus simple arr v14.vpmx") # use / or \\

# parameters to be extracted from the result set for documentation purposes
# parameters changed in setval statements in the runSetup will be added to this list automatically
# so this only needs to contain additional parameters of interest, including perhaps those in .cin files
parameters = ['r0',
              'high risk share'
              ]
# TODO: would be nice to extract parameters from .cin and .vsc files in use

# desired output metrics to be extracted from the result set
# note that venpy currently expects unsubscripted names,
# and will return all elements if given a variable that has subscripts
# the value judgment associated with each variable sets the heatmap orientation
# this can be set to 'neutral' or just '' to skip heatmapping
metrics = {'total deaths': 'bad',
           'active infectious': 'bad',
           'cumulative cost': 'bad',
           'susceptible': 'good',
           'total recovered': 'neutral',
           'peak prevalence': 'bad'
           }
#TODO: use this list to generate a savelist

# time slices for output extraction
# this is for data reduction, to limit the amount of output reported to just a few points
# often it may be desirable to create some reporting variables in the model, using
# SAMPLE IF TRUE to capture peak or trough behaviors, and
# reporting stocks (levels) to capture cumulative flows
# for sensitivity runs, only these times will be retrieved from Vensim
# for regular runs, the full data will be retrieved, then filtered to these values
# TODO: improve handling of roundoff error
timeSlices = [100, 365]

# run specs

# format is [ {'name': 'myname', 'group': 'mygroup', 'changes':['x.cin', ...], 'setvals': ['varname=123', ...], 'sensitivity': ('control.vsc','save.lst') }, ... ]
# name is the run name for Vensim (avoid illegal filename characters); should be unique but this is not checked
# group is the group to which a run belongs, used to create some hierarchical organization in tables
# changes is a list of .cin file names, like ['a.cin', 'b.cin']. Can be omitted.
# setvals is a list of variable changes, using the .cin format, like ['x=3.5','y=5.9']. Can be omitted.

runs = [{'name': 'Base', 'group': 'Status Quo'},
        {'name': 'Behavior', 'setvals': ['Behavioral Risk Reduction=.8'], 'group': 'Policy'},
        {'name': 'Quarantine', 'setvals': ['Relative Isolation Effectiveness for Low Risk=.8','Potential Quarantine Effectiveness=.8'], 'group': 'Policy'},
        {'name': 'Full', 'changes': ['control.cin'], 'group': 'Policy'},
        ]


# sensitivity is a tuple of control file and savelist. Can be =None.
# if provided, it applies to all runs, as it doesn't really make sense to mix regular and sensitivity runs
sensitivitySetup = ('ccv12 uncert.vsc', 'ccv9 save.lst')
# sensitivitySetup = None

resultFrames = []  # list of pandas dataframes returned by venpy containing results
paramFrames = []  # dataframes containing parameter values
runNames = []  # list of runs used, will be extracted from the run setups in runs[]
groups = []  # list of run group headers, will be extracted from the run setups in runs[]
indexTuples = []  # list of index tuples, consisting of a run name and a group


# first iterate to populate the paramlist
for runSetup in runs:
    if 'setvals' in runSetup.keys():
        for change in runSetup['setvals']:
            param = change.split("=")[0].strip() # note that this doesn't handle a quoted variable name containing = but that should be rare
            if param not in parameters:
                parameters.append(param)

print("Parameter list:")
print(*parameters, sep=', ')  # syntax from https://favtutor.com/blogs/print-list-python


# now do the runs
# collect all the results in a list of dataframes
for runSetup in runs:
    if sensitivitySetup:
        result, param = runSensitivity(model, runSetup, metrics.keys(), parameters, sensitivitySetup, timeSlices)
        # print(rx)
    else:
        result, param = run(model, runSetup, metrics.keys(), parameters)

    # result.info(verbose=True)
    print(result)

    resultFrames.append(result)
    paramFrames.append(param.head(1))  # only need the first row for constants
    runNames.append(runSetup['name'])

    # keep track of the run groups
    if runSetup['group'] not in groups:
        groups.append(runSetup['group'])  # note this means groups list is shorter, and contains only unique groups
    indexTuples.append((runSetup['group'], runSetup['name']))
    model.cmd('SPECIAL>CLEARRUNS')  # prevent annoying overload errors without having to turn off all interaction


# concatenate the resulting dataframes from the runs
# adding the runNames as an hierarchical key, since times will repeat
# name the index columns for clarity
if sensitivitySetup:
    results = pd.concat(resultFrames, keys=indexTuples, names=['Group', 'Run', 'Time', 'Simulation'])
else:
    results = pd.concat(resultFrames, keys=indexTuples, names=['Group', 'Run', 'Time'])

results.info(verbose=True)
print(results)

params = pd.concat(paramFrames, keys=indexTuples, names=['Group', 'Run', 'Time'])
params.info(verbose=True)
print(params)


# report the metrics with heatmap styling
# export to Excel
resultsAtTime = None
for t in timeSlices:
    resultsAtTime = results.query('Time==' + str(t))  # this is clearer than using loc[], but TODO converting number to string probably fragile
    resultsAtTime = resultsAtTime.droplevel('Time')  # drop the time index as it now has only one value

    if sensitivitySetup:
        meanResults = resultsAtTime.groupby(['Group', 'Run']).mean()
        meanResults = meanResults.reindex(indexTuples)  # groupby changes the sort order; this undoes it
    else:
        meanResults = resultsAtTime

    meanResults = meanResults.transpose()
    # print(resultsAtTime)
    styledOutput = meanResults.style

    # color palette will be used to indicate variable value within the range of scenarios
    # here we associate "warm" with "bad" and "cool" with "good"
    # c = sns.color_palette("coolwarm", as_cmap=True)
    # cr = sns.color_palette("coolwarm_r", as_cmap=True)
    # 'RdYlGn' and 'RdYlGn_r' another good pair of options

    nc = len(meanResults.columns)
    idx = unsubbedList(meanResults.index)  # this retrieves the dataframe index as a list of variable names (with subscripts if they exist)
    for metric in metrics.keys():
        if metric in idx.keys():  # should be true unless there's some error
            # orient the color palette
            if metrics[metric] == 'bad':
                c = sns.color_palette("coolwarm", as_cmap=True)
            elif metrics[metric] == 'good':
                c = sns.color_palette("coolwarm_r", as_cmap=True)
            else:
                c = sns.light_palette("seagreen", as_cmap=True)

            # now iterate over the subscripted elements of the named variable
            for element in idx[metric]:
                styledOutput = styledOutput.background_gradient(cmap=c, axis='columns', subset=pd.IndexSlice[element, indexTuples])

    with pd.ExcelWriter("Consequence " + str(t) + ".xlsx") as ex:
        styledOutput.to_excel(ex, sheet_name="Scenarios")

# As long as we have the final timeslice of results, use Seaborn to generate a plot
# https://seaborn.pydata.org/examples/scatterplot_categorical.html
resultsAtTime = resultsAtTime.reset_index()  # seaborn wants data in columns, so this converts index to cols
# print(resultsAtTime)
fig, (ax1, ax2) = plt.subplots(2)  # https://matplotlib.org/3.5.0/gallery/subplots_axes_and_figures/subplots_demo.html
fig.suptitle('Results at Time '+str(t))
sns.set_theme(style="whitegrid", palette="muted")
sns.boxenplot(data=resultsAtTime, x='Run', y="total deaths", hue='Group', ax=ax1, dodge=False)  # dodge avoids decentering of the runs by hue-group
#ax1.set(title="Total Deaths")
sns.swarmplot(data=resultsAtTime, x='Run', y="cumulative cost", hue='Group', size=3, ax=ax2)
#ax2.set(title="Cumulative Cost")
# ax.set(ylabel="")
plt.show()

# dump parameters to a separate Excel file
# with distinct heatmap
resultsAtTime = params.droplevel('Time')  # drop the time index as it now has only one value
resultsAtTime = resultsAtTime.transpose()
print(resultsAtTime)
styledOutput = resultsAtTime.style

nc = len(resultsAtTime.columns)
idx = resultsAtTime.index
# using a more value-neutral color scale
c = sns.color_palette("RdYlGn", as_cmap=True)

for param in parameters:
    if param in idx:
        styledOutput = styledOutput.background_gradient(cmap=c, axis='columns', subset=pd.IndexSlice[param, indexTuples])

with pd.ExcelWriter("Parameters.xlsx") as ex:
    styledOutput.to_excel(ex, sheet_name="Parameters")
