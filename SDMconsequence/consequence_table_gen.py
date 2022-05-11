# -*- coding: utf-8 -*-
# import unittest
import numpy as np
import sys
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import openpyxl

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
def runSensitivity(mdl, setup, savelist, parmlist, scontrol, ssavelist, stimes):
    setChanges(mdl, setup)

    model.run(runname=setup['name'], sensitivity=(scontrol, ssavelist), debug=True)

    rtimes = []
    for st in stimes:
        r = model.result(names=savelist, sensitivitytime=st)
        r['Time'] = st
        rtimes.append(r)
    rres = pd.concat(rtimes, keys=stimes, names=['Time', 'Simulation'])
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

# time slices for output extraction
# this is for data reduction, to limit the amount of output reported to just a few points
# often it may be desirable to create some reporting variables in the model, using
# SAMPLE IF TRUE to capture peak or trough behaviors, and
# reporting stocks (levels) to capture cumulative flows
timeSlices = ['100', '365']

# run specs

# format is [ {'name': 'myname', 'group': 'mygroup', 'changes':['x.cin', ...], 'setvals': ['varname=123', ...] }, ... ]
# name is the run name for Vensim (avoid illegal filename characters)
# group is the group to which a run belongs, used to create some hierarchical organization in tables
# changes is a list of .cin file names, like ['a.cin', 'b.cin']. Can be omitted.
# setvals is a list of variable changes, using the .cin format, like ['x=3.5','y=5.9']. Can be omitted.
runs = [{'name': 'Base', 'group': 'Status Quo'},
        {'name': 'Behavior', 'setvals': ['Behavioral Risk Reduction=.8'], 'group': 'Policy'},
        {'name': 'Quarantine', 'setvals': ['Relative Isolation Effectiveness for Low Risk=.8','Potential Quarantine Effectiveness=.8'], 'group': 'Policy'},
        {'name': 'Full', 'changes': ['control.cin'], 'group': 'Policy'},
        ]

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
    result, param = run(model, runSetup, metrics.keys(), parameters)
    # rx, px = runsensitivity(model, runSetup, metrics.keys(), parameters, "Notional.vsc", "Consequence.lst", [2030, 2040])
    # result['Run'] = runSetup['name']  # add a column with the run name as a constant
    # result.info(verbose=True)
    print(result)
    #print(rx)
    # result['Group'] = 'Group'
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
results = pd.concat(resultFrames, keys=indexTuples, names=['Group', 'Run', 'Time'])
results.info(verbose=True)
print(results)

params = pd.concat(paramFrames, keys=indexTuples, names=['Group', 'Run', 'Time'])
params.info(verbose=True)
print(params)


# report the metrics with heatmap styling
# export to Excel
for t in timeSlices:
    for group in groups:
        resultsAtTime = results.query('Time==' + t)  # this is clearer than using loc[]
        resultsAtTime = resultsAtTime.droplevel('Time')  # drop the time index as it now has only one value
        q = "Group==" + repr(group)

        resultsAtTime = resultsAtTime.transpose()
        print(resultsAtTime)
        styledOutput = resultsAtTime.style

        # color palette will be used to indicate variable value within the range of scenarios
        # here we associate "warm" with "bad" and "cool" with "good"
        # c = sns.color_palette("coolwarm", as_cmap=True)
        # cr = sns.color_palette("coolwarm_r", as_cmap=True)
        # 'RdYlGn' and 'RdYlGn_r' another good pair of options

        nc = len(resultsAtTime.columns)
        idx = unsubbedList(resultsAtTime.index)  # this retrieves the dataframe index as a list of variable names (with subscripts if they exist)
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

        with pd.ExcelWriter("Consequence " + t + ".xlsx") as ex:
            styledOutput.to_excel(ex, sheet_name="Scenarios")

# dump parameters to a separate Excel file
# with distinct heatmap
for group in groups:
    resultsAtTime = params.droplevel('Time')  # drop the time index as it now has only one value
    q = "Group==" + repr(group)
    resultsAtTime = resultsAtTime.transpose()
    print(resultsAtTime)
    styledOutput = resultsAtTime.style

    nc = len(resultsAtTime.columns)
    idx = resultsAtTime.index
    # using a different color scale
    c = sns.color_palette("RdYlGn", as_cmap=True)

    for param in parameters:
        if param in idx:
            styledOutput = styledOutput.background_gradient(cmap=c, axis='columns', subset=pd.IndexSlice[param, indexTuples])

    with pd.ExcelWriter("Parameters.xlsx") as ex:
        styledOutput.to_excel(ex, sheet_name="Scenarios")
