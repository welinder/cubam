import os, sys, pickle
from numpy import random
from matplotlib.pylab import figure

sys.path.append("../install")
from cubam import Binary1dSignalModel
from cubam.utils import generate_data

############################################################################
# TASKS
############################################################################
tasks = ['gen-data', 'run-model', 'show-results']
#tasks = ['gen-data', 'run-model']
tasks = ['show-results']

############################################################################
# DEMO PARAMETERS
############################################################################
numWkrs = 5
numImgs = 2000

############################################################################
# OUTPUT LOCATION
############################################################################
resDir = 'results'
filePrefix = '%s/%s' % (resDir, __file__[:-3])

############################################################################
# GENERATE SYNTHETIC DATA FOR THE DEMO
############################################################################
task = 'gen-data'
random.seed(7)
model = Binary1dSignalModel()
if not os.path.exists(resDir): os.makedirs(resDir)
if task in tasks:
    print "Generating Synthetic Data..."
    prm = generate_data(model, numImgs, numWkrs, '%s.txt' % filePrefix)
    pickle.dump(prm, open('%s-gt-prm.pickle' % filePrefix, 'w'))
else:
    prm = pickle.load(open('%s-gt-prm.pickle' % filePrefix))

############################################################################
# RUN THE MODEL
############################################################################
task = 'run-model'
if task in tasks:
    print "Running Model..."
    model.load_data('%s.txt' % filePrefix)
    model.set_model_param(prm={
        'muw' : 2.0, 'sigt' : 10.0, 'sigw' : 10.0, 'sigx' : 2.0,
    })
    model.optimize_param()
    eprm = { 'wkr' : model.get_worker_param(), 
             'img' : model.get_image_param() }
    pickle.dump(eprm, open('%s-est-prm.pickle' % filePrefix, 'w'))

############################################################################
# SHOW RESULTS
############################################################################
task = 'show-results'
if task in tasks:
    print "Creating Results Plots..."
    eprm = pickle.load(open('%s-est-prm.pickle' % filePrefix))
    plots = [('img', 0, 'xi'), ('wkr', 0, 'wj'), ('wkr', 1, 'tj')]
    for (labelType, idx, prmName) in plots:
        n = len(prm[labelType])
        gt = [prm[labelType][id][idx] for id in range(n)]
        est = [eprm[labelType][id][idx] for id in range(n)]
        # plot the correlation between ground truth and estimate
        fig = figure(1); fig.clear(); ax = fig.add_subplot(1,1,1)
        ax.plot(gt, est, '.')
        ax.set_title('estimating %s' % prmName)
        ax.set_xlabel('ground truth %s' % prmName)
        ax.set_ylabel('estimated truth %s' % prmName)
        fig.savefig('%s-%s.pdf' % (filePrefix, prmName))
        # plot the distribution of parameters
        fig = figure(2); fig.clear(); ax = fig.add_subplot(1,1,1)
        ax.hist(gt, max(10, n/20))
        ax.set_title('distribution of %s' % prmName)
        ax.set_xlabel('%s' % prmName)
        ax.set_ylabel('number of instances')
        fig.savefig('%s-%s-gt-dist.pdf' % (filePrefix, prmName))
