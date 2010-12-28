import os, sys, pickle
from numpy import *
from matplotlib.pylab import *

sys.path.append("../install")
from cubam import Binary1dSignalModel
from cubam.utils import create_data_file


# parameters
outDir = 'out'
filePrefix = '%s/demo-Binary1dSignal' % outDir
numWkrs = 5
numImgs = 2000
tasks = ['gen-data', 'run-model', 'show-results']
#tasks = ['show-results']
#tasks = ['cleanup']

# set up model
numpy.random.seed(7)
model = Binary1dSignalModel()
if not os.path.exists(outDir): os.makedirs(outDir)

## Generate Data
task = 'gen-data'
if task in tasks:
    print "Generating Data..."
    prm = generate_data(model, numImgs, numWkrs, '%s.txt' % filePrefix)
    pickle.dump(prm, open('%s.pickle' % filePrefix, 'w'))
else:
    prm = pickle.load(open('%s.pickle' % filePrefix))

## Run Model
task = 'run-model'
if task in tasks:
    print "Running Model..."
    model.load_data('%s.txt' % filePrefix)
    model.set_model_param(prm={
        'muw' : 2.0, 'sigt' : 10.0, 'sigw' : 10.0, 'sigx' : 2.0,
    })
    for n in range(10):
        print "  - iteration %d/%d" % (n, 10)
        model.optimize_image_param()
        model.optimize_worker_param()

## Show Results
task = 'show-results'
if task in tasks and 'run-model' in tasks:
    print "Creating Results Plots"
    eprm = { 'wkr' : model.get_worker_param(), 
             'img' : model.get_image_param()
    }
    plots = [('img', 0, 'xi'), ('wkr', 0, 'wj'), ('wkr', 1, 'tj')]
    for (labelType, idx, prmName) in plots:
        n = len(prm[labelType])
        gt = [prm[labelType][id][idx] for id in range(n)]
        est = [eprm[labelType][id][idx] for id in range(n)]
        # plot the correlation between ground truth and estimate
        figure(1); clf()
        plot(gt, est, '.')
        title('estimating %s' % prmName)
        xlabel('ground truth %s' % prmName)
        ylabel('estimated truth %s' % prmName)
        savefig('%s-%s.pdf' % (filePrefix, prmName))
        # plot the distribution of parameters
        figure(2); clf()
        hist(gt, max(10, n/20))
        title('distribution of %s' % prmName)
        xlabel('%s' % prmName)
        ylabel('number of instances')
        savefig('%s-%s-gt-dist.pdf' % (filePrefix, prmName))

## Clean up
task = 'cleanup'
if task in tasks:
    os.system('bash -c "rm %s*.{txt,pdf,pickle}"' % filePrefix)