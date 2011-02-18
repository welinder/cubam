"""
This script simulates different numbers of workers and compares majority
voting against the NIPS 2010 model and a variant of the Dawid & Skene method.

You should just be able to run it:

  python synthetic.py

"""
import sys, os, pickle
from numpy import random, mean, std, sqrt
from matplotlib.pylab import figure

sys.path.append("../install")
from cubam import Binary1dSignalModel, BinaryBiasModel
from cubam.utils import generate_data, save_param_file, load_param_file, \
     majority_vote, read_data_file

############################################################################
# TASKS
############################################################################
tasks = ['gen-data','run-models','show-results']

############################################################################
# DEMO PARAMETERS
############################################################################
numWkrList = [4, 8, 12, 16, 20]
numWkrList = [4, 12, 20]
numImg = 500
numTrial = 40

############################################################################
# OUTPUT LOCATION
############################################################################
rndseed = 3
dataDir = 'data/%s' % __file__[:-3]
resDir = 'results/%s' % __file__[:-3]
imgPrm = { 'theta' : 0.5 } # for generating data

############################################################################
# GENERATE SYNTHETIC DATA FOR THE DEMO
############################################################################
task = 'gen-data'
random.seed(rndseed)
model = Binary1dSignalModel()
for tdir in [dataDir, resDir]:
    if not os.path.exists(tdir): os.makedirs(tdir)
# generate the data
dinfoFile = '%s-info.pickle' % dataDir # data info file
if task in tasks:
    dinfo = { 'numImg' : numImg, 'numTrial' : numTrial }
    dinfo['filemap'] = []
    dFn = lambda nw, tr, ext: "%s/w%02d-t%d.%s" % (dataDir, nw, tr, ext)
    for numWkr in numWkrList:
        trialList = []
        print "Generating trials for %d workers" % numWkr
        for trial in range(numTrial):
            print "  Generating data for trial %d" % (trial+1)
            # create model and synthetic data
            dfile = dFn(numWkr, trial, 'txt')
            prm = generate_data(model, numImg, numWkr, dfile, imgPrm=imgPrm)
            pfile = dFn(numWkr, trial, 'pickle')
            save_param_file(prm, pfile)
            trialList.append((dfile, pfile))
        dinfo['filemap'].append((numWkr, trialList))
    # save data info
    f = open(dinfoFile, 'w')
    pickle.dump(dinfo, f); f.close()

############################################################################
# RUN MODELS ON DATA
############################################################################
task = 'run-models'
rateFile = '%s-rates.pickle' % resDir
if task in tasks:
    dinfo = pickle.load(open(dinfoFile))
    errRates = { 'signal' : {}, 'majority' : {}, 'bias' : {} }
    getParameter = lambda prmdict, pidx: [prmdict[i][pidx] for i \
                                          in range(len(prmdict))]
    for (numWkr, trialList) in dinfo['filemap']:
        print "Processing %d workers" % numWkr
        for alg in errRates.keys(): errRates[alg][numWkr] = []
        for (dfile, pfile) in trialList:
            prm = load_param_file(pfile)
            gxi = getParameter(prm['img'], 0)
            gzi = [gxi[id] > 0. for id in range(dinfo['numImg'])]
            # Binary Signal Model
            m = Binary1dSignalModel(filename=dfile)
            m.optimize_param()
            exi = getParameter(m.get_image_param(), 0)
            comperr = lambda ez: float(sum([ez[id]!=gzi[id] for id \
              in range(dinfo['numImg'])]))/dinfo['numImg']
            err = comperr([exi[id]>0. for id in range(dinfo['numImg'])])
            errRates['signal'][numWkr].append(err)
            # Binary Bias Model
            m = BinaryBiasModel(filename=dfile)
            m.optimize_param()
            iprm = m.get_image_param_raw()
            err = comperr([iprm[id]>.5 for id in range(dinfo['numImg'])])
            errRates['bias'][numWkr].append(err)
            # majority
            labels = read_data_file(dfile)
            ezis = majority_vote(labels['image'])
            err = comperr([ezis[id] for id in range(dinfo['numImg'])])
            errRates['majority'][numWkr].append(err)
    # save result
    f = open(rateFile, 'w')
    pickle.dump(errRates, f); f.close()

############################################################################
# PLOT THE ERROR RATE VS NO OF WORKERS FOR EACH ALGORITHM
############################################################################
task = 'show-results'
if task in tasks:
    errRates = pickle.load(open(rateFile))
    exps = [('signal', 'NIPS 2010'), ('majority', 'majority'), ('bias', 'Dawid & Skene')]
    numWkrList = sorted(errRates[exps[0][0]].keys())
    fig = figure(1, (5.5,3)); fig.clear(); ax = fig.add_subplot(1,1,1)
    for (expt, legname) in exps:
        rates = [mean(errRates[expt][nw]) for nw in numWkrList]
        erbs = [std(errRates[expt][nw])/sqrt(numTrial) for nw in numWkrList]
        ax.errorbar(numWkrList, rates, yerr=erbs, label=legname, lw=3)
    ax.set_xlabel('number of annotators', fontsize=16)
    ax.set_ylabel('error rate', fontsize=16)
    ax.set_title('synthetic annotator data', fontsize=18)
    ax.set_xlim(2, 22)
    ax.set_ylim(0.0, .12)
    ax.set_xticks([4, 8, 12, 16, 20])
    ax.legend(loc=1)
    fig.savefig('%s-results.pdf' % resDir)
