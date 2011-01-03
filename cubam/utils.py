import yaml
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tempfile import mkstemp

###########################################################################
### MATHEMATICAL
###########################################################################
def randtn(minlim=-3., maxlim=3.):
    """
    Use rejection sampling to sample from a truncated 1-D Normal distribution.
    
    Inputs:
    - `minlim`: [-3] the lower bound to truncate at.
    - `maxlim`: [-3] the upper bound to truncate at.
    """
    rn = minlim-1 # initialize out of bounds to get the while loop going
    while (rn<minlim) or (rn>maxlim):
        rn = np.random.randn()
    return rn

def correlation(u, v):
    """
    Compute the Spearman and Person correlation coefficients.
    
    Input:
    - `u`: list of values to be correlated.
    - `v`: list of values used to correlate with.
    
    Output:
    1. Spearman correlation.
    2. Pearson correlation.
    """
    return [spearmanr(u,v)[0], pearsonr(u,v)[0]]

def tw2tsw(tj, wj):
    """
    Converts from the (tj, wj) convention to (tj, sj, wj).
    """
    sj = 1./np.sqrt(np.sum(np.array(wj)**2))
    wj1 = np.array(wj)*sj
    tj1 = tj*sj
    return (tj1, sj, wj1)
    ###########################################################################
### DATA GENERATION, WRITING, READING
###########################################################################
def normalize_data_file(filename, outpfx, skipFirst=False):
    """
    Normalizes a data file so that workers and images are indexed from 0.
    
    Reads an input file with lines of the following format:
    
        {image id} {worker id} {binary label (0/1)}
    
    Where the image and worker ids may be any integers. Creates two output
    files:
    - `{outpfx}.txt`: normalized version where ids are indexed from 0, and
      where the first line is: `{n images} {n workers} {n labels}`
    - `{outpfx}-mapping.yaml`: the original to normalized id mappings as two
      dictionaries (called `image` and `worker`) in the YAML data format.
    
    Input:
    - `filename`: input file to normalize.
    - `outpfx` : output path prefix for the output.
    - `skipFirst`: [False] skip the first line of the input file.
    """
    # summarize input file
    imgIds, wkrIds, numLbls = {}, {}, 0
    infile = open(filename, 'r')
    if skipFirst: infile.readline() # skip first line
    for line in infile:
        imgId, wkrId, label = [int(col) for col in line.rstrip().split(" ")]
        if not imgIds.has_key(imgId): imgIds[imgId] = len(imgIds)    
        if not wkrIds.has_key(wkrId): wkrIds[wkrId] = len(wkrIds)
        numLbls += 1
    # write new file
    outfile = open("%s.txt" % outpfx, 'w')
    outfile.write("%d %d %d\n" % (len(imgIds), len(wkrIds), numLbls))
    infile = open(filename, 'r')
    infile.readline()
    for line in infile:
        imgId, wkrId, label = [int(col) for col in line.rstrip().split(" ")]
        outfile.write("%d %d %d\n" % (imgIds[imgId], wkrIds[wkrId], label))
    outfile.close()
    # save mapping
    outfile = open("%s-mapping.yaml" % outpfx, 'w')
    yaml.dump({'image' : imgIds, 'worker' : wkrIds}, outfile)
    outfile.close()

def write_data_file(labels, filename=None):
    """
    Writes a text-based data file from a list of image-worker labels.
    
    Creates a text file where the first row is: 
      `{n images} {n workers} {n labels}`
    And the remaining rows are formatted as such:
      `{image id} {worker id} {binary label (0/1)}`
    
    Input:
    - `labels`: list of tuples, (image id, worker id, label [0/1])
    - `filename`: [None] filename output file. If `None`, a temporary file is
      created and written to.
    
    Output:
    1. The filename of the output file.
    """
    if filename is None: filename = mkstemp()[1]
    imgIds = set([row[0] for row in labels])    
    wkrIds = set([row[1] for row in labels])
    f = open(filename, 'w')
    f.write('%d %d %d\n' % (len(imgIds), len(wkrIds), len(labels)))
    for row in labels:
        # by convention the label can be a vector, so pick 1st element
        label = row[2][0] if type(row[2])==type(list()) else row[2]
        f.write('%d %d %d\n' % (row[0], row[1], label))
    f.close()
    return filename

def read_data_file(filename, skipFirst=True):
    """
    Reads a text-based data file and returns a structured dictionary.
    
    Input:
    - `filename`: filename for the input text file.
    - `skipFirst`: [True] skip the first line when reading the data file?
    
    Output: dictionary with the following keys:
    - `image`: dictionary with image ids as keys and (worker id -> label)
      dictionaries as values.
    - `worker`: dictionary with worker ids as keys and (image id -> label)
      dictionaries as values.
    - `labels`: list of [image id, worker id, label] elements.
    """
    f = open(filename)
    if skipFirst: f.readline() # skip first line
    wkrLbls, imgLbls, labels = {}, {}, []
    for line in f:
        if len(line.rstrip())==0 or line[0]=='#': continue
        iId, wId, label = [int(c) for c in line.rstrip().split(" ")]
        # record labels
        if not wkrLbls.has_key(wId): wkrLbls[wId] = {}
        wkrLbls[wId][iId] = label
        if not imgLbls.has_key(iId): imgLbls[iId] = {}
        imgLbls[iId][wId] = label
        labels.append([iId, wId, label])
    return {'image' : imgLbls, 'worker' : wkrLbls, 'labels' : labels}

def generate_data(model, numImgs, numWkrs, filename, wkrPrm={}, imgPrm={}):
    """
    Samples simulated image and worker parameters, and generates labels.
    
    Input:
    - `model`: model instance to use for sampling parameters and labels.
    - `numImgs`: number of image parameters to generate.
    - `numWkrs`: number of worker parameters to generate.
    - `filename`: filename of the output file.
    - `wkrPrm`: [{}] arguments used for sampling worker parameters using the
      model instance function `model.sample_worker_param`
    - `imgPrm`: [{}] arguments used for sampling image parameters using the
      model instance function `model.sample_image_param`
    
    Output:
    1. Dictionary of image and worker parameters, with two keys: img and wkr.
    """
    # sample the worker and image parameters
    wkrs = model.sample_worker_param(numWkrs, **wkrPrm)
    imgs = model.sample_image_param(numImgs, **imgPrm)
    # generate and save the labels
    labels = sample_labels(model, wkrs, imgs)
    write_data_file(labels, filename)
    # return the worker and image parameters
    return { 'wkr' : wkrs, 'img': imgs }

def sample_labels(model, wkrs, imgs):
    """
    Generate a full labeling by workers given worker and image parameters.
    
    Input:
    - `model`: model instance to use for sampling parameters and labels.
    - `wkrs`: list of worker parameters.
    - `imgs`: list of image parameters.
    
    Output:
    1. list [img id, wkr id, label] as provided by `model.sample_label`.
    """
    labels = [[ii, wi, model.sample_label(wkrs[wi], imgs[ii])] \
              for ii in range(len(imgs)) for wi in range(len(wkrs))]
    return labels

###########################################################################
### BENCHMARKING
###########################################################################
def majority_vote(imgLbls):
    """
    Use the majority vote rule to determine image labels.
    
    Inputs:
    - `imgLbls`: a dictionary with image ids as keys and (worker id -> label)
      dictionaries as values.
      
    Output:
    1. dictionary with (image id -> predicted label)
    """
    zi = {}
    for (imgId, labels) in imgLbls.iteritems():
        vote = 0.0 + (np.random.rand()-.5) # add some noise for ties
        for (wId, label) in labels.iteritems():
            vote += 1.0 if (label==1 or label==True) else -1.0
        zi[imgId] = vote>0.0
    return zi

def error_rates(exi, gxi):
    n = len(exi)
    gt = np.array([xi>0.0 for xi in gxi])
    est = np.array([xi>0.0 for xi in exi])
    # error rate
    er = 1.0-float(sum(gt==est))/n
    # false alarm rate
    far = float(sum(est[gt==False]))/sum(gt==False)
    # miss rate
    mr = float(sum(est[gt==True]==False))/sum(gt==True)
    return [er, far, mr]
