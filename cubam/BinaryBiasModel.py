from BinaryModel import *
from numpy import ones, log10, nonzero, flipud, diag
from numpy.random import multinomial
from scipy.stats import beta

class BinaryBiasModel(BinaryModel):
    def __init__(self, filename=None):
        self.mdlPrm = {
            'pz1' : 0.5,
            'res' : 100,
            'initAj' : (.7, .7),
            'priorPrm' : [ 
                ('beta', .9, 10, 2), ('beta', .05, 2, 10) , ('ridge', .05)
             ]
        }
        self.wkrIds = {}
        self.imgIds = {}
        if filename:
            self.load_data(filename)
        else:
            self._setup_prior()    
    
    def __del__(self):
        pass

    def load_data(self, filename, skipyaml=False):
        """
        Data is assumed to be in the format:
        imageId workerId label
        """
        yamlfile = "%s.yaml" % filename[:-4]
        if filename[-4:]=='.txt' and os.path.exists(yamlfile) \
          and not skipyaml:
            prm = yaml.load(open(yamlfile))
            self.imgIds = prm['imgIds']
            self.wkrIds = prm['wkrIds']
        # load the text data
        filein = open(filename)
        info = filein.readline().rstrip().split(' ')
        self.numLbls = int(info[2])
        self.numWkrs = int(info[1])
        self.numImgs = int(info[0])
        self.wkrPrm = zeros((self.numWkrs, 2)) # [a_1, a_0]
        self.wkrPrm[:,0] = self.mdlPrm['initAj'][0]
        self.wkrPrm[:,1] = self.mdlPrm['initAj'][1]
        self.imgPrm = [[0.0]]*self.numImgs # [pz1]
        self.wkrLbls = dict((id, []) for id in range(self.numWkrs))
        self.imgLbls = dict((id, []) for id in range(self.numImgs))
        self.labels = []
        for (lino, line) in enumerate(filein):
            cols = [int(c) for c in line.rstrip().split(' ')]
            iId = cols[0]; wId = cols[1]; lij = cols[2]==1
            self.wkrLbls[wId].append([iId, lij])
            self.imgLbls[iId].append([wId, lij])
            self.labels.append((iId, wId, lij))
        self._setup_prior()

    def _setup_prior(self):
        n = self.mdlPrm['res']
        ajs = linspace(1e-10, 1.-1e-10, n)
        aj0, aj1 = meshgrid(ajs, ajs); aj0=aj0.flatten(); aj1=aj1.flatten()
        prior = zeros( len(aj0) )
        for prm in self.mdlPrm['priorPrm']:
            tp = prm[0]
            if tp=='beta':
                a, b = prm[2:4]
                p = beta(a, b).pdf(aj0) * beta(a, b).pdf(aj1)
            elif tp=='ridge':
                # diagonal ridge (displaced to be a little bit optimistic
                # about the annotator parameters)
                p = flipud(diag(ones(n),1))[:n,:n].flatten()
            w = prm[1]
            prior += w * p/sum(p)
        self.prior = {}
        self.prior['logprior'] = log(prior/sum(prior))
        self.prior['aj0'] = aj0
        self.prior['aj1'] = aj1
        self.prior['ajs'] = ajs

    def get_num_wkrs(self):
        return self.numWkrs

    def get_num_imgs(self):
        return self.numImgs

    def get_num_lbls(self):
        return self.numLbls

    def set_model_param(self, raw=[], prm=None):
        """
        Sets model parameters.

        Arguments:
          - `raw`: raw parameter vector
          - `prm`: hash of model parameter values to be changed
        """
        if not prm is None:
            for (k, v) in prm.iteritems():
                self.mdlPrm[k] = v
        self._setup_prior()

    def set_worker_param(self, raw):
        self.wkrPrm = array(raw).reshape((self.numWkrs, 2))

    def set_image_param(self, raw):
        self.imgPrm = [[r] for r in raw]

    def get_model_param(self):
        return {}

    def get_worker_param_raw(self):
        return array(self.wkrPrm).flatten().tolist()

    def get_image_param_raw(self):
        return [p[0] for p in self.imgPrm]

    def get_worker_param(self, id=None):
        return self.wkrPrm

    def get_image_param(self, id=None):
        return self.imgPrm

    def get_labels(self):
        return [int(p[0]>0.5) for p in self.imgPrm]
    
    # TODO: load and save parameters

    def optimize_worker_param(self):
        for (wId, labels) in self.wkrLbls.iteritems():
            n = [[0.0, 0.0], [0.0, 0.0]] # [gt, label]
            # count no. of false alarms etc
            for (iId, lij) in self.wkrLbls[wId]:
                p1 = self.imgPrm[iId][0]
                if lij==True:
                    n[1][1] += p1
                    n[0][1] += (1.-p1)
                else:
                    n[1][0] += p1
                    n[0][0] += (1.-p1)
            # TODO PRIOR HERE
            opt = n[0][0]*log(self.prior['aj0']) + \
              n[0][1]*log(1.-self.prior['aj0']) + \
              n[1][1]*log(self.prior['aj1']) + \
              n[1][0]*log(1.-self.prior['aj1']) + \
              self.prior['logprior']
            idx = argmax(opt)
            self.wkrPrm[wId][0] = self.prior['aj1'][idx]
            self.wkrPrm[wId][1] = self.prior['aj0'][idx]

    def optimize_image_param(self):
        pz1 = self.mdlPrm['pz1']
        newPrm = log10(pz1/(1-pz1))*ones(self.numImgs)
        for (iId, wId, lij) in self.labels:
            aj = self.wkrPrm[wId]
            if lij==True:
                newPrm[iId] += log10(aj[1]/(1.-aj[0]))
            else:
                newPrm[iId] += log10((1.-aj[1])/aj[0])
        R = 10.**newPrm
        newPrm = R/(1.+R)
        self.imgPrm = [[p] for p in newPrm]

    def objective(self, prm=None):
        pass

    def image_objective(self, prm=None):
        pass
        
    def image_objective_range(self, imgId, prm):
        pass

    def worker_objective_range(self, wkrId, prm):
        pass

    def gradient(self, prm=None):
        return []

    def worker_gradient(self, prm=None):
        return []

    def image_gradient(self, prm=None):
        pass

    def get_num_wkr_lbls(self):
        return [len(self.wkrLbls[id]) for id in range(self.numWkrs)]

    def get_num_img_lbls(self):
        return [len(self.imgLbls[id]) for id in range(self.numImgs)]

    def sample_worker_param(self, numWkr, sklName=None):
        wkrPrm = []
        wkrGrps =  {
            'expert' : [.95, .95],
            'good' : [.7, .7],
            'bot' : [.5, .5],
            'adversary' : [.1, .1] 
        }
        wkrGrpDist = [.1, .6, .29, .01]
        sklNms = ['expert', 'good', 'bot', 'adversary']
        for wIdx in range(numWkr):
            if not sklName:
                sklIdx = int(nonzero(multinomial(1, wkrGrpDist))[0])
                sklName = sklNms[sklIdx]
                prm = wkrGrps[sklName]
                sklName = None
            else:
                prm = wkrGrps[sklName]
            wkrPrm.append(prm)
        return wkrPrm

    def sample_image_param(self, numImg):
        "Generate random image (parameters)."
        imgPrm = []
        for iIdx in range(numImg):
            imgPrm.append([1.0 if rand()<self.mdlPrm['pz1'] else 0.0])
        return imgPrm

    def sample_label(self, wkrPrm, imgPrm):
        "Let worker produce a label for an image based on img/wkr param."
        if(imgPrm[0] > 0.5):
            return [rand() < wkrPrm[1]]
        else:
            return [rand() > wkrPrm[0]]
