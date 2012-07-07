"""
This script shows an example of estimating a 2D latent space for the duck
experiment.

You should just be able to run it:

  python ducks.py

"""
import yaml, sys
from numpy import array, random
from matplotlib.pylab import figure
sys.path.append('..')
from cubam import BinaryNdSignalModel

############################################################################
# DEMO SETTINGS
############################################################################
random.seed(3)

############################################################################
# ESTIMATE MODEL PARAMETERS
############################################################################
model = BinaryNdSignalModel(filename='ducks/labels.txt')
model.init_param_from_1d()
model.optimize_param()
imgPrm = model.get_image_param()

############################################################################
# SHOW 2D LATENT SPACE
############################################################################
classInfo = yaml.load(open('ducks/classes.yaml'))
fig = figure(1, (6,6), dpi=100); fig.clear()
ax = fig.add_subplot(1,1,1)
markers = {'American Black Duck': 'd',
 'Canada Goose': '<',
 'Mallard': 's',
 'Non Bird': 'o',
 'Red-necked Grebe': '>'}
colors = {'American Black Duck': 'c',
 'Canada Goose': 'r',
 'Mallard': 'g',
 'Non Bird': 'k',
 'Red-necked Grebe': 'm'}
for s, sid in classInfo['species'].items():
    xi = array([imgPrm[iid] for iid, label in classInfo['labels'].items() \
                if label==sid])
    ax.scatter(xi[:,0], xi[:,1], marker=markers[s],
               color=colors[s], label=s)
ax.legend(loc='lower right')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('duck experiment')
