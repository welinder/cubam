"""
Contains functions that make use of the classes to produce label estimates.
"""
import yaml, os
from Binary1dSignalModel import Binary1dSignalModel
from BinaryBiasModel import BinaryBiasModel
from MajorityModel import MajorityModel

def run_model_on_file(modelName, filename=None, modelPrm=None, optimizePrm=None,
                      outputPrefix=None):
    # load model
    if modelName == 'signal':
        m = Binary1dSignalModel(filename=filename)
    elif modelName == 'bias':
        m = BinaryBiasModel(filename=filename)
    elif modelName == 'majority':
        m = MajorityModel(filename=filename)
    else:
        assert False, "Invalid model"
    # set the model parameters
    if not modelPrm is None:
        m.set_model_param(prm=modelPrm)
    # optimize the model parameters
    if optimizePrm is None:
        m.optimize_param()
    else:
        m.optimize_param(**optimizePrm)
    # convert lists to dictionaries
    imgPrm = m.get_image_param()
    if type(imgPrm)==type(list()):
        imgPrm = dict((i, p) for (i, p) in enumerate(imgPrm))
    labels = m.get_labels()
    if type(labels)==type(list()):
        labels = dict((i, p) for (i, p) in enumerate(labels))
    wkrPrm = m.get_worker_param()
    if type(wkrPrm)==type(list()):
        wkrPrm = dict((i, p) for (i, p) in enumerate(wkrPrm))
    # save predictions
    if not outputPrefix is None:
        dirName = os.path.dirname(outputPrefix)
        if not os.path.exists(dirName): os.makedirs(dirName)
        yaml.dump(labels, open(outputPrefix+'-est-labels.yaml', 'w'))
        yaml.dump(imgPrm, open(outputPrefix+'-est-img-prm.yaml', 'w'))
        yaml.dump(wkrPrm, open(outputPrefix+'-est-wkr-prm.yaml', 'w'))
    return { 'labels' : labels, 'workers' : wkrPrm, 'images' : imgPrm }

def run_model_on_files(modelName, files, modelPrm=None, optimizePrm=None,
                       outputDir=None):
    for filePath in files:
        fn = os.path.basename(filePath)
        if fn[-4:]=='.txt': fn = fn[:-4]
        outputPrefix = '%s/%s' % (outputDir, fn)
        run_model_on_file(modelName, filename=filePath, modelPrm=modelPrm,
                          optimizePrm=optimizePrm, outputPrefix=outputPrefix)
