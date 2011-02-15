"""
Contains functions that make use of the classes to produce label estimates.
"""
import yaml, os
from Binary1dSignalModel import Binary1dSignalModel
from BinaryBias import BinaryBias
from MajorityModel import MajorityModel

def run_model_on_file(modelName, filename=None, modelPrm=None, optimizePrm=None,
                      outputPrefix=None):
    # load model
    if modelName == 'Binary1dSignal':
        m = Binary1dSignalModel(filename=filename)
    elif modelName == 'BinaryBias':
        m = BinaryBias(filename=filename)
    elif modelName == 'MajorityModel':
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
        imgPrm = [(i, p) for (i, p) in enumerate(imgPrm)]
    labels = m.get_image_labels()
    if type(labels)==type(list()):
        labels = [(i, p) for (i, p) in enumerate(labels)]
    wkrPrm = m.get_worker_param()
    if type(wkrPrm)==type(list()):
        wkrPrm = [(i, p) for (i, p) in enumerate(wkrPrm)]
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
