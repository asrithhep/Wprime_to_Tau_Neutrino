import uproot3 as ROOT
import awkward as aw
import pandas as pd
import numpy as np

from keras.models import load_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--csvFile",help="csv file consists of test and mT variables")
parser.add_argument("-o","--outputFile",help="root file name to save Datacard")
parser.add_argument("-m","--model",help="dnn model")
parser.add_argument("xsec", type=int, help="cross section")
parser.add_argument("N", type=int, help="Total events")


args = parser.parse_args()

DataSet = pd.read_csv(args.csvFile,index_col=False)
model   = laod_model(args.model)

drop_cols = ['mT',"dtheta","pTratio"] #set the list with variable not use for the model testing.
TestDataSet = DataSet.drop(columns=drop_cols).to_numpy(dtype='float32')

dnn_mass     = model.predict(TestDataSet)
weight   = np.repeat(args.xsec*2.3/args.N,TestDataset.shape[0])


rootfile = ROOT.recreate(args.outputFile)

#Branch info
branch_name = {"weight":np.float32,"dnn_mass":np.float32
               "mT":np.float32,"pTratio":np.float32,"dtheta":np.float32}

branch_fill = {"weight":weight,"dnn_mass":dnn_mass
               "mT":DataSet['mT'],"pTratio":DataSet['pTratio'],"dtheta":DataSet['dtheta']}

rootfile['tree'] = ROOT.newtree(branch_name)
rootfile['tree'].extend(branch_fill)
rootfile.close()







