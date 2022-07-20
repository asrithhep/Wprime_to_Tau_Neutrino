import ROOT
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle


class makeDataset:
    def __init__(self,rootfile):
        tree_df = ROOT.RDataFrame('analysisTree',rootfile)
        self.train_df  = tree_df.Filter('gentau_vis_pt > 80 && genmet > 120 && TauTag >0.5')
        self.test_df  = tree_df.Filter('tau_vis_pt > 80 && met > 120 && TauTag > 0.5')

    def get_df(self,df):
        dataframe = pd.DataFrame(self.load_data(df),columns=self.getFeatures())
        return dataframe
        

    def setFeatures(self,*var):
        self.features = [v for v in var]
        return self.features

    def getFeatures(self):
        return self.features
    
    def load_data(self,df):
        feature_list = self.getFeatures()
        tree_ar = df.AsNumpy()
        feature_dict = self.get_Inputs(tree_ar,feature_list)
        data_ar = []
        for key in feature_dict.keys():
            data_ar.append(feature_dict[key])
        data = np.stack(tuple(data_ar),axis=1)
        return data

    def saveCSV(self,path,data,filename):
        feature_head = self.getFeatures()
        DataFrame = pd.DataFrame(data,columns=feature_head)
        DataFrame.to_csv(path+'/'+filename,index=False)
        
        
    def get_Inputs(self,df,col_list):
        input_dict = {}
        for col in col_list:
            item = df[col]
            input_dict[col] = item
        return input_dict


    
        
##
#os.chdir("../data")

# inputFile1 = input("1st root file: ")
# inputFile2 = input("2nd root file: ")

# trainFile = input("training filename: ")
# testFile1  = input("test filename for signal: ")
# testFile2  = input("test filename for bkg: ")

# traindataset1 = makeDataset(inputFile1)
# traindataset1.setFeatures(
#     'gentau_vis_px',
#     'gentau_vis_py',
#     'gentau_vis_pz',
#     'gentau_vis_e',
#     'genmet_px',
#     'genmet_py',
#     'boson_mass'
# )
# testdataset1 = makeDataset(inputFile1)
# testdataset1.setFeatures(
#     'tau_vis_px',
#     'tau_vis_py',
#     'tau_vis_pz',
#     'tau_vis_e',
#     'met_px',
#     'met_py',
# )
# traindataset2 = makeDataset(inputFile2)
# traindataset2.setFeatures(
#     'gentau_vis_px',
#     'gentau_vis_py',
#     'gentau_vis_pz',
#     'gentau_vis_e',
#     'genmet_px',
#     'genmet_py',
#     'boson_mass'
# )
# testdataset2 = makeDataset(inputFile2)
# testdataset2.setFeatures(
#     'tau_vis_px',
#     'tau_vis_py',
#     'tau_vis_pz',
#     'tau_vis_e',
#     'met_px',
#     'met_py',
#   )

# traincsvfile1 = traindataset1.get_df(traindataset1.train_df)
# traincsvfile2 = traindataset2.get_df(traindataset2.train_df)
# testcsvfile1  = testdataset1.get_df(testdataset1.test_df)
# testcsvfile2  = testdataset2.get_df(testdataset2.test_df)

# traincsvfile = pd.concat([traincsvfile1,traincsvfile2],ignore_index=True)
# #testcsvfile  = pd.concat([testcsvfile1,testcsvfile2],ignore_index=True)

# for _ in range(16):
#     traincsvfile = shuffle(traincsvfile)
#     #testcsvfile  = shuffle(testcsvfile)

# traincsvfile.to_csv(trainFile,index=False)
# testcsvfile1.to_csv(testFile1,index=False)
# testcsvfile2.to_csv(testFile2,index=False)


 
