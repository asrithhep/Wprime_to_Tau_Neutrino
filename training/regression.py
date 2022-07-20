import keras
import numpy as np
import pandas as pd
#import makeDataset 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers import LeakyReLU

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-is","--inputsigFile",help="give the input signal root file to Test/Train")
parser.add_argument("-ib","--inputbkgFile",help="give the input bkg    root file to Test/Train")
parser.add_argument("-t","--train",help="boolean set for train",action="store_true")
parser.add_argument("-s","--isSignal",help="boolean set for signal",action="store_true")
parser.add_argument("-n","--nepoch",type=int,help="no. of epochs",default=100)
parser.add_argument("-m","--model",help="model name to save or load with .h5 format",default="regression_model.h5")



args = parser.parse_args()


def build_model(layer_geom,learning_rate=3e-5,input_shapes=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shapes))
    for layer in layer_geom:
        model.add(keras.layers.Dense(layer_geom[layer],activation='relu',kernel_initializer="normal"))
    #he_avg_init = keras.initializers.VarianceScaling(scale=2.,mode='fan_avg',distribution='uniform')
    model.add(keras.layers.Dense(1,activation='linear',kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error',optimizer='adam')
    return model


def plot_loss(history):
    fig1,ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_ylim([0.5, 1.3])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error [MPG]')
    ax1.legend()
    fig1.savefig('../plots/training.png')
    ax1.grid(True)



sigdataset = makeDataset(args.inputsigFile)
bkgdataset = makeDataset(args.inputbkgFile)

if(args.train):
    sigdataset.setFeatures(
        'gentau_vis_px',
        'gentau_vis_py',
        'gentau_vis_pz',
        'gentau_vis_e',
        'genmet_px',
        'genmet_py',
        'boson_mass'
    )
    bkgdataset.setFeatures(
        'gentau_vis_px',
        'gentau_vis_py',
        'gentau_vis_pz',
        'gentau_vis_e',
        'genmet_px',
        'genmet_py',
        'boson_mass'
    )
    traincsvfile1 = sigdataset.get_df(sigdataset.train_df)
    traincsvfile2 = bkgdataset.get_df(bkgdataset.train_df)
    TrainDataSet  = pd.concat([traincsvfile1,traincsvfile2],ignore_index=True)
    for _ in range(16):
        TrainDataSet = shuffle(TrainDataSet)

else:
    sigdataset.setFeatures(
        'tau_vis_px',
        'tau_vis_py',
        'tau_vis_pz',
        'tau_vis_e',
        'met_px',
        'met_py',
    )
    if(args.isSignal):
        TestDataSet  = sigdataset.get_df(sigdataset.test_df)
    else:
        TestDataSet  = bkgdataset.get_df(bkgdataset.test_df)
    
if(args.train):
    features = TrainDataSet.drop(columns=['boson_mass']).to_numpy(dtype='float32')
    labels  = TrainDataSet['boson_mass'].to_numpy(dtype='float32')
    xtrain,xtest,ytrain,ytest = train_test_split(features,labels, test_size=0.33, random_state=42)
else:
    xstar  = TestDataSet.to_numpy(dtype='float32')

num_pipeline = Pipeline([
    ('std_scaler',StandardScaler()),
])


if(args.train):
    input_dim = xtrain.shape[1]
else:
    input_dim = xstar.shape[1]


hlayer_outline = {'hlayer1':32,'hlayer2':128,'hlayer3':128,'hlayer3':32}
model = build_model(hlayer_outline,input_shapes=[input_dim])
model.summary()
if args.train == True:
    checkpoint_cb = keras.callbacks.ModelCheckpoint("../models/"+args.model,save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True)
    model.fit(xtrain,ytrain,epochs=args.nepoch,batch_size =128,validation_data=(xtest,ytest),callbacks=[checkpoint_cb])#,early_stopping_cb])
    plot_loss(fit_model)
else:
    model = keras.models.load_model("../models/"+args.model)

plotname = ''
rootname = ''
branchname = ''
if(args.isSignal):
    plotname = 'invariant_mass_signal.png'
    rootname = 'signal.root'
    branchname = 'signal'
else:
    plotname = 'invariant_mass_background.png'
    rootname = 'background.root'
    branchname = 'background'
    

if(args.train == False):
    y_pred = model.predict(xstar)
   # y_pred = num_pipeline.inverse_transform(y_pred_tr)
    fig2,ax2 = plt.subplots()
    ax2.hist(y_pred,bins=80,log=True)
    ax2.set_xlabel('invariant mass')
    ax2.set_ylabel('Events [a.u]')
    fig2.savefig('../plots/'+plotname)
    filename = rootfile
    branch = np.array(y_pred,dtype=[(branchname,'float32')])
    array2root(branch,filename,'tree',mode='recreate')
