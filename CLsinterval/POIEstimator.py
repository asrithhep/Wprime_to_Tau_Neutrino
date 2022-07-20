from unittest import result
import pyhf
from pyhf.contrib.viz import brazil
import matplotlib.pyplot as plt
import numpy as np

class POIEstimator:
    def __init__(self,sig_list,bkg_list,bkg_uncert):
        self.sig_l = sig_list
        self.bkg_l = bkg_list
        self.bkg_un = bkg_uncert
        model = self.GetModel()
        self.obs = self.bkg_l + model.config.auxdata
    
    def GetModel(self):
        model = pyhf.simplemodels.hepdata_like(signal_data=self.sig_l,bkg_data=self.bkg_l,bkg_uncerts=self.bkg_un)
        return model

    def POIrange(self,val_lo,val_hi,nvals):
        return np.linspace(val_lo,val_hi,nvals)
    
    def hypo_test(self,poi_values):
        model = self.GetModel()
        results = [
        pyhf.infer.hypotest(
            poi_value,
            self.obs,
            model,
            test_stat="qtilde",
            return_expected_set=True,
        )
        for poi_value in poi_values
        ]
        return results

    def compute_limits(self,poi_values,alpha):
        model = self.GetModel()
        obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upperlimit(
            self.obs, model, poi_values, level=alpha, return_results=True
        )
        return exp_limits

    def plot_CLsPOI(self,poi_values):
        results = self.hypo_test(poi_values)
        fig, ax = plt.subplots()
        fig.set_size_inches(10.5, 7)
        ax.set_title("Hypothesis Tests")
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\mathrm{CL}_{s}$")

        brazil.plot_results(ax, poi_values, results)
        
