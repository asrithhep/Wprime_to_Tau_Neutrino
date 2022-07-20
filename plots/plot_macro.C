#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "TMath.h"
#include "TRandom.h"
#include "TFile.h"
#include <stdio.h>
#include <cstring>
#include "TLatex.h"


void plot_macro(Int_t nbins=80,Float_t xlow=0,Float_t xhigh=4000){

  vector <TString> rootfiles = {   //List of files 
    "SkimWprime-M3000ToTauNu",
  };
  TH1F* hgen[rootfiles.size()];
  TH1F* hrec[rootfiles.size()];
    
  int nfile =1;
  for(auto filename: rootfiles){
    
    TFile* file = new TFile(filename+".root");
    TTree* tree = (TTree*)file->Get("analysisTree");
    Long_t TotEvents = tree->GetEntries();
    TString histname_gen = filename+"_gen";
    TString histname_rec = filename+"_rec";

    hgen[nfile-1]  = new TH1F(histname_gen,"",nbins,xlow,xhigh);
    hrec[nfile-1]  = new TH1F(histname_rec,"",nbins,xlow,xhigh);
    
    hgen[nfile-1]->SetLineColor(nfile);
    hrec[nfile-1]->SetLineColor(nfile);
    hgen[nfile-1]->SetLineStyle(1);
    hrec[nfile-1]->SetLineStyle(2);
    hgen[nfile-1]->SetLineWidth(3);
    hrec[nfile-1]->SetLineWidth(3);

    float xsec_weight = 10.99*2.3/TotEvents;
    TCut gen_cut = "gen_pTratio > 0.7 && gen_pTratio < 1.3 && gen_dtheta > 2.4";
    TCut rec_cut = "pTratio > 0.7 && pTratio < 1.3 && dtheta > 2.4";
    TCanvas *dummy = new TCanvas();
    tree->Draw("gen_mT>>"+histname_gen,gen_cut);
    tree->Draw("mT>>"+histname_rec,rec_cut);
    hrec[nfile-1]->Scale(xsec_weight);
    hgen[nfile-1]->Scale(xsec_weight);
    dummy->Delete();
    ++nfile;
  }
  TCanvas *c1 = new TCanvas("c1","",700,700);
  gStyle->SetOptStat(0);
  gPad->SetLogy();
  TH1F* href = new TH1F("href","",nbins,xlow,xhigh);
  href->GetYaxis()->SetRangeUser(0.01,100);
  href->Draw("histo");
  for(int i=0; i < rootfiles.size(); i++){
    //cout<<hrec[i]->Integral(0,3000)<<endl;
    hrec[i]->Draw("histosame");
    hgen[i]->Draw("histosame");
  }
  
}


