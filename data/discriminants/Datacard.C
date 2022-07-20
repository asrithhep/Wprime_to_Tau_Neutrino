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


void DataCard(Int_t nbins=50,Float_t xlow=0,Float_t xhigh=1500){

  int nbackgrounds = 1;
  
  TFile* sigfile = new TFile("../data/discriminants/signalfile.root");
  TFile* bkgfile = new TFile("../data/discriminants/backgroundfile.root");
  
  TTree* sig_tree = (TTree*)sigfile->Get("tree");
  TTree* bkg_tree = (TTree*)bkgfile->Get("tree");
  
  TH1F* hsignal = new TH1F("hsignal","",nbins,xlow,xhigh);
  TH1F* hbkg    = new TH1F("hbkg","",nbins,xlow,xhigh);
    
    

  float sig_weight = 2.457*1000*2.3/500000;
  float bkg_weight = 1260*1000*2.3/9751000;
  
  TCanvas *dummy = new TCanvas();
  sig_tree->Draw("signal>>hsignal");
  bkg_tree->Draw("background1>>hbkg");
  hsignal->Scale(sig_weight);
  hbkg->Scale(bkg_weight);
  hbkg->SetFillColor(4);
  hbkg->SetFillStyle(3244);
  hsignal->SetFillStyle(3344);
  hsignal->SetFillColor(2);
  dummy->Delete();

  TCanvas *c1 = new TCanvas("c1","",700,700);
  gStyle->SetOptStat(0);
  gPad->SetLogy();
  TH1F* href = new TH1F("href","",nbins,xlow,xhigh);
  href->GetYaxis()->SetRangeUser(0.01,10000);
  href->GetXaxis()->SetTitle("dnn mass");
  href->GetYaxis()->SetTitle("Events/30 GeV");
  href->Draw("histo");
  hbkg->Draw("histosame");
  hsignal->Draw("histosame");

  auto legend = new TLegend(0.1,0.7,0.48,0.9);
  legend->SetHeader("Teststatics","C");
  legend->AddEntry(hsignal,"Signal","f");
  legend->AddEntry(hbkg,"Background","f");
  legend->Draw();
  
}


