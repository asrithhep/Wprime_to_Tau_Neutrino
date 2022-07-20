#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#include "external/ntuple.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif


void AnalysisWprimeToTauNu(const char *inputFile){
  gSystem->Load("libDelphes");
  
  TChain *chain = new TChain("Delphes");
  TString TPath = "/scratch1/WplusAnalysis/data/WjetsToLNuHighMsamples/WJetsHT100-200ToLNu/";
  chain->Add(TPath+TString(inputFile));

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  TTree* analysisTree = new TTree("analysisTree","analysisTree");
  
  ntuple* tuple = new ntuple(analysisTree);

  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchGenMissingET = treeReader->UseBranch("GenMissingET");
  TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");

  Long64_t allEntries = treeReader->GetEntries();

  cout << "** Chain contains " << allEntries << " events" << endl;

  bool background = true;

  for(Int_t entry = 0; entry < allEntries; ++entry)
  {
    treeReader->ReadEntry(entry);
    for(Int_t i = 0; i < branchParticle->GetEntriesFast(); i++){
      GenParticle *gen = (GenParticle *)branchParticle->At(i);
      if(gen->PID == -34 || gen->PID == 34 || gen->PID == 24 || gen->PID == -24){
	GenParticle *Tau = (GenParticle*)branchParticle->At(gen->D1);
	GenParticle *Nu  = (GenParticle*)branchParticle->At(gen->D2);
	if(abs(Tau->PID) == 15 || abs(Tau->PID) == 11 || abs(Tau->PID) == 13){
	  tuple->gentau_px = Tau->Px;
	  tuple->gentau_py = Tau->Py;
	  tuple->gentau_pz = Tau->Pz;
	  tuple->gentau_e  = Tau->E;
	  
	  tuple->gentau_pt  = Tau->PT;
	  tuple->gentau_eta = Tau->Eta;
	  tuple->gentau_phi = Tau->Phi;
	  
	  GenParticle* VisTau = (GenParticle*)branchParticle->At(Tau->D2);
	   if(!background){
	     if(abs(VisTau->PID) > 16){
	       tuple->gentau_vis_px = VisTau->Px;
	       tuple->gentau_vis_py = VisTau->Py;
	       tuple->gentau_vis_pz = VisTau->Pz;
	       tuple->gentau_vis_e  = VisTau->E;
	       
	       TLorentzVector taup4  = VisTau->P4();
	       tuple->gentau_vis_pt  = taup4.Pt();
	       tuple->gentau_vis_eta = taup4.Eta();
	       tuple->gentau_vis_phi = taup4.Phi();
	     }
	     
	   }
	   else{
	     tuple->gentau_vis_px = Tau->Px;
	     tuple->gentau_vis_py = Tau->Py;
	     tuple->gentau_vis_pz = Tau->Pz;
	     tuple->gentau_vis_e  = Tau->E;
	     
	     TLorentzVector taup4  = Tau->P4();
	     tuple->gentau_vis_pt  = taup4.Pt();
	     tuple->gentau_vis_eta = taup4.Eta();
	     tuple->gentau_vis_phi = taup4.Phi();
	   }
	}
	if(!background){
	  if(abs(Nu->PID) == 16){
	    tuple->neutrino_px = Nu->Px;
	    tuple->neutrino_py = Nu->Py;
	    tuple->neutrino_pz = Nu->Pz;
	    tuple->neutrino_e  = Nu->E;
	  }
	}
	else{
	  tuple->neutrino_px = Nu->Px;
	  tuple->neutrino_py = Nu->Py;
	  tuple->neutrino_pz = Nu->Pz;
	  tuple->neutrino_e  = Nu->E;
	}
      }
    }

    tuple->boson_px = tuple->gentau_px + tuple->neutrino_px;
    tuple->boson_py = tuple->gentau_py + tuple->neutrino_py;
    tuple->boson_pz = tuple->gentau_pz + tuple->neutrino_pz;
    tuple->boson_e  = tuple->gentau_e  + tuple->neutrino_e;

    tuple->boson_mass = TMath::Sqrt(tuple->boson_e*tuple->boson_e - tuple->boson_px*tuple->boson_px -tuple->boson_py*tuple->boson_py - tuple->boson_pz*tuple->boson_pz);

    for(int i = 0; i < branchGenMissingET->GetEntriesFast(); ++i)
    {
      MissingET* MET = (MissingET *)branchGenMissingET->At(i);
      TLorentzVector genMET_mom = MET->P4();
      tuple->genmet_phi = genMET_mom.Phi();
      tuple->genmet_px = genMET_mom.Px();
      tuple->genmet_py = genMET_mom.Py();
      tuple->genmet = MET->MET;
    }
    for(int i=0; i < branchJet->GetEntriesFast(); ++i){
      Jet* jet = (Jet*)branchJet->At(i);
      tuple->TauTag = jet->TauTag;
      if(jet->TauTag > 0.5){
	TLorentzVector mom = jet->P4();
	tuple->tau_vis_pt  = jet->PT;
	tuple->tau_vis_eta = jet->Eta;
	tuple->tau_vis_phi = jet->Phi;
	tuple->tau_vis_px = mom.Px();
	tuple->tau_vis_py = mom.Py();
	tuple->tau_vis_pz = mom.Pz();
	tuple->tau_vis_e  = mom.E();
      }
    }
    for(int i = 0; i < branchMissingET->GetEntriesFast(); ++i)
    {
      MissingET* MET = (MissingET *)branchMissingET->At(i);
      TLorentzVector MET_mom = MET->P4();
      tuple->met_phi = MET_mom.Phi();
      tuple->met_px = MET_mom.Px();
      tuple->met_py = MET_mom.Py();
      tuple->met = MET->MET;
    }
    
    
    tuple->dtheta = TMath::ACos((tuple->tau_vis_px*tuple->met_px + tuple->tau_vis_py*tuple->met_py)/(tuple->tau_vis_pt*tuple->met));
    tuple->mT = TMath::Sqrt(2*tuple->tau_vis_pt*tuple->met*(1-TMath::Cos(tuple->dtheta)));
    tuple->pTratio = tuple->tau_vis_pt/tuple->met;

    tuple->gen_dtheta = TMath::ACos((tuple->gentau_vis_px*tuple->genmet_px + tuple->gentau_vis_py*tuple->genmet_py)/(tuple->gentau_vis_pt*tuple->genmet));    
    tuple->gen_mT = TMath::Sqrt(2*tuple->gentau_vis_pt*tuple->genmet*(1-TMath::Cos(tuple->gen_dtheta)));
    tuple->gen_pTratio = tuple->gentau_vis_pt/tuple->genmet;


    tuple->default_tree->Fill();
    tuple->Initialise();
  }
  
  TFile *fnew = new TFile("./WJetsHT100To200/Skim"+TString(inputFile), "recreate");
  tuple->default_tree->Write();
  fnew->Close();


}
void makeAll(){
  // AnalysisWprimeToTauNu("WprimeM-1000GeVToTauNu.root");
  // AnalysisWprimeToTauNu("WprimeM-2000GeVToTauNu.root");
  // AnalysisWprimeToTauNu("WprimeM-3000GeVToTauNu.root");
  // AnalysisWprimeToTauNu("WprimeM-4000GeVToTauNu.root");
  // AnalysisWprimeToTauNu("WprimeM-5000GeVToTauNu.root");

  AnalysisWprimeToTauNu("WJetsToLNu_HT400-600_run1.root");
}
  
void LoopOver(int nfiles,int intex){
  for(int i=intex; i<nfiles; i++){

    TString file_name = "WJetsHT100-200ToLNu_run"+to_string(i)+".root";
    cout<<"Opening "<<file_name<<endl;
    AnalysisWprimeToTauNu(file_name);
    cout<<"Closing "<<file_name<<endl;

  }
}
