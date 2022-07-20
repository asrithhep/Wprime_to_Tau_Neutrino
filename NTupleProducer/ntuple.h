#ifndef ntuple_h
#define ntuple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>



class ntuple
{
 public:
  TTree* default_tree;

  Float_t gentau_px;
  Float_t gentau_py;
  Float_t gentau_pz;
  Float_t gentau_e;

  Float_t gentau_pt;
  Float_t gentau_eta;
  Float_t gentau_phi;

  Float_t genmet_px;
  Float_t genmet_py;

  Float_t genmet;
  Float_t genmet_phi;
  Float_t gen_mT;
  Float_t gen_pTratio;
  Float_t gen_dtheta;
 
  Float_t gentau_vis_px;
  Float_t gentau_vis_py;
  Float_t gentau_vis_pz;
  Float_t gentau_vis_e;

  Float_t gentau_vis_pt;
  Float_t gentau_vis_eta;
  Float_t gentau_vis_phi;

  Float_t tau_vis_px;
  Float_t tau_vis_py;
  Float_t tau_vis_pz;
  Float_t tau_vis_e;

  Float_t tau_vis_pt ;
  Float_t tau_vis_eta;
  Float_t tau_vis_phi;

  Float_t met_px;
  Float_t met_py;

  Float_t met;
  Float_t met_phi;
  Float_t mT;
  Float_t pTratio;
  Float_t dtheta;
 

  Float_t boson_px;
  Float_t boson_py;
  Float_t boson_pz;
  Float_t boson_e;
  Float_t boson_mass;
  
  Float_t minv;
  
  Float_t neutrino_px;
  Float_t neutrino_py;
  Float_t neutrino_pz;
  Float_t neutrino_e;
  
  Bool_t TauTag;
  
  
  ntuple(TTree* tree)
    {
      tree->Branch("gentau_px",&gentau_px,"gentau_px/F");
      tree->Branch("gentau_py",&gentau_py,"gentau_py/F");
      tree->Branch("gentau_pz",&gentau_pz,"gentau_pz/F");
      tree->Branch("gentau_e",&gentau_e,"gentau_e/F");
      tree->Branch("gentau_pt",&gentau_pt,"gentau_pt/F");
      tree->Branch("gentau_eta",&gentau_eta,"gentau_eta/F");
      tree->Branch("gentau_phi",&gentau_phi,"gentau_phi/F");
      tree->Branch("genmet_px",&genmet_px,"genmet_px/F");
      tree->Branch("genmet_py",&genmet_py,"genmet_py/F");
      tree->Branch("genmet_phi",&genmet_phi,"genmet_phi/F");
      tree->Branch("genmet",&genmet,"genmet/F");
      tree->Branch("gen_mT",&gen_mT,"gen_mT/F");
      tree->Branch("gen_pTratio",&gen_pTratio,"gen_pTratio/F");
      tree->Branch("gen_dtheta",&gen_dtheta,"gen_dtheta/F");
      
      tree->Branch("neutrino_px",&neutrino_px,"neutrino_px/F");
      tree->Branch("neutrino_py",&neutrino_py,"neutrino_py/F");
      tree->Branch("neutrino_pz",&neutrino_pz,"neutrino_pz/F");
      tree->Branch("neutrino_e",&neutrino_e,"neutrino_e/F");
      
      tree->Branch("gentau_vis_px",&gentau_vis_px,"gentau_vis_px/F");
      tree->Branch("gentau_vis_py",&gentau_vis_py,"gentau_vis_py/F");
      tree->Branch("gentau_vis_pz",&gentau_vis_pz,"gentau_vis_pz/F");
      tree->Branch("gentau_vis_e",&gentau_vis_e,"gentau_vis_e/F");
      tree->Branch("gentau_vis_pt",&gentau_vis_pt,"gentau_vis_pt/F");
      tree->Branch("gentau_vis_eta",&gentau_vis_eta,"gentau_vis_eta/F");
      tree->Branch("gentau_vis_phi",&gentau_vis_phi,"gentau_vis_phi/F");

      tree->Branch("tau_vis_px",&tau_vis_px,"tau_vis_px/F");
      tree->Branch("tau_vis_py",&tau_vis_py,"tau_vis_py/F");
      tree->Branch("tau_vis_pz",&tau_vis_pz,"tau_vis_pz/F");
      tree->Branch("tau_vis_e",&tau_vis_e,"tau_vis_e/F");
      tree->Branch("tau_vis_pt",&tau_vis_pt,"tau_vis_pt/F");
      tree->Branch("tau_vis_eta",&tau_vis_eta,"tau_vis_eta/F");
      tree->Branch("tau_vis_phi",&tau_vis_phi,"tau_vis_phi/F");
      tree->Branch("met_px",&met_px,"met_px/F");
      tree->Branch("met_py",&met_py,"met_py/F");
      tree->Branch("met_phi",&met_phi,"met_phi/F");
      tree->Branch("met",&met,"met/F");
      tree->Branch("mT",&mT,"mT/F");
      tree->Branch("pTratio",&pTratio,"pTratio/F");
      tree->Branch("dtheta",&dtheta,"dtheta/F");

      tree->Branch("boson_px",&boson_px,"boson_px/F");
      tree->Branch("boson_py",&boson_py,"boson_py/F");
      tree->Branch("boson_pz",&boson_pz,"boson_pz/F");
      tree->Branch("boson_e",&boson_e,"boson_e/F");
      tree->Branch("boson_mass",&boson_mass,"boson_mass/F");
      tree->Branch("minv",&minv,"minv/F");
      tree->Branch("TauTag",&TauTag,"TauTag/O");
      
      default_tree = tree;
      
    }

  void Fill(){
    default_tree->Fill();
  }  
  void Initialise(){
    gentau_vis_pt = -1;
    gentau_pt     = -1;
    tau_vis_pt    = -1;
  }
  
  ~ntuple()
    {
      std::cout<<"Tree created"<<endl;
    }
  
};
#endif
