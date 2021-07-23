# NN settings

# Higgs mass range in GeV
min_mass = 50
max_mass = 250

# channels to process
channels = "inclusive"

# NN structure
Nlayers = 10
Nneurons = 100

# NN training
loss = "mae"
optimizer = "Adam"
w_init_mode = "glorot_uniform"
activation = "relu"
regularizer = None
epochs = 200

# Dataset splitting
train_frac = 0.7
valid_frac = 0.2
random_seed = 2020

# Target and inputs
target = "tauH_SVFIT_mass"

default_model_inputs_file = "PuppiMET_with_METcov_j1j2jr_Nnu_Npu"
model_inputs_file = __import__('DL_for_HTT.common.model_inputs.{}'.format(default_model_inputs_file), fromlist=[''])
inputs = model_inputs_file.inputs

inputs_from_Heppy = {
    "tau1_pt_reco" : "l1_pt",
    "tau1_eta_reco" : "l1_eta",
    "tau1_phi_reco" : "l1_phi",
    "tau2_pt_reco" : "l2_pt",
    "tau2_eta_reco" : "l2_eta",
    "tau2_phi_reco" : "l2_phi",
    "jet1_pt_reco" : "j1_pt",
    "jet1_eta_reco" : "j1_eta",
    "jet1_phi_reco" : "j1_phi",
    "jet2_pt_reco" : "j2_pt",
    "jet2_eta_reco" : "j2_eta",
    "jet2_phi_reco" : "j2_phi",
    "MET_pt_reco" : "met",
    "MET_phi_reco" : "metphi",
    "mT1_reco" : "l1_mt",
    "mT2_reco" : "l2_mt",
    "mTtt_reco" : "mt_tt",
    "mTtot_reco" : "mt_tot",
}
