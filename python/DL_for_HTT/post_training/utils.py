import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd

import DL_for_HTT.common.NN_settings as NN_default_settings

import DL_for_HTT.post_training.macros as macros

def load_model_from_json(input_json):
    model_name = input_json.split('/')[-1].replace('.json', '')
    if  model_name[:3] == 'XGB':
        model_type = model_name.split("-")[0]
        import xgboost as xgb
        loaded_model = xgb.XGBRegressor()
        loaded_model.load_model(input_json)
        
        # ... -inclusive-max_depth-5-eta-0.1-n_estimators-500-es-5-gamma-0-min_child_weight-1-loss-rmse
        # Get infos on the trained XGB
        infos = model_name
        infos = infos.replace('.json', '')

        objective = infos.split("-")[-1]
        eval_ = infos.split("-")[-3]
        min_child_weight = infos.split("-")[-5]
        gamma = infos.split("-")[-7]
        early_stopping_rounds = infos.split("-")[-9]
        n_estimators = infos.split("-")[-11]
        eta = infos.split("-")[-13]
        max_depth = infos.split("-")[-15]
        channel = infos.split("-")[-17]
        
        print("Properties:")
        
        print(
            "\t{} channel, eta = {}, max_depth = {}, n_estimators = {}, early_stopping_rounds = {},".format(
                channel,
                eta,
                max_depth,
                n_estimators,
                early_stopping_rounds,
            )
        )
        print(
            "\t objective = {}, eval = {}, gamma = {}, min_child_weight = {}".format(
                objective,
                eval_,
                gamma,
                min_child_weight,
            )
        )
    else:
        model_type = 'DNN'
        from keras.models import model_from_json
        
        # load json and create model
        NN_weights_path_and_file = input_json.split('/')
        NN_weights_path_and_file[-1] = "NN_weights-{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
        NN_weights_file = "/".join(NN_weights_path_and_file)
        
        json_file = open(input_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(NN_weights_file)
        print("Loaded model from disk:")
        print("\t{}".format(input_json))
        
        # Get infos on the trained NN
        infos = NN_weights_path_and_file[-1]
        infos = infos.replace('.h5', '')
        infos = infos.replace('NN_weights-', '')
        
        is_bottleneck = ("-bottleneck" == infos[-11:])
        
        bottleneck = ""
        if is_bottleneck:
            infos = infos.replace('-bottleneck', '')
            bottleneck = "-bottleneck"
            
        Nneurons = infos.split("-")[-2]
        Nlayers = infos.split("-")[-4]
        channel = infos.split("-")[-5]
        
        w_init_mode = infos.split("-")[-6]
        optimizer = infos.split("-")[-7]
        loss = infos.split("-")[-8]
        
        print("Properties:")
        
        print(
            "\t{} channel, {} hidden layers of {} neurons with{} bottleneck".format(
                channel,
                Nlayers,
                Nneurons,
                "" if is_bottleneck else "out",
            )
        )
        print(
            "\ttrained with {} optimizer, w_init {} and {} loss.".format(
                optimizer,
                w_init_mode,
                loss,
            )
        )
    return loaded_model, model_type, model_name

def load_h5_file_and_predict(input_h5, loaded_model, model_type, model_name, only=None, inputs = NN_default_settings.inputs, target = NN_default_settings.target):
    df = pd.read_hdf(input_h5)

#    from sklearn.preprocessing import StandardScaler
#    scaler  = StandardScaler()
#    scaler.fit(df[inputs])
#    df[inputs] = pd.DataFrame(scaler.transform(df[inputs]),columns=df[inputs].columns)
    
    if only != None:
        df = df.loc[df['is_{}'.format(only)] == 1]
    
    if "N_neutrinos_reco" in inputs:
        df["N_neutrinos_reco"] = 2*np.ones(len(df["channel_reco"]), dtype='int')
        df.loc[(df["channel_reco"] == "mt"), ["N_neutrinos_reco"]] = 3
        df.loc[(df["channel_reco"] == "et"), ["N_neutrinos_reco"]] = 3
        df.loc[(df["channel_reco"] == "mm"), ["N_neutrinos_reco"]] = 4
        df.loc[(df["channel_reco"] == "em"), ["N_neutrinos_reco"]] = 4
        df.loc[(df["channel_reco"] == "ee"), ["N_neutrinos_reco"]] = 4

    if "tau1_px_reco" in inputs:
        for ptc in ["tau1", "tau2", "jet1", "jet2", "remaining_jets", "MET", "PuppiMET"]:
            if "{}_eta_reco".format(ptc) in df.keys():
                df["{}_pz_reco".format(ptc)] = df["{}_pt_reco".format(ptc)] * np.sinh(df["{}_eta_reco".format(ptc)])
            df["{}_px_reco".format(ptc)] = df["{}_pt_reco".format(ptc)] * np.cos(df["{}_phi_reco".format(ptc)])
            df["{}_py_reco".format(ptc)] = df["{}_pt_reco".format(ptc)] * np.sin(df["{}_phi_reco".format(ptc)])

    for leg in ["leg1", "leg2"]:
        for variable in ["pt", "eta", "phi"]:
            for subsample in ["is_train", "is_valid", "is_test"]:
                if "{leg}_{variable}_gen".format(leg=leg, variable=variable) in inputs:
                    df.loc[(df["{leg}_{variable}_gen".format(leg=leg, variable=variable)] == -10), [subsample]] = False

    df["pairType"] = np.zeros(len(df["channel"]), dtype='int')
    df.loc[(df["channel"] == "mt"), ["pairType"]] = 0
    df.loc[(df["channel"] == "et"), ["pairType"]] = 1
    df.loc[(df["channel"] == "tt"), ["pairType"]] = 2
     
    df["is_train"] = df["is_train"].astype(int)
    df["is_valid"] = df["is_valid"].astype(int)
    df["is_test"] = df["is_test"].astype(int)
        
    if model_type == None:
        df["predictions"] = df[model_name]
    elif model_type == 'XGBoost':
        from xgboost import DMatrix
        df["predictions"] = loaded_model.predict(DMatrix(data = np.r_[df[inputs]], feature_names=inputs))
    else:
        df["predictions"] = loaded_model.predict(df[inputs])
    return df

from scipy.optimize import curve_fit

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def make_gaussian_fit(ax_hist):
    x, y = (ax_hist[1][1:]+ax_hist[1][:-1])/2, ax_hist[0]
    popt,pcov = curve_fit(gaus, x, y, p0=[1,1,1])
    return x, popt

def tester(df, channel, model_name, min_mass, max_mass, prefix = '', target = None, **kwargs):

    df1 = macros.filter_channel(df, channel)
        
    medians_model = []
    CL68s_model_up = []
    CL68s_model_do = []
    CL95s_model_up = []
    CL95s_model_do = []
    xpos = []
    
    mHcuts = np.arange(min_mass, max_mass, 10) # [.200, .350]
    mHranges = [[min_mass, mHcuts[0]]]
    for mHcut in mHcuts[1:]:
        mHranges.append([mHranges[-1][1], mHcut])
    mHranges.append([mHranges[-1][1], max_mass])
    for mHrange in mHranges:
        mHrange[0] = np.round(mHrange[0],3)
        mHrange[1] = np.round(mHrange[1],3)
        
        df2 = df1.loc[(df1[target] >= mHrange[0]) & (df1[target] < mHrange[1])]
        
        predictions = np.r_[df2["predictions"]]
        if len(predictions) == 0:
            continue

        xpos.append((mHrange[1]+mHrange[0])/2)

        mHs = np.r_[df2[target]]
        values_model = predictions/mHs
        
        values_model = [v for v in values_model]
        values_model.sort()

        try:
            medians_model.append(values_model[int(len(values_model)/2)])
        except:
            import pdb; pdb.set_trace()

        above_model = [v for v in values_model if v >= medians_model[-1]]
        below_model = [v for v in values_model if v <= medians_model[-1]]

        above_model.sort()
        below_model.sort(reverse = True)

        CL68s_model_up.append(above_model[int(0.68 * len(above_model))])
        CL68s_model_do.append(below_model[int(0.68 * len(below_model))])
        CL95s_model_up.append(above_model[int(0.95 * len(above_model))])
        CL95s_model_do.append(below_model[int(0.95 * len(below_model))])

    median_diff = 0
    CL68_width = 0
    CL95_width = 0
    CL68_calibr_width = 0
    CL95_calibr_width = 0

    for k in range(len(medians_model)):
        median_diff += abs(medians_model[k] - 1)
        CL68_width += CL68s_model_up[k] - CL68s_model_do[k]
        CL95_width += CL95s_model_up[k] - CL95s_model_do[k]
        CL68_calibr_width += (CL68s_model_up[k] - CL68s_model_do[k])/medians_model[k]
        CL95_calibr_width += (CL95s_model_up[k] - CL95s_model_do[k])/medians_model[k]

    df2= df1.loc[(df1[target] >= min_mass) & (df1[target] < max_mass)]
    N = len(df2)
    
    median_diff *= 1./len(medians_model)
    CL68_width *= 1./len(medians_model)
    CL95_width *= 1./len(medians_model)
    CL68_calibr_width *= 1./len(medians_model)
    CL95_calibr_width *= 1./len(medians_model)
    
    y_true = df2[target].array
    y_pred = df2["predictions"].array
    
    mse = ((y_pred-y_true)**2).mean()
    mae = (np.abs(y_pred-y_true)).mean()
    mape = (np.abs(y_pred-y_true)/y_true).mean() * 100
        
    return median_diff, CL68_width, CL95_width, CL68_calibr_width, CL95_calibr_width, mse, mae, mape, N, len(medians_model)

def create_scores_database(args):
    command = "find {} -type f -name \*.perfs".format(args.basedir)
    for filter_to_apply in args.filters_to_match.split(','):
        command = "{} | grep {}".format(command, filter_to_apply)
    for filter_to_apply in args.filters_to_not_match.split(','):
        command = "{} | grep -ve {}".format(command, filter_to_apply)

    perf_files = os.popen(command).readlines()
    perf_files = [f[:-1] for f in perf_files]

    all_data = []
    print("Processing on {} perf files...".format(len(perf_files)))
    for model_perf in perf_files:
        data = {}

        data["file"] = model_perf
        data["type"] = "XGB" if "xgboosts" in model_perf else "DNN"
        data["model_inputs"] = model_perf.split("/")[-2]
        data["training_dataset"] = model_perf.split("/")[-3]
        model_name = model_perf.split("/")[-1].replace(".perfs", "")
        if data["type"] == "XGB":
            data["max_depth"] = int(model_name.split("-")[-15])
            data["eta"] = float(model_name.split("-")[-13])
            data["n_estimators"] = int(model_name.split("-")[-11])
            data["early_stopping_rounds"] = int(model_name.split("-")[-9])
            data["gamma"] = float(model_name.split("-")[-7])
            data["min_child_weight"] = float(model_name.split("-")[-5])
            data["eval"] = model_name.split("-")[-3]
            data["loss"] = model_name.split("-")[-1]
        elif data["type"] == "DNN":
            is_bottleneck = ("-bottleneck" == model_name.split("-")[-1])
            data["bottleneck"] = is_bottleneck
            if is_bottleneck:
                model_name.replace('-bottleneck', '')
            data["Nneurons"] = int(model_name.split("-")[-2])
            data["Nlayers"] = int(model_name.split("-")[-4])
            data["loss"] = model_name.split("-")[-8]
            data["optimizer"] = model_name.split("-")[-7]
            data["w_init_mode"] = model_name.split("-")[-6]
            data["activation"] = model_name.split("-")[-11]
            if "ADAM_glorot_uniform" in model_perf:
                data["optimizer"] = "Adam"
                data["w_init_mode"] = "gu"
                        
        for region in ["low", "medium", "high", "full"]:
            for perf in ["median_diff", "CL68_width", "CL95_width", "CL68_calibr_width", "CL95_calibr_width", "mse", "mae", "mape"]:
                key = "_".join([region, perf])
                key_for_data = "_".join([region, perf])
                key_for_data = key_for_data.replace("CL68", "1sig")
                key_for_data = key_for_data.replace("CL95", "2sig")
                try:
                    data[key_for_data] = float(os.popen('grep {} {}'.format(key, model_perf)).readlines()[0][:-1].split(" ")[1])
                except:
                    print("{} not found for {}".format(key, model_perf))

        all_data.append(data)

    print("Building DataFrame...")
    df = pd.DataFrame(all_data)
    print("DataFrame created, saving...")
    df.to_hdf("{}/{}.h5".format(args.database_path, args.database_name), key='df')
