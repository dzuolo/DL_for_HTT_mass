import DL_for_HTT.post_training.utils as utils

from DL_for_HTT.common.NN_settings import target
from DL_for_HTT.common.labels import labels

default_language = 'fr'
import locale
import matplotlib.pyplot as plt
plt.rcdefaults()

import numpy as np

from xgboost import plot_importance

plt.rcParams["figure.figsize"] = [7, 7]
plt.rcParams['axes.formatter.use_locale'] = True

vars_with_y_log_scale = [
    'tau1_pt_reco',
    'tau2_pt_reco',
]

def filter_channel(df, channel = None):
    df1 = df
    if channel in set(df['channel_reco']):
        df1 = df.loc[(df['channel_reco'] == channel)]
    elif channel == "lt":
        df1 = df.loc[(df['channel_reco'] == "mt") | (df['channel_reco'] == "et")]
    elif channel == "ll":
        df1 = df.loc[(df['channel_reco'] == "mm") | (df['channel_reco'] == "em") | (df['channel_reco'] == "ee")]
    return df1    

def gen_vs_reco(df, channel, model_name, min_mass, max_mass, language=default_language, prefix = '', file_format = 'png', **kwargs):
    gen_vars = [k for k in df.keys() if "_gen" in k]
    reco_vars = [k for k in df.keys() if "_reco" in k]
    for reco_var in reco_vars:
        gen_var = reco_var.replace("_reco", "_gen")
        if not gen_var in gen_vars:
            continue

        var = reco_var.replace("_reco", "")
        if any([v in var for v in ["phi", "channel", "Nevt", "pdgId", "Event"]]):
            continue

        print("\t gen_vs_reco on {}".format(var))

        df1 = filter_channel(df, channel)
        
        medians_model = []
        CL68s_model_up = []
        CL68s_model_do = []
        CL95s_model_up = []
        CL95s_model_do = []
        xpos = []
        xerr = []
    
        mHcuts = np.arange(min_mass, max_mass, 10) # [.200, .350]
        mHranges = [[min_mass, mHcuts[0]]]
        for mHcut in mHcuts[1:]:
            mHranges.append([mHranges[-1][1], mHcut])
        mHranges.append([mHranges[-1][1], max_mass])
        for mHrange in mHranges:
            mHrange[0] = np.round(mHrange[0],3)
            mHrange[1] = np.round(mHrange[1],3)
        
            df2 = df1.loc[(df1[target] >= mHrange[0]) & (df1[target] <= mHrange[1])]
        
            predictions = np.r_[df2[reco_var]]
            if len(predictions) == 0:
                continue

            xpos.append((mHrange[1]+mHrange[0])/2)
            xerr.append((mHrange[1]-mHrange[0])/2)

            # mTtots = np.r_[df2["mTtot_reco"]]
            mHs = np.r_[df2[gen_var]]
            values_model = predictions/mHs
            # values_mTtot = mTtots/mHs
        
            values_model = [v for v in values_model]
            # values_mTtot = [v for v in values_mTtot]
            values_model.sort()
            # values_mTtot.sort()

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

        if language == 'fr':
            locale.setlocale(locale.LC_NUMERIC, 'fr_FR.UTF-8')
            
        fig, ax = plt.subplots()
        #fig.suptitle(model_name)
        plt.xlabel(labels["GenHiggsMassGeV"][language])
        plt.ylabel("{} reco/gen".format(var))

        ax.fill_between(
            xpos, CL95s_model_do, CL68s_model_do,
            color = "yellow", alpha = .5, label = "$\pm2\sigma$",
        )
        ax.fill_between(
            xpos, CL68s_model_up, CL95s_model_up,
            color = "yellow", alpha = .5,
        )
        ax.fill_between(
            xpos, CL68s_model_do, CL68s_model_up,
            color = "green", alpha = .5, label = "$\pm1\sigma$",
        )
        ax.errorbar(
            xpos, medians_model, xerr = xerr, #yerr = sigmas,
            marker='.', markersize=4, linewidth=0, elinewidth=1,
            fmt=' ', capsize = 3, capthick = 0, color = "black", label = labels["median"][language],
        )
    
        plt.plot([min_mass, max_mass], [1,1], color='C3')    
        
        plt.ylim(0,3)
        plt.xlim(min_mass, max_mass)
        
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        fig.savefig("gen_vs_reco-{}{}{}.{}".format(prefix,var, "-en" if language=='en' else "", file_format))

def model_response_tau_filtered(df, channel, model_name, min_mass, max_mass, language = default_language,prefix = '', file_format = 'png', **kwargs):
    channel = "tt"

    dR_max_gen = 0.1
    gen_type = "tau"
    
    dR_min_jet = 0.5
    dR_max_jet = 0.1

    pT_max_jet_frac = 2./3

    for filter in ['gen', "r_gen", 'jet', 'r_jet']:
        df1 = filter_channel(df, channel)

        if filter == 'gen':
            df1 = df1.loc[df1[gen_type+"1_phi_gen"] != -10 ]
            df1 = df1.loc[df1[gen_type+"2_phi_gen"] != -10 ]
            Delta_eta_1 = abs(df1[gen_type+"1_eta_gen"]-df1["tau1_eta_reco"])
            Delta_eta_2 = abs(df1[gen_type+"2_eta_gen"]-df1["tau2_eta_reco"])
            Delta_phi_1 = abs(df1[gen_type+"1_phi_gen"]-df1["tau1_phi_reco"])
            Delta_phi_2 = abs(df1[gen_type+"2_phi_gen"]-df1["tau2_phi_reco"])
            
            Delta_phi_1[Delta_phi_1 > np.pi] -= 2*np.pi
            Delta_phi_2[Delta_phi_2 > np.pi] -= 2*np.pi
            
            df1 = df1.loc[(Delta_eta_1**2+Delta_phi_1**2) < dR_max_gen**2]
            df1 = df1.loc[(Delta_eta_2**2+Delta_phi_2**2) < dR_max_gen**2]

        if filter == 'r_gen':
            df1 = df1.loc[df1[gen_type+"1_phi_gen"] != -10 ]
            df1 = df1.loc[df1[gen_type+"2_phi_gen"] != -10 ]
            Delta_eta_1 = abs(df1[gen_type+"1_eta_gen"]-df1["tau1_eta_reco"])
            Delta_eta_2 = abs(df1[gen_type+"2_eta_gen"]-df1["tau2_eta_reco"])
            Delta_phi_1 = abs(df1[gen_type+"1_phi_gen"]-df1["tau1_phi_reco"])
            Delta_phi_2 = abs(df1[gen_type+"2_phi_gen"]-df1["tau2_phi_reco"])
            
            Delta_phi_1[Delta_phi_1 > np.pi] -= 2*np.pi
            Delta_phi_2[Delta_phi_2 > np.pi] -= 2*np.pi
            
            df1 = df1.loc[(Delta_eta_1**2+Delta_phi_1**2) > dR_max_gen**2]
            df1 = df1.loc[(Delta_eta_2**2+Delta_phi_2**2) > dR_max_gen**2]

        if filter in ["jet", "r_jet"]:
            Delta_eta_11 = abs(df1["jet1_eta_reco"]-df1["tau1_eta_reco"])
            Delta_eta_12 = abs(df1["jet1_eta_reco"]-df1["tau2_eta_reco"])
            Delta_eta_21 = abs(df1["jet2_eta_reco"]-df1["tau1_eta_reco"])
            Delta_eta_22 = abs(df1["jet2_eta_reco"]-df1["tau2_eta_reco"])
            Delta_phi_11 = abs(df1["jet1_phi_reco"]-df1["tau1_phi_reco"])
            Delta_phi_12 = abs(df1["jet1_phi_reco"]-df1["tau2_phi_reco"])
            Delta_phi_21 = abs(df1["jet2_phi_reco"]-df1["tau1_phi_reco"])
            Delta_phi_22 = abs(df1["jet2_phi_reco"]-df1["tau2_phi_reco"])
            
            Delta_phi_11[Delta_phi_11 > np.pi] -= 2*np.pi
            Delta_phi_21[Delta_phi_21 > np.pi] -= 2*np.pi
            Delta_phi_12[Delta_phi_12 > np.pi] -= 2*np.pi
            Delta_phi_22[Delta_phi_22 > np.pi] -= 2*np.pi

            if filter == "jet":
                df1 = df1.loc[(Delta_eta_11**2+Delta_phi_11**2) > dR_min_jet**2]
                df1 = df1.loc[(Delta_eta_12**2+Delta_phi_12**2) > dR_min_jet**2]
                df1 = df1.loc[(Delta_eta_21**2+Delta_phi_21**2) > dR_min_jet**2]
                df1 = df1.loc[(Delta_eta_22**2+Delta_phi_22**2) > dR_min_jet**2]

            if filter == "r_jet":
                df1 = df1.loc[((Delta_eta_11**2+Delta_phi_11**2) < dR_max_jet**2) | ((Delta_eta_12**2+Delta_phi_12**2) < dR_max_jet**2)]
                df1 = df1.loc[((Delta_eta_21**2+Delta_phi_21**2) < dR_max_jet**2) | ((Delta_eta_22**2+Delta_phi_22**2) < dR_max_jet**2)]

                df1 = df1.loc[
                    (
                        (
                            df1["jet1_pt_reco"] > df1["tau1_pt_reco"] * pT_max_jet_frac
                        ) & (
                            df1["jet1_pt_reco"] < df1["tau1_pt_reco"] / pT_max_jet_frac
                        )
                    ) | (
                        (
                            df1["jet1_pt_reco"] > df1["tau2_pt_reco"] * pT_max_jet_frac
                        ) & (
                            df1["jet1_pt_reco"] < df1["tau2_pt_reco"] / pT_max_jet_frac
                        )
                    )
                ]

                df1 = df1.loc[
                    (
                        (
                            df1["jet2_pt_reco"] > df1["tau1_pt_reco"] * pT_max_jet_frac
                        ) & (
                            df1["jet2_pt_reco"] < df1["tau1_pt_reco"] / pT_max_jet_frac
                        )
                    ) | (
                        (
                            df1["jet2_pt_reco"] > df1["tau2_pt_reco"] * pT_max_jet_frac
                        ) & (
                            df1["jet2_pt_reco"] < df1["tau2_pt_reco"] / pT_max_jet_frac
                        )
                    )
                ]
                            
                            
        
        model_response(df1, channel, model_name, min_mass, max_mass, language, prefix = prefix+"FilterBy"+filter)

def model_response(df, channel, model_name, min_mass, max_mass, language = default_language, prefix = '', file_format = 'png', **kwargs):
    df1 = filter_channel(df, channel)
        
    medians_model = []
    averages = []
    averages_diff = []
    CL68s_model_up = []
    CL68s_model_do = []
    CL95s_model_up = []
    CL95s_model_do = []
    medians_model_diff = []
    CL68s_model_diff_up = []
    CL68s_model_diff_do = []
    CL95s_model_diff_up = []
    CL95s_model_diff_do = []
    medians_mTtot = []
    CL68s_mTtot_up = []
    CL68s_mTtot_do = []
    CL95s_mTtot_up = []
    CL95s_mTtot_do = []
    xpos = []
    xerr = []
    
    mHcuts = np.arange(min_mass, max_mass, 10) # [.200, .350]
    mHranges = [[min_mass, mHcuts[0]]]
    for mHcut in mHcuts[1:]:
        mHranges.append([mHranges[-1][1], mHcut])
    mHranges.append([mHranges[-1][1], max_mass])
    for mHrange in mHranges:
        mHrange[0] = np.round(mHrange[0],3)
        mHrange[1] = np.round(mHrange[1],3)
        
        df2 = df1.loc[(df1[target] >= mHrange[0]) & (df1[target] <= mHrange[1])]
        
        predictions = np.r_[df2["predictions"]]
        if len(predictions) == 0:
            continue

        xpos.append((mHrange[1]+mHrange[0])/2)
        xerr.append((mHrange[1]-mHrange[0])/2)

        # mTtots = np.r_[df2["mTtot_reco"]]
        mHs = np.r_[df2[target]]
        values_model = predictions/mHs
        values_model_diff = predictions - mHs
        # values_mTtot = mTtots/mHs
        
        values_model = [v for v in values_model]
        values_model_diff = [v for v in values_model_diff]
        # values_mTtot = [v for v in values_mTtot]
        values_model.sort()
        values_model_diff.sort()
        # values_mTtot.sort()

        try:
            averages.append(np.mean(values_model))
            averages_diff.append(np.mean(values_model_diff))
            medians_model.append(values_model[int(len(values_model)/2)])
            medians_model_diff.append(values_model_diff[int(len(values_model_diff)/2)])
            # medians_mTtot.append(values_mTtot[int(len(values_mTtot)/2)])
        except:
            import pdb; pdb.set_trace()

        above_model = [v for v in values_model if v >= medians_model[-1]]
        below_model = [v for v in values_model if v <= medians_model[-1]]
        above_model_diff = [v for v in values_model_diff if v >= medians_model_diff[-1]]
        below_model_diff = [v for v in values_model_diff if v <= medians_model_diff[-1]]
        # above_mTtot = [v for v in values_mTtot if v >= medians_mTtot[-1]]
        # below_mTtot = [v for v in values_mTtot if v <= medians_mTtot[-1]]

        above_model.sort()
        below_model.sort(reverse = True)
        above_model_diff.sort()
        below_model_diff.sort(reverse = True)
        # above_mTtot.sort()
        # below_mTtot.sort(reverse = True)

        CL68s_model_up.append(above_model[int(0.68 * len(above_model))])
        CL68s_model_do.append(below_model[int(0.68 * len(below_model))])
        CL95s_model_up.append(above_model[int(0.95 * len(above_model))])
        CL95s_model_do.append(below_model[int(0.95 * len(below_model))])
        CL68s_model_diff_up.append(above_model_diff[int(0.68 * len(above_model_diff))])
        CL68s_model_diff_do.append(below_model_diff[int(0.68 * len(below_model_diff))])
        CL95s_model_diff_up.append(above_model_diff[int(0.95 * len(above_model_diff))])
        CL95s_model_diff_do.append(below_model_diff[int(0.95 * len(below_model_diff))])
        # CL68s_mTtot_up.append(above_mTtot[int(0.68 * len(above_mTtot))])
        # CL68s_mTtot_do.append(below_mTtot[int(0.68 * len(below_mTtot))])
        # CL95s_mTtot_up.append(above_mTtot[int(0.95 * len(above_mTtot))])
        # CL95s_mTtot_do.append(below_mTtot[int(0.95 * len(below_mTtot))])

    if language == 'fr':
        locale.setlocale(locale.LC_NUMERIC, 'fr_FR.UTF-8')

    fig, ax = plt.subplots()
    #fig.suptitle(model_name)
    plt.xlabel(labels["GenHiggsMassGeV"][language])
    plt.ylabel(labels["Pred_on_HiggsGenMass"][language])
    
    ax.fill_between(
        xpos, CL95s_model_do, CL68s_model_do,
        color = "yellow", alpha = .5, label = "$\pm2\sigma$",
    )
    ax.fill_between(
        xpos, CL68s_model_up, CL95s_model_up,
        color = "yellow", alpha = .5,
    )
    ax.fill_between(
        xpos, CL68s_model_do, CL68s_model_up,
        color = "green", alpha = .5, label = "$\pm1\sigma$",
    )
    ax.errorbar(
        xpos, medians_model, xerr = xerr, #yerr = sigmas,
        marker='.', markersize=4, linewidth=0, elinewidth=1,
        fmt=' ', capsize = 3, capthick = 0, color = "black", label = labels["median"][language],
    )
    ax.errorbar(
        xpos, averages, xerr = xerr,
        marker='+', markersize=5, linewidth=0, elinewidth=1,
        fmt=' ', capsize = 3, capthick = 0, color = "C4", label = labels["average"][language],
    )
    # ax.errorbar(
    #     xpos, medians_model, xerr = xerr, #yerr = sigmas,
    #     marker='+', markersize=4, linewidth=.4, elinewidth=1,
    #     fmt=' ', capsize = 3, capthick = .4, color = "black", #label = "DNN",
    # )

    # ax.plot(
    #     xpos, CL95s_mTtot_do,
    #     color = "C7", #alpha = .5,
    #     dashes = [1,1],
    # )
    # ax.plot(
    #     xpos, CL95s_mTtot_up,
    #     color = "C7", #alpha = .5,
    #     dashes = [1,1],
    # )
    # ax.plot(
    #     xpos, CL68s_mTtot_do,
    #     color = "C7", #alpha = .5,
    #     dashes = [2,2],
    # )
    # ax.plot(
    #     xpos, CL68s_mTtot_up,
    #     color = "C7", #alpha = .5,
    #     dashes = [2,2],
    # )
    # ax.plot(
    #     xpos, medians_mTtot,
    #     color = "C7", #alpha = .5,
    #     #dashes = [2,1],
    #     label = "mTtot",
    # )
    
    plt.plot([min_mass, max_mass], [1,1], color='C3')    

    plt.ylim(0,3)
    plt.xlim(min_mass, max_mass)

    ax.legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig("model_response-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))

    plt.xlim(min_mass, 200)
    plt.xticks(np.arange(min_mass, 201, step=10))
    fig.savefig("model_response_lowmass-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))

    plt.clf()
    fig, ax = plt.subplots()
    #fig.suptitle(model_name)
    plt.xlabel(labels["GenHiggsMassGeV"][language])
    plt.ylabel(labels["Pred_minus_HiggsGenMassGeV"][language])
    
    ax.fill_between(
        xpos, CL95s_model_diff_do, CL68s_model_diff_do,
        color = "yellow", alpha = .5, label = "$\pm2\sigma$",
    )
    ax.fill_between(
        xpos, CL68s_model_diff_up, CL95s_model_diff_up,
        color = "yellow", alpha = .5,
    )
    ax.fill_between(
        xpos, CL68s_model_diff_do, CL68s_model_diff_up,
        color = "green", alpha = .5, label = "$\pm1\sigma$",
    )
    ax.errorbar(
        xpos, medians_model_diff, xerr = xerr, #yerr = sigmas,
        marker='.', markersize=4, linewidth=0, elinewidth=1,
        fmt=' ', capsize = 3, capthick = 0, color = "black", label = labels["median"][language],
    )
    ax.errorbar(
        xpos, averages_diff, xerr = xerr,
        marker='+', markersize=4, linewidth=0, elinewidth=1,
        fmt=' ', capsize = 3, capthick = 0, color = "C4", label = labels["average"][language],
    )
    
    plt.plot([min_mass, max_mass], [0,0], color='C3')    

    plt.ylim(-500,500)
    plt.xlim(min_mass, max_mass)

    ax.legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig("model_response_diff-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))

    plt.xlim(min_mass, 200)
    plt.ylim(-100, 100)
    plt.xticks(np.arange(min_mass, 201, step=10))
    fig.savefig("model_response_diff_lowmass-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))

    plt.clf()
    fig, ax = plt.subplots()

    plt.xlabel(labels["GenHiggsMassGeV"][language])
    plt.ylabel(labels["Calibr_response"][language])

    CL68s_model_up = [CL68s_model_up[k]/medians_model[k] for k in range(len(medians_model))]
    CL68s_model_do = [CL68s_model_do[k]/medians_model[k] for k in range(len(medians_model))]
    CL95s_model_up = [CL95s_model_up[k]/medians_model[k] for k in range(len(medians_model))]
    CL95s_model_do = [CL95s_model_do[k]/medians_model[k] for k in range(len(medians_model))]
    
    ax.fill_between(
        xpos, CL95s_model_do, CL68s_model_do,
        color = "yellow", alpha = .5, label = "$\pm2\sigma$",
    )
    ax.fill_between(
        xpos, CL68s_model_up, CL95s_model_up,
        color = "yellow", alpha = .5,
    )
    ax.fill_between(
        xpos, CL68s_model_do, CL68s_model_up,
        color = "green", alpha = .5, label = "$\pm1\sigma$",
    )

    plt.plot([min_mass, max_mass], [1,1], color='C3')    

    plt.ylim(0,3)
    plt.xlim(min_mass, max_mass)

    ax.legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig("model_response_calibrated-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))

    plt.close('all')

def predicted_vs_answer_histo(df, channel, model_name, min_mass, max_mass, language = default_language, prefix = '', file_format = 'png', **kwargs):
    df1 = filter_channel(df, channel)

    min_mass, max_mass = 0, 1000

    bins_x = [k for k in range(min_mass, max_mass, 10)]
    bins_y = [k for k in range(min_mass, max_mass, 10)]
    vmax = 0.25

    if language == 'fr':
        locale.setlocale(locale.LC_NUMERIC, 'fr_FR.UTF-8')
    
    fig, ax = plt.subplots()
    ax.hist2d(
        df1[target],
        df1["predictions"],
        bins = [bins_x, bins_y],
        weights = df1["sample_weight"],
        density = True,
        cmap = "ocean_r",
        vmax = vmax/(len(bins_x)*len(bins_y)),
    )

    plt.xlabel(labels["GenHiggsMassGeV"][language])
    plt.ylabel(labels["ModelPredGeV"][language])
        
    plt.plot([min_mass, max_mass], [min_mass, max_mass], color='C3')    

    plt.ylim(min_mass, max_mass)
    plt.xlim(min_mass, max_mass)

    fig.tight_layout()
    fig.savefig("predicted_vs_answer_histo-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))

    plt.ylim(min_mass, 200)
    plt.xlim(min_mass, 200)

    fig.tight_layout()
    fig.savefig("predicted_vs_answer_histo_lowmass-{}{}{}.{}".format(prefix,model_name, "-en" if language=='en' else "", file_format))
            
def predicted_vs_answers(df, channel, model_name, min_mass, max_mass, language = default_language, prefix = '', cmap="ocean_r", file_format = 'png', **kwargs):

    df = filter_channel(df, channel=channel)

    df = df.copy()
    df["use"] = np.ones(len(df))
    step = 2
    cut = 500
    for mH in range(int(min_mass), int(max_mass)+1, step):
        population = len(df.loc[(df[target] >= mH - step/2) & (df[target] <= mH + step/2)])
        if population > cut:
            to_keep = np.concatenate([np.ones(cut), np.zeros(population-cut)])
            np.random.shuffle(to_keep)
            df.loc[(df[target] >= mH - step/2) & (df[target] <= mH + step/2), ["use"]] = to_keep

    df = df.loc[df["use"]==1]

    predictions = df["predictions"]
    answers = df[target]
    
    # Plot predicted vs answer on a test sample
    if language == 'fr':
        locale.setlocale(locale.LC_NUMERIC, 'fr_FR.UTF-8')
    plt.clf()
    fig, ax = plt.subplots()

    import seaborn as sns
    try:
        sns.kdeplot(answers, predictions, cmap=cmap, n_levels=30, shade=True, bw=.15)
    except:
        print("Singular matrix?")

    ax.plot(answers, answers, color="C3")
    plt.xlabel(labels["GenHiggsMassGeV"][language])
    plt.ylabel(labels["ModelPredGeV"][language])
    
    #plt.show()
    plt.xlim(min_mass, max_mass)
    plt.ylim(min_mass, max_mass)

    fig.tight_layout()
    fig.savefig("predicted_vs_answers-{}{}{}.{}".format(prefix, model_name, "-en" if language=='en' else "", file_format))

    # Plot predicted vs answer on a test sample
    plt.clf()
    fig, ax = plt.subplots()

    try:
        sns.kdeplot(answers, predictions/answers, cmap=cmap, n_levels=30, shade=True, bw=.15)
    except:
        print("Singular matrix?")

    ax.plot([min_mass, max_mass], [1,1], color='C3')
    plt.ylabel(labels["Pred_on_HiggsGenMass"][language])
        
    #plt.show()
    plt.xlim(min_mass, max_mass)
    plt.ylim(0, 3)

    fig.tight_layout()
    fig.savefig("predicted_on_answers-{}{}{}.{}".format(prefix, model_name, "-en" if language=='en' else "", file_format))

def predictions_distributions(df_all, channel, model_name, language = default_language, prefix = '', file_format = 'png', **kwargs):
    mass_points_GeV = [90, 125, 300, 500, 700, 800]
    width_GeV = 1.0
    for data_category in ["is_train", "is_valid", "is_test"]:
        for mass_point_GeV in mass_points_GeV:
            df = filter_channel(df_all, channel=channel)
            df = df.loc[df[data_category] == 1]
            df = df.loc[abs(df[target]-mass_point_GeV) <= width_GeV]
            _variable_distribution(df, "predictions", channel, data_category, model_name = model_name, language = language, prefix="mH_{}GeV".format(mass_point_GeV)+'-', weighted = False, density = True)
    
def variables_distributions(df_all, channel, model_name, language = default_language, prefix = '', variables_list = [target], file_format = 'png', **kwargs):
    df1 = filter_channel(df_all, channel=channel)
    for var in variables_list:
        _variables_distribution(df1, var, channel, "all_events", model_name = model_name, language = language)
        for data_category in ["is_train", "is_valid", "is_test"]:
            df2 = df1.loc[df_all[data_category] == 1]
            _variables_distribution(df2, var, channel, data_category, model_name = model_name, language = language, prefix = '')

def _variables_distribution(df, var, channel, data_category, model_name = None, language = default_language, prefix = ''):
    _variable_distribution(df, var, channel, data_category, model_name = model_name, language = language, prefix = prefix, weighted = False)
    _variable_distribution(df, var, channel, data_category, model_name = model_name, language = language, prefix = prefix, weighted = True)

def _variable_distribution(df, var, channel, data_category, model_name = None, language = default_language, prefix = '', weighted = False, density = False):
    plt.clf()
    fig, ax = plt.subplots()
    weights = None
    weights_in_output_name = 'raw'
    if weighted:
        weights = df["sample_weight"]
        weights_in_output_name = 'weighted'
    binning_default = 50
    bin_width = 2
    min_value = df[var].min()
    max_value = df[var].max()
    binning = np.arange(min_value, max_value, 10)
    if len(binning) < 50:
        binning = binning_default
    n, bins, patches = ax.hist(df[var], binning, weights = weights, log = (var in vars_with_y_log_scale), density = density)
    if var in labels:
        xlabel = labels[var][language]
    else:
        xlabel = var
    ax.set_xlabel(xlabel)
    ax.set_ylabel(labels["Nevents"][language])
    if density:
        ax.set_ylabel(labels["Probability"][language])
    if var in [target, 'predictions']:
        plt.xlim(0, 1000)
    fig.tight_layout()
    if var == "predictions":
        plt.savefig('distribution-{}-{}-{}-{}{}.{}'.format(channel, "-".join([var, "{}{}".format(prefix,model_name)]), weights_in_output_name, data_category, "-en" if language=='en' else "", file_format))
    else:
        plt.savefig('distribution-{}-{}-{}-{}{}.{}'.format(channel, var, weights_in_output_name, data_category, "-en" if language=='en' else "", file_format))

def feature_importance(model, inputs, model_name, prefix = '', file_format = 'png', **kwargs):
    plt.clf()
    fig, ax = plt.subplots()
    plot_importance(
        model,
        title = None,
        xlabel = labels["Score"][language],
        ylabel = labels["Variable"][language],
        grid = False,
    )
    plt.subplots_adjust(left=0.25)
    fig.tight_layout()
    plt.savefig('feature_importance-{}{}.{}'.format(prefix, model_name, file_format))

available_plots = {
    'model_response' : model_response,
    'predicted_vs_answers' : predicted_vs_answers,
    'feature_importance' : feature_importance,
    'variables_distributions': variables_distributions,
    'predictions_distributions' : predictions_distributions,
    'gen_vs_reco' : gen_vs_reco,
    'model_response_tau_filtered' : model_response_tau_filtered,
    'predicted_vs_answer_histo' : predicted_vs_answer_histo,
}
