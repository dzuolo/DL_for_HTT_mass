import DL_for_HTT.post_training.utils as utils

from DL_for_HTT.common.NN_settings import target

import locale; locale.setlocale(locale.LC_NUMERIC, 'fr_FR.UTF-8')
import matplotlib.pyplot as plt
plt.rcdefaults()

import numpy as np

from xgboost import plot_importance

plt.rcParams["figure.figsize"] = [7, 7]
plt.rcParams['axes.formatter.use_locale'] = True

def filter_channel(df, channel = None):
    df1 = df
    if channel in set(df['channel_reco']):
        df1 = df.loc[(df['channel_reco'] == channel)]
    elif channel == "lt":
        df1 = df.loc[(df['channel_reco'] == "mt") | (df['channel_reco'] == "et")]
    elif channel == "ll":
        df1 = df.loc[(df['channel_reco'] == "mm") | (df['channel_reco'] == "em") | (df['channel_reco'] == "ee")]
    return df1    

def model_response(df, channel, model_name, min_mass, max_mass, prefix = '', **kwargs):
    df1 = filter_channel(df, channel)
        
    medians_model = []
    CL68s_model_up = []
    CL68s_model_do = []
    CL95s_model_up = []
    CL95s_model_do = []
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
        # values_mTtot = mTtots/mHs
        
        values_model = [v for v in values_model]
        # values_mTtot = [v for v in values_mTtot]
        values_model.sort()
        # values_mTtot.sort()

        try:
            medians_model.append(values_model[int(len(values_model)/2)])
            # medians_mTtot.append(values_mTtot[int(len(values_mTtot)/2)])
        except:
            import pdb; pdb.set_trace()

        above_model = [v for v in values_model if v >= medians_model[-1]]
        below_model = [v for v in values_model if v <= medians_model[-1]]
        # above_mTtot = [v for v in values_mTtot if v >= medians_mTtot[-1]]
        # below_mTtot = [v for v in values_mTtot if v <= medians_mTtot[-1]]

        above_model.sort()
        below_model.sort(reverse = True)
        # above_mTtot.sort()
        # below_mTtot.sort(reverse = True)

        CL68s_model_up.append(above_model[int(0.68 * len(above_model))])
        CL68s_model_do.append(below_model[int(0.68 * len(below_model))])
        CL95s_model_up.append(above_model[int(0.95 * len(above_model))])
        CL95s_model_do.append(below_model[int(0.95 * len(below_model))])
        # CL68s_mTtot_up.append(above_mTtot[int(0.68 * len(above_mTtot))])
        # CL68s_mTtot_do.append(below_mTtot[int(0.68 * len(below_mTtot))])
        # CL95s_mTtot_up.append(above_mTtot[int(0.95 * len(above_mTtot))])
        # CL95s_mTtot_do.append(below_mTtot[int(0.95 * len(below_mTtot))])
        
    fig, ax = plt.subplots()
    #fig.suptitle(model_name)
    plt.xlabel("Masse générée du Higgs (GeV)")
    plt.ylabel("Prédicition du modèle / Masse générée du Higgs (1/GeV)")
    
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
        fmt=' ', capsize = 3, capthick = 0, color = "black", label = "Médiane",
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
    fig.savefig("model_response_{}.png".format(model_name))
    plt.close('all')

def mean_sigma_mae(df, channel, Nneurons_list, Nlayers_list, bottleneck_list, min_mass, max_mass):
    for bottleneck in bottleneck_list:
        for Nneurons in Nneurons_list:
            mean_sigma_mae_fct_Nlayers(df, channel, Nneurons, Nlayers_list, bottleneck, min_mass, max_mass)
        for Nlayers in Nlayers_list:
            mean_sigma_mae_fct_Nneurons(df, channel, Nneurons_list, Nlayers, bottleneck, min_mass, max_mass)

def mean_sigma_mae_fct_Nlayers(df, channel, Nneurons, Nlayers_list, bottleneck, min_mass, max_mass):
    mean_sigma_mae_fct(df, channel, Nlayers_list, bottleneck, min_mass, max_mass, fixed = "{} neurons per layer".format(str(Nneurons)), at = Nneurons, type = "n")

def mean_sigma_mae_fct_Nneurons(df, channel, Nneurons_list, Nlayers, bottleneck, min_mass, max_mass):
    mean_sigma_mae_fct(df, channel, Nneurons_list, bottleneck, min_mass, max_mass, fixed = "{} hidden layers".format(str(Nlayers)), at = Nlayers, type = "l")

def mean_sigma_mae_fct(df, channel, list, bottleneck, min_mass, max_mass, fixed = "?", at = 0, type = "?"):
    df1 = filter_channel(df, channel)
        
    means = []
    sigmas = []
    maes = []
    xpos = []

    for val in list:

        if bottleneck == "_bottleneck":
            if (type == "l" and val == 2000 and at == 2) or (type == "n" and val == 2 and at == 2000):
                continue
            
        if type == "n":
            var = "{}_{}_layers_{}_neurons{}_output".format(channel, str(val), str(at), bottleneck)
        elif type == "l":
            var = "{}_{}_layers_{}_neurons{}_output".format(channel, str(at), str(val), bottleneck)
            
        predictions = np.r_[df1[var]]
        mHs = np.r_[df1[target]]
        values = predictions/mHs
        
        fig, ax = plt.subplots()

        hist = ax.hist(values, bins=200, range = [0,2], label = 'Deep NN output', alpha=0.5, color = 'C0')
        x, popt = utils.make_gaussian_fit(hist)
        
        means.append(popt[1])
        sigmas.append(popt[2])
        maes.append(abs(predictions - mHs).sum()/len(predictions))
        xpos.append(val)

    if type == "n":
        uxerr = .5
    elif type == "l":
        uxerr = 100
    xerr = [uxerr for x in xpos]

    fig, ax = plt.subplots()
    fig.suptitle("{} performances with {}{}".format(channel, fixed, " and bottleneck" if bottleneck != "" else ""))
    if type == "n":
        plt.xlabel("Number of hidden layers")
    elif type == "l":
        plt.xlabel("Number of neurons per hidden layer")
    
    
    ax.errorbar(
        xpos, means, xerr = xerr, yerr = sigmas,
        marker='+', markersize=4, linewidth=.4, elinewidth=1,
        fmt=' ', capsize = 6, capthick = 1,
        color = "C0", label = "$\\sigma$")
    ax.errorbar(
        xpos, means, xerr = xerr, yerr = maes,
        marker='+', markersize=4, linewidth=.4, elinewidth=1,
        fmt=' ', capsize = 6, capthick = 1,
        color = "C3", label = "MAE")
    ax.legend(loc='upper right')

    xmin = int(xpos[0]-xerr[0])
    xmax = int(xpos[-1]+xerr[-1])+1
    
    plt.plot([xmin, xmax], [1,1], color='C3')

    plt.ylim(0.75,1.25)
    plt.xlim(xmin, xmax)
    
    fig.tight_layout()
    if type == "n":
        fig.savefig("NN_mean_{}_at_fixed_{}_Nneurons{}.png".format(channel, str(at), bottleneck))
    elif type == "l":
        fig.savefig("NN_mean_{}_at_fixed_{}_Nlayers{}.png".format(channel, str(at), bottleneck))    
    plt.close('all')

def predicted_vs_answers(df, channel, model_name, min_mass, max_mass, prefix = '', **kwargs):

    df = filter_channel(df, channel=channel)

    predictions = df["predictions"]
    answers = df[target]
    
    # Plot predicted vs answer on a test sample
    plt.clf()
    fig, ax = plt.subplots()

    import seaborn as sns
    sns.kdeplot(answers, predictions, cmap="viridis", n_levels=30, shade=True, bw=.15)

    ax.plot(answers, answers, color="C3")
    plt.xlabel("Masse générée du Higgs (GeV)")
    plt.ylabel("Prédicition du modèle")
    
    #plt.show()
    plt.xlim(min_mass, max_mass)
    plt.ylim(min_mass, max_mass)

    fig.tight_layout()
    fig.savefig("predicted_vs_answers-{}{}.png".format(prefix, model_name))

    # Plot predicted vs answer on a test sample
    plt.clf()
    fig, ax = plt.subplots()

    import seaborn as sns
    sns.kdeplot(answers, predictions/answers, cmap="viridis", n_levels=30, shade=True, bw=.15)

    ax.plot([min_mass, max_mass], [1,1], color='C3')
    plt.xlabel("Masse générée du Higgs (GeV)")
    plt.ylabel("Prédicition du modèle / Masse générée du Higgs (1/GeV)")
    
    #plt.show()
    plt.xlim(min_mass, max_mass)
    plt.ylim(0, 3)

    fig.tight_layout()
    fig.savefig("predicted_on_answers-{}{}.png".format(prefix, model_name))
    
def variables_distributions(df_all, channel, model_name, prefix = '', variables_list = [target], **kwargs):
    df1 = filter_channel(df_all, channel=channel)
    for var in variables_list:
        _variables_distribution(df1, var, channel, "all_events")
        for data_category in ["is_train", "is_valid", "is_test"]:
            df2 = df1.loc[df_all[data_category] == 1]
            _variables_distribution(df2, var, channel, data_category, prefix = '')

var_name_to_label = {
    'Higgs_mass_gen' : "Masse générée du Higgs (GeV)",
}

vars_with_y_log_scale = [
    'tau1_pt_reco',
    'tau2_pt_reco',
]

def _variables_distribution(df, var, channel, data_category, prefix = ''):
    plt.clf()
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(df[var], 50, log = (var in vars_with_y_log_scale))
    if var in var_name_to_label:
        xlabel = var_name_to_label[var]
    else:
        xlabel = var
    ax.set_xlabel(xlabel)
    ax.set_ylabel('N events')
    if var == target:
        plt.xlim(0, 1000)
    fig.tight_layout()
    plt.savefig('distribution-{}-{}-{}.png'.format(channel, var, data_category))

def feature_importance(model, inputs, model_name, prefix = '', **kwargs):
    plt.clf()
    fig, ax = plt.subplots()
    plot_importance(
        model,
        title = None,
        xlabel = 'Score',
        ylabel = 'Variable',
        grid = False,
    )
    plt.subplots_adjust(left=0.25)
    fig.tight_layout()
    plt.savefig('feature_importance-{}{}.png'.format(prefix, model_name))

available_plots = {
    'model_response' : model_response,
    'predicted_vs_answers' : predicted_vs_answers,
    'feature_importance' : feature_importance,
    'variables_distributions': variables_distributions,
}
