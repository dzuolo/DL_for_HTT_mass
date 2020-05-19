#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from optparse import OptionParser
usage = "usage: %prog [options] <input root file> <output file name>"
parser = OptionParser(usage=usage)
parser.add_option("-v", "--verbose", dest = "verbose",
                                    default=0)
parser.add_option("-N", "--Nmax", dest = "Nmax",
                                    default=None)

(options,args) = parser.parse_args()

options.verbose = int(options.verbose)

if len(args) == 2:
    input_file_str = args[0]
    output_file_str = args[1]
else:
    raise RuntimeError("Please give one input file and one output file")

import ROOT
import libPyROOT as _root
ROOT.gSystem.Load("libDelphes")

input_file = ROOT.TFile(input_file_str, "READ")
tree = input_file.Get("Delphes")

if options.Nmax is None:
    Nmax = tree.GetEntries()
else:
    Nmax = min([int(options.Nmax), tree.GetEntries()])
                        
def match(ptc1, ptc2, dR = .5):
    ''' Check that two particles instance can be linked to the same physic object.'''
    if ptc1.PID != ptc2.PID:
        return False
    # if abs((ptc1.PT - ptc2.PT)/ptc1.PT) > .1:
    #     return False
    if DR2(ptc1, ptc2)**.5 > dR:
        return False
    return True

Nevt_in = 0
Nevt_out_gen= 0
Nevt_out_reco= 0
Nevt_out_gen_and_reco= 0

channel_stats_gen = {
    "tt":0,
    "mt":0,
    "et":0,
    "mm":0,
    "ee":0,
    "em":0,
}

channel_stats_reco = {}
channel_identification = {}

for c1 in channel_stats_gen.keys():
    channel_stats_reco[c1] = 0
    for c2 in channel_stats_gen.keys():
        channel_identification["{} as {}".format(c1, c2)] = 0

from Delphes_to_NN.analysis.HTT_gen import HTT_analysis as HTT_analysis_gen
from Delphes_to_NN.analysis.HTT_reco import HTT_analysis as HTT_analysis_reco

analysis_results = []

for evt in tree:
    Nevt_in += 1
    passed_gen = False
    passed_reco = False
    
    if options.verbose > 0 and Nevt_in%10 == 0:
        print("\nEvent {}:".format(Nevt_in))

    gen_analysis = HTT_analysis_gen(evt, verbose = options.verbose)
    if gen_analysis != {}:
        passed_gen = True
        Nevt_out_gen += 1

    reco_analysis = {}
    if passed_gen:
        reco_analysis = HTT_analysis_reco(evt, verbose = options.verbose)

    if reco_analysis != {}:
        passed_reco = True
        Nevt_out_reco += 1


    if passed_gen and passed_reco:
        analysis_results.append({"gen" : gen_analysis, "reco" : reco_analysis})
        
    if options.verbose > 0:
        print("")
    if Nevt_in >= Nmax:
        break

print("Processed on {Nevt_in} events.".format(Nevt_in=Nevt_in))

data = {}
for analysis_result in analysis_results:
    for ana in analysis_result:
        for k in analysis_result[ana]:
            key = "_".join([k, ana])
            if key not in data:
                data[key] = []
            data[key].append(analysis_result[ana][k])

file = open('{}.txt'.format(output_file_str), 'w')
cols = list(data.keys())
cols.sort()
cols = ["Event"] + cols
s = " ".join(cols)
file.write(s+"\n")
for k in range(len(data[cols[1]])):
    s= " ".join([str(k+1)]+[str(data[c][k]) for c in cols[1:]])
    file.write(s+"\n")
file.close()
