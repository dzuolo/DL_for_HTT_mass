#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from optparse import OptionParser
usage = "usage: %prog [options] <input root file> <output file name>"
parser = OptionParser(usage=usage)
parser.add_option("-v", "--verbose", dest = "verbose",
                                    default=False,
                                    action='store_true')
parser.add_option("-N", "--Nmax", dest = "Nmax",
                                    default=None)

(options,args) = parser.parse_args()

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

def find_HTT(evt):
    ''' Find the Higgs boson in the generated particles,
    check that it decays in tau leptons,
    return the Higgs and the two taus.'''
    Higgs_IDs = [25, 35, 36, 37]
    for ptc in evt.Particle:
        if abs(ptc.PID) in Higgs_IDs:
            if ptc.D1 != -1:
                if abs(evt.Particle.At(ptc.D1).PID) == 15:
                    if ptc.D2 != -1:
                        if evt.Particle.At(ptc.D2).PID == - evt.Particle.At(ptc.D1).PID:
                            return ptc, evt.Particle.At(ptc.D1), evt.Particle.At(ptc.D2)

def DR2(ptc1, ptc2):
    return (ptc1.Eta - ptc2.Eta)**2 + (ptc1.Phi - ptc2.Phi)**2
                        
def match(ptc1, ptc2, dR = .5):
    ''' Check that two particles instance can be linked to the same physic object.'''
    if ptc1.PID != ptc2.PID:
        return False
    # if abs((ptc1.PT - ptc2.PT)/ptc1.PT) > .1:
    #     return False
    if DR2(ptc1, ptc2)**.5 > dR:
        return False
    return True

def get_daughters(evt, ptc):
    ''' Sometime a gen ptc decays into a particle and itself (software feature),
    so this function tries to get the real decays.'''
    daughters = []
    if ptc.D1 != -1:
        D1 = evt.Particle.At(ptc.D1)
        if D1.PID == ptc.PID:
            daughters += get_daughters(evt, D1)
        else:
            daughters.append(D1)
    if ptc.D2 != -1:
        D2 = evt.Particle.At(ptc.D2)
        if D2.PID == ptc.PID:
            daughters += get_daughters(evt, D2)
        else:
            daughters.append(D2)
    return list(set(daughters))

def check_decays_from(decays, ptc):
    final_decays = []
    correct_decays = set()
    bad_decays = set()
    for decay in decays:
        chain = []
        matched = False
        index = decay.M1
        while index != -1 and not matched:
            previous_decay = evt.Particle.At(index)
            if previous_decay in bad_decays:
                break
            chain.append(previous_decay)
            index = previous_decay.M1
            if previous_decay == ptc or previous_decay in correct_decays:
                matched = True
        if matched:
            final_decays.append(decay)
            correct_decays.update(chain)
        else:
            bad_decays.update(chain)
    return final_decays
    
def find_tau_decays(evt, tau):
    decays = []
    channel = "t"
    DM = None
    decays = check_decays_from(evt.Particle, tau)
    decays = [p for p in set(decays) if p.PID != tau.PID]
    if any(abs(ptc.PID) == 11 for ptc in decays):
        channel = "e"
    elif any(abs(ptc.PID) == 13 for ptc in decays):
        channel = "m"
    if channel == "t":
        neutrinos = [neutrino for neutrino in decays if abs(neutrino.PID) in [12, 14, 16]]
        if len(neutrinos) > 1:
            string = "Hadronic tau with more than 1 neutrino at generator level??"
            #print(string)
            import pdb; pdb.set_trace()
            #raise RuntimeError(string)
        pi0s = [pi0 for pi0 in decays if abs(pi0.PID) in [111]]
        photons = [p for p in decays if abs(p.PID) == 22]
        prongs = [p for p in decays if p.Charge != 0]
        DM = "{}prong{}pi0".format(len(prongs), len(pi0s))
        if not len(prongs) % 2:
            print(DM)
            import pdb; pdb.set_trace()
    return decays, channel, DM
                        
Nevt = 0
channel_stats = {
    "tt":0,
    "mt":0,
    "et":0,
    "mm":0,
    "ee":0,
    "em":0,
}

for evt in tree:
    Nevt += 1
    print("\nEvent {}:".format(Nevt))

    Higgs, tau1, tau2 = find_HTT(evt)

    #print("\tHiggs energy is {} GeV = 2 * {} GeV.".format(Higgs.E, Higgs.E/2))
    
    decays1, channel1, DM1 = find_tau_decays(evt, tau1)
    decays2, channel2, DM2 = find_tau_decays(evt, tau2)

    channel = channel1+channel2
    if channel == "me":
        channel = "em"
    elif channel == "te":
        channel = "et"
    elif channel == "tm":
        channel = "mt"

    channel_stats[channel] += 1
        
    print("\tChannel: {}".format(channel))

    if any(DM != None for DM in [DM1, DM2]):
        DMstr = "\ttauh decay modes:"
        for DM in [DM1, DM2]:
            if DM is not None:
                DMstr += " {}".format(DM)
        print(DMstr)

    print("")
    print("\tleg1:")
    print("\ttau pT: {}, eta: {}, phi: {}, E: {}".format(tau1.PT, tau1.Eta, tau1.Phi, tau1.E))

    print("")
    print("\tleg2:")
    print("\ttau pT: {}, eta: {}, phi: {}, E: {}".format(tau2.PT, tau2.Eta, tau2.Phi, tau1.E))
                
        
    print("")
    if Nevt >= Nmax:
        break

print("Processed on {Nevt} events.".format(Nevt=Nevt))
for channel in channel_stats:
    print("\t{} proportion: {} +/- {} pct".format(channel, 100*channel_stats[channel]/Nevt, 100*channel_stats[channel]**.5/Nevt))
