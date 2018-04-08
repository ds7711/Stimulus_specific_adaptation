# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:39:52 2015

SSA network model
@author: MIngwen Dong

"""

#%% function used to visualize synaptic connection
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

#%% Define a function that converts spectrogram into Poisson input (firing rate)
# 1ST: convert spectrogram into a spiketrain matrix whose temporal resolution is 0.1 ms
# 2ND: adding some random noise and then have a logical comparison so that firing rate has some variance
# 3RD: if necessary, purturb the timing of the spike within a certain window
# 4TH: convert the spike train matrix into dummy current so that the TimedInput array could be used
# 


#%% import brian2 & other required modules
from brian2 import *
start_scope() # clear magic functions from previous session
defaultclock.dt = 0.1 * ms # set default numeric resolution

##%% Create TimedArray input
Num_Poisson = 100
refractory_period = 0.0 * ms
III = 1 * nA
# Create timed input
Stim_pulse_dur = 10 
Stim_baseline_dur = 990
noise_lvl = 1.0

Real_input = III / nA * ones((1, Stim_pulse_dur), dtype = float64)
Dummy_input = zeros((1, Stim_baseline_dur), dtype = float64)
Timed_input_unit = concatenate((Dummy_input, Real_input), axis = 1)
Timed_III = tile(squeeze(Timed_input_unit), (Num_Poisson, 7))
Timed_III = squeeze(Timed_III)

#%% GroupA_Neuron: input
# Poisson input neuron group
#Num_Poisson = 2
Temp_FR = 20 * Hz
# implement PoissonGroup with NeuronGroup (Dummy neurons)
# dt = defaultclock.dt = 0.1 * ms
tau_AAA = 10.0 ** (0)  # for unit purpose
TH_AAA = 0.8 * mV
resting_AAA = 0 * mV
refractory_AAA = 0.02 * ms
Timed_III = Timed_III * nA
#III = TimedArray(Timed_III.T, defaultclock.dt)
temp = float64(rand(7000, Num_Poisson) > 0.99)
III_AAA = TimedArray(temp * nA, dt = defaultclock.dt)
CCC = 281.0 / 2 * pF # membrance capacitance
#I = 1
eqs_AAA = '''
# dv/dt = III_AAA(t, i) / ms / tau_AAA: 1 #(unless refractory)
dv/dt = (1.0 / CCC) * (III_AAA(t, i)): volt
'''
# Neuron_AAA = NeuronGroup(Num_Poisson, 'rates: Hz', threshold = 'rand() < Temp_FR * dt')
Neuron_AAA = NeuronGroup(Num_Poisson, 
                         eqs_AAA, 
                         threshold = 'v > TH_AAA',
                         reset = 'v = resting_AAA', 
                         refractory = refractory_AAA)
Neuron_AAA.v = resting_AAA
PoissonInput_monitor = SpikeMonitor(Neuron_AAA, record = 1)
Monitor_AAA = StateMonitor(Neuron_AAA, 'v', record = 1)


#%% Define AdEx neuron population B
#start_scope()
# Time constant 
gL = 30.0 * nS # leak conductance
EL = -70.6 * mV # resting potential 
Delta_T = 2.0 * mV # slope factor 
V_T = -50.4 * mV # threshold 
Reset_Thresh = V_T + 5 * Delta_T # Reset to resting potential when the potential reaches 20 mV
tau_w = 144.0 * ms # adaptation time constant
aaa = 4.0 * nS # subthreshold adaptation
bbb = 0.0805 * nA # spike-triggered adaptation
CCC = 281.0 * pF # membrance capacitance
#www = 2 * nA; # adaptation current
# III_BBB = 0 * nA
refractory_period = 0.0 * ms
num_inGroup_BBB = 10

# Create timed input
#Stim_pulse_dur = 10 
#Stim_baseline_dur = 990
#
#Real_input = III_BBB / nA * ones((1, Stim_pulse_dur), dtype = float64)
#Dummy_input = zeros((1, Stim_baseline_dur), dtype = float64)
#Timed_input_unit = concatenate((Dummy_input, Real_input), axis = 1)
#Timed_III = tile(squeeze(Timed_input_unit.T), 7) * nA
#III_BBB = TimedArray(Timed_III, defaultclock.dt)

eqs = ''' 
dv/dt = (1.0 / CCC) * (III_BBB + (gL * Delta_T * exp((v - V_T) / Delta_T) - gL * (v - EL)) - www) : volt (unless refractory) 
dwww/dt = (aaa*(v - EL) - www)/tau_w : amp
''' 
Group_BBB = NeuronGroup(num_inGroup_BBB, eqs, 
                        threshold = 'v > Reset_Thresh', 
                        reset='v = EL; www += bbb', 
                        method = 'exponential_euler',
                        refractory = refractory_period) 
M_BBB = StateMonitor(Group_BBB, 'v', record = 1) 
spikemon_BBB = SpikeMonitor(Group_BBB, record = 1)

Group_BBB.v[0]= -60 * mV # EL set initial membrane potential to resting potential


#%% Connect the neurons

# Parameters for A2B synapses
tau_ir = 0.0 * ms # tau_ir = 800.0 * ms
tau_re = 0.9 * ms
tau_ei = 5.3 * ms
g_syn_A2B = 0.001 * nS # ????parameters to be determined????
E_syn_A2B = 0 * mV # reversal potential of the synapse
Vm_post = -70.0 * mV # Post-synaptic potential ????parameters to be determined????

#xeee = 0.8
#xiii = 0.1
#xrrr = 0.1
M = 0.0
M_step = 1.0
# dummy equation, M will be a step function
eqs_A2B = '''
# I_syn_A2B = g_syn_A2B * xeee * (E_syn_A2B - Vm_post): amp # Vm_post: post-synaptic potential
dxrrr/dt = -1.0 * M * xrrr / tau_re + xiii / tau_ir: 1
dxeee/dt = M * xrrr / tau_re - xeee / tau_ei: 1
dxiii/dt = xeee / tau_ei - xiii / tau_ir: 1
'''

Synapses_A2B = Synapses(Neuron_AAA, Group_BBB, eqs_A2B, 
                        on_pre = '''
                        xrrr -= xrrr / (tau_re * dt) * ms * ms
                        xeee += xrrr / (tau_re * dt) * ms * ms
                        III_BBB_post = g_syn_A2B * xeee * (E_syn_A2B - Vm_post)''') # , pre = 'M += 1'
Synapses_A2B.connect('i != j', p = 0.5)
A2B_Monitor = StateMonitor(Synapses_A2B, ['www', 'xeee', 'xrrr', 'xiii'], record = True)

#visualise_connectivity(Synapses_A2B)


#%% Run simulation and check results

# Plot for input neuron
run(700 * ms)
figure
plot(A2B_Monitor.t / ms, A2B_Monitor.w[0], label = 'w')
plot(A2B_Monitor.t / ms, A2B_Monitor.xeee[0], label = 'xe')

figure
plot(Monitor_AAA.t/ms, Monitor_AAA.v[0] / mV) 
title('Dummy current visualization')
xlabel('Time (ms)') 
xlim([0, 800])
ylabel('v')
show()

# Plot spikes
plot(PoissonInput_monitor.t/ms, PoissonInput_monitor.i, '.k')
title('Poisson input visualization')
xlabel('Time (ms)')
ylabel('Neuron index')
xlim([0, 800])
ylim([-1, Num_Poisson])
show()

# Plot for neuron in Group_BBB
figure
plot(M_BBB.t/ms, M_BBB.v[0] / mV) 
xlabel('Time (ms)') 
xlim([0, 800])
#ylim([-1, num_inGroup_BBB])
ylabel('v')
show()

# Plot spikes
plot(spikemon_BBB.t/ms, spikemon_BBB.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
xlim([0, 800])
ylim([-1, num_inGroup_BBB])
show()
show()

