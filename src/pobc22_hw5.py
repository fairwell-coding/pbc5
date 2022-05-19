#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, PoissonInput, SpikeGeneratorGroup, SpikeMonitor, StateMonitor, Synapses
from brian2 import mV, pA, pF, ms, second, Hz, Gohm
import brian2.numpy_ as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from os.path import join
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def poisson_generator(rate, t_lim, unit_ms=False):
    """
    Draw events from a Poisson point process.

    Note: the implementation assumes at t=t_lim[0], although this spike is not
    included in the spike list.

    :param rate: the rate of the discharge in Hz
    :param t_lim: tuple containing start and end time of the spike
    :param unit_ms: use ms as unit for times in t_lim and resulting events
    :returns: numpy array containing spike times in s (or ms, if unit_ms is set)
    """

    assert len(t_lim) == 2

    if unit_ms:
        t_lim = (t_lim[0] / 1000, t_lim[1] / 1000)

    if rate > 0.:
        events_ = [t_lim[0]]

        while events_[-1] < t_lim[1]:
            T = t_lim[1] - events_[-1]

            # expected number of events
            num_expected = T * rate

            # number of events to generate
            num_generate = np.ceil(num_expected + 3 * np.sqrt(num_expected))
            num_generate = int(max(num_generate, 1000))

            beta = 1. / rate
            isi_ = np.random.exponential(beta, size=num_generate)
            newevents_ = np.cumsum(isi_) + events_[-1]
            events_ = np.append(events_, newevents_)

        lastind = np.searchsorted(events_, t_lim[1])
        events_ = events_[1:lastind]  # drop ghost spike at start

        if unit_ms:
            events_ *= 1000.

    elif rate == 0.:
        events_ = np.asarray([])

    else:
        raise ValueError('requested negative rate.')

    return events_


def generate_stimulus(t_sim, stim_len=50, stim_dt=500, num_input=3, rate=200, dt=.1):
    """
    Generate input spikes.

    :param t_sim: total time for stimulus generation in ms
    :param stim_len: duration of each stimulus
    :param stim_dt: stimulus spacing
    :param num_input: number of input signals (i.e. number of input neurons)
    :param rate: firing rate of active neurons in Hz
    :param dt: simulation time step for rounding
    :returns: list contain a list of spike times for each input neuron
    """

    num_stim = int(np.floor(t_sim / stim_dt) - 1)
    bits = np.random.randint(2, size=(num_stim, num_input))

    t_stim_ = stim_dt * np.arange(1, num_stim + 1)
    assert len(t_stim_) == bits.shape[0]

    spikes = [np.array([]) for n in range(num_input)]

    for n in range(num_input):
        for t, bit in zip(t_stim_, bits[:,n]):
            if bit == 1:
                spikes[n] = np.append(spikes[n], poisson_generator(rate, t_lim=(t, t + stim_len), unit_ms=True))

    # round to dt, clip, sort out duplicates
    spikes = [np.round(sp / dt).astype(int) for sp in spikes]
    spikes = [np.clip(sp, 1, t_sim / dt) for sp in spikes]
    spikes = [np.unique(sp) * dt for sp in spikes]

    # remove possible duplicates

    # brian data format
    ids = np.concatenate([k * np.ones(len(sp), dtype=int) for k, sp in enumerate(spikes)])
    times = np.concatenate(spikes)
    assert len(times) == len(ids)

    return bits, spikes, ids, times


def check_stp(outdir=None):
    """
    Visualize the implementation of STP.

    :param outdir: output directory for plots
    """

    net = brian2.Network()

    # setup neuron

    tau_syn = 5 * ms
    w0 = 100 * pA

    # facilitation
    U1 = .1
    tau_fac1 = 100 * ms
    tau_rec1 = 5 * ms

    # depression
    U2 = .5
    tau_fac2 = 5 * ms
    tau_rec2 = 100 * ms

    eqs = '''
    dI/dt = -I / tau_syn : ampere
    '''

    neuron_post = NeuronGroup(2, eqs, method='exact')
    neuron_post.I = 0

    state_mon = StateMonitor(neuron_post, 'I', record=True)

    # setup inputs

    stim_times = np.concatenate((np.arange(10, 80, 5), [220]))
    stim_ids = np.zeros_like(stim_times, dtype=int)

    inputs = SpikeGeneratorGroup(1, stim_ids, stim_times * ms)

    input_spike_mon = SpikeMonitor(inputs)

    # setup synapses

    syn_eqs = '''
    w : ampere
    U : 1
    tau_fac : second
    tau_rec : second
    
    du/dt = - u / tau_fac : 1 (clock-driven)
    dz/dt = - z / tau_rec : 1 (clock-driven)
    '''

    on_pre = '''
    u += U * (1 - u) * 1
    w = w0 * (1 - z) * u
    z += (1 - z) * u * 1
    I += w
    '''

    synapses = Synapses(inputs, neuron_post, syn_eqs, on_pre=on_pre, method='exact')
    synapses.connect()

    synapses.w = w0

    synapses.U[0] = U1
    synapses.tau_fac[0] = tau_fac1
    synapses.tau_rec[0] = tau_rec1

    synapses.U[1] = U2
    synapses.tau_fac[1] = tau_fac2
    synapses.tau_rec[1] = tau_rec2

    synapse_state_mon = StateMonitor(synapses, ['u', 'z'], record=True)

    # run

    net.add([neuron_post, state_mon, inputs, input_spike_mon, synapses, synapse_state_mon])

    t_sim = 250 * ms

    net.run(t_sim)

    # plot

    fig = plt.figure(figsize=(6, 6))

    ax = plt.subplot(3, 2, 1)
    ax.set_title('facilitation')
    ax.plot(state_mon.t / ms, state_mon.I[0,:] / pA, c='C2', lw=2)
    ax.locator_params(nbins=2)
    ax.set_ylabel(r'$\mathrm{PSC}(t)$ / pA')

    ax = plt.subplot(3, 2, 2, sharex=plt.gca(), sharey=plt.gca())
    ax.set_title('depression')
    ax.plot(state_mon.t / ms, state_mon.I[1,:] / pA, c='C3', lw=2)
    ax.locator_params(nbins=2)
    plt.setp([ax.spines['top'], ax.spines['right']], visible=False)

    ax = plt.subplot(3, 2, 3)
    ax.plot(synapse_state_mon.t / ms, synapse_state_mon.u[0,:], c='C2', lw=2)
    ax.locator_params(nbins=2)
    ax.set_ylabel(r'$u(t)$')

    ax = plt.subplot(3, 2, 4, sharex=plt.gca(), sharey=plt.gca())
    ax.plot(synapse_state_mon.t / ms, synapse_state_mon.u[1,:], c='C3', lw=2)
    ax.locator_params(nbins=2)

    ax = plt.subplot(3, 2, 5)
    ax.plot(synapse_state_mon.t / ms, 1 - synapse_state_mon.z[0,:], c='C2', lw=2)
    ax.locator_params(nbins=2)
    ax.set_xlabel(r'$t$ / ms')
    ax.set_ylabel(r'$R(t)$')

    ax = plt.subplot(3, 2, 6, sharex=plt.gca(), sharey=plt.gca())
    ax.plot(synapse_state_mon.t / ms, 1 - synapse_state_mon.z[1,:], c='C3', lw=2)
    ax.locator_params(nbins=2)
    ax.set_xlabel(r'$t$ / ms')

    [plt.setp([ax.spines['top'], ax.spines['right']], visible=False) for ax in fig.axes]
    plt.tight_layout()

    if outdir:
        plt.savefig(join(outdir, 'check_stp.png'), dpi=200)


def get_neuron_liquid_states(spike_times, readout_times, tau):
    """
    :param spike_times: list containing spike times for one neuron
    :param readout_times: times at which to extract liquid states
    :param tau: exponential decay time constant
    :returns: array containing liquid state at each requested time
    """
    t_window = 3. * tau
    states = np.zeros_like(readout_times)

    for nt, t in list(enumerate(readout_times))[::-1]:
        # advance mask
        spike_times = spike_times[spike_times < t]

        # only use spikes within window
        spikes_cur_ = spike_times[spike_times >= t - t_window]
        states[nt] = np.exp(-(t - spikes_cur_) / tau).sum()

    return states


def train_readout(states, targets, train_size=.8, reg_fact=1):
    """
    Train and test a readout. Assumes labels are integers (0, ..., N_classes-1).

    :param states: liquid states
    :param targets: target values
    :param train_size: fraction of liquid states to use for training, default: 0.8
    :param reg_fact: inverse of regularization strength, default: 1
    :returns: tuple of training and test error
    """

    np.random.seed()

    X, Xt, y, yt = train_test_split(states, targets, train_size=train_size, test_size=1 - train_size)

    readout = LogisticRegression(solver='sag', max_iter=10000, multi_class='multinomial', C=reg_fact).fit(X, y)

    test = lambda X, y: sum(readout.predict(X) != y) / len(y)

    return test(X, y), test(Xt, yt)


def train_readouts(spike_trains, targets, readout_times, *, num_discard=5, num_readouts=20, tau_filter=20, reg_factors=None, train_size=.8, use_mp=True, outdir=None, title=None):
    '''
    Train and test readouts. Assumes labels are integers (0, ..., N_classes-1).

    :param spike_trains: spike trains to extract liquid states from
    :param targets: target values
    :param readout_times: times at which liquid states should be generated
    :param num_discard: number of initial data points to drop to allow dynamics to settle
    :param num_readouts: number of classifiers to train
    :param tau_filter: time constant for filtering spikes in ms
    :param reg_factors: list of regularization factors to test, default: 1
    :param train_size: fraction of liquid states to use for training, default: 0.8
    :param use_mp: whether to use mulitprocessing, default: True
    :returns: dicts containing means and standard deviations of training and test errors
    '''

    # extract liquid states

    if not use_mp:
        ret = [get_neuron_liquid_states(st, readout_times, tau_filter) for st in spike_trains]

    else:
        delayed_ = (delayed(get_neuron_liquid_states)(st, readout_times, tau_filter) for st in spike_trains)
        ret = Parallel(n_jobs=mp.cpu_count())(delayed_)

    states = np.asarray(ret).T
    assert states.shape[0] == len(readout_times)
    assert states.shape[1] == len(spike_trains)

    # discard first few states
    states = states[num_discard:,:]
    targets = targets[num_discard:]
    targets = targets[:len(states)]

    # train readouts

    if reg_factors is None:
        reg_factors = [1]

    train_error = dict(mean=[], std=[])
    test_error = dict(mean=[], std=[])

    for reg_fact in reg_factors:
        print('training readouts using regularization = {0:e}'.format(reg_fact))

        args = dict(states=states, targets=targets, train_size=train_size, reg_fact=reg_fact)

        if not use_mp:
            ret = [train_readout(**args) for _ in range(num_readouts)]

        else:
            delayed_ = (delayed(train_readout)(**args) for _ in range(num_readouts))
            ret = Parallel(n_jobs=mp.cpu_count())(delayed_)

        train_errors = [r[0] for r in ret]
        test_errors = [r[1] for r in ret]

        train_mean, train_std = np.mean(train_errors), np.std(train_errors)
        test_mean, test_std = np.mean(test_errors), np.std(test_errors)

        print('  train error mean: {0:.3f}, std: {1:.3f}'.format(train_mean, train_std))
        print('  test error mean: {0:.3f}, std: {1:.3f}'.format(test_mean, test_std))

        train_error['mean'] += [train_mean]
        train_error['std'] += [train_mean]
        test_error['mean'] += [test_mean]
        test_error['std'] += [test_mean]

    return train_error, test_error


def lsm_experiment(task, simulate=True, reg_fact=1, outdir=None, title=''):
    """
    Perform an LSM experiment.

    :param task: target task for learning
    :param simulate: whether SNN should be simulated, otherwise load data from pickle
    :param reg_fact: inverse of regularization strength, default: 1
    :param outdir: output directory for plots
    :param title: title for plots
    """

    assert outdir is not None, 'need to pass outdir'

    if simulate:
        t_sim = 120 * second

        N_E = 1000
        N_I = 250

        tau_m = 30 * ms
        C_m = 30 * pF
        R_m = tau_m / C_m
        u_rest = -65 * mV
        u_th = -60 * mV
        u_reset = -72 * mV
        Delta_abs = 3 * ms
        tau_syn = 5 * ms

        J_input = 660 * pA
        J_EE = 205 * pA
        J_EI = 95 * pA
        J_IE = -450 * pA
        J_II = -370 * pA

        C_E = 2
        C_I = 1

        f_noise = 25 * Hz
        J_noise = 5 * pA

        dt = .1 * ms

        # setup brian2

        brian2.defaultclock.dt = dt

        net = brian2.Network()

        # setup neuron

        eqs = '''
        du_m/dt = ( -(u_m - u_rest) + R_m * I) / tau_m : volt (unless refractory)
        dI/dt = -I / tau_syn : ampere
        '''

        thres = 'u_m >= u_th'
        reset = 'u_m = u_reset'

        neurons_exc = NeuronGroup(N_E, eqs, threshold=thres, reset=reset, refractory=Delta_abs, method='exact')
        neurons_exc.u_m = u_rest

        neurons_inh = NeuronGroup(N_I, eqs, threshold=thres, reset=reset, refractory=Delta_abs, method='exact')
        neurons_inh.u_m = u_rest

        # connect pools

        spike_mon_exc = SpikeMonitor(neurons_exc)
        spike_mon_inh = SpikeMonitor(neurons_inh)

        # setup inputs

        stim_len = 50
        stim_dt = 250
        num_input = 3

        input_bits, stim_spike_trains, stim_ids, stim_times = generate_stimulus(t_sim / ms, stim_len=stim_len, stim_dt=stim_dt, num_input=num_input, dt=dt / ms)

        num_input = input_bits.shape[1]

        inputs = SpikeGeneratorGroup(num_input, stim_ids, stim_times * ms)

        input_spike_mon = SpikeMonitor(inputs)

        # setup synapses

        # TODO: use the same model here as in check_stp() (but use
        # event-driven updates here if you didn't do it already), wire
        # the circuit as specified in the assigment sheet

        syn_eqs = ...

        on_pre = ...

        syn_in = ...
        ... # set the synapse parameters

        syn_EE = ...
        syn_EI = ...
        syn_IE = ...
        syn_II = ...

        # TODO end

        # background input

        poisson_input_exc = PoissonInput(neurons_exc, 'I', 1, f_noise, weight=J_noise)
        poisson_input_inh = PoissonInput(neurons_inh, 'I', 1, f_noise, weight=J_noise)

        # run

        units = [inputs, input_spike_mon, neurons_exc, neurons_inh, spike_mon_exc,
                spike_mon_inh, syn_in, syn_EE, syn_EI, syn_IE, syn_II,
                poisson_input_exc, poisson_input_inh]

        net.add(units)

        net.run(t_sim, report='stdout', report_period=10 * second)

        # analysis

        f_exc = len(spike_mon_exc) / t_sim / len(spike_mon_exc.source)
        f_inh = len(spike_mon_inh) / t_sim / len(spike_mon_inh.source)

        print(f'mean firing rate (exc.): {f_exc/Hz:.1f} Hz')
        print(f'mean firing rate (inh.): {f_inh/Hz:.1f} Hz')

        # plots

        plot_neuron_exc = 50
        plot_neuron_inh = 50
        plot_t = 2000 * ms

        times_in, ids_in = input_spike_mon.t[:], input_spike_mon.i[:]
        m_in = (times_in <= plot_t) & (ids_in < plot_neuron_inh)

        times_exc, ids_exc = spike_mon_exc.t[:], spike_mon_exc.i[:]
        m_exc = (times_exc <= plot_t) & (ids_exc < plot_neuron_exc)

        times_inh, ids_inh = spike_mon_inh.t[:], spike_mon_inh.i[:]
        m_inh = (times_inh <= plot_t) & (ids_inh < plot_neuron_inh)

        def get_rates(spike_mon, t_max, bin_size=20 * ms):
            spikes = spike_mon.t[spike_mon.t < t_max]

            t_ = np.arange(0, t_max / ms, bin_size / ms) * ms
            f_ = np.zeros_like(t_)

            for k, t1 in enumerate(t_[1:]):
                t0 = t1 - bin_size
                f_[k + 1] = sum((t0 < spikes) & (spikes <= t1)) / bin_size / len(spike_mon.source)

            return t_, f_

        t_exc, f_exc = get_rates(spike_mon_exc, plot_t)
        t_inh, f_inh = get_rates(spike_mon_inh, plot_t)

        fig = plt.figure(figsize=(6, 6))

        h_in = 1
        h_n = 3
        h_f = 2
        grid_h_ = [h_in, h_n, h_n, h_f]
        grid_h0_ = [sum(grid_h_[:n]) for n in range(len(grid_h_))]
        grid = (sum(grid_h_), 1)

        ax = plt.subplot2grid(grid, (grid_h0_[0], 0), rowspan=grid_h_[0])
        ax.scatter(times_in[m_in] / second, ids_in[m_in], c='k', marker='.', s=1)
        ax.set_ylim(-1, len(inputs))
        ax.locator_params(axis='x', nbins=3)
        ax.set_yticks([])
        ax.set_ylabel('inputs')

        ax = plt.subplot2grid(grid, (grid_h0_[1], 0), rowspan=grid_h_[1], sharex=plt.gca())
        ax.scatter(times_exc[m_exc] / second, ids_exc[m_exc], c='C0', marker='.', s=1)
        ax.set_ylim(-1, plot_neuron_exc)
        ax.locator_params(axis='x', nbins=3)
        ax.set_yticks([])
        ax.set_ylabel('exc. neurons')

        ax = plt.subplot2grid(grid, (grid_h0_[2], 0), rowspan=grid_h_[2], sharex=plt.gca())
        ax.scatter(times_inh[m_inh] / second, ids_inh[m_inh], c='C3', marker='.', s=1)
        ax.set_ylim(-1, plot_neuron_inh)
        ax.locator_params(axis='x', nbins=3)
        ax.set_yticks([])
        ax.set_ylabel('inh. neurons')

        ax = plt.subplot2grid(grid, (grid_h0_[3], 0), rowspan=grid_h_[3], sharex=plt.gca())
        ax.plot(t_exc / second, f_exc / Hz, c='C0', lw=2, label='exc.')
        ax.plot(t_inh / second, f_inh / Hz, c='C3', lw=2, label='inh.')
        ax.set_xlim(0, plot_t / second)
        ax.locator_params(nbins=3)
        ax.set_xlabel('$t$ / s')
        ax.set_ylabel('$f$ / Hz')
        ax.legend(loc='upper right', ncol=2, borderaxespad=0)

        [plt.setp(ax.get_xticklabels(), visible=False) for ax in fig.axes[:-1]]
        [ax.spines['top'].set_visible(False) for ax in fig.axes]
        [ax.spines['right'].set_visible(False) for ax in fig.axes]
        plt.tight_layout()

        if outdir and title:
            plt.savefig(join(outdir, title + '_activity.png'), dpi=200)

        # save data

        data = {
                'inputs': {
                    'bits': input_bits,
                    'times': stim_times,
                    'ids': stim_ids,
                    'spike_trains': stim_spike_trains,
                },
                'exc_spike_trains': [sp / ms for sp in spike_mon_exc.spike_trains().values()],
                'inh_spike_trains': [sp / ms for sp in spike_mon_exc.spike_trains().values()],
                'simulation': {
                        't_sim': t_sim / second,
                        'num_input': num_input,
                        'stim_dt': stim_dt,
                        'stim_len': stim_len,
                },
        }

        with open(join(outdir, 'data.pkl'), 'wb') as f:
            pkl.dump(data, f)

    else:
        with open(join(outdir, 'data.pkl'), 'rb') as f:
            data = pkl.load(f)

    if task is None or task == 'none':
        return

    # ----------------------------------------------------------------------
    # generate targets

    num_input = data['simulation']['num_input']  # number of input units
    inputs = data['inputs']['bits']  # shape: (sample, num_input)

    # TODO: define targets

    if task == 'xor':
        targets = ...

    elif task == 'mem1':
        targets = ...

    elif task == 'memall':
        targets = ...

    elif task == 'sum':
        targets = ...  # only needed for bonus task

    else:
        raise ValueError()

    # TODO end

    assert len(targets) == inputs.shape[0]

    # ----------------------------------------------------------------------
    # train readouts

    stim_dt = data['simulation']['stim_dt']
    stim_len = data['simulation']['stim_len']
    t_sim = data['simulation']['t_sim'] * 1000  # in ms

    # subsample neurons

    num_output_neurons = 200
    spike_trains = data['exc_spike_trains'][:num_output_neurons]

    # train

    if task == 'xor':

        readout_delay = 20  # ms
        rec_time_start = stim_dt + stim_len + readout_delay  # time of first liquid state
        readout_times = np.arange(rec_time_start, t_sim, stim_dt)

        reg_factors = [1]

        train_error, test_error = train_readouts(spike_trains, targets, readout_times, reg_factors=reg_factors)

        # plot results

        if len(reg_factors) > 1:
            fig = plt.figure(figsize=(6, 2.5))

            ax = plt.subplot(1, 2, 1)
            ax.errorbar(np.log(reg_factors), train_error['mean'], train_error['std'], c='C2', capsize=2)
            ax.set_ylim(None, min(1, ax.get_ylim()[1]))
            ax.locator_params(axis='y', nbins=4)
            ax.set_xlabel(r'$\log\,C$')
            ax.set_ylabel('train error')

            ax = plt.subplot(1, 2, 2)
            ax.errorbar(np.log(reg_factors), test_error['mean'], test_error['std'], c='C0', capsize=2)
            ax.set_ylim(None, min(1, ax.get_ylim()[1]))
            ax.locator_params(axis='y', nbins=4)
            ax.set_xlabel(r'$\log\,C$')
            ax.set_ylabel('test error')

            [ax.spines['top'].set_visible(False) for ax in fig.axes]
            [ax.spines['right'].set_visible(False) for ax in fig.axes]

            plt.tight_layout()
            if outdir and title:
                plt.savefig(join(outdir, title + '_errors.png'), dpi=200)

    elif task in ['mem1', 'memall']:
        if task == 'memall':
            extra_readout_time = 0  # TODO: bonus task: change this to 250
        else:
            extra_readout_time = 0

        readout_delays = np.arange(10, stim_dt - stim_len + 10 + extra_readout_time, 10)

        train_error = dict(mean=[], std=[])
        test_error = dict(mean=[], std=[])

        for readout_delay in readout_delays:
            print('using delay = {0:g}'.format(readout_delay))

            rec_time_start = stim_dt + stim_len + readout_delay  # time of first liquid state
            readout_times = np.arange(rec_time_start, t_sim, stim_dt)

            reg_factors = [1]

            d_train_error, d_test_error = train_readouts(spike_trains, targets, readout_times, reg_factors=reg_factors)

            train_error['mean'] += d_train_error['mean']
            train_error['std'] += d_train_error['std']
            test_error['mean'] += d_test_error['mean']
            test_error['std'] += d_test_error['std']

        # plot means and standard deviations

        fig = plt.figure(figsize=(8, 2.5))

        ax = plt.subplot(1, 2, 1)
        ax.errorbar(readout_delays, train_error['mean'], train_error['std'], c='C2', capsize=2)
        ax.set_xlim(0, None)
        ax.set_ylim(None, min(1, ax.get_ylim()[1]))
        ax.locator_params(axis='y', nbins=4)
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel('train error')

        ax = plt.subplot(1, 2, 2)
        ax.errorbar(readout_delays, test_error['mean'], test_error['std'], c='C0', capsize=2)
        ax.set_xlim(0, None)
        ax.set_ylim(None, min(1, ax.get_ylim()[1]))
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel('test error')

        [ax.spines['top'].set_visible(False) for ax in fig.axes]
        [ax.spines['right'].set_visible(False) for ax in fig.axes]

        plt.tight_layout()
        if outdir and title:
            plt.savefig(join(outdir, title + '_errors.png'), dpi=200)

    elif task == 'sum':
        input_spike_trains = data['inputs']['spike_trains']

        # TODO: bonus task
        #
        # train readouts on spike trains from liquid and on input spike trains

        ...

        # TODO end


if __name__ == '__main__':

    outdir = 'out'
    outdir = join(os.path.dirname(__file__), outdir)
    os.makedirs(outdir, exist_ok=True)

    # check your stp implementation

    check_stp(outdir=outdir)

    # run the lsm and record all spikes

    # lsm_experiment(task=None, simulate=True, outdir=outdir, title='run')

    # use the generated data to train and test readouts for the different tasks

    # lsm_experiment(task='xor', simulate=False, outdir=outdir, title='xor')
    #
    # lsm_experiment(task='mem1', simulate=False, outdir=outdir, title='mem1')
    #
    # lsm_experiment(task='memall', simulate=False, outdir=outdir, title='memall')

    plt.show()  # avoid having multiple plt.show()s in your code
