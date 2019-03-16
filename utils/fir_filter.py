import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz, lfilter_zi


class fir_filter(object):
    def __init__(self, fs, cutoff, ripple_db):
        self.fs = fs  # sample_rate
        # The Nyquist rate of the signal.
        nyq_rate = self.fs / 2.0
        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 5.0 / nyq_rate
        # The desired attenuation in the stop band, in dB.
        self.ripple_db = 10
        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)
        #print('N = ', N)
        # The cutoff frequency of the filter.
        self.cutoff = cutoff
        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        self.taps = firwin(N, self.cutoff / nyq_rate, window=('kaiser', beta))

        self.N = N
        self.x_buffer = []
        self.index = 0

        self.zi =  lfilter_zi(self.taps, 1.0)

        self.init_done = False
    def apply(self, x):
        self.x_buffer.append(x)

        if not self.init_done:

            if self.index < self.N-1:
                filtered_x = x
                self.index += 1
            else:
                self.init_done = True
                self.index =0

        if self.init_done:
            y = lfilter(self.taps, 1.0, np.array(self.x_buffer))
            filtered_x = y[-1]


        return filtered_x

    def ouput_filtered_x(self, x):
        y = lfilter(self.taps, 1.0, np.array(x))

        return y


    def reset(self):
        self.x_buffer = []
        self.index = 0
        self.init_done = False

from utils.Logger import  IO
from matplotlib.pylab import plt
def test_fir():

    rlt = IO('RewardDuty/fitness5_param.pkl').read_pickle()
    (rewards, commands, v_e) = rlt

    x = rewards[:,0]

    T = 1000

    fir = fir_filter(100,10,10)

    filter_x_list = []
    for i in range(T):
        state = x[i]
        state_e = fir.apply(state)


        filter_x_list.append(state_e)


    filter_x_list = np.array(filter_x_list)
    N = fir.N
    t = np.arange(0,10,0.01)
    plt.plot(t, x[:T], 'b--')
    #plot(y)
    plt.plot(t, filter_x_list, 'r')
    #plt.plot(t[N-1:] , filter_x_list[N-1:], 'g', linewidth=4)
    plt.show()
