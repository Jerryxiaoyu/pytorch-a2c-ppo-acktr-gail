from math import pi, cos, sin

class matsuoka_oscillator(object):
    def __init__(self, kf=1, dt=0.01):
        
        # Set up the oscillator constants
        self.tau = 0.2800
        self.tau_prime = 0.4977
        self.beta = 2.5000
        self.w_0 = 2.2829
        self.u_e = 0.4111
        self.m1 = 1.0
        self.m2 = 1.0
        self.a = 1.0
    
        # Modify the time constants based on kf
        self.tau *= kf
        self.tau_prime *= kf
    
        # Step time
        self.dt = dt

    def oscillator_fun(self, u1, u2, v1, v2, y1, y2, f1, f2, s1, s2, bias, gain ):
        """
        Calculates the state variables in the next time step
        """
        d_u1_dt = (-u1 - self.w_0 *y2 -self.beta * v1 + self.u_e + f1 + self.a * s1) / self.tau
        d_v1_dt = (-v1 + y1) / self.tau_prime
        y1 = max([0.0, u1])
        
        d_u2_dt = (-u2 - self.w_0 * y1 - self.beta * v2 + self.u_e + f2 + self.a * s2) / self.tau
        d_v2_dt = (-v2 + y2) / self.tau_prime
        y2 = max([0.0, u2])
        
        u1 += d_u1_dt * self.dt
        u2 += d_u2_dt * self.dt
        v1 += d_v1_dt * self.dt
        v2 += d_v2_dt * self.dt
        
        o = bias + gain * (-self.m1 * y1 + self.m2 * y2)
        
        return u1, u2, v1, v2, y1, y2, o

 
    
    
    
class CPG_MatsuokaNeutron(object):
    def __init__(self, id, master_nuron, param ,dt=0.01, kf=1, w_ms = 1):
        self.id = id
        self.parm = {'kf': kf, 'u1':param[0], 'u2':param[1], 'v1':param[2], 'v2':param[3],
                     'y1':param[4], 'y2':param[5], 'o':param[6], 'gain':param[7], 'bias':param[8]}
        self.w_ms = w_ms
        

        osillator = matsuoka_oscillator(self.parm['kf'], dt=dt)
        self.osillator_fun = osillator.oscillator_fun

        self.master_nuron = master_nuron
        
        
    def next_output(self,  f1, f2  ):
        
        if self.master_nuron is not None:
            s1 = self.w_ms * self.master_nuron.parm['u1']
            s2 = self.w_ms * self.master_nuron.parm['u2']
        else:
            s1 = 0
            s2 = 0
        
        self.parm['u1'],self.parm['u2'], self.parm['v1'], self.parm['v2'], self.parm['y1'], self.parm['y2'], self.parm['o'] = \
            self.osillator_fun(self.parm['u1'],self.parm['u2'], self.parm['v1'], self.parm['v2'], self.parm['y1'], self.parm['y2'],
                               f1, f2, s1, s2, self.parm['bias'], self.parm['gain'] )
        
    
