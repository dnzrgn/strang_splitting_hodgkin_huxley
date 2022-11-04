import numpy as np

class StrangSplittingHodgkinHuxley():
    """
    A class implementing the strang splitting solver for the hodgkin huxley neuron model.
    
    Exploits the fact that the hodgkin huxley model is a conditionally linear system, 
    i.e. is described by a system of ODEs of the form:
        x_i' = a_i(x)*x_i + b_i(x),     i = 1,...,d
    where a_i and b_i only depend on x_j for j != i.
    
    Mathematical formulation taken from:
    DOI: 10.1137/18M123390X

    Attributes
    ----------
    C_m:        Membrane capacitance [uF/cm^2] (default: 1)
    g_Na:       Maximum sodium conductance [mS/cm^2] (default: 120)
    g_K:        Maximum Potassium conductance [mS/cm^2] (default: 36)
    g_L:        Maximum Leak conductance [mS/cm^2] (default: 0.3)
    E_Na:       Sodium reversial potential [mV] (default: 50)
    E_K:        Potassium reversial potential [mV] (default: -77)
    E_L:        Leak reversal potential [mV] (default: -54.387)
    """
    def __init__(self,
        C_m=1, 
        g_Na=120, 
        g_K=36, 
        g_L=0.3, 
        E_Na=50, 
        E_K=-77, 
        E_L=-54.387,
    ):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

    def alpha_m(self, V):
        """
        Channel gating kinetics.

        Parameter:
        V: Membrane Potential [mV]
        """
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """
        Channel gating kinetics.

        Parameter:
        V: Membrane Potential [mV]
        """
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """
        Channel gating kinetics.

        Parameter:
        V: Membrane Potential [mV]
        """
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """
        Channel gating kinetics.

        Parameter:
        V: Membrane Potential [mV]
        """
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """
        Channel gating kinetics.

        Parameter:
        V: Membrane Potential [mV]
        """
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """
        Channel gating kinetics.

        Parameter:
        V: Membrane Potential [mV]
        """
        return 0.125*np.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Function calculating membrane current for Sodium ions.

        Parameters:
        V:  Membrane Potential [mV]
        m:  Gating variable m
        h:  Gating variable h

        Returns:
        Membrane current [uA/cm^2]
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Function calculating membrane current for Potassium ions.

        Parameters:
        V:  Membrane Potential [mV]
        n:  Gating variable n

        Returns:
        Membrane current [uA/cm^2]
        """
        return self.g_K  * n**4 * (V - self.E_K)
    
    def I_L(self, V):
        """
        Function calculating leak membrane current.

        Parameters:
        V:  Membrane Potential [mV]

        Returns:
        Membrane current [uA/cm^2]
        """
        return self.g_L * (V - self.E_L)

    def __call__(self, I, delta_t=0.01, state=None):
        """
        Function calculating one solver step.

        Parameters:
        I:          Input current [uA/cm^2]
        delta_t:    Time to integrate over [ms] (default: 0.01)
        state:      Tuple of state variables (V, m, h, n) 
            If None, state is initialized as (-65, 0.05, 0.6, 0.32) 
            (default: None)

        Returns
        Tuple of state variables (V, m, h, n)
        """

        def exp_euler_step(x, dt, a, b):
            """
            Function calculating one step of exponential euler method.
            Used for ODE of form x' = ax + b

            Parameters:
            x:  previous function value
            a:  Constant coefficient of ODE
            b:  Constant coefficient of ODE

            Returns
            New function value at t+dt
            """
            e_a = np.exp(dt*a)
            return e_a*x + ( (e_a-1)*b )/a

        # Initialize state, if not given
        if state is None:
            V, m, h, n = -65, 0.05, 0.6, 0.32
        else:
            V, m, h, n = state

        # Calculate midpoint for mhn gating variables
        alphas = np.array([self.alpha_m(V), self.alpha_h(V), self.alpha_n(V)])
        betas = np.array([self.beta_m(V), self.beta_h(V), self.beta_n(V)])
        mhn_half = exp_euler_step((m, h, n), delta_t/2, -(alphas+betas), alphas)

        # Calculate full step for membrane potential, using midpoint of mhn variables.
        V = exp_euler_step(
            V, 
            delta_t, 
            -( self.g_Na*(mhn_half[0]**3)*mhn_half[1] + self.g_K*(mhn_half[2]**4) + self.g_L ), 
            I + self.g_Na*(mhn_half[0]**3)*mhn_half[1]*self.E_Na + self.g_K*(mhn_half[2]**4)*self.E_K + self.g_L*self.E_L
        )

        # Calculate remaining half step for mhn gating variables
        alphas = np.array([self.alpha_m(V), self.alpha_h(V), self.alpha_n(V)])
        betas = np.array([self.beta_m(V), self.beta_h(V), self.beta_n(V)])
        m, h, n = exp_euler_step(mhn_half, delta_t/2, -(alphas+betas), alphas)

        return V, m, h, n