from DimerDetuneNew import DimerDetune

class Operations(DimerDetune):
    """input class dimer needs:
    Methods:
        liouvillian
        hamiltonian
        init_state
    Arguments:
        dimH - Hilbert space dimenstion
        omega - some frequency of interest to be resolved by time step
        """

    def steady_state(self):
        l = self.liouvillian()
        a = sl.eig(l) #eigenvectors of liovillian
        eners = a[0]
        eigstates = a[1]
        print('max eigenvalue is ', max(np.real(eners)))
        max_index = np.argmax(eners)
        rho_ss = eigstates[max_index, :].reshape([self.dimH, self.dimH])
        rho_ss = rho_ss / np.trace(rho_ss)
        return rho_ss

    def time_evol_me(self, tmax_ps):
        count1 = time.time()
        tmax = tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12
        steps = np.int((tmax - self.t0) / self.dt)  # total number of steps. Must be int.

        # initial state
        rho0 = self.init_state()
        rho0_l = rho0.reshape(1, (rho0.shape[1]) ** 2)  # initial state in Liouville space

        # function to be integrated
        L = self.liouvillian()

        def f(t, y):
            pt = L.dot(y.transpose()).transpose()  # Master equation in Liouville space
            return pt

        evo = integrate.complex_ode(f)
        evo.set_initial_value(rho0_l, self.t0)  # initial conditions
        t = np.zeros(steps)

        t[0] = self.t0
        rhoT = np.zeros((steps, rho0.shape[1] ** 2), dtype=complex)  # time is 3rd dim.
        rhoT[0, :] = rho0_l
        # now do the iteration.
        k = 1
        while evo.successful() and k < steps:
            evo.integrate(evo.t + self.dt)  # time to integrate at at each loop
            t[k] = evo.t  # save current loop time
            rhoT[k, :] = evo.y  # save current loop data
            k += 1  # keep index increasing with time.
        count2 = time.time()
        print('Integration =', count2 - count1)

        return rhoT, t

    def oper_evol(self, operator, rhoT, t, tmax_ps):
        """calculates the time evolution of an operator"""
        steps = len(rhoT[:, 0])
        N = int(np.sqrt(len(rhoT[0, :])))
        oper = np.zeros(steps)
        for i in np.arange(steps):
            oper[i] = np.real(np.trace(operator.dot(rhoT[i, :].reshape(N, N))))
        return oper

    @staticmethod
    def corrfunc(f1, f2, delta):
        f1bar = np.zeros(np.size(f1) - delta)
        f2bar = np.zeros(np.size(f1) - delta)
        df1df2bar = np.zeros(np.size(f1) - 2 * delta)
        df1sqbar = np.zeros(np.size(f1) - 2 * delta)
        df2sqbar = np.zeros(np.size(f1) - 2 * delta)
        for i in np.arange(np.size(f1) - delta):
            f1bar[i] = (1 / delta) * integrate.trapz(f1[i:i + delta + 1])
            f2bar[i] = (1 / delta) * integrate.trapz(f2[i:i + delta + 1])
        df1 = f1[0:(np.size(f1) - delta)] - f1bar
        df2 = f2[0:(np.size(f1) - delta)] - f2bar
        df1df2 = df1 * df2
        df1sq = df1 ** 2
        df2sq = df2 ** 2
        for i in np.arange(np.size(f1) - 2 * delta):
            df1df2bar[i] = integrate.trapz(df1df2[i:i + delta + 1])
            df1sqbar[i] = integrate.trapz(df1sq[i:i + delta + 1])
            df2sqbar[i] = integrate.trapz(df2sq[i:i + delta + 1])
        C = df1df2bar / np.sqrt(df1sqbar * df2sqbar)
        return C