
"""
Code for two coupled dimer chromophores, strongly coupled to modes.
Tensor order is excitonic, mode 1, mode 2.
@author: Charlie Nation
based on code by stefan siwiak-jaszek

___________________________________________________________________________________________________________________

To do:

Replace np.matrix (depreciated) with np.array throughout - this doesn't have a getH method, so will need
.conj().transpose() I think

Define regulary used arrays inside init

rhoT needs to be parsed into each plot to avoid repeated calculations
"""
import numpy as np
import scipy.constants as constant
import scipy.integrate as integrate
import time
import scipy.sparse as sp
import scipy.linalg as sl
import matplotlib.pylab as plt
import QCcorrelations as QC
#from plots import Plots
#from foo import FooClass


class DimerDetune:
    """Defines properties and functions of the vibronic dimer system"""
    def __init__(self, rate_swap, n_cutoff=5, temperature=298):
        #initialise properties of dimer 
        self.n_cutoff =n_cutoff
        self.temperature =  temperature

        #unsure
        self.huang = 0.0578
        #initialise with no detuning
        self.omega = 1111
        self.detuning = 1.00
        self.w1 = self.omega
        self.w2 = self.detuning * self.omega
        
        #electronic states
        self.e1 = 0 + self.w1 * self.huang
        self.e2 = 1042 + self.w2 * self.huang  # 946.412
        self.de = self.e2 - self.e1

        #dipole-dipole coupling strength
        self.V = 92  # 236.603 # cm-1

        #excitonic states 
        self.dE = np.sqrt(self.de ** 2 + 4 * self.V ** 2)  # energy between excitonic states
        #rotation angle to create excitonic Hamiltonian
        self.theta = 0.5 * np.arctan(2 * self.V / self.de)

        #exciton-vibration coupling strength
        self.a = 1
        self.g1 = self.w1 * np.sqrt(self.huang) * self.a
        self.g2 = self.w2 * np.sqrt(self.huang) * self.a
        
        #decoherence properties
        self.kBT = (constant.k * temperature) / (constant.h * constant.c * 100)  # cm-1
        #rates.. where do these numbers come from?

        if rate_swap ==True:
            self.thermal_dissipation = 333.3564  # 70
            self.electronic_dephasing = 33.564
        else:
            self.thermal_dissipation = 33.3564  # 70
            self.electronic_dephasing = 333.564
        #0.1, 1 ps rates - 
        self.taudiss = 1 / (1e-12 * self.thermal_dissipation * 100 * constant.c)
        self.taudeph = 1 / (1e-12 * self.electronic_dephasing * 100 * constant.c)
        
         # %% scaling effects from setting 2pi x c = 1 and hbar = 1

        self.r_el = self.electronic_dephasing / (2 * constant.pi)
        self.r_th = self.thermal_dissipation / (2 * constant.pi)
        self.r_v1 = self.r_th  # 6 # cm-1
        self.r_v2 = self.r_th  # 6 # cm-1

        #hilbert space defined according to max number of vibrational modes
        self.N = n_cutoff  # 7
        self.dimH = 2 * n_cutoff**2  # Hilbert space dimension

        # Exciton Vectors

        self.E1 = sp.lil_matrix(np.matrix([[1.], [0.]])).tocsr()
        self.E2 = sp.lil_matrix(np.matrix([[0.], [1.]])).tocsr()

        self.t0 = 0
        self.dt = ((2 * constant.pi) / self.omega) / 100

        

    def exciton_operator(self, e1, e2):
        """returns |E_e1><E_e2|"""
        if e1 == 1:
            e_1 = self.E1
        elif e1 == 2:
            e_1 = self.E2
        else:
            raise Exception('e1 should be 1 or 2 for |E_1> or |E_2>')
        if e2 == 1:
            e_2 = self.E1
        elif e2 == 2:
            e_2 = self.E2
        else:
            raise Exception('e2 should be 1 or 2 for <E_1| or <E_2|')
        return sp.kron(e_1, e_2.getH())

    def rotate(self):
        """Unitary rotation from site to excitonic bases"""
        #why sparse matrix here? - You're right its not really sparse, but we need it to do operations with other sparse matrices
        # so put it in that format.
        return sp.lil_matrix(np.matrix([[np.cos(self.theta), np.sin(self.theta)],
                                        [-np.sin(self.theta), np.cos(self.theta)]])).tocsr()

    def vib_mode_operator(self, k, j):
        """returns |k><j| in the vibrational Fock space"""
        mode1 = np.matrix(np.zeros([self.N, 1]))
        mode2 = np.matrix(np.zeros([self.N, 1]))
        mode1[k, 0] = 1
        mode2[j, 0] = 1

    def destroy(self):
        """Annihilation operator on vibrational Fock space"""
        #defined in thesis as b
        return sp.diags(np.sqrt(np.arange(1, self.N)), 1).tocsr()

    def identity_vib(self):
        """identity operator on vibrational Fock space"""
        #sp.eye returns 1s on diag
        return sp.eye(self.N, self.N).tocsr()

    def thermal(self, w):
        """Thermal density operator on vibrational Fock space"""
        p = np.matrix(np.zeros((self.N, self.N)))
        for i in np.arange(self.N):
            p[i, i] = (1 - np.exp(-w / self.kBT)) * np.exp(-w * i / self.kBT)
        return sp.lil_matrix(p).tocsr()

    def a_k(self, k):
        #not sure about this one - rotates an excitonic operator back into the site basis
        return self.rotate() * self.exciton_operator(k, k) * self.rotate().getH()

    @staticmethod
    def dissipator(operator, rate):
        """Dissipator in Liouville space"""
        d = rate * (sp.kron(operator, operator.conj()).tocsr()
                    - 0.5 * sp.kron(operator.getH() * operator, sp.eye(operator.shape[1])).tocsr()
                    - 0.5 * sp.kron(sp.eye(operator.shape[1]), (operator.getH() * operator).transpose()).tocsr())
        return d

    @staticmethod
    def liouv_commutator(operator):
        """Commutator in Liouville space"""
        h = sp.kron(operator, sp.eye(operator.shape[1])).tocsr() - sp.kron(sp.eye(operator.shape[1]), operator.transpose()).tocsr()
        return h

    def hamiltonian(self):
        """dE/2 sigma_z + omega_1 n_1 + omega_2 n_2 + g_1 |1><1| (b_1 + b^dag_1) + g_2 |2><2| (b_2 + b^dag_2 )
        |1(2)><1(2)| is the site basis ground (excited) state"""
        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        sigmaz = oE2 - oE1   # note that this is backwards compared to definition in quantum information approaches
        b = self.destroy()
        Iv = self.identity_vib()
        eldis1 = self.a_k(1)
        eldis2 = self.a_k(2)
        Ie = sp.eye(2, 2).tocsr()

        H = sp.kron((self.dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
            + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
            + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
            + self.g1 * sp.kron(eldis1, sp.kron((b + b.getH()), Iv)) \
            + self.g2 * sp.kron(eldis2, sp.kron(Iv, (b + b.getH())))

        return H

    def liouvillian(self):
        H = self.hamiltonian()
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()
        ELdis1 = sp.kron(self.a_k(1), sp.kron(Iv, Iv)).tocsr()
        ELdis2 = sp.kron(self.a_k(1), sp.kron(Iv, Iv)).tocsr()
        b = self.destroy()
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        nw1 = 1 / (np.exp(self.w1 / self.kBT) - 1)  # thermal distribution
        nw2 = 1 / (np.exp(self.w2 / self.kBT) - 1)

        L = -1j * self.liouv_commutator(H) + self.dissipator(ELdis1, self.r_el) + self.dissipator(ELdis2, self.r_el) \
            + self.dissipator(oB1, self.r_v1 * (nw1 + 1)) + self.dissipator(oB1.getH(), self.r_v1 * nw1) \
            + self.dissipator(oB2, self.r_v2 * (nw2 + 1)) + self.dissipator(oB2.getH(), self.r_v2 * nw2)
        return L

    def init_state(self):
        oE2 = self.exciton_operator(2, 2)
        return sp.kron(oE2, sp.kron(self.thermal(self.w1), self.thermal(self.w2))).todense()


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

class Plots(Operations):

    def __init__(self, rate_swap, n_cutoff=5, temperature=298):
        DimerDetune.__init__(self, rate_swap, n_cutoff=5, temperature=298)
        tmax_ps = 4
        self.rhoT , self.t  = self.time_evol_me(tmax_ps)

    def sync_evol(self):

        b = self.destroy()
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        oX2 = oB2 + oB2.getH()
        oX1 = oB1 + oB1.getH()

        tmax_ps = 4
        
        #time evo of operators, REPRODUCE THIS IN COHERENCES
        x1 = self.oper_evol(oX1,self.rhoT, self.t, tmax_ps)
        x2 = self.oper_evol(oX2,self.rhoT, self.t, tmax_ps)  # can also pass a time step if necessary

        t_cm = self.t / (2 * constant.pi)
        t_ps = (t_cm * 1e12) / (100 * constant.c)

        elta = np.int(np.round(((2 * constant.pi) / self.omega) / self.dt))
        c_X12 = self.corrfunc(x1, x2, elta)

        FigureA = plt.figure(14)
        en = 13000
        st = 0000
        itvl = 5
        axA = FigureA.add_subplot(111)
        axA.plot(t_ps[np.arange(st, en, itvl)], x2[np.arange(st, en, itvl)], label=r'$\langle X_2\rangle$')
        axA.plot(t_ps[np.arange(st, en, itvl)], x1[np.arange(st, en, itvl)], label=r'$\langle X_1\rangle$')

        plt.ylabel('$<x>$')
        plt.ylabel('$C_{<X_1><X_2>}$',fontsize=12)
        axA.set_xlabel('Time (ps)')
        # axA.set_xlim([0,10])
        axA.set_yticks([])
        axB = axA.twinx()
        axB.plot(t_ps[np.arange(st, en, itvl)], c_X12[np.arange(st, en, itvl)], 'r-o', markevery=0.05, markersize=5,
                 label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')
        axB.grid()
        axA.grid()
        axB.legend(bbox_to_anchor=(0.9, 0.6))
        axA.legend()
        # plt.savefig('cXX_dw004_QC.pdf',bbox_inches='tight',dpi=600,format='pdf',transparent=True)


    def coherences(self):
        """Complex magnitude of exciton-vibration coherences scaled by the absolute value
        of their corresponding position matrix element in the open quantum system evolution."""
        b = self.destroy()
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        oX2 = oB2 + oB2.getH()
        oX1 = oB1 + oB1.getH()


        H= self.hamiltonian()
        vals, eigs = np.linalg.eigh(H.todense())
        oX1eig = eigs.getH() * oX1 * eigs

        M1thermal = self.thermal(self.w1)
        M2thermal = self.thermal(self.w2)
        oE2 = sp.kron(self.E2, self.E2.getH()).tocsr()
        P0 = sp.kron(oE2, sp.kron(M1thermal, M2thermal)).todense()

        ########################################
        # def oper_evol(self, operator, rhoT, t, tmax_ps):
        #     """calculates the time evolution of an operator"""
        #     steps = len(rhoT[:, 0])
        #     N = int(np.sqrt(len(rhoT[0, :])))
        #     oper = np.zeros(steps)
        #     for i in np.arange(steps):
        #         oper[i] = np.real(np.trace(operator.dot(rhoT[i, :].reshape(N, N))))
        #     return oper

        # eigs_evo = oper_evol(eigs,self.rhoT, self.t, tmax_ps)
        #############################################

        opsi01 = np.kron(eigs[:, 0], eigs[:, 1].getH())
        opsi10 = np.kron(eigs[:, 1], eigs[:, 0].getH())
        opsi03 = np.kron(eigs[:, 0], eigs[:, 3].getH())
        opsi14 = np.kron(eigs[:, 1], eigs[:, 4].getH())
        opsi13 = np.kron(eigs[:, 1], eigs[:, 3].getH())
        opsi16 = np.kron(eigs[:, 1], eigs[:, 6].getH())
        opsi25 = np.kron(eigs[:, 2], eigs[:, 5].getH())
        opsi36 = np.kron(eigs[:, 3], eigs[:, 6].getH())
        opsi27 = np.kron(eigs[:, 2], eigs[:, 7].getH())
        opsi28 = np.kron(eigs[:, 2], eigs[:, 8].getH())
        opsi38 = np.kron(eigs[:, 3], eigs[:, 8].getH())
        opsi02 = np.kron(eigs[:, 0], eigs[:, 2].getH())
        opsi15 = np.kron(eigs[:, 1], eigs[:, 5].getH())
        opsi26 = np.kron(eigs[:, 2], eigs[:, 6].getH())
        opsi37 = np.kron(eigs[:, 3], eigs[:, 7].getH())


        tmax_ps = 4 #2
        tmax = tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  # 3 end time
        
        t_cm = self.t / (2 * constant.pi)
        t_ps = (t_cm * 1e12) / (100 * constant.c)

        ##################################################
        #eigs_evo = self.oper_evol(eigs,self.rhoT, self.t, tmax_ps)
        #############################################

        steps = len(self.rhoT[:, 0]) # np.int((tmax - self.t0) / self.dt)  # total number of steps. Must be int.

        
        psi01 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi01[i] = np.trace(opsi01.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))
        
        psi02 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi02[i] = np.trace(opsi02.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

        psi03 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi03[i] = np.trace(opsi03.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

        psi14 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi14[i] = np.trace(opsi14.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

        psi15 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi15[i] = np.trace(opsi15.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

        psi37 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi37[i] = np.trace(opsi37.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

        psi38 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi38[i] = np.trace(opsi38.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

        psi13 = np.zeros((steps), dtype=complex)
        for i in np.arange(steps):
            psi13[i] = np.trace(opsi13.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))
        #fn on operations - oper_evol FIGURE THIS OUT

        FigureB = plt.figure(5)
        plt.xlabel('Time ($ps$)')
        plt.grid()

        st = 0000
        en = 12000

        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2).transpose()

        f13 = np.round(np.abs(omegaarray[1, 3]), decimals=2)
        f01 = np.round(np.abs(omegaarray[0, 1]), decimals=2)
        f02 = np.round(np.abs(omegaarray[0, 2]), decimals=2)
        f03 = np.round(np.abs(omegaarray[0, 3]), decimals=2)
        f14 = np.round(np.abs(omegaarray[1, 4]), decimals=2)
        f15 = np.round(np.abs(omegaarray[1, 5]), decimals=2)
        f37 = np.round(np.abs(omegaarray[3, 7]), decimals=2)
        f38 = np.round(np.abs(omegaarray[3, 8]), decimals=2)
        f13 = np.round(np.abs(omegaarray[1, 3]), decimals=2)

        plt.plot(t_ps[st:en], np.abs(oX1eig[0, 1]) * np.abs(psi01[st:en]), label=r'$\Omega_{01} =$' + str(f01))
        plt.plot(t_ps[st:en], np.abs(oX1eig[0, 2]) * np.abs(psi02[st:en]), label=r'$\Omega_{02} =$' + str(f02))
        plt.plot(t_ps[st:en], np.abs(oX1eig[0, 3]) * np.abs(psi03[st:en]), label=r'$\Omega_{03} =$' + str(f03))
        plt.plot(t_ps[st:en], np.abs(oX1eig[1, 4]) * np.abs(psi14[st:en]), label=r'$\Omega_{14} =$' + str(f14))
        plt.plot(t_ps[st:en], np.abs(oX1eig[1, 5]) * np.abs(psi15[st:en]), label=r'$\Omega_{15} =$' + str(f15))
        plt.plot(t_ps[st:en], np.abs(oX1eig[3, 7]) * np.abs(psi37[st:en]), label=r'$\Omega_{37} =$' + str(f37))
        plt.plot(t_ps[st:en], np.abs(oX1eig[3, 8]) * np.abs(psi38[st:en]), label=r'$\Omega_{38} =$' + str(f38))
        plt.plot(t_ps[st:en], np.abs(oX1eig[1, 3]) * np.abs(psi13[st:en]), label=r'$\Omega_{13} =$' + str(f13))

        #plt.ylim(-0.005,0.02)
        
        plt.legend(bbox_to_anchor=([1, 1]), title="Oscillatory Frequencies")
        plt.title(r'$\omega_2$ = ' + np.str(np.round(self.w2, decimals=2)) + ' $\omega_1$ = ' + np.str(
            np.round(self.w1, decimals=2)))  # $\omega=1530cm^{-1}$')

        # plt.savefig('Eigcoherences_1p75g_w2_1113',bbox_inches='tight',dpi=600)

    def energy_transfer(self):
        plt.figure(7)

        Iv = self.identity_vib()
        b = self.destroy()
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        oX2 = oB2 + oB2.getH()
        oX1 = oB1 + oB1.getH()

        st = 0000
        en = 13000  # P_el.shape[2]
        itvl = 3    #time interval?


        tmax_ps = 4
        tmax = tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  # 3 end time
        # dt = ((2 * constant.pi) / self.omega) / 100  # 0.0001 # time steps at which we want to record the data. The solver will
                                                        # automatically choose the best time step for calculation.
        steps = np.int((tmax - self.t0) / self.dt)  # total number of steps. Must be int.

        # rhoT, t = self.time_evol_me(tmax_ps)

        t_cm = self.t / (2 * constant.pi)
        t_ps = (t_cm * 1e12) / (100 * constant.c)

        #COULD MOVE THESE INTO INIT
        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE1E2 = self.exciton_operator(1, 2)

        oE1mImI = sp.kron(oE1, sp.kron(Iv, Iv)).tocsr()
        oE2mImI = sp.kron(oE2, sp.kron(Iv, Iv)).tocsr()
        oE1E2mImI = sp.kron(oE1E2, sp.kron(Iv, Iv)).tocsr()

        M1thermal = self.thermal(self.w1)
        M2thermal = self.thermal(self.w2)
        oE2 = sp.kron(self.E2, self.E2.getH()).tocsr()
        P0 = sp.kron(oE2, sp.kron(M1thermal, M2thermal)).todense()

        ex1 = np.zeros((steps))
        for i in np.arange(steps):
            ex1[i] = np.real(np.trace(oE1mImI.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

        ex2 = np.zeros((steps))
        for i in np.arange(steps):
            ex2[i] = np.real(np.trace(oE2mImI.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

        ex12 = np.zeros((steps))
        for i in np.arange(steps):
            ex12[i] = np.abs(np.trace(oE1E2mImI.dot(self.rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

        x1 = self.oper_evol(oX1, self.rhoT, self.t,  tmax_ps)
        x2 = self.oper_evol(oX2, self.rhoT, self.t,  tmax_ps)  # can also pass a time step if necessary
        elta = np.int(np.round(((2 * constant.pi) / self.omega) / self.dt))
        c_X12 = self.corrfunc(x1, x2, elta)


        print(t_ps.shape)
        print(c_X12.shape)
        print(np.arange(0,en,itvl).shape)
        print(np.arange(0,en,itvl).shape)

        plt.plot(t_ps[0:en], ex1[0:en], label=r'$|E_{1}\rangle\langle E_{1}|$')
        plt.plot(t_ps[0:en], ex2[0:en], label=r'$|E_{2}\rangle\langle E_{2}|$')
        plt.plot(t_ps[0:en], ex12[0:en], label=r'$||E_{1}\rangle\langle E_{2}||$')
        plt.plot(t_ps[np.arange(0,en,itvl)],c_X12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')

        plt.xlabel('Time ($ps$)')
        # plt.xlim([0,5])
        plt.grid()
        # plt.legend(bbox_to_anchor=[1,1])
        plt.legend()

    def fourier(self):

        H= self.hamiltonian()
        vals, eigs = np.linalg.eigh(H.todense())

        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()
        b = self.destroy()

        M1thermal = self.thermal(self.w1)
        M2thermal = self.thermal(self.w2)
        oE2 = sp.kron(self.E2, self.E2.getH()).tocsr()
        P0 = sp.kron(oE2, sp.kron(M1thermal, M2thermal)).todense()
        P0eig = eigs.getH() * P0 * eigs
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()

        oX1 = oB1 + oB1.getH()
        oX2 = oB2 + oB2.getH()
        oX1eig = eigs.getH() * oX1 * eigs
        oX2eig = eigs.getH() * oX2 * eigs
        
        coefx1 = np.multiply(oX1eig, P0eig)
        coefx2 = np.multiply(oX2eig, P0eig)
        
        coefx1chop = np.tril(coefx1,k=-1)
        coefx2chop = np.tril(coefx2,k=-1)
        
        tmax_ps = 4
        tmax = tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  # 3 end time
        # dt = ((2 * constant.pi) / self.omega) / 100  # 0.0001 # time steps at which we want to record the data. The solver will
                                                        # automatically choose the best time step for calculation.
        steps = np.int((tmax - self.t0) / self.dt)  # total number of steps. Must be int.
        # rhoT, t = self.time_evol_me(tmax_ps)
        t_cm = self.t / (2 * constant.pi)

        
        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
        2 * N ** 2, 2 * N ** 2).transpose()
        omegachop = np.tril(omegaarray,k=-1)

        count4 = time.time()

        sampleratecm = 1/(t_cm[1]-t_cm[0])
        freqres1 = 0.5
        ftlen = (t_cm[1]-t_cm[0])*np.arange(int(sampleratecm/freqres1))
        anaX1array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])
        anaX2array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])
        anaX1 = np.zeros(np.size(ftlen))
        anaX2 = np.zeros(np.size(ftlen))

        for a in np.arange(np.size(ftlen)):
            anaX1array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefx1chop)
            anaX1[a] = np.sum(anaX1array[:,:,a]) + np.trace(coefx1)
            anaX2array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefx2chop)
            anaX2[a] = np.sum(anaX2array[:,:,a]) + np.trace(coefx2)

        freqres2 = 0.1
        pads = int((sampleratecm/freqres2)-np.shape(anaX1)[0]) #30000
        x1pad = np.append(anaX1,np.zeros(pads))
        x2pad = np.append(anaX2,np.zeros(pads))
        fr10 = np.fft.rfft(x1pad,int(np.size(x1pad)))
        fr20 = np.fft.rfft(x2pad,int(np.size(x2pad)))
        freq_cm0 = sampleratecm*np.arange(0,1-1/np.size(x1pad),1/np.size(x1pad))

        count5 = time.time()
        print('Measurements =',count5-count4)

        st = int(1100/(sampleratecm/np.size(x1pad)))
        en = int(1130/(sampleratecm/np.size(x1pad)))

        plt.figure(30)
        plt.plot(freq_cm0[st:en],np.real(fr10)[st:en],label=r'$\langle X_1\rangle$')
        plt.plot(freq_cm0[st:en],np.real(fr20)[st:en],label=r'$\langle X_2\rangle$')
        plt.ylabel('Real Part of FT')
        plt.xlabel('Frequency ($cm^{-1}$)')
        plt.legend()
        plt.grid(True,which='both')
        plt.minorticks_on()
        plt.yticks([0])
        plt.title(r'Components of $\langle X\rangle$ at $T=2ps$')

if __name__ == "__main__":
  
    # print(foo.FooClass.blah())
    #plot = plots.Plots(rate_swap=True)
    # tmax_ps = 4
    plot = Plots(rate_swap=True)

    #plot.sync_evol()
    plot.coherences()
    #plot.energy_transfer()
    #plot.fourier()
    plt.show()