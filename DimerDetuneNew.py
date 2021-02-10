
"""
Code for two coupled dimer chromophores, strongly coupled to modes.
Tensor order is excitonic, mode 1, mode 2.
@author: Charlie Nation, Pip Benjamin
based on code by stefan siwiak-jaszek

___________________________________________________________________________________________________________________

To do:

Replace np.matrix (depreciated) with np.array throughout - this doesn't have a getH method, so will need
.conj().transpose() I think
"""
import numpy as np
import scipy.constants as constant
import scipy.integrate as integrate
import time
import cmath
import scipy.sparse as sp
import scipy.linalg as sl
import matplotlib.pylab as plt
import QCcorrelations as QC
import pandas
# from Plots import Plots

class DimerDetune:
    """Defines properties and functions of the vibronic dimer system"""
    def __init__(self, hamiltonian, r_th, r_el, phi1, phi2, detuning, n_cutoff, temperature, tmax_ps):
        """Initialise variables assosciated with the dimer system
        Arguments:
            self - instance of DimerDetune class
            hamiltonian - options are 'original' and 'militello'
            phi1, phi2 - synchronisation phase parameters for militello hamiltonian
            thermal_dissipation, electronic_dephasing - rates in ps^-1
            n_cutoff - cutoff for vibrational mode maximum 
            temperature"""
        #initialise properties of dimer 
        self.hamiltonian = hamiltonian 
        self.n_cutoff =n_cutoff
        self.temperature =  temperature

        #unsure
        self.huang = 0.0578
        #initialise with no detuning
        self.omega = 1111
        self.detuning = detuning
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
        
        #NEW- input in [ps]^-1
        self.r_th = 1/(r_th* 1e-12 * 100 * constant.c * 2*constant.pi )
        self.r_el = 1/(r_el*1e-12 * 100 * constant.c * 2*constant.pi )

        self.r_v1 = self.r_th  # 6 # cm-1
        self.r_v2 = self.r_th  # 6 # cm-1

        #hilbert space defined according to max number of vibrational modes
        self.N = n_cutoff  # 7
        self.dimH = 2 * n_cutoff**2  # Hilbert space dimension

        # Exciton Vectors - Exciton Basis & site vectors in site basis
        self.E1 = sp.lil_matrix(np.matrix([[1.], [0.]])).tocsr()
        self.E2 = sp.lil_matrix(np.matrix([[0.], [1.]])).tocsr()

        self.t0 = 0
        self.dt = ((2 * constant.pi) / self.omega) / 100
        self.tmax_ps = tmax_ps
        tmax = self.tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12
        self.steps = np.int((tmax - self.t0) / self.dt)  # total number of steps. Must be int.

        self.phi1 = phi1
        self.phi2 = phi2

    def electron_operator(self, e1, e2):
        """returns |e_1><e_2|"""
        if e1 == 1:
            e_1 = self.E1
        elif e1 == 2:
            e_1 = self.E2
        else:
            raise Exception('e1 should be 1 or 2 for |e_1> or |e_2>')
        if e2 == 1:
            e_2 = self.E1
        elif e2 == 2:
            e_2 = self.E2
        else:
            raise Exception('e2 should be 1 or 2 for <e_1| or <e_2|')
        return sp.kron(e_1, e_2.getH())


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
        return sp.kron(mode1,mode2.getH())

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
        """|e><e|"""
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

    def original_hamiltonian(self):
        """dE/2 sigma_z + omega_1 n_1 + omega_2 n_2 + g_1 |1><1| (b_1 + b^dag_1) + g_2 |2><2| (b_2 + b^dag_2 )
        |1(2)><1(2)| is the site basis ground (excited) state"""
        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        sigmaz = oE2 - oE1   # note that this is backwards compared to definition in quantum information approaches
        b = self.destroy() #relation
        Iv = self.identity_vib()
        eldis1 = self.a_k(1)
        eldis2 = self.a_k(2)
        Ie = sp.eye(2, 2).tocsr()

        # if(self.phase_change == True):


        H = sp.kron((self.dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
            + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
            + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
            + self.g1 * sp.kron(eldis1, sp.kron(cmath.exp(1j*self.phi1)*b + cmath.exp(-1j*self.phi1)*b.getH(), Iv)) \
            + self.g2 * sp.kron(eldis2, sp.kron(Iv, cmath.exp(1j*self.phi2) * b + cmath.exp(-1j*self.phi2)*b.getH()))    

            
        # else:
        #     H = sp.kron((self.dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
        #         + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
        #         + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
        #         + self.g1 * sp.kron(eldis1, sp.kron((b + b.getH()), Iv)) \
        #         + self.g2 * sp.kron(eldis2, sp.kron(Iv, (b + b.getH())))

        return H

    def militello_hamiltonian(self):
        oe11 = self.electron_operator(1,1)
        oe12 = self.electron_operator(1,2)
        oe22 = self.electron_operator(2,2)
        oe21 = self.electron_operator(2,1)
        sigmaz = oe22 - oe11
        sigmax = oe21 +oe12
        b = self.destroy()
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()

        H=  sp.kron((self.de / 2) * sigmaz, sp.kron(Iv, Iv)) \
            + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
            + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
            + self.g1 * sp.kron(sigmax, sp.kron(cmath.exp(-1j*self.phi1) * b + cmath.exp(1j*self.phi1) * b.getH(), Iv)) \
            + self.g2 * sp.kron(sigmax, sp.kron(Iv, cmath.exp(-1j*self.phi2) * b + cmath.exp(1j*self.phi2) * b.getH()))
            
        return H

    def liouvillian(self):
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()
        ELdis1 = sp.kron(self.a_k(1), sp.kron(Iv, Iv)).tocsr()
        ELdis2 = sp.kron(self.a_k(1), sp.kron(Iv, Iv)).tocsr()
        b = self.destroy()
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        nw1 = 1 / (np.exp(self.w1 / self.kBT) - 1)  # thermal distribution
        nw2 = 1 / (np.exp(self.w2 / self.kBT) - 1)

        if(self.hamiltonian == "militello"):
            H = self.militello_hamiltonian()
            L = -1j * self.liouv_commutator(H) + self.dissipator(sp.kron(self.electron_operator(1,2),sp.kron(Iv,Iv)).tocsr() , 0.2*self.de)
            
        else:
            H = self.original_hamiltonian()
            L = -1j * self.liouv_commutator(H) + self.dissipator(ELdis1, self.r_el) + self.dissipator(ELdis2, self.r_el) \
            + self.dissipator(oB1, self.r_v1 * (nw1 + 1)) + self.dissipator(oB1.getH(), self.r_v1 * nw1) \
            + self.dissipator(oB2, self.r_v2 * (nw2 + 1)) + self.dissipator(oB2.getH(), self.r_v2 * nw2)
        
        return L

    def init_state(self):
        oE2 = self.exciton_operator(2, 2)
        oe1  = self.electron_operator(1,1)
        Iv = self.identity_vib()
        if(self.hamiltonian == "militello"):
            #rho0 = sp.kron(oe1, sp.kron(Iv,Iv)).todense()
            rho0 = sp.kron(oe1, sp.kron(self.thermal(self.w1), self.thermal(self.w2))).todense()
        else:
            rho0 = sp.kron(oE2, sp.kron(self.thermal(self.w1), self.thermal(self.w2))).todense()
        return rho0

class Operations(DimerDetune):

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
        # tmax = tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12
        # steps = np.int((tmax - self.t0) / self.dt)  # total number of steps. Must be int.

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
        t = np.zeros(self.steps)

        t[0] = self.t0
        rhoT = np.zeros((self.steps, rho0.shape[1] ** 2), dtype=complex)  # time is 3rd dim.
        rhoT[0, :] = rho0_l
        # now do the iteration.
        k = 1
        while evo.successful() and k < self.steps:
            evo.integrate(evo.t + self.dt)  # time to integrate at at each loop
            t[k] = evo.t  # save current loop time
            rhoT[k, :] = evo.y  # save current loop data
            k += 1  # keep index increasing with time.
        count2 = time.time()
        print('Integration =', count2 - count1)

        return rhoT, t

    def oper_evol(self, operator, rhoT, t, tmax_ps):
        """calculates the time evolution of an operator"""
        #steps = len(rhoT[:, 0])
        N = int(np.sqrt(len(rhoT[0, :])))
        oper = np.zeros(self.steps, dtype=complex)
        for i in np.arange(self.steps):
            #may need a complex verison with np.real here?
            oper[i] = np.trace(operator.dot(rhoT[i, :].reshape(N, N)))
        return oper

    @staticmethod
    def corrfunc(f1, f2, delta):
        f1bar = np.zeros(np.size(f1) - delta, dtype =complex)
        f2bar = np.zeros(np.size(f1) - delta, dtype=complex)
        df1df2bar = np.zeros(np.size(f1) - 2 * delta,dtype =complex)
        df1sqbar = np.zeros(np.size(f1) - 2 * delta,dtype =complex)
        df2sqbar = np.zeros(np.size(f1) - 2 * delta,dtype =complex)
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

    def matrix_element(self,bra, operator, ket):
        return np.real(bra.getH() * operator * ket)

class Plots(Operations):

    def __init__(self, hamiltonian, r_th, r_el, phi1, phi2, detuning, j_k, save_plots, n_cutoff, temperature, tmax_ps):
        DimerDetune.__init__(self, hamiltonian, r_th, r_el, phi1, phi2, detuning,  n_cutoff, temperature, tmax_ps)
        if(hamiltonian == "militello"):
            self.H = self.militello_hamiltonian()
        else:
            self.H = self.original_hamiltonian()
        self.tmax_ps = tmax_ps
        self.rhoT , self.t  = self.time_evol_me(self.tmax_ps)

        b = self.destroy()
        self.Iv = self.identity_vib()
        self.Ie = sp.eye(2, 2).tocsr()
        
        self.oB1 = sp.kron(self.Ie, sp.kron(b, self.Iv)).tocsr()
        self.oB2 = sp.kron(self.Ie, sp.kron(self.Iv, b)).tocsr()
        self.oX2 = self.oB2 + self.oB2.getH()
        self.oX1 = self.oB1 + self.oB1.getH()

        
        #time evo of operators, REPRODUCE THIS IN COHERENCES
        self.x1 = self.oper_evol(self.oX1,self.rhoT, self.t, self.tmax_ps)
        self.x2 = self.oper_evol(self.oX2,self.rhoT, self.t, self.tmax_ps)  # can also pass a time step if necessary

        self.t_cm = self.t / (2 * constant.pi)
        self.t_ps = (self.t_cm * 1e12) / (100 * constant.c)

        self.elta = np.int(np.round(((2 * constant.pi) / self.omega) / self.dt))
        self.c_X12 = self.corrfunc(self.x1, self.x2, self.elta)
        self.save_plots = save_plots
        self.j_k = j_k


    def sync_evol(self):

        fig = plt.figure(1)
        en = self.steps -200
        st = 0000
        itvl = 5

        axA = fig.add_subplot(111)
        #axA.set_xlim(0,4)
        print(self.t_ps[np.arange(st, en, itvl)])
        axA.plot(self.t_ps[np.arange(st, en, itvl)], self.x2[np.arange(st, en, itvl)], label=r'$\langle X_2\rangle$')
        axA.plot(self.t_ps[np.arange(st, en, itvl)], self.x1[np.arange(st, en, itvl)], label=r'$\langle X_1\rangle$')
        axA.set_ylabel('$<X_i>$', fontsize =13)
        axA.set_xlabel('Time (ps)', fontsize =13)

        axB = axA.twinx()
        axB.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        axB.plot(self.t_ps[np.arange(st, en, itvl)], self.c_X12[np.arange(st, en, itvl)], 'r-o', markevery=0.05, markersize=5,
                 label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')
        
        fig.legend(loc=(0.6,0.6))
        fig.show()

        if(self.save_plots == True):
            fig.savefig('sync_evol.png',bbox_inches='tight',dpi=600)


    def matrix_elements(self):
        vals, eigs = np.linalg.eigh(self.H.todense())
        j_k = self.j_k
        
        #elec
        oe12 = self.electron_operator(1,2)    
        oe21 = self.electron_operator(2,1)
        sigmax = oe21 + oe12
        Ie = sp.eye(2, 2).tocsr()

        #vib
        vib_gs_op = self.vib_mode_operator(0, 0)
        Iv = self.identity_vib()
        
       
        columns = []
        for i in range(len(j_k)):
            columns.append(str(j_k[i]))

        rows = ["X1", "X2", "sigmax", "vib gs"]
        
        oX1_kj = np.zeros(len(j_k))
        oX2_kj = np.zeros(len(j_k))
        sigmax_kj = np.zeros(len(j_k))
        gs_kj = np.zeros(len(j_k))

        for i in range(len(j_k)):
            j = j_k[i][0]
            k = j_k[i][1]

            oX1_kj[i] = self.matrix_element(eigs[:, k], self.oX1, eigs[:, j])
            oX2_kj[i] = self.matrix_element(eigs[:, k], self.oX2, eigs[:, j])
            sigmax_kj[i] = self.matrix_element(eigs[:, k], sp.kron(sigmax, sp.kron(Iv, Iv)), eigs[:, j])
            gs_kj[i] = self.matrix_element(eigs[:, k], sp.kron(Ie, sp.kron(vib_gs_op,vib_gs_op)), eigs[:, j])

        print("sigmax = ", sigmax)
    
        elements = np.stack((oX1_kj, oX2_kj, sigmax_kj, gs_kj))
        df = pandas.DataFrame(data = elements, index = rows, columns=columns)
        print(df)

    def coherences(self):
        """Complex magnitude of exciton-vibration coherences scaled by the absolute value
        of their corresponding position matrix element in the open quantum system evolution."""

        vals, eigs = np.linalg.eigh(self.H.todense())
        oX1eig = eigs.getH() * self.oX1 * eigs
        j_k = self.j_k

        fig = plt.figure(2)
        axA = fig.add_subplot(111)
        axA.set_xlabel('Time ($ps$)', fontsize =13)
        axA.set_ylabel('$|X_{i,jk}| ||\\rho_{jk}(t)||$', fontsize =13)
        axA.grid()

        st = 0000
        en = self.steps -200
        itvl = 3 

        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2).transpose()
        
        for i in range(len(j_k)):
            j = j_k[i][0]
            k = j_k[i][1]
            f = np.round(np.abs(omegaarray[j, k]), decimals=2)
            opsi_jk = np.kron(eigs[:, j], eigs[:, k].getH())
            psi_jk = self.oper_evol(opsi_jk,self.rhoT, self.t, self.tmax_ps)

            plt.plot(self.t_ps[st:en], np.abs(oX1eig[j, k]) * np.abs(psi_jk[st:en]), label="$\Omega_{" +str(j)+str(k) +"} =$" + str(f)) 

        #plt.xlim(0,2.5)
        # plt.ylim(0,0.05)
        plt.legend(bbox_to_anchor=([0.5, 1]), title="Oscillatory Frequencies / $cm^-1$", fontsize =13)

        # sync plot
        # axB = axA.twinx()
        # axB.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        # axB.plot(self.t_ps[np.arange(st, en, itvl)], self.c_X12[np.arange(st, en, itvl)], 'c-o', markevery=0.05, markersize=5,
        #          label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')

        if(self.save_plots == True):
            fig.savefig('Eigcoherences.png',bbox_inches='tight',dpi=600)
        fig.show()

    def energy_transfer(self):
        fig = plt.figure(3)

        st = 0000
        en = self.steps -200
        itvl = 3  

        #COULD MOVE THESE INTO INIT
        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE1E2 = self.exciton_operator(1, 2)

        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        oE2mImI = sp.kron(oE2, sp.kron(self.Iv, self.Iv)).tocsr()
        oE1E2mImI = sp.kron(oE1E2, sp.kron(self.Iv, self.Iv)).tocsr()

        ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)
        ex2 = self.oper_evol(oE2mImI,self.rhoT, self.t, self.tmax_ps)
        ex12 = self.oper_evol(oE1E2mImI,self.rhoT, self.t, self.tmax_ps)

        axA = fig.add_subplot(111)
        axA.plot(self.t_ps[0:en], ex1[0:en], label=r'$|E_{1}\rangle\langle E_{1}|$')
        axA.plot(self.t_ps[0:en], ex2[0:en], label=r'$|E_{2}\rangle\langle E_{2}|$')
        #axA.plot(self.t_ps[0:en], ex12[0:en], label=r'$|E_{1}\rangle\langle E_{2}|$')
        
        axC = axA.twinx()
        axC.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        #axC.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        axC.plot(self.t_ps[np.arange(st, en, itvl)], self.c_X12[np.arange(st, en, itvl)], 'c-o', markevery=0.05, markersize=5,
                 label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')
    
        #plot coherences
        #axB = axA.twinx()
        #axB.set_ylabel('$|X_{i,jk}| ||\\rho_{jk}(t)||$', fontsize =13)        

        # vals, eigs = np.linalg.eigh(self.H.todense())
        # oX1eig = eigs.getH() * self.oX1 * eigs

        # N= self.n_cutoff
        # omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2).transpose()

        # n_m = [] #coherences to plot

        # for i in range(len(n_m)):
        #     n = n_m[i][0]
        #     m = n_m[i][1]
        #     f = np.round(np.abs(omegaarray[n, m]), decimals=2)
        #     opsi_nm = np.kron(eigs[:, n], eigs[:, m].getH())
        #     psi_nm = self.oper_evol(opsi_nm,self.rhoT, self.t, self.tmax_ps)

        #     axB.plot(self.t_ps[st:en], np.abs(oX1eig[n, m]) * np.abs(psi_nm[st:en]), label="$\Omega_{" +str(n)+str(m) +"} =$" + str(f)) 
    
        #plt.xlim(0,4)
        fig.legend(loc=(0.6,0.6), fontsize =13)
        axA.set_xlabel('Time ($ps$)', fontsize =13)
        axA.grid(axis='x')
        fig.show()
        if(self.save_plots == True):
            fig.savefig('ET_withsync.png',bbox_inches='tight',dpi=600)

    def fourier(self):
        vals, eigs = np.linalg.eigh(self.H.todense())

        M1thermal = self.thermal(self.w1)
        M2thermal = self.thermal(self.w2)
        oE2 = sp.kron(self.E2, self.E2.getH()).tocsr()
        P0 = sp.kron(oE2, sp.kron(M1thermal, M2thermal)).todense()
        P0eig = eigs.getH() * P0 * eigs

        oX1eig = eigs.getH() * self.oX1 * eigs
        oX2eig = eigs.getH() * self.oX2 * eigs
        
        coefx1 = np.multiply(oX1eig, P0eig)
        coefx2 = np.multiply(oX2eig, P0eig)
        
        coefx1chop = np.tril(coefx1,k=-1)
        coefx2chop = np.tril(coefx2,k=-1)
     
        tmax = self.tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  # 3 end time
        
        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
        2 * N ** 2, 2 * N ** 2).transpose()
        omegachop = np.tril(omegaarray,k=-1)

        count4 = time.time()

        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        freqres1 = 0.5
        ftlen = (self.t_cm[1]-self.t_cm[0])*np.arange(int(sampleratecm/freqres1))
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

        fig = plt.figure(5)
        plt.plot(freq_cm0[st:en],np.real(fr10)[st:en],label=r'$\langle X_1\rangle$')
        plt.plot(freq_cm0[st:en],np.real(fr20)[st:en],label=r'$\langle X_2\rangle$')
        plt.ylabel('Real Part of FT', fontsize =13)
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        plt.legend()
        plt.grid(True,which='both')
        plt.minorticks_on()
        plt.yticks([0])
        plt.title(r'Components of $\langle X\rangle$ at $T=2ps$', fontsize =13)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('FT_original.png',bbox_inches='tight',dpi=600)


    def q_correlations(self):

        ##put these in init
        M1thermal = self.thermal(self.w1)
        M2thermal = self.thermal(self.w2)

        oE2 = sp.kron(self.E2, self.E2.getH()).tocsr()
        P0 = sp.kron(oE2, sp.kron(M1thermal, M2thermal)).todense()
        print("shape(P0) = ", np.shape(P0))

        counta = time.time()

        q_mutual = []
        c_info = []
        q_discord = []
        corr_times = []

        #dtperps = (100 * constant.c * 2 * constant.pi * 1e-12) / self.dt
        maxstep = np.shape(self.rhoT)[0] #np.int(np.round(12*dtperps))
        N= self.n_cutoff
                            #maxstep
        for i in np.arange(0,maxstep,2000):

            test_matrix = self.rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1])
            quantum_mutual_info, classical_info, quantum_discord = QC.correlations(test_matrix, 2, N, N, 1, 2)
            q_mutual.append(quantum_mutual_info)
            c_info.append(classical_info)
            q_discord.append(quantum_discord)
            corr_times.append(self.t_ps[i])
            print(i)

        q_mutual = np.array(q_mutual)
        c_info = np.array(c_info)
        q_discord = np.array(q_discord)
        corr_times = np.array(corr_times)

        countb = time.time()

        print('Quantum Correlation Measures =',countb-counta)

        #QUANTUM PLOT
        fig = plt.figure(6)

        en = self.steps -200
        st = 000

        itvl = 5
        axA = fig.add_subplot(111)
        axA.plot(corr_times,c_info,label=r'Classical Info')
        axA.plot(corr_times,q_mutual,label=r'Q Mutual Info')
        axA.plot(corr_times,q_discord,label=r'Discord')
        axA.set_xlabel('Time (ps)', fontsize =13)
        #axA.set_xlim([0,4])

        axB = axA.twinx()
        axB.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        axB.plot(self.t_ps[np.arange(st,en,itvl)],self.c_X12[np.arange(st,en,itvl)],'r-o',markevery=0.05,markersize=5,label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')
        
        print(np.arange(st,en,itvl))
        
        #axB.grid()
        #axA.grid()
        axB.legend(bbox_to_anchor=([0.3,0.8]), fontsize =13)
        axA.legend(bbox_to_anchor=([0.9,0.8]), fontsize =13)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('Q_Correlations.png',bbox_inches='tight',dpi=600)

    def test(self):
        print("oX1= ", np.shape(self.oX1.toarray()))
        oe12 = self.electron_operator(1,2)       
        oe21 = self.electron_operator(2,1)
        sigmax = oe21 + oe12
        print("sigmax= ", sigmax.toarray())
        print("dt = ", self.dt)
        print("t_ps shape = ", np.shape(self.t_ps))
        print("rhoT size = ", np.shape(self.rhoT))
        print("c_X12 size  = ", np.shape(self.c_X12))
        dtperps = (100 * constant.c * 2 * constant.pi * 1e-12) / self.dt
        print("dtperps = ", dtperps)
        print("dephrate = ",self.r_el)
        print("dissrate = ",self.r_th)



if __name__ == "__main__":

    #j_k = [[1,3],[0,1],[0,2],[0,3]]
    #j_k = [[2,1],[1,1],[0,1],[0,2],[0,3],[1,4],[1,5],[3,7],[3,8],[1,3]]
    j_k = [[0,1],[0,2],[0,3],[1,4],[1,5],[3,7],[3,8],[1,3]] #originals
    #j_k = [[0,4],[0,5],[0,6],[1,2],[2,1],[2,3],[2,2],[1,1]]
    #j_k = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6]]
    #j_k = [[2,1],[1,1],[0,1],[0,2],[0,3],[1,4],[1,5],[3,7],[3,8],[1,3]]
    #j_k = [[1,3],[0,1],[0,2],[0,3]]

    #all in range
    # j_k = []
    # for j in range(4):
    #     for k in range(4):
    #         if(j!=k):
    #             j_k.append([j,k])
    # print(j_k)


    #original  r_th =[1ps]^-1, r_el = [0.1ps]^-1
    plot = Plots(hamiltonian="original", r_th =0.1, r_el =1, phi1 = 0 , phi2 =0, detuning =1, j_k=j_k, save_plots = True, n_cutoff=5, temperature=298, tmax_ps = 30)
    plot.test()
    plot.matrix_elements()
    plot.sync_evol()
    plot.coherences()
    plot.energy_transfer()
    # #plot.fourier()
    
    # plot.q_correlations()
    plt.show()

    