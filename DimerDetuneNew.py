
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
import cmath
import scipy.sparse as sp
import scipy.linalg as sl
import matplotlib.pylab as plt
import QCcorrelations as QC
from Plots import Plots

class DimerDetune:
    """Defines properties and functions of the vibronic dimer system"""
    def __init__(self, hamiltonian,  rate_swap, n_cutoff=5, temperature=298):
        #initialise properties of dimer 
        self.hamiltonian = hamiltonian 
        self.n_cutoff =n_cutoff
        self.temperature =  temperature

        #unsure
        self.huang = 0.0578
        #initialise with no detuning
        self.omega = 1111
        self.detuning = 1
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

        # Exciton Vectors - Exciton Basis & site vectors in site basis
        self.E1 = sp.lil_matrix(np.matrix([[1.], [0.]])).tocsr()
        self.E2 = sp.lil_matrix(np.matrix([[0.], [1.]])).tocsr()

        self.t0 = 0
        self.dt = ((2 * constant.pi) / self.omega) / 100

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

        H = sp.kron((self.dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
            + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
            + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
            + self.g1 * sp.kron(eldis1, sp.kron((b + b.getH()), Iv)) \
            + self.g2 * sp.kron(eldis2, sp.kron(Iv, (b + b.getH())))

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
        phi1 = 0
        phi2 = 0

            #sp.kron(self.e1 * oe11, Iv) + sp.kron(self.e2 * oe22, Iv)\
        H=  sp.kron((self.de / 2) * sigmaz, sp.kron(Iv, Iv)) \
            + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
            + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
            + self.g1 * sp.kron(sigmax, sp.kron(cmath.exp(1j*phi1) * b + cmath.exp(-1j*phi1) * b.getH(), Iv)) \
            + self.g2 * sp.kron(sigmax, sp.kron(Iv, cmath.exp(1j*phi2) * b + cmath.exp(-1j*phi2) * b.getH()))
            
        return H

    def liouvillian(self):
        if(self.hamiltonian == "militello"):
            H = self.militello_hamiltonian()
        else:
            H = self.original_hamiltonian()
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



if __name__ == "__main__":
  
    # tmax_ps = 4
    plot = Plots(hamiltonian="militello", rate_swap=False)

    plot.sync_evol()
    plot.coherences()
    plot.energy_transfer()
    #plot.fourier()
    #plot.test()
    #plot.q_correlations()
   
    plt.show()