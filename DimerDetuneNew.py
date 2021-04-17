
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
from scipy.fft import fft, ifft, rfft, fftfreq
import matplotlib
import matplotlib.pylab as plt
import QCcorrelations as QC
import pandas
# from Plots import Plots

class DimerDetune:
    """Defines properties and functions of the vibronic dimer system"""
    def __init__(self, r_th, r_el, phi1, phi2, detuning, n_cutoff, temperature, tmax_ps):
        """Initialise variables assosciated with the dimer system
        Arguments:
            self - instance of DimerDetune class
            hamiltonian - options are 'original' and 'militello'
            phi1, phi2 - synchronisation phase parameters for militello hamiltonian
            thermal_dissipation, electronic_dephasing - rates in ps^-1
            n_cutoff - cutoff for vibrational mode maximum 
            temperature"""
        #initialise properties of dimer 
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
        self.tmax = self.tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12
        self.steps = np.int((self.tmax - self.t0) / self.dt)  # total number of steps. Must be int.

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
        """returns |E_1><E_2|"""
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
        """U|e><e|U"""
        return self.rotate() * self.electron_operator(k, k) * self.rotate().getH()

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
        #not sure about this
        # sigmaz = self.rotate() * sigmaz_electron * self.rotate().getH()
        b = self.destroy() #relation
        Iv = self.identity_vib()
        eldis1 = self.a_k(1)
        eldis2 = self.a_k(2)
        Ie = sp.eye(2, 2).tocsr()

        H = sp.kron((self.dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
            + sp.kron(Ie, sp.kron(self.w1 * b.getH() * b, Iv)) \
            + sp.kron(Ie, sp.kron(Iv, self.w2 * b.getH() * b)) \
            + self.g1 * sp.kron(eldis1, sp.kron(cmath.exp(1j*self.phi1)*b + cmath.exp(-1j*self.phi1)*b.getH(), Iv)) \
            + self.g2 * sp.kron(eldis2, sp.kron(Iv, cmath.exp(1j*self.phi2) * b + cmath.exp(-1j*self.phi2)*b.getH()))    
        
        return H

    def collective_hamiltonian(self):
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()

        I = sp.kron(Ie, sp.kron(Iv,Iv)).tocsr()
        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE12 = self.exciton_operator(1, 2)
        oE21 = self.exciton_operator(2, 1)
        sigmaz = oE2 - oE1
        #sigmaz_hamiltonian =  sp.kron(sigmaz, sp.kron(Iv, Iv)).tocsr()
        sigmax = oE12 + oE21

        #SORT OUT THESE SIGMAZ DEFINITIONS
        self.theta = 0.5 * np.arctan(2 * self.V / self.de)
        Usigmaz_e = np.sin(2*self.theta) * sigmax +  np.cos(2*self.theta) * sigmaz
        Usigmaz = sp.kron(Usigmaz_e, sp.kron(Iv, Iv)).tocsr()

        b = self.destroy()
        ob1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        ob2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        ob_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
        ob_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

        # 
        # \

        H = sp.kron((self.dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
            +self.w1 * (ob_rd.getH()*ob_rd + ob_cm.getH()*ob_cm) \
            -(self.g1/np.sqrt(2)) * (Usigmaz * (ob_rd + ob_rd.getH())) \
            +(self.g1/np.sqrt(2))* (ob_cm + ob_cm.getH())

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
        ELdis2 = sp.kron(self.a_k(2), sp.kron(Iv, Iv)).tocsr()
        b = self.destroy()
        oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        nw1 = 1 / (np.exp(self.w1 / self.kBT) - 1)  # thermal distribution
        nw2 = 1 / (np.exp(self.w2 / self.kBT) - 1)

        H = self.original_hamiltonian()
        L = -1j * self.liouv_commutator(H) + self.dissipator(ELdis1, self.r_el) + self.dissipator(ELdis2, self.r_el) \
        + self.dissipator(oB1, self.r_v1 * (nw1 + 1)) + self.dissipator(oB1.getH(), self.r_v1 * nw1) \
        + self.dissipator(oB2, self.r_v2 * (nw2 + 1)) + self.dissipator(oB2.getH(), self.r_v2 * nw2)

        fig = plt.figure(389)
        plt.title('Original Liouvillian')
        im =plt.imshow(np.real(L.todense()), cmap = 'rainbow')
        cb = fig.colorbar(im)
        
        return L

    def collective_liouvillian(self):
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()

        I = sp.kron(Ie, sp.kron(Iv,Iv)).tocsr()
        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE12 = self.exciton_operator(1, 2)
        oE21 = self.exciton_operator(2, 1)
        sigmaz_e = oE2 - oE1
        sigmaz =  sp.kron(sigmaz_e, sp.kron(Iv, Iv)).tocsr()
        sigmax_e = oE12 + oE21
        sigmax = sp.kron(sigmax_e, sp.kron(Iv, Iv)).tocsr()

        Theta_1 = 0.5*(I - np.sin(2*self.theta) * sigmax - np.cos(2*self.theta) * sigmaz)
        Theta_2 = 0.5*(I + np.sin(2*self.theta) * sigmax + np.cos(2*self.theta) * sigmaz)

        #vibrational
        B = 1 / (np.exp(self.w1 / self.kBT) - 1)
        b = self.destroy()
        ob1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        ob2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        b_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
        b_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

        H = self.collective_hamiltonian()
        L = -1j * self.liouv_commutator(H) + self.dissipator(b_cm.getH(), self.r_th*B) + self.dissipator(b_cm, self.r_th*(1+B)) \
            + self.dissipator(b_rd.getH(), self.r_th*B) + self.dissipator(b_rd, self.r_th*(B+1))  + self.dissipator(Theta_1, self.r_el) \
            + self.dissipator(Theta_2, self.r_el) 

        # fig = plt.figure(389)
        # plt.title('Collective Liouvillian')
        # im =plt.imshow(np.real(L.todense()), cmap = 'rainbow')
        # cb = fig.colorbar(im)

        return L

    def collective_liouvillian_BAD(self):
        """only defined for w1=w2"""
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()

        I = sp.kron(Ie, sp.kron(Iv,Iv)).tocsr()

        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE12 = self.exciton_operator(1, 2)
        oE21 = self.exciton_operator(2, 1)
        sigmaz = oE2 - oE1
        sigmaz_hamiltonian =  sp.kron(sigmaz, sp.kron(Iv, Iv)).tocsr()
        sigmax = oE12 + oE21
        

        #SORT OUT THESE SIGMAZ DEFINITIONS
        self.theta = 0.5 * np.arctan(2 * self.V / self.de)
        sigmaz = np.sin(2*self.theta) * sigmax +  np.cos(2*self.theta) * sigmaz
        sigmaz = sp.kron(sigmaz, sp.kron(Iv, Iv)).tocsr()
        sigmax = sp.kron(sigmax, sp.kron(Iv, Iv)).tocsr()
        print(self.a_k(1).toarray())
        print(self.a_k(2).toarray())

        print("sigmaz = ", sigmaz.toarray())

        #vibrational
        B = 1 / (np.exp(self.w1 / self.kBT) - 1)
        b = self.destroy()
        ob1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        ob2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        ob_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
        ob_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

        # L = (self.r_el/2 * ( sp.kron(sigmaz, sigmaz.transpose()) - sp.kron(I,I)) \

        L = ((self.r_el/2) * (np.cos(2*self.theta)**2 * sp.kron(sigmaz_hamiltonian, sigmaz_hamiltonian.transpose()) \
            + np.sin(2*self.theta)**2 * sp.kron(sigmax, sigmax.transpose()) \
            + 0.5 * np.sin(4*self.theta) * (sp.kron(sigmaz_hamiltonian, sigmax.transpose()) + sp.kron(sigmax, sigmaz_hamiltonian.transpose())) - sp.kron(I,I)) \

        + self.r_th * ((B*(sp.kron(ob_cm.getH(),ob_cm.transpose()) + (sp.kron(ob_rd.getH(),ob_rd.transpose()))) \
        + (1+B)*(sp.kron(ob_cm, ob_cm.conjugate()) + sp.kron(ob_rd,ob_rd.conjugate()))) \

        -(B/2) * (sp.kron(I, (ob_cm*ob_cm.getH() + ob_rd*ob_rd.getH()).transpose()) \
        + sp.kron((ob_cm*ob_cm.getH() + ob_rd*ob_rd.getH()), I) ) \
        
        -((1+B)/2) * (sp.kron(I, (ob_cm.getH()*ob_cm + ob_rd.getH()*ob_rd).transpose()) \
        + sp.kron((ob_cm.getH()*ob_cm + ob_rd.getH()*ob_rd), I) )) \

        +1j * ((self.dE/2) * (-sp.kron(sigmaz_hamiltonian, I) + sp.kron(I,sigmaz_hamiltonian.transpose())) \

        +self.w1 * (sp.kron(I, (ob_cm.getH()*ob_cm + ob_rd.getH()*ob_rd).transpose()) \
        - sp.kron((ob_cm.getH()*ob_cm + ob_rd.getH()*ob_rd), I) ) \

        +(self.g1/np.sqrt(2)) * (sp.kron(I, (-(ob_cm+ob_cm.getH()) - sigmaz * (ob_rd+ob_rd.getH())).transpose())
        + sp.kron(-(ob_cm+ob_cm.getH()) + sigmaz * (ob_rd+ob_rd.getH()), I) ) )).tocsr()
        
        return L
        
    def init_state(self):
        oE2 = self.exciton_operator(2, 2)
        rho0 = sp.kron(oE2, sp.kron(self.thermal(self.w1), self.thermal(self.w2))).todense()
        return rho0

class Operations(DimerDetune):

    def steady_state(self):
        #l = self.liouvillian()
        l=self.collective_liouvillian()
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
     
        # initial state
        rho0 = self.init_state()
        rho0_l = rho0.reshape(1, (rho0.shape[1]) ** 2)  # initial state in Liouville space
        print("rho0_l.shape = ", rho0_l.shape)
        # function to be integrated
        #L = self.liouvillian()
        L = self.collective_liouvillian()

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
        # print("rhoT = ", rhoT)
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

    #FOURIER TRANSFORM FUNCTIONS
    def padding(self, y_t, t, padding_multiplier):
        """pads oscillating function with zeros and extends time axis
            y_t - Oscillating Function
            t- time array
            padding multiplier (int)- pad the signal by a multiple of its length"""
        dt = t[1] - t[0]
        y_t_len = len(y_t)

        pad_value = y_t[y_t_len-1]
        #pad_value =0

        pad_size = padding_multiplier * y_t_len
        padded_y = pad_value * np.ones(pad_size)
        padded_t = np.zeros(pad_size)

        for i in range(y_t_len):
            padded_y[i] = y_t[i]
        
        for j in range(pad_size):
            padded_t[j] = j*dt

        return padded_y, padded_t

    def fft_time_to_freq(self,time):
        """Convert array of time values to frequencies"""
        dt = time[1] - time[0]
        N = time.shape[0]
        freq = 2.*np.pi*np.fft.fftfreq(N, dt)  #
        return np.append(freq[N//2:], freq[:N//2])

    def fft(self,y_t, t, return_frequencies=False):
        """return Fourier transform of an oscillating function"""
        n_times = len(t)
        ft = t[-1] * np.fft.fft(y_t, n_times)
        ft = np.append(ft[n_times // 2:], ft[:n_times // 2])
        if return_frequencies ==True:
            frequencies = self.fft_time_to_freq(t)
            return frequencies, ft
        else:
            return ft

    

class Plots(Operations):

    def __init__(self, r_th, r_el, phi1, phi2, detuning, j_k, save_plots, n_cutoff, temperature, tmax_ps):
        DimerDetune.__init__(self, r_th, r_el, phi1, phi2, detuning,  n_cutoff, temperature, tmax_ps)
    
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

        self.tmax_cm = self.tmax / (2 * constant.pi)
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
        axA.plot(self.t_ps[np.arange(st, en, itvl)], self.x1[np.arange(st, en, itvl)], '-', label=r'$\langle X_1\rangle$')
        
        
        axA.set_ylabel('$<X_i>$', fontsize =13)
        axA.set_xlabel('Time (ps)', fontsize =13)
        axA.grid(axis='x')
        axA.set_xlim(0,4)

        axB = axA.twinx()
        axB.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        axB.plot(self.t_ps[np.arange(st, en, itvl)], self.c_X12[np.arange(st, en, itvl)], 'r-o', markevery=0.05, markersize=5,
                 label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')
        
        fig.legend(loc=(0.6,0.2))
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
        #oX1eig = eigs.getH() * self.oX1 * eigs
        j_k = self.j_k

        fig = plt.figure(2)
        axA = fig.add_subplot(111)
        # axA.set_xlabel('Time ($ps$)', fontsize =13)
        # axA.set_ylabel('$|X_{i,kj}| ||\\rho_{jk}(t)||$', fontsize =13)
        axA.grid()
        axA.set_xlim(0,4)
        # plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20) 
        plt.rc('xtick', labelsize=20) 

        plt.xticks(fontsize= 15)
        plt.yticks(fontsize= 15)
        st = 0000
        en = self.steps -200
        itvl = 3 

        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2).transpose()
        oX1_kj = np.zeros(len(j_k))
        oX2_kj = np.zeros(len(j_k))

        pos_cohs = np.zeros(en)

        neg_cohs = np.zeros(en)

        for i in range(len(j_k)):
            j = j_k[i][0]
            k = j_k[i][1]
            oX1_kj[i] = self.matrix_element(eigs[:, k], self.oX1, eigs[:, j])
            oX2_kj[i] = self.matrix_element(eigs[:, k], self.oX2, eigs[:, j])

            f = np.round(np.abs(omegaarray[j, k]), decimals=2)
            opsi_jk = np.kron(eigs[:, j], eigs[:, k].getH())
            psi_jk = self.oper_evol(opsi_jk,self.rhoT, self.t, self.tmax_ps)
            
            #+ve sync
            if (oX1_kj[i] * oX2_kj[i] > 0):
                plt.plot(self.t_ps[st:en], np.abs(psi_jk[st:en]) * np.abs(oX1_kj[i]) , ls='-', label=r"$|\psi_{" +str(j)+r"}\rangle\langle\psi_{"+str(k) +r"}|$") 
                pos_cohs += np.abs(psi_jk[st:en] * np.abs(oX1_kj[i]))

            else:
                plt.plot(self.t_ps[st:en], np.abs(psi_jk[st:en]) * np.abs(oX1_kj[i]), ls='--', label=r"$|\psi_{" +str(j)+r"}\rangle\langle\psi_{"+str(k) +r"}|$")
                neg_cohs += np.abs(psi_jk[st:en] * np.abs(oX1_kj[i]))
            

        # plt.xlim(3.5,4)
        # plt.ylim(0,0.004)
        plt.legend(bbox_to_anchor=([0.5, 1]), title="Coherences", fontsize =13)

        # fig2 = plt.figure(255)
        # axA2 = fig.add_subplot(111)
        # plt.plot(self.t_ps[st:en], neg_cohs, ls='--', label="Negative")
        # plt.plot(self.t_ps[st:en], pos_cohs, label ="Positive")
        # plt.xlim(0,4)
        # #plt.ylim(0.002,0.004)
        # plt.legend(fontsize=20)

        if(self.save_plots == True):
            fig.savefig('Eigcoherences_ZOOM.png',bbox_inches='tight',dpi=600)
        fig.show()

    def coherences_sigmax_scaling(self):
        """Complex magnitude of exciton-vibration coherences scaled by the absolute value
        of their corresponding position matrix element in the open quantum system evolution."""

        vals, eigs = np.linalg.eigh(self.H.todense())
        print("COHERENCE EIG SHAPE = ", eigs[1].shape)
        oe12 = self.exciton_operator(1,2)       
        oe21 = self.exciton_operator(2,1)
        sigmax = oe21 + oe12
        j_k = self.j_k

        fig = plt.figure(19)
        axA = fig.add_subplot(111)
        axA.set_xlabel('Time ($ps$)', fontsize =13)
        axA.set_ylabel('$|\sigma_x,jk| ||\\rho_{jk}(t)||$', fontsize =13)
        axA.grid()

        st = 0000
        en = self.steps -200

        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2).transpose()
        osigmax_kj = np.zeros(len(j_k))
       

        for i in range(len(j_k)):
            j = j_k[i][0]
            k = j_k[i][1]
            osigmax_kj[i] = self.matrix_element(eigs[:, k], sp.kron(sigmax, sp.kron(self.Iv, self.Iv)), eigs[:, j])
            # oX2_kj[i] = self.matrix_element(eigs[:, k], self.oX2, eigs[:, j])

            f = np.round(np.abs(omegaarray[j, k]), decimals=2)
            opsi_jk = np.kron(eigs[:, j], eigs[:, k].getH())
            psi_jk = self.oper_evol(opsi_jk,self.rhoT, self.t, self.tmax_ps)
            plt.plot(self.t_ps[st:en],np.abs(osigmax_kj[i]) * np.abs(psi_jk[st:en]), label="$\Omega_{" +str(j)+str(k) +"} =$" + str(f)) 
            
        plt.legend(bbox_to_anchor=([0.5, 1]), title="Oscillatory Frequencies / $cm^-1$", fontsize =13)

        if(self.save_plots == True):
            fig.savefig('Coherences_sigmax.png',bbox_inches='tight',dpi=600)
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
    
    def vib_collec_evol(self):
        Iv = self.identity_vib()
        Ie = sp.eye(2, 2).tocsr()

        I = sp.kron(Ie, sp.kron(Iv,Iv)).tocsr()
        b = self.destroy()
        ob1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
        ob2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
        ob_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
        ob_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE12 = self.exciton_operator(1, 2)
        oE21 = self.exciton_operator(2, 1)
        sigmaz = oE2 - oE1
        sigmax = oE12 + oE21

        ex_coupling = (np.sin(2*self.theta) * sigmax +  np.cos(2*self.theta) * sigmaz)
        ex_coupling = sp.kron(ex_coupling, sp.kron(Iv,Iv))
        ex_evol = self.oper_evol(ex_coupling, self.rhoT, self.t, self.tmax_ps)


        #POPULATIONS
        # cm = self.oper_evol(ob1.getH()*ob1 ,self.rhoT, self.t, self.tmax_ps)#self.omega*(ob_cm.getH()*ob_cm) + (self.g1/np.sqrt(2))*  + (self.g1/np.sqrt(2))*(ob_cm.getH() + ob_cm)
        # rd = self.oper_evol(ob2.getH()*ob2 ,self.rhoT, self.t, self.tmax_ps)#self.omega*(ob_rd.getH()*ob_rd) -(self.g1/np.sqrt(2))* ex_coupling *  -(self.g1/np.sqrt(2))* ex_coupling *(ob_rd.getH()+ob_rd)

        cm = self.oper_evol(ob_cm.getH() + ob_cm ,self.rhoT, self.t, self.tmax_ps)#self.omega*(ob_cm.getH()*ob_cm) + (self.g1/np.sqrt(2))*  + (self.g1/np.sqrt(2))*(ob_cm.getH() + ob_cm)
        rd = self.oper_evol(ob_rd.getH() + ob_rd ,self.rhoT, self.t, self.tmax_ps)#self.omega*(ob_rd.getH()*ob_rd) -(self.g1/np.sqrt(2))* ex_coupling *  -(self.g1/np.sqrt(2))* ex_coupling *(ob_rd.getH()+ob_rd)

        st = 0000
        en = self.steps -200
    
        fig = plt.figure(900)
        axA = fig.add_subplot(111)

        axA.plot(self.t_ps[0:en], cm[0:en], label=r'$<(b^{\dagger}_{cm}+b_{cm})>$')
        axA.plot(self.t_ps[0:en], rd[0:en], label=r'$<(b^{\dagger}_{rd}+b_{rd})>$') 
        # axA.plot(self.t_ps[0:en], ex_evol[0:en], label=r'$EX EVOL$') 

        fig.legend(loc=(0.6,0.6), fontsize =13)
        axA.set_xlabel('Time ($ps$)', fontsize =13)
        axA.grid(axis='x')
        plt.xlim(0,4)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('b_ops_evol.png',bbox_inches='tight',dpi=600)


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
        for i in np.arange(0,maxstep,500):

            test_matrix = self.rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1])
            quantum_mutual_info, classical_info, quantum_discord = QC.correlations(test_matrix, 2, N, N, 0, 1)
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
        axA.plot(corr_times,c_info,color='k',label=r'Classical Information')
        #axA.plot(corr_times,q_mutual,label=r'Q Mutual Info')
        axA.plot(corr_times,q_discord, color='C2', label=r'Quantum Discord')
        axA.set_xlabel('Time (ps)', fontsize =13)
        axA.set_xlim([0,4])
        axA.set_ylim([0,0.4])

        # axB = axA.twinx()
        # axB.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        # axB.set_ylim([-1,1])
        # axB.plot(self.t_ps[np.arange(st,en,itvl)],self.c_X12[np.arange(st,en,itvl)],'r-o',markevery=0.05,markersize=5,label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')
        
        print(np.arange(st,en,itvl))
        
        #axB.legend(bbox_to_anchor=([0.3,0.8]), fontsize =13)
        axA.legend(bbox_to_anchor=([0.9,0.8]), fontsize =13)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('Q_Correlations.png',bbox_inches='tight',dpi=600)

    def test(self):
        print("oX1= ", self.oX1.shape)
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

        oE1 = self.exciton_operator(1, 1)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        print("oE1mImI size= ", np.shape(oE1mImI))
        ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)
        print("ex1 size = ", np.shape(ex1))

        print("ox1 size = ", np.shape(self.oX1))
        
        #sample rate in cm^-1
        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        #frequency resolution?
        freqres1 = 0.2
        ftlen = (self.t_cm[1]-self.t_cm[0])*np.arange(int(sampleratecm/freqres1))
        print("ftlen = ", ftlen )
        print( "self.t_cm= ", self.t_cm)

        print("(self.H).shape[1] = ", (self.H).shape[1])

    def hilbert_space_plotter(self):
        

        Iv = self.identity_vib()
        Iv_dense =Iv.todense()
        Ie = sp.eye(2, 2).tocsr()
        I = sp.kron(Ie,sp.kron(Iv,Iv)).todense()
        

        b = self.destroy()
        ob1 = sp.kron(b, Iv).tocsr()
        ob2 = sp.kron(Iv, b).tocsr()
        ob_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
        ob_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

        brdbrd = (ob_rd.getH() * ob_rd).todense() #sp.kron(Ie,
        bcmbcm = (ob_cm.getH() * ob_cm).todense() #sp.kron(Ie,

        

        fig1, ax1 = plt.subplots(1,1)
        img = ax1.imshow(I,cmap='rainbow')
        cb2 = fig1.colorbar(img)
        plt.title("I")
        #cb2.set_label(r"Synchronisation Time $(ps)$")

        fig3, ax3 = plt.subplots(1,1)
        img = ax3.imshow(bcmbcm,cmap='rainbow')
        cb2 = fig3.colorbar(img)
        plt.title(r"$b_{cm}^\dag b_{cm}$")

        print(I.shape)

        rho0 = self.rhoT[3000,:]
        rho0 = rho0.reshape(I.shape[0], I.shape[0])
        fig4, ax4 = plt.subplots(1,1)
        img = ax4.imshow(np.abs(rho0),cmap='rainbow')
        cb2 = fig4.colorbar(img)
        plt.title(r"$\rho_0$")
        
        #FULL
        fig2, ax2 = plt.subplots(1,1)
        img = ax2.imshow(brdbrd,cmap='rainbow')
        cb2 = fig2.colorbar(img)
        plt.title(r"$b_{rd}^\dag b_{rd}$")

        print(brdbrd.shape)
        print(brdbrd[0,5])
        print(len(brdbrd[:,0]))
        brdbrd_ones = np.zeros((len(brdbrd[:,0]), len(brdbrd[:,0])))
        bcmbcm_ones = np.zeros((len(bcmbcm[:,0]), len(bcmbcm[:,0])))

        fig5, ax5 = plt.subplots(1,1)
        img = ax5.imshow(Iv_dense,cmap='rainbow')
        cb5 = fig5.colorbar(img)
        plt.title("I")

        fig6, ax6 = plt.subplots(1,1)
        img = ax6.imshow(ob_rd.todense(),cmap='rainbow')
        cb6 = fig6.colorbar(img)
        plt.title(r"$b_{rd}$")

        fig7, ax7 = plt.subplots(1,1)
        img = ax7.imshow(ob_cm.todense(),cmap='rainbow')
        cb7 = fig7.colorbar(img)
        plt.title(r"$b_{cm}$")

        fig8, ax8 = plt.subplots(1,1)
        img = ax8.imshow(ob1.todense(),cmap='rainbow')
        cb8 = fig8.colorbar(img)
        plt.title(r"$b_{1}$")

        fig9, ax9 = plt.subplots(1,1)
        img = ax9.imshow(ob2.todense(),cmap='rainbow')
        cb9 = fig9.colorbar(img)
        plt.title(r"$b_{2}$")



        for i in range(len(brdbrd[:,0])):
            for j in range(len(brdbrd[:,0])):
                if(np.abs(brdbrd[i,j]) > 0):
                    brdbrd_ones[i,j] = 1

        fig10, ax10 = plt.subplots(1,1)
        img = ax10.imshow(brdbrd_ones,cmap='rainbow')
        cb10 = fig10.colorbar(img)
        plt.title(r"$b_{rd}^\dag b_{rd}$ ONES")


        for i in range(len(bcmbcm[:,0])):
            for j in range(len(bcmbcm[:,0])):
                if(np.abs(bcmbcm[i,j]) > 0):
                    bcmbcm_ones[i,j] = 1

        fig11, ax11 = plt.subplots(1,1)
        img = ax11.imshow(bcmbcm_ones,cmap='rainbow')
        cb11 = fig11.colorbar(img)
        plt.title(r"$b_{cm}^\dag b_{cm}$ ONES")

        fig12, ax12 = plt.subplots(1,1)
        img = ax12.imshow(np.abs(self.collective_hamiltonian().todense()), cmap = 'rainbow')
        cb12 = fig12.colorbar(img)
        plt.title("Hamiltonian")
        
        fig13, ax13 = plt.subplots(1,1)
        img = ax13.imshow(np.abs(self.collective_liouvillian().todense()), cmap = 'rainbow')
        cb13 = fig13.colorbar(img)
        plt.title("Liouvillian")


        








class MultiPlots(Operations):
    """Functions for plotting multiple evolutions for different parameter regimes on the same axes"""

    def __init__(self, el_rates, th_rates, phi1, phi2, detuning, save_plots, n_cutoff, temperature, tmax_ps):
        dummy_rate=0.1
        self.el_rates = el_rates
        self.th_rates = th_rates
        self.phi1 =phi1
        self.phi2= phi2
        self.detuning =detuning
        self.save_plots = save_plots
        self.j_k = j_k
        self.n_cutoff =n_cutoff
        self.temperature =temperature
        self.tmax_ps =tmax_ps
        #dummy class of DimerDetune to bring in neccesarry attributes
        DimerDetune.__init__(self, dummy_rate, dummy_rate, self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
        self.H = self.original_hamiltonian()

        self.tmax_ps = tmax_ps
    
        b = self.destroy()
        self.Iv = self.identity_vib()
        self.Ie = sp.eye(2, 2).tocsr()
        
        self.oB1 = sp.kron(self.Ie, sp.kron(b, self.Iv)).tocsr()
        self.oB2 = sp.kron(self.Ie, sp.kron(self.Iv, b)).tocsr()
        self.oX2 = self.oB2 + self.oB2.getH()
        self.oX1 = self.oB1 + self.oB1.getH()
        self.elta = np.int(np.round(((2 * constant.pi) / self.omega) / self.dt))
        
    def Multi_sync_evol(self):

        fig = plt.figure(25)
        en = self.steps -200
        st = 0000
        itvl = 5

        ax = fig.add_subplot(111)
        ax.set_ylabel('$C_{<X_1><X_2>}$',fontsize=13)
        ax.set_xlabel('Time (ps)', fontsize =13)

        for i in range( len(self.th_rates)):
            for j in range(len(self.el_rates)):
                DimerDetune.__init__(self, self.th_rates[i], self.el_rates[j], self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
                rhoT , t  = self.time_evol_me(self.tmax_ps)

                self.t_cm = t / (2 * constant.pi)
                self.tmax = self.tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  
                self.tmax_cm = self.tmax / (2 * constant.pi)
                self.t_ps = (self.t_cm * 1e12) / (100 * constant.c)
        
                #time evo of operators, REPRODUCE THIS IN COHERENCES
                x1 = self.oper_evol(self.oX1,rhoT, t, self.tmax_ps)
                x2 = self.oper_evol(self.oX2,rhoT, t, self.tmax_ps)
                c_X12 = self.corrfunc(x1, x2, self.elta)

                ax.plot(self.t_ps[np.arange(st, en, itvl)], c_X12[np.arange(st, en, itvl)], '-o', markevery=0.05, markersize=5,
                        label="$\Gamma_{deph} = [" +str(self.el_rates[j])+"ps]^{-1}, \Gamma_{th} = [" +str(self.th_rates[i])+ "ps]^{-1}$")
                
        fig.legend(bbox_to_anchor=([0.45, 0.6]),loc ='center left')
        fig.show()
        ax.set_xlim(0,2)
        ax.grid(axis='x')
        if(self.save_plots == True):
            fig.savefig('Multi_Sync 0.1,1.png',bbox_inches='tight',dpi=600)

    def Multi_ET(self):
        fig = plt.figure(26)
        en = self.steps -200
        st = 0000
        itvl = 5

        # ax = fig.add_subplot(111)

        # ax.set_xlabel('Time (ps)', fontsize =13)
        transfers = np.zeros(len(self.th_rates)*len(self.el_rates))
        n =0 
        #ax.set_ylabel("$|E_1\rangle\langle E_1 |$", fontsize=13)

        for i in range( len(self.th_rates)):
            for j in range(len(self.el_rates)):
                DimerDetune.__init__(self, self.th_rates[i], self.el_rates[j], self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
                rhoT , t  = self.time_evol_me(self.tmax_ps)

                self.t_cm = t / (2 * constant.pi)
                self.tmax = self.tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  
                self.tmax_cm = self.tmax / (2 * constant.pi)
                self.t_ps = (self.t_cm * 1e12) / (100 * constant.c)

                oE1 = self.exciton_operator(1, 1)
        
                oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()

                ex1 = self.oper_evol(oE1mImI,rhoT, t, self.tmax_ps)
                transfers[n] = ex1[en]
                n+=1
    
                #ax.plot(self.t_ps[0:en], ex1[0:en], label="$\Gamma_{deph} = [" +str(self.el_rates[j])+"ps]^{-1}, \Gamma_{th} = [" +str(self.th_rates[i])+ "ps]^{-1}$")
        
        fig, ax = plt.subplots(1,1)
        #plt.title('Energy Transfer')
        plt.xlabel("$\Gamma_{th}$", fontsize =13)
        plt.ylabel("$\Gamma_{deph}$", fontsize =13)
        transfer_square= transfers.reshape(len(self.th_rates),len(self.el_rates), order='F')
        inv = self.cmap_map(lambda x: 1-x, matplotlib.cm.PRGn)
        img = ax.imshow(transfer_square,extent=[0.025,1.025,0.025,1.025], cmap='Greens')
        ax.set_xticks(self.th_rates[1::2])
        ax.set_yticks(self.el_rates[::2])
        cb2 = fig.colorbar(img)
        #cb2.set_label(r"$|E_1\rangle\langleE_1|$ Population")
        if(self.save_plots == True):
            fig.savefig('Populations Heatmap.png',bbox_inches='tight',dpi=600)

        # ax.grid(axis='x')
        # ax.set_xlim(0,2)
        # fig.legend(bbox_to_anchor=([0.45, 0.45]), loc = 'upper left')
        # fig.show()
        if(self.save_plots == True):
            fig.savefig('Multi_ET 0.1,1.png',bbox_inches='tight',dpi=600)

    def Multi_coherence(self):
        vals, eigs = np.linalg.eigh(self.H.todense())
        #oX1eig = eigs.getH() * self.oX1 * eigs
        j_k = [[0,2]]

        fig = plt.figure(31)
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time ($ps$)', fontsize =13)
        ax.set_ylabel('$||\\rho_{1,3}(t)||$', fontsize =13)
        ax.grid()

        st = 0000
        en = self.steps -200
        itvl = 3 

        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2).transpose()
        for i in range( len(self.th_rates)):
            for j in range(len(self.el_rates)):

                DimerDetune.__init__(self, self.th_rates[i], self.el_rates[j], self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
                rhoT , t  = self.time_evol_me(self.tmax_ps)   

                self.t_cm = t / (2 * constant.pi)
                self.tmax = self.tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  
                self.tmax_cm = self.tmax / (2 * constant.pi)
                self.t_ps = (self.t_cm * 1e12) / (100 * constant.c)
 

                f = np.round(np.abs(omegaarray[0, 2]), decimals=2)
                opsi_jk = np.kron(eigs[:, 0], eigs[:, 2].getH())
                psi_jk = self.oper_evol(opsi_jk,rhoT, t, self.tmax_ps)
                

                ax.plot(self.t_ps[st:en], np.abs(psi_jk[st:en]), ls='-', label="$\Gamma_{deph} = [" +str(self.el_rates[j])+"ps]^{-1}, \Gamma_{th} = [" +str(self.th_rates[i])+ "ps]^{-1}$")

        ax.set_xlim(0,2)
        fig.legend(bbox_to_anchor=([0.45, 0.9]), loc = 'upper left')
        if(self.save_plots == True):
            fig.savefig('1,3 coherence 0.1,0.5,1.png',bbox_inches='tight',dpi=600)
        fig.show()

    def cmap_map(self,function, cmap):
        """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
        This routine will break any discontinuous points in a colormap.
        """
        cdict = cmap._segmentdata
        step_dict = {}
        # Firt get the list of points where the segments start or end
        for key in ('red', 'green', 'blue'):
            step_dict[key] = list(map(lambda x: x[0], cdict[key]))
        step_list = sum(step_dict.values(), [])
        step_list = np.array(list(set(step_list)))
        # Then compute the LUT, and apply the function to the LUT
        reduced_cmap = lambda step : np.array(cmap(step)[0:3])
        old_LUT = np.array(list(map(reduced_cmap, step_list)))
        new_LUT = np.array(list(map(function, old_LUT)))
        # Now try to make a minimal segment definition of the new LUT
        cdict = {}
        for i, key in enumerate(['red','green','blue']):
            this_cdict = {}
            for j, step in enumerate(step_list):
                if step in step_dict[key]:
                    this_cdict[step] = new_LUT[j, i]
                elif new_LUT[j,i] != old_LUT[j, i]:
                    this_cdict[step] = new_LUT[j, i]
            colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
            colorvector.sort()
            cdict[key] = colorvector

        return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

    def eigs_nocoupling(self):
        th_data = np.zeros(len(self.th_rates)*len(self.el_rates))
        el_data = np.zeros(len(self.th_rates)*len(self.el_rates))
        eigval_differences = np.zeros(len(self.th_rates)*len(self.el_rates))
        run = 0

        for i in range( len(self.th_rates)):
            for j in range(len(self.el_rates)):

                th_data[run] = self.th_rates[i]
                el_data[run] = self.el_rates[j]
                count1 = time.time()

                DimerDetune.__init__(self, self.th_rates[i], self.el_rates[j], self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
           
                vals, eigs = np.linalg.eig(self.collective_liouvillian().todense())
                print(eigs.shape)
                Iv = self.identity_vib()
                Ie = sp.eye(2, 2).tocsr()
                b = self.destroy()
                ob1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
                ob2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
                ob_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
                ob_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

                k = np.argmin(np.abs(np.imag(vals)))
                vals2 = np.delete(vals, k)
                eigs2 = np.delete(eigs, [k],1)

                l = np.argmin(np.abs(np.imag(vals2)))
                if(vals2[l] == vals[k] or np.imag(vals2[l] - vals[k])<0.01):
                    vals2 = np.delete(vals2, l)
                    eigs2 = np.delete(eigs2, [l],1)
                    l = np.argmin(np.abs(np.imag(vals2)))
                print("vals[k] = ", vals[k])
                print("vals2[l] = ", vals2[l])
                eigval_differences[run] = np.abs(vals[k]) - np.abs(vals2[l])
                run +=1

        fig, ax = plt.subplots(1,1)
        plt.title('Eigenvalue Differences')
        plt.xlabel("$\Gamma_{th} / [ps]^{-1}$", fontsize =13)
        plt.ylabel("$\Gamma_{deph} / [ps]^{-1}$", fontsize =13)
        eigval_square = eigval_differences.reshape(len(self.th_rates),len(self.el_rates), order='F')
        inv = self.cmap_map(lambda x: 1-x, matplotlib.cm.PRGn)
        img = ax.imshow(eigval_square,extent=[0.025,1.025,0.025,1.025], cmap='Greens')
        ax.set_xticks(self.th_rates[1::2])
        ax.set_yticks(self.el_rates[::2])
        cb2 = fig.colorbar(img)
        cb2.set_label(r"Eigenvalue Difference")
        if(self.save_plots == True):
            fig.savefig('eigvals_heatmap.png',bbox_inches='tight',dpi=600)

                

    def liouv_eigs(self):
        """find equilibrium state- plot evolutions
        DEFINED SEPERATE RHOT EVOLUTION HERE"""
        th_data = np.zeros(len(self.th_rates)*len(self.el_rates))
        el_data = np.zeros(len(self.th_rates)*len(self.el_rates))
        eigval_differences = np.zeros(len(self.th_rates)*len(self.el_rates))
        run = 0

        for i in range( len(self.th_rates)):
            for j in range(len(self.el_rates)):

                th_data[run] = self.th_rates[i]
                el_data[run] = self.el_rates[j]
                count1 = time.time()

                DimerDetune.__init__(self, self.th_rates[i], self.el_rates[j], self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
           
                vals, eigs = np.linalg.eig(self.collective_liouvillian().todense())
                print(eigs.shape)
                Iv = self.identity_vib()
                Ie = sp.eye(2, 2).tocsr()
                b = self.destroy()
                ob1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()
                ob2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()
                ob_cm = (1/np.sqrt(2))*(ob1 + ob2).tocsr()
                ob_rd = (1/np.sqrt(2))*(ob1 - ob2).tocsr()

                # eigID = 0
                # val = 50

                #no. opp in full H space
                brdbrd = (ob_rd.getH() * ob_rd).todense()
                brdbrd_ones = np.zeros((len(brdbrd[:,0]), len(brdbrd[:,0])))
                
                # #get rid of zero real value
                # for y in range(len(vals)):
                #     if(np.real(vals[y]) < 0.000001):
                #         vals = np.delete(vals, y)


                
                for g in range(len(brdbrd[:,0])):
                    for h in range(len(brdbrd[:,0])):
                        if(np.abs(brdbrd[g,h]) > 0):
                            brdbrd_ones[g,h] = 1
                            
                
              
                
                N=np.int(np.sqrt(np.shape(eigs[1])[1]))
                print("N= ", N)

                
                print("np.amin(vals) = ", np.amin(np.abs(np.imag(vals))))
                #lowest eigenvalue
                k = np.argmin(np.abs(np.real(vals)))
                print("vals[k] = ", vals[k])
                if(np.abs(np.real(vals[k]))< 0.000001):
                    vals = np.delete(vals,k)
                    eigs = np.delete(eigs, [k],1)
                    k = np.argmin(np.abs(np.real(vals)))
                #     zero = True
                #     while(zero == True):
                #         vals = np.delete(vals,k)
                #         eigs = np.delete(eigs, [k],1)
                #         k = np.argmin(np.abs(np.real(vals)))
                #         if(np.abs(np.imag(vals[k])) >0.000001):
                #             zero=False

                    
                #find trace of operator dot rho 
                #k_coupling = np.trace((ob_rd).dot(eigs[k].reshape(N, N)))
                eigs_square = np.asarray(eigs[:,k].reshape(N, N))
                k_coupling = 0
                # for b in range(len(brdbrd[:,0])):
                #     for d in range(len(brdbrd[:,0])):
                        #k_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])
                k_coupling = np.trace(brdbrd_ones.dot(eigs[:,k].reshape(N, N)))
                # figk, axk = plt.subplots(1,1)
                # img = axk.imshow(np.abs(eigs_square),cmap='rainbow')
                # cbk = figk.colorbar(img)
                # plt.title(r"k")
                
                # print(eigs[:,k])

                vals2 = np.delete(vals, k)
                eigs2 = np.delete(eigs, [k],1)

                
                l_coupling = 0

                l = np.argmin(np.abs(np.real(vals2)))
                print("l = ", l)

                # if(np.abs(np.imag(vals2[l]))< 0.000001):
                #     zero = True
                #     while(zero == True):
                #         vals2 = np.delete(vals2,l)
                #         eigs2 = np.delete(eigs2, [l],1)
                #         l = np.argmin(np.abs(np.real(vals2)))
                #         print("l = ", l)
                #         print("vals2[l] = ", vals2[l])

                #         if(np.abs(np.imag(vals2[l])) >0.000001):
                #             zero=False

                if(vals2[l] == vals[k] or np.real(vals2[l] - vals[k])<0.01):
                    vals2 = np.delete(vals2, l)
                    eigs2 = np.delete(eigs2, [l],1)

                    l =np.argmin(np.abs(np.real(vals2)))
                    #print("l = ", l)
                    #print(eigs2[:,l])
                    eigs_square = np.asarray(eigs2[:,l].reshape(N, N))
                    for b in range(len(brdbrd[:,0])):
                        for d in range(len(brdbrd[:,0])):
                            #l_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])
                            l_coupling = np.trace(brdbrd_ones.dot(eigs2[:,l].reshape(N, N)))
                else:
                    eigs_square = np.asarray(eigs2[:,l].reshape(N, N))

                    # for b in range(len(brdbrd[:,0])):
                    #     for d in range(len(brdbrd[:,0])):
                            #l_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])
                l_coupling = np.trace(brdbrd_ones.dot(eigs2[:,l].reshape(N, N)))

                # figl, axl = plt.subplots(1,1)
                # img = axl.imshow(np.abs(eigs_square),cmap='rainbow')
                # cbk = figl.colorbar(img)
                # plt.title(r"l")

                print("vals2[l]=", vals2[l])
                
                # #second lowest eigenvalue
                # vals2 = np.delete(vals,k)
                # l = np.argmin(np.abs(vals2))
                # #l_coupling = np.trace((ob_rd).dot(eigs[l].reshape(N, N)))
                # l_coupling = np.trace(brdbrd_ones.dot(eigs[l].reshape(N, N)))


                #third lowest eigenvalue
                vals3 = np.delete(vals2, l)
                eigs3 = np.delete(eigs2, [l],1)
                m_coupling = 0

                m = np.argmin(np.abs(np.real(vals3)))
                # if(np.abs(np.imag(vals3[m]))< 0.000001):
                #     zero = True
                #     while(zero == True):
                #         vals3 = np.delete(vals3,m)
                #         eigs3 = np.delete(eigs3, [m],1)
                #         m = np.argmin(np.abs(np.real(vals3)))
                #         if(np.abs(np.imag(vals3[m])) >0.000001):
                #             zero=False
                
                print("m = ", m)
                print("vals3[m] = ", vals3[m])
                print("vals2[l] = ", vals2[l])
                print("vals3[l] =",  vals3[l])
                if(vals3[m] == vals2[l] or np.real(vals3[m] - vals2[l])<0.01):
                    vals3 = np.delete(vals3, m)
                    eigs3 = np.delete(eigs3, [m],1)

                    m =np.argmin(np.abs(np.real(vals3)))
                    print("m = ", m)
                    eigs_square = np.asarray(eigs3[:,m].reshape(N, N))
                    for b in range(len(brdbrd[:,0])):
                        for d in range(len(brdbrd[:,0])):
                            #m_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])
                            m_coupling = np.trace(brdbrd_ones.dot(eigs3[:,m].reshape(N, N)))
                    
                else:
                    eigs_square = np.asarray(eigs3[:,m].reshape(N, N))
                    # for b in range(len(brdbrd[:,0])):
                    #     for d in range(len(brdbrd[:,0])):
                            #m_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])
                m_coupling = np.trace(brdbrd_ones.dot(eigs3[:,m].reshape(N, N)))
                #m_coupling = np.trace((ob_rd).dot(eigs[m].reshape(N, N)))
                # figm, axm = plt.subplots(1,1)
                # img = axm.imshow(np.abs(eigs_square),cmap='rainbow')
                # cbk = figm.colorbar(img)
                # plt.title(r"m")
                print("vals[k] =",  vals[k])
                print("vals2[l] = ", vals2[l])
                print("vals3[m] = ", vals3[m])
                

                # vals4 = np.delete(vals3,m)
                # eigs4 = np.delete(eigs3, [m],1)

                # n = np.argmin(np.abs(vals4))
                # n_coupling = 0
                # # print("m = ", m)
                # # print("vals3[m] = ", vals3[m])
                # # print("vals2[l] = ", vals2[l])
                # # print("vals3[l] =",  vals3[l])
                # if(vals4[n] == vals3[m] or np.real(vals4[n] - vals3[m])<0.01):
                #     vals4 = np.delete(vals4, n)
                #     eigs4 = np.delete(eigs4, [n],1)

                #     n =np.argmin(np.abs(vals4))
                #     print("n = ", n)
                #     eigs_square = np.asarray(eigs4[:,n].reshape(N, N))
                #     for b in range(len(brdbrd[:,0])):
                #         for d in range(len(brdbrd[:,0])):
                #             n_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])
                #     #m_coupling = np.trace(brdbrd_ones.dot(eig3s[:,m].reshape(N, N)))
                    
                # else:
                #     eigs_square = np.asarray(eigs4[:,n].reshape(N, N))
                #     for b in range(len(brdbrd[:,0])):
                #         for d in range(len(brdbrd[:,0])):
                #             m_coupling += brdbrd_ones[b,d] * np.abs(eigs_square[b,d])

                

                #sort couplings by absolute value
                small_vals = np.asarray([vals[k],vals2[l],vals3[m]])
                #for x in range(len(small_vals))
                print("small_vals=", small_vals)
                sort_smallvals = sorted(np.real(small_vals), key=abs)
                print ("sort_smallvals= ", sort_smallvals)
                couplings = np.asarray([k_coupling, l_coupling, m_coupling])
                sort_couplings = sorted(couplings, key=abs)
                print("couplings: ", couplings)
                print("sort couplings: ", sort_couplings)



                #find real part of eigenvalue corresponding to second lowest coupling
                l1= 0 
                if(sort_couplings[1] == k_coupling):
                    l1 = np.real(vals[k])
                elif(sort_couplings[1] == l_coupling): 
                    l1 = np.real(vals2[l])
                elif(sort_couplings[1] == m_coupling): 
                    l1 = np.real(vals3[m])
                # elif(sort_couplings[2] == n_coupling): 
                #     l1 = np.abs(vals3[n])


                #real part of eigenvalue corresponsing to highest coupling
                l2= 0 
                if(sort_couplings[2] == k_coupling):
                    l2 = np.real(vals[k])
                elif(sort_couplings[2] == l_coupling): 
                    l2 = np.real(vals2[l])
                elif(sort_couplings[2] == m_coupling): 
                    l2 = np.real(vals3[m])
                # elif(sort_couplings[3] == n_coupling): 
                #     l2 = np.abs(vals4[n])

                #sort 2 eigvals by abs
                sorted_l12 = sorted([l1,l2], key=abs)
                print(sorted_l12)
                #smallest vs largest
                eigval_differences[run] = np.abs(np.abs(sorted_l12[0]) - np.abs(sorted_l12[1]))
                run+=1
                # print("eig_difference = ", eig_difference)
                count2 = time.time()
                print('Difference Time =', count2 - count1)

        # plt.figure()
        # plt.title('Eigenvalue differences')
        # plt.xlabel("$\Gamma_{th}$", fontsize =13)
        # plt.ylabel("$\Gamma_{deph}$", fontsize =13)
        # # im =plt.imshow(, cmap = 'rainbow')
        # print(th_data)
        # print(el_data)
        # print(eigval_differences)
        # plt.scatter(th_data,el_data, c=eigval_differences, s =50)
        # cb = plt.colorbar()
        # cb.set_label(r"$\lambda_2^R - \lambda_1^R$")


        fig, ax = plt.subplots(1,1)
        plt.title('Eigenvalue Differences')
        plt.xlabel("$\Gamma_{th}$", fontsize =13)
        plt.ylabel("$\Gamma_{deph}$", fontsize =13)
        eigval_square = eigval_differences.reshape(len(self.th_rates),len(self.el_rates), order='F')
        inv = self.cmap_map(lambda x: 1-x, matplotlib.cm.PRGn)
        img = ax.imshow(eigval_square,extent=[0.025,1.025,0.025,1.025], cmap='Greens')
        ax.set_xticks(self.th_rates[1::2])
        ax.set_yticks(self.el_rates[::2])
        cb2 = fig.colorbar(img)
        cb2.set_label(r"$\lambda_2^R - \lambda_1^R$")
        if(self.save_plots == True):
            fig.savefig('eigvals_heatmap.png',bbox_inches='tight',dpi=600)

    
    

    def get_sync_times(self):
        th_data = np.zeros(len(self.th_rates)*len(self.el_rates))
        el_data = np.zeros(len(self.th_rates)*len(self.el_rates))
        sync_times = np.zeros(len(self.th_rates)*len(self.el_rates))
        n = 0 

        for i in range( len(self.th_rates)):
            for j in range(len(self.el_rates)):

                th_data[n] = self.th_rates[i]
                el_data[n] = self.el_rates[j]
        
                DimerDetune.__init__(self, self.th_rates[i], self.el_rates[j], self.phi1, self.phi2, self.detuning,  self.n_cutoff, self.temperature, self.tmax_ps)
                self.rhoT , self.t  = self.time_evol_me(self.tmax_ps)

                self.t_cm = self.t / (2 * constant.pi)
                self.t_ps = (self.t_cm * 1e12) / (100 * constant.c)

                self.x1 = self.oper_evol(self.oX1,self.rhoT, self.t, self.tmax_ps)
                self.x2 = self.oper_evol(self.oX2,self.rhoT, self.t, self.tmax_ps) 

                self.elta = np.int(np.round(((2 * constant.pi) / self.omega) / self.dt))
                c_X12 = self.corrfunc(self.x1, self.x2, self.elta)
                # print(len(c_X12))
                C=0.5
                C_new =0.5
                nan_time = np.nan
                sync_times[n] = nan_time
                sync_reached =False
                for m in range(len(c_X12[::500])-1):
                    #SORT THIS CODE OUT
                    #print("len(self.c_X12[::200]) = ", len(self.c_X12[::200]))
                    # print(m)
                    C_new = c_X12[500*m]
                    # print(c_X12[500*m])
                    C_new_2 = c_X12[500*m+400]
                    
                    #TEST FOR SYNCHRONISATION
                    if((np.abs(np.abs(np.real(C_new)/np.real(C)) -1))< 0.005 and np.abs(np.abs(np.real(C)) - 1)< 0.005 and np.abs(np.abs(np.real(C_new_2)/np.real(C)) -1)< 0.005 and sync_reached == False):
                        # print("C = ", C)
                        # print("C_new = ", C_new)
                        time = self.t_ps[500*m]
                        # print("time =", time)
                        if(C>0):
                            sync_times[n] = time
                        if(C<0):
                            sync_times[n] = -time
                        sync_reached = True
                        # break
                    #sync lost
                    if(sync_reached ==True and np.abs(C_new)<0.96):
                        sync_times[n] = nan_time
                        break
                    C= C_new

                n+=1
        
        fig, ax = plt.subplots(1,1)
        plt.title('Synchronisation Times')
        plt.xlabel("$\Gamma_{th} / [ps]^{-1}$", fontsize =13)
        plt.ylabel("$\Gamma_{deph} / [ps]^{-1}$", fontsize =13)
        times_square = sync_times.reshape(len(self.th_rates),len(self.el_rates), order='F')
        print(sync_times)
        print(times_square)
        inv = self.cmap_map(lambda x: 1-x, matplotlib.cm.PRGn)
        # im =plt.imshow(times_square, cmap = inv)
        img = ax.imshow(times_square,extent=[0.025,1.025,0.025,1.025], cmap=inv, vmin=-6, vmax=6)
        #x_label_list = ['A2', 'B2', 'C2', 'D2']
        ax.set_xticks(self.th_rates[1::2])
        ax.set_yticks(self.el_rates[::2])
        # plt.xticks(self.th_rates)
        # plt.yticks(self.el_rates)
        #im.show()
        cb2 = fig.colorbar(img)
        cb2.set_label(r"Synchronisation Time $(ps)$")
        if(self.save_plots == True):
            fig.savefig('Sync_times_heatmap.png',bbox_inches='tight',dpi=600)
        

        # plt.figure(901)
        # plt.title('Synchronisation Times')
        # plt.xlabel("$\Gamma_{th}$", fontsize =13)
        # plt.ylabel("$\Gamma_{deph}$", fontsize =13)
        # print(sync_times)
        # plt.scatter(th_data,el_data, c=sync_times, s =50)
        # cb = plt.colorbar()
        # cb.set_label(r"Synchronisation Time $(ps)$")


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
    plot = Plots(r_th =1, r_el =0.1, phi1 = 0 , phi2 =0, detuning =1, j_k=j_k, save_plots = True, n_cutoff=5, temperature=298, tmax_ps = 4.1)
    # # plot.test()
    # # plot.matrix_elements()
    #plot.sync_evol()
    plot.coherences()
    #plot.hilbert_space_plotter()
    # plot.coherences_sigmax_scaling()
    # plot.energy_transfer()
    # plot.vib_collec_evol()
    # plot.q_correlations()

    #el_rates = [0.05,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    el_rates = [1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
    th_rates = [0.05,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # el_rates = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    # th_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # el_rates = [0.1,0.25,0.5,0.75,1]
    # el_rates = [1,0.75,0.5,0.25,0.1]
    # th_rates = [0.1,0.25,0.5,0.75,1]
    # el_rates = [1,0.1]
    # th_rates = [0.1,1]
    
    # el_rates=[0.1]
    # th_rates = [0.05]
    # multiPlot =MultiPlots(el_rates = el_rates, th_rates =th_rates, phi1 = 0 , phi2 =0, detuning =1, save_plots=True, n_cutoff=5, temperature=298, tmax_ps=8.1) # 0.22 div by 8
    # multiPlot.get_sync_times() #el_rates must be descending for sync_times
    # # multiPlot.liouv_eigs()
    # multiPlot.eigs_nocoupling()
    # multiPlot.Multi_sync_evol()
    # multiPlot.Multi_coherence()
    # multiPlot.Multi_ET()
    

    plt.show()



    