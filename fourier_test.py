import numpy as np
import matplotlib.pylab as plt

def fft_time_to_freq(time):
    dt = time[1] - time[0]
    N = time.shape[0]
    freq = 2.*np.pi*np.fft.fftfreq(N, dt)  #
    return np.append(freq[N//2:], freq[:N//2])

def fft(y_t, t, return_frequencies=False):
    n_times = len(t)
    ft = t[-1] * np.fft.fft(y_t, n_times)
    ft = np.append(ft[n_times // 2:], ft[:n_times // 2])
    if return_frequencies ==True:
        frequencies = fft_time_to_freq(t)
        return frequencies, ft
    else:
        return ft

times = np.linspace(0, 100000, 100000)
omega = 1

yt = np.sin(omega * times)
freqs, ft = fft(yt, times, True)


plt.figure()

plt.plot(freqs, np.real(ft))
plt.show()

for j in range(6):
    print(j)


     def fourier(self):
        vals, eigs = np.linalg.eigh(self.H.todense())

        rho0 = self.init_state()
        rho0eig = eigs.getH() * rho0 * eigs

        oX1eig = eigs.getH() * self.oX1 * eigs
        oX2eig = eigs.getH() * self.oX2 * eigs
        
        #X_j,k rho0_j,k
        coefx1 = np.multiply(oX1eig, rho0eig)
        coefx2 = np.multiply(oX2eig, rho0eig)
        
        #bottom triangular?
        coefx1chop = np.tril(coefx1,k=-1)
        coefx2chop = np.tril(coefx2,k=-1)
     
        N= self.n_cutoff
        #all frequencies
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
        2 * N ** 2, 2 * N ** 2).transpose()
        omegachop = np.tril(omegaarray,k=-1)

        count4 = time.time()

        #sample rate in cm^-1
        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        #frequency resolution?
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


    def ET_FT_Charlie(self):
        oE1 = self.exciton_operator(1, 1)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)

        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        freqres1 = 0.5
        ftlen = (self.t_cm[1]-self.t_cm[0])*np.arange(int(sampleratecm/freqres1))
        print("pre ex1 len = ", len(ex1))
        ex1 = ex1[0:len(ftlen)]

        print("ftlen len = ", len(ftlen))
        print("ex1 len = ", len(ex1))

        figtest =plt.figure(9000)
        plt.plot(ftlen, ex1)

        freqres2 = 0.1
        # ex1_pad, t_pad = self.padding(ex1, self.t,3)
        pads = int((sampleratecm/freqres2)-np.shape(ex1)[0]) #30000
        print("np.shape(ex1) = ", np.shape(ex1) )
        ex1_pad = np.append(ex1,np.ones(pads))
   
        freq_cm0 = sampleratecm*np.arange(0,1-1/np.size(ex1_pad),1/np.size(ex1_pad))

        # freqs, ft = self.fft(ex1_pad, t_pad, True)
        ft = np.fft.rfft(ex1_pad,int(np.size(ex1_pad)))
        fig = plt.figure(20)

        st = int(0/(sampleratecm/np.size(ex1_pad)))
        en = int(10000/(sampleratecm/np.size(ex1_pad)))
               
        plt.plot(freq_cm0[st:en],np.abs(ft)[st:en]**2)

        if(self.save_plots == True):
            fig.savefig('ET_FT_Zoom_25ps.png',bbox_inches='tight',dpi=600)


    def X_FT(self):

        X1 = self.oper_evol(self.oX1, self.rhoT, self.t, self.tmax_ps)
        X2 = self.oper_evol(self.oX2, self.rhoT, self.t, self.tmax_ps)

        # X1 = X1[6000:7000]
        # X2 = X2[6000:7000]

        X1_pad, t_pad1 = self.padding(X1, self.t_cm,1)
        X2_pad, t_pad2 = self.padding(X2, self.t_cm,1)

        figtest = plt.figure(100)
        plt.plot(t_pad1,X1_pad)
        figtest.show()

        freqs1, ft1 = self.fft(X1_pad, t_pad1, True)
        freqs2, ft2 = self.fft(X2_pad, t_pad2, True)

        fig = plt.figure(21)

        plt.plot(freqs1, np.abs(ft1)**2)
        plt.plot(freqs2, np.abs(ft2)**2)
        # plt.xlim(1100,1300)

        if(self.save_plots == True):
            fig.savefig('X_FT_Zoom_25ps.png',bbox_inches='tight',dpi=600)

    
    def fourier_ET(self):
        vals, eigs = np.linalg.eigh(self.H.todense())
        print("vals shape =", np.shape(vals))


        rho0 = self.init_state()
        rho0eig = eigs.getH() * rho0 * eigs

        oE1 = self.exciton_operator(1, 1)
        oE2 = self.exciton_operator(2, 2)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        oE2mImI = sp.kron(oE2, sp.kron(self.Iv, self.Iv)).tocsr()
        # ex2 = self.oper_evol(oE2mImI,self.rhoT, self.t, self.tmax_ps)
        #Ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)

        oEx1eig = eigs.getH() * oE1mImI * eigs
        oEx2eig = eigs.getH() * oE2mImI * eigs
        coefEx1 = np.multiply(oEx1eig, rho0eig)
        coefEx2 = np.multiply(oEx2eig, rho0eig)
    
        coefEx1chop = np.tril(coefEx1,k=-1)
        coefEx2chop = np.tril(coefEx2,k=-1)

        N= self.n_cutoff
        print("N = ", N)
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
        2 * N ** 2, 2 * N ** 2).transpose()
        omegachop = np.tril(omegaarray,k=-1)

        count4 = time.time()

        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        freqres1 = 0.5
        ftlen = (self.t_cm[1]-self.t_cm[0])*np.arange(int(sampleratecm/freqres1))

        anaEx1array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])
        anaEx2array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])

        anaEx1 = np.zeros(np.size(ftlen))
        anaEx2 = np.zeros(np.size(ftlen))


        for a in np.arange(np.size(ftlen)):
            anaEx1array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefEx1chop)
            anaEx2array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefEx2chop)

            anaEx1[a] = np.sum(anaEx1array[:,:,a]) + np.trace(coefEx1)
            anaEx2[a] = np.sum(anaEx1array[:,:,a]) + np.trace(coefEx1)

        freqres2 = 0.1
        pads = int((sampleratecm/freqres2)-np.shape(anaEx1)[0]) #30000


        Ex1pad = np.append(anaEx1,np.zeros(pads))
        Ex2pad = np.append(anaEx2,np.zeros(pads))

     
        fr10 = np.fft.rfft(Ex1pad,int(np.size(Ex1pad)))
        fr20 = np.fft.rfft(Ex2pad,int(np.size(Ex2pad)))

        freq_cm0 = sampleratecm*np.arange(0,1-1/np.size(Ex1pad),1/np.size(Ex1pad))

        count5 = time.time()
        print('Measurements =',count5-count4)

        en = self.steps -200

        fig = plt.figure(7)
        plt.plot(freq_cm0[0:en],(np.abs(fr10))[0:en],label=r'$\langle X_1\rangle$')
        plt.plot(freq_cm0[0:en],(np.abs(fr20))[0:en],label=r'$\langle X_1\rangle$')

    
        plt.ylabel('Real Part of FT', fontsize =13)
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        plt.xlim(0,150)
        plt.legend()
        plt.grid(True,which='both', axis='x')
        plt.minorticks_on()
        #plt.yticks([0])
        plt.title(r'Components of $|E_1\rangle\langle E_1 |$', fontsize =13)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('ET_FT_original.png',bbox_inches='tight',dpi=600)