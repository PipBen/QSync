def fourier_ET(self):
        vals, eigs = np.linalg.eigh(self.H.todense())

        rho0 = self.init_state()
        rho0eig = eigs.getH() * rho0 * eigs

        oE1 = self.exciton_operator(1, 1)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        #Ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)

        oEx1eig = eigs.getH() * oE1mImI * eigs
        coefEx1 = np.multiply(oEx1eig, rho0eig)
        coefEx1chop = np.tril(coefEx1,k=-1)
        
        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
        2 * N ** 2, 2 * N ** 2).transpose()
        omegachop = np.tril(omegaarray,k=-1)

        count4 = time.time()

        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        freqres1 = 0.5
        ftlen = (self.t_cm[1]-self.t_cm[0])*np.arange(int(sampleratecm/freqres1))

        anaEx1array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])
        anaEx1 = np.zeros(np.size(ftlen))

        for a in np.arange(np.size(ftlen)):
            anaEx1array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefEx1chop)
            anaEx1[a] = np.sum(anaEx1array[:,:,a]) + np.trace(coefEx1)
 
        freqres2 = 0.1
        pads = int((sampleratecm/freqres2)-np.shape(anaEx1)[0]) #30000
        Ex1pad = np.append(anaEx1,np.zeros(pads))
     
        fr10 = np.fft.rfft(Ex1pad,int(np.size(Ex1pad)))
   
        freq_cm0 = sampleratecm*np.arange(0,1-1/np.size(Ex1pad),1/np.size(Ex1pad))

        count5 = time.time()
        print('Measurements =',count5-count4)

        en = self.steps -200

        fig = plt.figure(7)
        # print("fourier y axis = ", [0:en])
        plt.plot(freq_cm0[0:en],(np.abs(fr10)**2)[0:en],label=r'$\langle X_1\rangle$')
    
        plt.ylabel('Real Part of FT', fontsize =13)
        #plt.yticks()
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        #plt.xlim(0,150)
        plt.legend()
        plt.grid(True,which='both', axis='x')
        plt.minorticks_on()
        #plt.yticks([0])
        plt.title(r'Components of $|E_1\rangle\langle E_1 |$', fontsize =13)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('ET_FT_original.png',bbox_inches='tight',dpi=600)

    def ET_FT2(self):
        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        oE1 = self.exciton_operator(1, 1)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        Ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)

        freqres = 0.5
        pads = int((sampleratecm/freqres)-np.shape(Ex1)[0]) #30000
        Ex1pad = np.append(Ex1,np.zeros(pads))
        FT = np.fft.rfft(Ex1pad,int(np.size(Ex1pad)))

        freq_cm = sampleratecm*np.arange(0,1-1/np.size(Ex1pad),1/np.size(Ex1pad))

        st = int(0/(sampleratecm/np.size(Ex1pad)))
        en = int(1200/(sampleratecm/np.size(Ex1pad)))

        fig = plt.figure(15)
        # plt.plot(freq_cm[st:en],(np.abs(FT1)**2)[st:en],label='<$X_1$>')
        # plt.plot(freq_cm[st:en],(np.abs(FT2)**2)[st:en],label='<$X_2$>')
        plt.plot(freq_cm[st:en],np.real(FT)[st:en],label='<$E_1$>')
        plt.ylabel('Real Part of FT', fontsize =13)
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        plt.legend()
        plt.grid()
        fig.show()
        if(self.save_plots == True):
            fig.savefig('ET_FT.png',bbox_inches='tight',dpi=600)

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


def fourier_ET(self):
        vals, eigs = np.linalg.eigh(self.H.todense())

        rho0 = self.init_state()
        rho0eig = eigs.getH() * rho0 * eigs

        oE1 = self.exciton_operator(1, 1)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        #Ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)

        oEx1eig = eigs.getH() * oE1mImI * eigs
        coefEx1 = np.multiply(oEx1eig, rho0eig)
        coefEx1chop = np.tril(coefEx1,k=-1)
        
        N= self.n_cutoff
        omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
        2 * N ** 2, 2 * N ** 2).transpose()
        omegachop = np.tril(omegaarray,k=-1)

        count4 = time.time()

        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        freqres1 = 0.5
        ftlen = (self.t_cm[1]-self.t_cm[0])*np.arange(int(sampleratecm/freqres1))

        anaEx1array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])
        anaEx1 = np.zeros(np.size(ftlen))

        for a in np.arange(np.size(ftlen)):
            anaEx1array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefEx1chop)
            anaEx1[a] = np.sum(anaEx1array[:,:,a]) + np.trace(coefEx1)
 
        freqres2 = 0.1
        pads = int((sampleratecm/freqres2)-np.shape(anaEx1)[0]) #30000
        Ex1pad = np.append(anaEx1,np.zeros(pads))
     
        fr10 = np.fft.rfft(Ex1pad,int(np.size(Ex1pad)))
   
        freq_cm0 = sampleratecm*np.arange(0,1-1/np.size(Ex1pad),1/np.size(Ex1pad))

        count5 = time.time()
        print('Measurements =',count5-count4)

        en = self.steps -200

        fig = plt.figure(7)
        # print("fourier y axis = ", [0:en])
        plt.plot(freq_cm0[0:en],(np.abs(fr10)**2)[0:en],label=r'$\langle X_1\rangle$')
    
        plt.ylabel('Real Part of FT', fontsize =13)
        #plt.yticks()
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        #plt.xlim(0,150)
        plt.legend()
        plt.grid(True,which='both', axis='x')
        plt.minorticks_on()
        #plt.yticks([0])
        plt.title(r'Components of $|E_1\rangle\langle E_1 |$', fontsize =13)
        fig.show()
        if(self.save_plots == True):
            fig.savefig('ET_FT_original.png',bbox_inches='tight',dpi=600)


    #OLD
    def ET_FT2(self):
        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])
        oE1 = self.exciton_operator(1, 1)
        oE1mImI = sp.kron(oE1, sp.kron(self.Iv, self.Iv)).tocsr()
        Ex1 = self.oper_evol(oE1mImI,self.rhoT, self.t, self.tmax_ps)

        freqres = 0.5
        pads = int((sampleratecm/freqres)-np.shape(Ex1)[0]) #30000
        Ex1pad = np.append(Ex1,np.zeros(pads))
        FT = np.fft.rfft(Ex1pad,int(np.size(Ex1pad)))

        freq_cm = sampleratecm*np.arange(0,1-1/np.size(Ex1pad),1/np.size(Ex1pad))

        st = int(0/(sampleratecm/np.size(Ex1pad)))
        en = int(1200/(sampleratecm/np.size(Ex1pad)))

        fig = plt.figure(15)
        # plt.plot(freq_cm[st:en],(np.abs(FT1)**2)[st:en],label='<$X_1$>')
        # plt.plot(freq_cm[st:en],(np.abs(FT2)**2)[st:en],label='<$X_2$>')
        plt.plot(freq_cm[st:en],np.real(FT)[st:en],label='<$E_1$>')
        plt.ylabel('Real Part of FT', fontsize =13)
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        plt.legend()
        plt.grid()
        fig.show()
        if(self.save_plots == True):
            fig.savefig('ET_FT.png',bbox_inches='tight',dpi=600)
    #OLD
    def Full_sync_FT(self):
        sampleratecm = 1/(self.t_cm[1]-self.t_cm[0])

        X1 = self.oper_evol(self.oX1, self.rhoT, self.t, self.tmax_ps)
        X2 = self.oper_evol(self.oX2, self.rhoT, self.t, self.tmax_ps)

        X1 = X1[6000:7000]
        X2 =X2[6000:7000]

        freqres = 0.5
        pads = int((sampleratecm/freqres)-np.shape(X1)[0]) #30000
        x1pad = np.append(X1,np.zeros(pads))
        x2pad = np.append(X2,np.zeros(pads))

        FT1 = np.fft.rfft(x1pad,int(np.size(x1pad)))
        FT2 = np.fft.rfft(x2pad,int(np.size(x2pad)))

        freq_cm = sampleratecm*np.arange(0,1-1/np.size(x1pad),1/np.size(x1pad))

        st = int(0/(sampleratecm/np.size(x1pad)))
        en = int(6000/(sampleratecm/np.size(x1pad)))
        

        fig = plt.figure(13)
        plt.plot(freq_cm[st:en],(np.abs(FT1)**2)[st:en],label='<$X_1$>')
        plt.plot(freq_cm[st:en],(np.abs(FT2)**2)[st:en],label='<$X_2$>')
        # plt.plot(freq_cm[st:en],np.real(FT1)[st:en],label='<$X_1$>')
        # plt.plot(freq_cm[st:en],np.real(FT2)[st:en],label='<$X_2$>')
        plt.ylabel('Real Part of FT', fontsize =13)
        plt.xlabel('Frequency ($cm^{-1}$)', fontsize =13)
        plt.legend()
        plt.grid()
        fig.show()
        if(self.save_plots == True):
            fig.savefig('Full_sync_FT.png',bbox_inches='tight',dpi=600)