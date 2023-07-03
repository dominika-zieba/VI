import numpy as np
import lal
from scipy.interpolate import interp1d
from scipy.special import i0, i0e
from matplotlib import pyplot as pl
from jimgw.PE.detector_projection import make_antenna_response
from jimgw.PE.detector_projection import time_delay_geocentric
import jax.numpy as jnp


class Interferometer(object):
    """Class for the Interferometer & strain data frequency series """


    def __init__(self, name, sensitivity, f_min, T, Fs, t_start, seed):
        """
        Instantiate an Interferometer object.

        Parameters
        ----------
        name: str
            Interferometer name, e.g., 'H1'.
        sensitivity: str
            Detector sensitivity, e.g. 'O1'
        minimum_frequency: float
            Minimum frequency to analyse for detector.
        """
        self.name = name
        self.sensitivity = sensitivity
        self.laldetector = lal.cached_detector_by_prefix[self.name]
        self.seed=seed
        self.tensor = self.laldetector.response
        self.location = self.laldetector.location
        
        #asd_data = np.genfromtxt('data/' + self.name + '_'+ self.sensitivity + '_strain.txt')
        #self.asd = interp1d(asd_data[:, 0], asd_data[:, 1])
        #self.f_max = asd_data[-1, 0]
        
        self.f_min = f_min
        self.T = T  #data segment duration
        self.Fs = Fs #sampling frequency  
        self.df = 1/T #frequency resolution in frequency domain
        self.Fn = Fs/2 #Nyquist frequency
        self.t_start = t_start #start time of the analyzed data segment for the detector 
        
        self.noise = 0.
        self.signal = 0.
        self.strain = 0.
        
        #get the frequency array for the complex frequency series
        N = int(self.T*self.Fs) #number of time domain data points 
        dt = 1/self.Fs #sample spacing in the time domain
        
        self.times = np.arange(0,self.T,dt)
        self.dt=dt
        self.N=N

        freqs = np.fft.rfftfreq(N,dt)
        
        start=np.where(freqs==self.f_min)[0][0]
        self.freqs=freqs[start:]
        
        #self.psd = self.asd(self.freqs)**2
        self.psd = 1.
        

    def get_time_delay(self, ra, dec, t_gps):
        """ get time delay from geocenter 
        """
        return lal.TimeDelayFromEarthCenter(self.laldetector.location, ra, dec, t_gps)

    def get_time_delay_jax(self, ra, dec, t_gps):
        """ get time delay from geocenter (jax)
        """
        gps = lal.LIGOTimeGPS(t_gps)
        gmst_rad = lal.GreenwichMeanSiderealTime(gps)

        return time_delay_geocentric(self.laldetector.location, jnp.array([0.,0.,0.]), ra, dec, gmst_rad)
    
    def get_antenna_response_jax(self, ra, dec, psi, t_gps):
        """ get the plus and cross polarizations antenna response (jax) 
        """
        gps = lal.LIGOTimeGPS(t_gps)
        gmst_rad = lal.GreenwichMeanSiderealTime(gps)

        antenna_response_plus = make_antenna_response(self.laldetector.response, 'plus')
        antenna_response_cross = make_antenna_response(self.laldetector.response, 'cross')

        fp = antenna_response_plus(ra, dec, gmst_rad, psi)
        fc = antenna_response_cross(ra, dec, gmst_rad, psi)

        return fp, fc
    
    def get_antenna_response(self, ra, dec, psi, t_gps):
        """ get the plus and cross polarizations antenna response 
        """
        gps = lal.LIGOTimeGPS(t_gps)
        gmst_rad = lal.GreenwichMeanSiderealTime(gps)
        
        response = self.laldetector.response

        # computation of plus and cross antenna factors
        fp, fc = lal.ComputeDetAMResponse(response, ra, dec, psi, gmst_rad)

        return fp, fc

    def get_frequency_array(self):
        """ get frequency array corresponding to the complex frequency series
        """
        N = int(self.T*self.Fs) #number of time domain data points 
        dt = 1/self.Fs #sample spacing in the time domain
        
        freqs = np.fft.rfftfreq(N,dt)
        
        start=np.where(freqs==self.f_min)[0][0]
        freqs=freqs[start:]
        
        return freqs
    
    def get_noise_realization_from_psd(self):
        """ a complex frequency series noise realization corresponding to the detector psd
        """
        state=np.random.get_state()
        np.random.seed(self.seed)
        
        noise_psd = self.psd                                 #noise psd     
        sigma_noise = 1/np.sqrt(2) * np.sqrt(self.T/2 * noise_psd)          #noise stds array, one per freq bin
        
        noise = np.zeros(len(self.freqs)) + 0j                              #array to hold the noise realization
        
        for i in range(len(noise)):
            noise[i] = sigma_noise[i]*np.random.randn() + 1j * sigma_noise[i]*np.random.randn()    #random noise ralization
        
        np.random.set_state(state)
        return noise
    
    def inject_signal_into_noise(self, signal):
        self.noise = self.get_noise_realization_from_psd() 
        self.signal = signal
        self.strain = self.signal + self.noise
    
    def get_signal_snr(self):

        SNR_squared=4*self.df*np.sum((np.abs(self.signal)**2/self.psd))
        SNR=np.sqrt(SNR_squared)
            
        return SNR
    
    def plot_signal(self):
        freqs = self.freqs
        fig, ax = pl.subplots(1, 1, figsize=(9,6))
        ax.set_title(self.name)
        ax.loglog(freqs, self.asd(freqs), 'b')
        ax.loglog(freqs, abs(self.signal), 'g', label = 'SNR =' + str(self.get_signal_snr()))
        ax.set_ylim([5e-25, 1e-20])
        ax.set_xlim([self.f_min, self.Fn])
        ax.set_xlabel('frequency /Hz')
        ax.set_ylabel(r'h(f)')
        ax.legend()
    
    def loglikelihood(self, model):
        """ calculates loglikelihood
        """    
        return - 2 * self.df * jnp.sum(jnp.abs(self.strain - model)**2 / self.psd)
    
    def h_inner_h_plus_d_inner_d(self, model):
        return np.sum((np.abs(model)**2 + np.abs(self.strain)**2) / self.psd) 
    
    def complex_h_inner_d(self, model):
        return np.sum(model * np.conj(self.strain) / self.psd) 
    
    
    
class Network(object):
    """Class for the Interferometer Network """

    def __init__(self, detectors):
        """
        Instantiate an Interferometer object.

        Parameters
        ----------
        name: str
            Interferometer name, e.g., 'H1'.
        sensitivity: str
            Detector sensitivity, e.g. 'O1'
        minimum_frequency: float
            Minimum frequency to analyse for detector.
        """
        self.detectors = detectors
        
    def network_phase_marg_loglikelihood(self, models):
        """ calculates network phase marginalized loglikelihood
        """
        Y = -2 * self.detectors[0].df * sum([det.h_inner_h_plus_d_inner_d(models[det.name]) for det in self.detectors])
        X = 4 * self.detectors[0].df * np.abs(sum([det.complex_h_inner_d(models[det.name]) for det in self.detectors]))
        
        #print(sum([det.complex_h_inner_d(models[det.name]) for det in self.detectors]))

        return Y + np.log(i0e(X)) + X            #i0e(x) = exp(-abs(X))*i0(X), where i0 is the modified bessel functionof first kind
    
    def snr(self):
        return np.sqrt(sum([det.get_signal_snr()**2 for det in self.detectors]))

    
    
    
        
