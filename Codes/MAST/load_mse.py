import pyuda
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

class MSE(object):

    def __init__(self, shot):

        """
        Pulls some useful parameters from the database for the MSE diagnostic. Requires a shotnumber as an input.
        :param shot: (int) Shot number.
        """

        self.client = pyuda.Client() #client to retrieve data

        self.beam_present = self.client.get('ams_beam_ok', shot)

        if self.beam_present.data == 0.:
            print('Beam not present - no MSE data available!')
            return
        else:
            print('Beam present - MSE data available!')
            pass

        self.channels = self.client.get('ams_ch', shot)

        # Fibers connected to each channel

        self.fibers = self.client.get('ams_md', shot)

        # Polarization angle (proportional to pitch angle in absence of Er)

        self.gamma = self.client.get('ams_gamma', shot)

        self.gamma_noise = self.client.get('ams_gammanoise', shot)

        # Magnetic Pitch angle tan(gamma)*(A5/A0) = tan(pitch_angle) where A0 and A5 are geometry coefficients

        self.pitch_angle = self.client.get('ams_pitcha', shot)

        self.pitch_angle_noise = self.client.get('ams_pitchanoise', shot)

        # Major radius and Time axis

        self.time_axis = self.gamma.dims[0].data

        self.time_range = self.time_axis.min(), self.time_axis.max()

        self.major_radius = self.client.get('ams_rpos', shot)

        # Stokes vectors and errors

        self.S0 = self.client.get('ams_s0', shot)

        self.S0_noise = self.client.get('ams_s0noise', shot)

        self.S1 = self.client.get('ams_s1noise', shot)

        self.S1_noise = self.client.get('ams_s1noise', shot)

        self.S2 = self.client.get('ams_s2noise', shot)

        self.S2_noise = self.client.get('ams_s2noise', shot)

        self.S3 = self.client.get('ams_s3noise', shot)

        self.S3_noise = self.client.get('ams_s3noise', shot)

        # View geometry data

        self.view_geometry = self.client.get('ams_viewstr', shot)

        # Horizontal co-ordinates of the MSE fibres on the focal plane of the collection lens

        self.vx0 = self.client.get('ams_vx0', shot)

        self.vy0 = self.client.get('ams_vy0', shot)

        # Cosine between the line of sight and beam direction for each sightline

        self.cosbeam = self.client.get('ams_cosbeam', shot)

        # Photoelastic Modulator parameters (driving frequency and their retardance)

        self.PEM1_frequency = self.client.get('ams_pem1_freq', shot)

        self.PEM1_retardance = self.client.get('ams_retar1', shot)

        self.PEM2_frequency = self.client.get('ams_pem2_freq', shot)

        self.PEM2_retardance = self.client.get('ams_retar2', shot)

        self.A_coefficients = self.client.get('AMS_ACOEFF', shot)

        #A coefficients related to the geometry of the diagnostic. There is a separate A coefficient for each line of sight, so 36x6 coefficients

        self.A0 = self.A_coefficients.data[0,0,:]

        self.A1 = self.A_coefficients.data[0,1,:]

        self.A2 = self.A_coefficients.data[0,2,:]

        self.A3 = self.A_coefficients.data[0,3,:]

        self.A4 = self.A_coefficients.data[0,4,:]

        self.A5 = self.A_coefficients.data[0,5,:]

    def time(self, time):
        """
        Returns an MSE object for the time slice closest to the requested time.

        :param time:Time point
        :return: An MSE object
        """
        try:
            #Find the nearest index which corresponds to the time closest to the specified time
            index = self._find_nearest(self.time_axis, time)
        except IndexError:
            raise ValueError('Requested time lies outside the range of the data: [{}, {}]s.'.format(*self.time_range))

        time = self.time_axis[index]

        gamma = self.gamma.data[index,:]
        gamma_noise = self.gamma_noise.data[index,:]

        major_radius = self.major_radius.data[0,:]

        pitch_angle = self.pitch_angle.data[index,:]
        pitch_angle_noise = self.pitch_angle_noise.data[index,:]

        S0 = self.S0.data[index,:]
        S0_noise = self.S0_noise.data[index,:]

        S1 = self.S1.data[index,:]
        S1_noise = self.S1_noise.data[index,:]

        S2 = self.S2.data[index,:]
        S2_noise = self.S2_noise.data[index,:]

        S3 = self.S3.data[index,:]
        S3_noise = self.S3_noise.data[index,:]

        #Store all the values for one time slice in a named tuple - just access them like a class (ie. gamma = mse_timeslice.gamma)

        mse_tuple = namedtuple('mse_timeslice', 'time major_radius gamma gamma_noise pitch_angle pitch_angle_noise S0 S0_noise S1 S1_noise S2 S2_noise S3 S3_noise')

        mse_timeslice = mse_tuple(time, major_radius, gamma, gamma_noise, pitch_angle, pitch_angle_noise, S0, S0_noise, S1, S1_noise, S2, S2_noise, S3, S3_noise)

        return mse_timeslice

    @staticmethod
    #Method to find the index of a time slice nearest to the requested time
    def _find_nearest(array, value):

        if value < array.min() or value > array.max():
            raise IndexError("Requested value is outside the range of the data.")

        index = np.searchsorted(array, value, side="left")

        if (value - array[index]) ** 2 < (value - array[index + 1]) ** 2:
            return index
        else:
            return index + 1

#How to use the code

mse = MSE(28101)
mse_timeslice = mse.time(0.2)

plt.figure()
plt.plot(mse_timeslice.major_radius, np.rad2deg(mse_timeslice.pitch_angle))
plt.xlabel('Time [s]')
plt.ylabel('Pitch Angle [Degrees]')
plt.show()











