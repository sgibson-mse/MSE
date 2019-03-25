
'''
This code is used to take the basic EFIT run used for a shot on MAST. Take the necessary parameters in order to
calculate B field components"

NB: To load OLD MAST data with pyIDAM. in the terminal: module load idam/1.6.3
    To load MAST-U data using pyuda: module unload idam/1.6.3, module load uda

'''

import pyuda
import numpy as np
import matplotlib.pyplot as plt

client = pyuda.Client() #get the pyuda client

class EFit:

    def __init__(self, shot_number, time, load_mast=True):

        self.shot_number = shot_number
        self.time = time

        self.Bphi_Rgeom = None
        self.Bphi_Rmag = None
        self.Bvac_R = None
        self.Bvac_Rgeom = None
        self.Bvac_Rmag = None
        self.Bvac_val = None

        self.chisq_magnetic = None
        self.chisq_total = None

        self.current_centeredR = None
        self.current_centeredZ = None
        self.current_fluxfunc = None

        self.ff_prime = None

        self.geom_axisR = None
        self.geom_axisZ = None
        self.R_grid = None
        self.Z_grid = None
        self.LCFS_R = None
        self.LCFS_Z = None
        self.mag_axisR = None
        self.mag_axisZ = None
        self.Rvals = None
        self.Z_boundary = None
        self.R_boundary = None

        self.psin_boundary = None
        self.psi_2d = None
        self.psi_R  = None
        self.psi_axis = None
        self.psi_boundary = None

        self.q_axis = None
        self.q_95 = None
        self.q_R = None
        self.time_index = None

        self.BR = None
        self.BZ = None
        self.Bp = None
        self.Bt = None

        if load_mast:
            self.load_MASTefit()
            self.time_indices()
            self.get_psi_normalised()
            self.calc_psi_derivatives()
            self.calc_bfield()
            self.current_flux_function_2d()


    def load_MASTefit(self):
	
        self.all_times = client.get('EFM_ALL_TIMES', self.shot_number)           #All time slices, regardless of whether they have converged (A)
        self.converged_times = client.get('EFM_CNVRGD_TIMES ', self.shot_number) #All times of converged reconstruction (time base B)
        self.plasma_times = client.get('EFM_IP_TIMES', self.shot_number) #Times where a plasma current is present (time base C)

        self.Bphi_Rgeom = client.get('EFM_BPHI_RGEOM', self.shot_number)         #Toroidal B field (total) at geometric axis; f(B)
        self.Bphi_Rmag = client.get('EFM_BPHI_RMAG ', self.shot_number)          #Toroidal B field (total) at magnetic axis; f(B)

        self.Bvac_R = client.get('EFM_BVAC_R', self.shot_number)                  #Reference radius for efm_bvac_val; f(A)

        self.Bvac_Rgeom = client.get('EFM_BVAC_RGEOM', self.shot_number)          #Vacuum toroidal B field at geometric axis; f(B)
        self.Bvac_Rmag = client.get('EFM_BVAC_RMAG', self.shot_number)            #Vacuum toroidal B field at magnetic axis; f(B)
        self.Bvac_val = client.get('EFM_BVAC_VAL', self.shot_number)              #Reference vacuum toroidal B field at efm_bvac_r; f(A)

        self.chisq_magnetic = client.get('EFM_CHISQ_MAGNETIC', self.shot_number)  #Magnetic fit total chi-squared for each iteration; f(num_iterations, A)
        self.chisq_total = client.get('EFM_FINAL_CHISQ', self.shot_number)        #Total chi-squared of fit; f(A)

        self.psin_boundary = client.get('EFM_CM_BDRY', self.shot_number)          #Normalised psi at detected boundary surface; f(B)

        self.current_centeredR = client.get('EFM_CURRENT_CENTRD_R', self.shot_number) #R co-ordinate of current centroid; f(B)
        self.current_centeredZ = client.get('EFM_CURRENT_CENTRD_Z', self.shot_number) #Z co-ordinate of current centroid; f(B)
        self.plasma_current = client.get('EFM_PLASMA_CURR(C)', self.shot_number)     # Value of the plasma current Ip

        self.current_fluxfunc = client.get('EFM_F(PSI)_(C)', self.shot_number)        #Poloidal current flux function, f=R*Bphi; f(psin, B)

        self.ff_prime = client.get('EFM_FFPRIME', self.shot_number)                   #ff' profile; f(npsi, B)

        self.geom_axisR = client.get('EFM_GEOM_AXIS_R(C)', self.shot_number)          #R of geometric axis of plasma; f(B)
        self.geom_axisZ = client.get('EFM_GEOM_AXIS_Z(C)', self.shot_number)          #Z of geometric axis of plasma; f(B)

        self.R_grid = client.get('EFM_GRID(R)', self.shot_number)                     #R grid for 2D outputs; f(nr)
        self.Z_grid = client.get('EFM_GRID(Z)', self.shot_number)                     #Z grid for 2D outputs; f(nz)

        self.LCFS_R = client.get('EFM_LCFS(R)_(C)', self.shot_number)                 #LCFS R coordinate values; f(nlcfs, B)
        self.LCFS_Z = client.get('EFM_LCFS(Z)_(C)', self.shot_number)                 #LCFS Z coordinate values; f(nlcfs, B)

        self.mag_axisR = client.get('EFM_MAGNETIC_AXIS_R', self.shot_number)          #R co-ordinate of magnetic axis; f(B)
        self.mag_axisZ = client.get('EFM_MAGNETIC_AXIS_Z', self.shot_number)          #Z co-ordinate of magnetic axis; f(B)

        self.psi_2d = client.get('EFM_PSI(R,Z)', self.shot_number)              #Poloidal magnetic flux per toroidal radian as a function of radius and height; f(nw, nh, B)
        self.psi_R = client.get('EFM_PSI(R)', self.shot_number)                 #Poloidal magnetic flux per toroidal radian as a function of radius at Z=0; f(nw, B)
        self.psi_axis = client.get('EFM_PSI_AXIS', self.shot_number)            #Poloidal magnetic flux per toroidal radian at the magnetic axis; f(B)
        self.psi_boundary = client.get('EFM_PSI_BOUNDARY', self.shot_number)    #Poloidal magnetic flux per toroidal radian at the plasma boundary; f(B)

        self.q_axis = client.get('EFM_Q_AXIS', self.shot_number)                #Safety factor at the magnetic axis; f(B)
        self.q_95 = client.get('EFM_Q_95', self.shot_number)                    #Safety factor at 95% normalised magnetic flux; f(B)
        self.q_R = client.get('EFM_Q(R)', self.shot_number)                     #Safety factor as a function of radius at Z=0; f(npsi, B)

        self.Rvals = client.get('EFM_RVALS', self.shot_number)	              #Radial co-ordinates used for radial profiles; f(nw)

        self.Z_boundary = client.get('EFM_ZBDRY', self.shot_number)             #Height of boundary position constraints; f(nbdry, A)
        self.R_boundary = client.get('EFM_RBDRY', self.shot_number)              #Radius of boundary position constraints; f(nbdry, A)


    def time_indices(self):
        self.time_index = np.abs(self.converged_times.data - self.time).argmin()
        self.tind_axis = np.abs(self.plasma_times.data - self.time).argmin()
        self.tind_boundary = np.abs(self.plasma_times.data-self.time).argmin()
        self.tind_plasmacurrent = np.abs(self.plasma_times.data-self.time).argmin()
        self.tind_bvac = np.abs(self.all_times.data-self.time).argmin()

    def get_psi_normalised(self):
        #R and Z are the same regardless of the time slice.
        self.R = self.R_grid.data[0,:]
        self.Z = self.Z_grid.data[0,:]

        #Get the parameters at the time slice we want
        self.psi2d = self.psi_2d.data[self.time_index,:,:]
        self.psi_axis = self.psi_axis.data[self.tind_axis]
        self.psi_bound = self.psi_boundary.data[self.tind_boundary]
        self.Ipsign = self.plasma_current.data[self.tind_plasmacurrent]

        #Calculate normalised poloidal flux, such that psi_n **2 = rho (where rho = 1 at the last closed flux surface)
        self.psi_n = (self.psi2d - self.psi_axis) / (self.psi_bound - self.psi_axis)

    def calc_psi_derivatives(self):
        self.dpsidZ = np.gradient(self.psi2d, axis=1) / np.gradient(self.Z)
        self.dpsidR = np.gradient(self.psi2d, axis=0) / np.gradient(self.R)

    def calc_bfield(self):
        self.BR = -1.0*self.dpsidZ/self.R
        self.BZ = self.dpsidR/self.R
        self.Bp = np.sign(self.Ipsign)*np.sqrt(self.BR**2 + self.BZ**2) #Sign of the plasma current determines sign of Bp

    def current_flux_function_2d(self):

        psi_grid = np.linspace(self.psi_axis, self.psi_bound, len(self.current_fluxfunc.data))

        self.f_RZ = np.zeros((len(self.R), len(self.Z)))

        for r in range(len(self.R)):
            for z in range(len(self.Z)):
                if self.psi2d[r, z] < psi_grid[-1] and self.psi2d[r, z] > psi_grid[0]:
                    self.f_RZ[r, z] = self.current_fluxfunc.data(self.psi2d[r, z])
                else:
                    self.f_RZ[r, z] = self.Bvac_Rmag.data[self.tind_bvac] * 1.0

        self.Bt = self.f_RZ/self.R
        self.B = np.sqrt(self.BR**2 + self.BZ**2 + self.Bt**2)

efit = EFit(24409, 0.35)

import matplotlib.pyplot as plt

rr, zz = np.meshgrid(efit.R, efit.Z)

plt.figure()
CS = plt.contour(rr, zz, efit.psi_n)
plt.colorbar()
plt.clabel(CS,inline=1, fontsize=10)
plt.show()



