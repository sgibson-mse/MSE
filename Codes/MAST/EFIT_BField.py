
''' This code is used to calculate the magnetic field vector using the flux co-ordinates provided by EFIT.
We need to derive the components of the magnetic field in flux co-ordinates. This means changing from
machine co-ordinates (R, Z, phi) to (psi, theta, phi) which are flux co-ordinates. Then we calculate the normalised
flux function using the poloidal flux function psi_p and at the last closed flux surface (LCFS) psi_lcfs.'''

import idlbridge as idl #import idl bridge to get data output from EFIT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

getdata = idl.export_function('getdata')			 #export getdata function from IDL

def EFIT_DATA(shotno, sliceno):

	psi_data = getdata('EFM_PSI(R,Z)', shotno, noecho=1)			 #flux function, which is defined as the poloidal flux function per radiant toroidal angle (psi_p / 2pi)
	#print(psi_data.keys())							 #gives a dictionary with data and other parameters
	psi = psi_data['data'] 						 # 120,65,65 (120 time slices so pick one around the middle)
	
	psi_slice = psi[sliceno,:,:]
	np.savetxt('psi.txt', psi_slice, delimiter=' ')

	#time at which slice is taken
	time_data = psi_data['time']
	time = time_data[sliceno]
	print('The time is', time,)

	psi_lcfs = getdata('EFM_PSI_BOUNDARY', shotno, noecho=1)	 #poloidal flux at the last closed flux surface f(t)
	psi_lcfs = psi_lcfs['data']
	np.savetxt('psi_lcfs.txt', psi_lcfs, delimiter=' ')	

	psi_0 = getdata('EFM_PSI_AXIS', shotno, noecho=1) 		 #poloidal flux at the magnetic axis f(t)
	psi_0 = psi_0['data']      
	np.savetxt('psi_0.txt', psi_0, delimiter=' ')                                     

	R_0   = getdata('EFM_MAGNETIC_AXIS_R', shotno, noecho=1)	 #R-coordinate of magnetic axis f(t)
	R_0   = R_0['data']
	np.savetxt('R_0.txt', R_0, delimiter=' ')


	Z_0   = getdata('EFM_MAGNETIC_AXIS_Z', shotno, noecho=1) 	 #Z co-ordinate of magnetic axis f(t)       
	Z_0   = Z_0['data']
	np.savetxt('Z_0.txt', Z_0, delimiter=' ')

	f     = getdata('EFM_F(PSI)_(C)', shotno, noecho=1)		 # Radial profile of f=R*B_phi in equal increments of poloidal flux from centre to edge f(x,t) (73,65)
	f     = f['data']
	np.savetxt('f.txt', f, delimiter=' ')

	print('shape of f',np.shape(f))

	B_vac = getdata('EFM_BVAC_VAL', shotno, noecho=1)		#vacuum toroidal field at R=bvac_r f(t)
	B_vac = B_vac['data']
	np.savetxt('B_vac.txt', B_vac, delimiter=' ')

	R_ref = getdata('EFM_BVAC_R', shotno, noecho=1)			#reference radius where vaccuum field is defined f(t)
	R_ref = R_ref['data']
	np.savetxt('R_ref.txt', R_ref, delimiter=' ')

	return psi, time, psi_lcfs, psi_0, R_0, Z_0, f, B_vac, R_ref


def poloidal_plane(shotno):

	grid_R = getdata('EFM_GRID(R)', shotno, noecho=1)
	grid_R = grid_R['data']
	print('shape of grid_r',np.shape(grid_R))
	np.savetxt('grid_R.txt', grid_R, delimiter=' ')

	grid_Z = getdata('EFM_GRID(Z)', shotno, noecho=1)
	grid_Z = grid_Z['data']
	np.savetxt('grid_Z.txt', grid_Z, delimiter=' ')

	Rp, Zp = np.meshgrid(grid_R,grid_Z) 				#grid of poloidal plane
	print('the shape of Rp,Zp is:', np.shape(Rp), np.shape(Zp))

	return Rp, Zp, grid_R, grid_Z


def flux_functions(sliceno, psi_lcfs, psi_0):

	lcfspsi_takeref = psi_lcfs[sliceno] - psi_0[sliceno] 		#outer most flux function take reference from magnetic axis

	psi_mod = psi[sliceno,:,:] - psi_0[sliceno]		
	psi_n   = np.divide(psi_mod,lcfspsi_takeref) 			#normalised flux function

	rho    = np.sqrt(psi_n)
	nans   = np.isnan(rho)						#nans from subtracting psi_0
	rho[nans] = 0

	z_mod = grid_Z - Z_0[sliceno]					#modify R and Z values by subtracting the R/Z values at magnetic axis
	r_mod = grid_R - R_0[sliceno]

	r = np.sqrt(z_mod**2 + r_mod**2) 				#minor radius

	return z_mod, r_mod, r, rho, psi_n


def angular_coordinates(z_mod, r_mod, grid_R, grid_Z):

	theta = np.arctan(z_mod/r_mod)

	return theta

#calculate the mod of the basis vectors grad psi, grad theta and grad phi


def gradients(psi, r):

	psi_slice = psi[sliceno,:,:] 					#take one time slice from psi

	print('the shape of psi_slice:', np.shape(psi_slice))

	grad_psi_R = np.gradient(psi_slice, axis=0)			

	print('the shape of grad psi r is:', np.shape(grad_psi_R))

	grad_psi_Z = np.gradient(psi_slice, axis=1) 

	grad_psi = np.sqrt(grad_psi_R**2 + grad_psi_Z**2)

	grad_theta = np.sqrt( (z_mod**2 + r_mod**2) / (z_mod**2 + r_mod**2)**2 )

	grad_phi = (grid_R)**-1

	return grad_psi_R, grad_psi_Z, grad_psi, grad_theta, grad_phi


def jacobian(grad_psi, r, grad_psi_R, grad_psi_Z, r_mod, z_mod):

	prefactor = grad_psi * r

	jacobian = (1/prefactor) * (r_mod*grad_psi_R + z_mod*grad_psi_Z)

	return jacobian


def B_field_RZphi(grad_phi, grad_psi_Z, grad_psi_R):

	B_Rp = grad_phi * grad_psi_Z

	B_Zp = -1 * grad_phi * grad_psi_R

	return B_Rp, B_Zp

# calculate the magnetic field components in flux co-ordinates
# Bphi - toroidal component of magnetic field, depends on the value psi_n
# 0 <= psi_n <= 1 - B phi is given by the current flux function (ie. within the plasma), outside B phi is the vacuum field as a function of radius
#Contravariant poloidal component of B theta = B_poloidal dotted with basis vector e_theta

def B_field_fluxcoords(z_mod, r_mod, grad_psi_R, grad_psi_Z, grad_phi, grad_theta, grid_R, f, mu_0, rho):

	b_theta = ((z_mod*grad_psi_Z) + (r_mod*grad_psi_R))  * grad_phi * grad_theta

	print('the shape of poloidal B field is', np.shape(b_theta))

	#impose condition on f such that it only uses values where rho**2 <=1 

	#rho_coords = np.where(rho**2 <=1)
	#print('the co-ordinates are:',rho_coords,np.shape(rho_coords))

	b_phiplasma = grid_R[:,np.where(rho[:,0]<=1)] * f[sliceno,np.where(rho[:,0]**2<=1)] * mu_0

	print('the shape of b phi plasma field is', np.shape(b_phiplasma))
	print('f is:', f[sliceno,np.where(rho[:,0]<=1)])

	b_phiout    = (B_vac[sliceno] * R_ref[sliceno]) / grid_R

	return b_theta, b_phiplasma, b_phiout


#some initial values required

shotno = 30200			 			 #shot number
sliceno = 65						 #time slice
mu_0 = 4*np.pi*10**-7					 #constant mu_0




psi, time, psi_lcfs, psi_0, R_0, Z_0, f, B_vac, R_ref = EFIT_DATA(shotno,sliceno)
Rp, Zp, grid_R, grid_Z = poloidal_plane(shotno)
z_mod, r_mod, r, rho, psi_n = flux_functions(sliceno, psi_lcfs, psi_0)
theta = angular_coordinates(z_mod, r_mod, grid_R, grid_Z)
grad_psi_R, grad_psi_Z, grad_psi, grad_theta, grad_phi = gradients(psi, r)
jacobian = jacobian(grad_psi, r, grad_psi_R, grad_psi_Z, r_mod, z_mod)
B_Rp, B_Zp = B_field_RZphi(grad_phi, grad_psi_Z, grad_psi_R)
b_theta, b_phiplasma, b_phiout = B_field_fluxcoords(z_mod, r_mod, grad_psi_R, grad_psi_Z, grad_phi, grad_theta, grid_R, f, mu_0,rho)





#######plotting###########

levels = np.arange(0.97,1.0,0.01)

plt.figure(1)
CS = plt.contour(Rp, Zp, rho, levels=levels, aspect='square')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Normalised flux function ($\psi_{n}$ = 0 to $\psi_{n}$ = 1)')
plt.xlabel('R')
plt.ylabel('Z')

plt.figure(2)
CS = plt.contour(Rp, Zp, b_theta, levels=levels, aspect='square')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Normalised flux function ($\psi_{n}$ = 0 to $\psi_{n}$ = 1)')
plt.xlabel('R')
plt.ylabel('Z')


plt.show()










