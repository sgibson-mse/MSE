
# Copyright 2014-2017 United Kingdom Atomic Energy Authority
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

"""
JET equilibrium data reading routines
"""

import pyuda
import numpy as np

from raysect.core import Point2D
from cherab.tools.equilibrium import EFITEquilibrium


class MASTUEquilibrium:
    """
    Reads MAST-U EFIT equilibrium data and provides object access to each timeslice.

    :param pulse: MAST-U pulse number.
    """

    def __init__(self, pulse):

        self.client = pyuda.Client()  # get the pyuda client

        # Poloidal magnetic flux per toroidal radian as a function of (Z,R) and timebase
        self.psi = self.client.get("efm_psi(r,z)", pulse)

        self.time_slices = self.psi.dims[0].data

        # Psi grid axes f(nr), f(nz)
        self.r = self.client.get("efm_grid(r)", pulse)
        self.z = self.client.get("efm_grid(z)", pulse)

        # f profile
        self.f = self.client.get("efm_f(psi)_(c)", pulse)# Poloidal current flux function, f=R*Bphi; f(psin, C)

        self.psi_r = self.client.get("efm_psi(r)", pulse) #poloidal magnetic flux per toroidal radian as a function of radius at Z=0

        # Poloidal magnetic flux per toroidal radian at the plasma boundary and magnetic axis
        self.psi_lcfs = self.client.get("efm_psi_boundary", pulse)
        self.psi_axis = self.client.get("efm_psi_axis", pulse)

        # Plasma current
        self.plasma_current = self.client.get("efm_plasma_curr(C)", pulse)

        # Reference vaccuum toroidal B field at R = efm_bvac_r
        self.b_vacuum_magnitude = self.client.get("efm_bvac_val", pulse)

        self.b_vacuum_radius = self.client.get("efm_bvac_r", pulse)

        # Magnetic axis co-ordinates
        self.axis_coord_r = self.client.get("efm_magnetic_axis_r", pulse)
        self.axis_coord_z = self.client.get("efm_magnetic_axis_z", pulse)

        #minor radius
        self.minor_radius = self.client.get("EFM_MINOR_RADIUS", pulse)

        #lcfs boundary polygon
        self.lcfs_poly_r = self.client.get("efm_lcfs(r)_(c)", pulse)
        self.lcfs_poly_z = self.client.get("efm_lcfs(z)_(c)", pulse)

        # Number of LCFS co-ordinates
        self.nlcfs = self.client.get("EFM_LCFS(N)_(C)", pulse)

        # time slices when plasma is present
        self.plasma_times = self.client.get("EFM_IP_TIMES", pulse)

        self.time_range = self.time_slices.min(), self.time_slices.max()


    def time(self, time):
        """
        Returns an equilibrium object for the time-slice closest to the requested time.

        The specific time-slice returned is held in the time attribute of the returned object.

        :param time: The equilibrium time point.
        :returns: An EFIT Equilibrium object.
        """

        # locate the nearest time point and fail early if we are outside the time range of the data

        try:
            index = self._find_nearest(self.time_slices, time)
            # Find the index in the time array defined as when the plasma is present
            plasma_index = self._find_nearest(self.plasma_times.data, time)
        except IndexError:
            raise ValueError('Requested time lies outside the range of the data: [{}, {}]s.'.format(*self.time_range))

        B_VACUUM_RADIUS = self.b_vacuum_radius.data[index]

        time = self.time_slices[index]

        psi = np.transpose(self.psi.data[index,:,:]) #transpose psi to get psi(R,Z) instead of psi(Z,R)

        psi_lcfs = self.psi_lcfs.data[plasma_index]

        psi_axis = self.psi_axis.data[plasma_index]

        print('psi_axis', psi_axis)

        psi_r = self.psi_r.data[:,plasma_index]

        f_profile_psin = self.f.dims[1].data
        self.f_profile_psin = f_profile_psin

        f_profile_magnitude = self.f.data[plasma_index, :]

        axis_coord = Point2D(self.axis_coord_r.data[plasma_index], self.axis_coord_z.data[plasma_index])

        b_vacuum_magnitude = self.b_vacuum_magnitude.data[index]

        lcfs_poly_r = self.lcfs_poly_r.data[plasma_index,:]
        lcfs_poly_z = self.lcfs_poly_z.data[plasma_index,:]

        # Get the actual co-ordinates of the LCFS
        lcfs_points = self.nlcfs.data[plasma_index]

        #Filter out padding in the LCFS coordinate arrays
        lcfs_poly_r = lcfs_poly_r[0:lcfs_points]
        lcfs_poly_z = lcfs_poly_z[0:lcfs_points]

        # convert raw lcfs poly coordinates into a polygon object
        lcfs_polygon = self._process_efit_lcfs_polygon(lcfs_poly_r, lcfs_poly_z)
        self.lcfs_polygon = lcfs_polygon

        r = self.r.data[0,:]
        z = self.z.data[0,:]

        minor_radius = self.minor_radius.data[plasma_index]

        print('minor radius', minor_radius)

        return EFITEquilibrium(r, z, psi, psi_axis, psi_lcfs, axis_coord, f_profile_psin,
                               f_profile_magnitude, B_VACUUM_RADIUS, b_vacuum_magnitude, lcfs_polygon, time)

    @staticmethod
    def _find_nearest(array, value):

        if value < array.min() or value > array.max():
            raise IndexError("Requested value is outside the range of the data.")

        index = np.searchsorted(array, value, side="left")

        if (value - array[index])**2 < (value - array[index + 1])**2:
            return index
        else:
            return index + 1

    @staticmethod
    def _process_efit_lcfs_polygon(poly_r, poly_z):

        if poly_r.shape != poly_z.shape:
            raise ValueError("EFIT LCFS polygon coordinate arrays are inconsistent in length.")

        n = poly_r.shape[0]
        if n < 2:
            raise ValueError("EFIT LCFS polygon coordinate contain less than 2 points.")

        # boundary polygon contains redundant points that must be removed
        unique = (poly_r != poly_r[0]) | (poly_z != poly_z[0])
        unique[0] = True  # first point must be included!
        poly_r = poly_r[unique]
        poly_z = poly_z[unique]

        # generate single array containing coordinates
        poly_coords = np.zeros((len(poly_r), 2))
        poly_coords[:, 0] = poly_r
        poly_coords[:, 1] = poly_z

        # magic number for vocel_coef from old code
        return poly_coords

