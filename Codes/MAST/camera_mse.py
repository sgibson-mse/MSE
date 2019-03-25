
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

# External imports
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy.constants import electron_mass, atomic_mass
from jet.data import sal
from raysect.core import Point3D, Vector3D, translate, rotate_basis
from raysect.optical import Ray, d65_white, World, Point3D, Vector3D, translate, rotate
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import AbsorbingSurface
from raysect.primitive import Sphere, Box, Intersect

import pyuda

# Internal imports
from cherab.core.math import Interpolate1DCubic, IsoMapper2D, IsoMapper3D, AxisymmetricMapper, Blend2D, Constant2D, \
    VectorAxisymmetricMapper
from cherab.core import Plasma, Maxwellian, Species, Beam
from cherab.core.atomic import elements
from cherab.core.atomic import Line, deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.core.model import SingleRayAttenuator, BeamCXLine
from cherab.openadas import OpenADAS

from cherab.mastu.equilibrium.MASTU_equilibrium import MASTUEquilibrium
from cherab.mastu.machine import cad_files, wall_outline

client = pyuda.Client()

PULSE = 28101
PULSE_PLASMA = 79503  # /!\ Plasma configuration is from pulse 79503!
TIME = 0.25

world = World()

adas = OpenADAS(permit_extrapolation=True)  # create atomic data source

#import_jet_mesh(world)


# ########################### PLASMA EQUILIBRIUM ############################ #
print('Plasma equilibrium')

equilibrium = MASTUEquilibrium(PULSE)
equil_time_slice = equilibrium.time(TIME)
psin_2d = equil_time_slice.psi_normalised
psin_3d = AxisymmetricMapper(equil_time_slice.psi_normalised)
inside_lcfs = equil_time_slice.inside_lcfs


# ########################### PLASMA CONFIGURATION ########################## #
print('Plasma configuration')

plasma = Plasma(parent=world)
plasma.atomic_data = adas
plasma.b_field = VectorAxisymmetricMapper(equil_time_slice.b_field)

DATA_PATH = '/pulse/{}/ppf/signal/{}/{}/{}:{}'
user = 'cgiroud'
sequence = 0

psi_coord = equilibrium.f_profile_psin
mask = np.argwhere(psi_coord <= 1.0)[:,0]

# Ignore flow velocity, set to zero vector.
flow_velocity = lambda x, y, z: Vector3D(0,0,0)

# Set Ti = Te
# swap this to load the thomson data
ion_temperature = client.get("AYC_TE", PULSE)
time_index = equilibrium._find_nearest(ion_temperature.dims[0].data, TIME)
ion_temperature_data = ion_temperature.data[time_index,mask]
print("Ti between {} and {} eV".format(ion_temperature_data.min(), ion_temperature_data.max()))
ion_temperature_psi = Interpolate1DCubic(psi_coord, ion_temperature_data)
ion_temperature = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(psin_2d, ion_temperature_psi), inside_lcfs))

electron_density = client.get("AYC_NE", PULSE)
time_index = equilibrium._find_nearest(electron_density.dims[0].data, TIME)
electron_density = electron_density.data[time_index,mask]
print("Ne between {} and {} m-3".format(electron_density.min(), electron_density.max()))
electron_density_psi = Interpolate1DCubic(psi_coord, ion_temperature_data)
electron_density_data = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(psin_2d, electron_density_psi), inside_lcfs))

# set to 1% electron density
density_c6_data = 0.01*electron_density
density_c6_psi = Interpolate1DCubic(psi_coord, density_c6_data)
density_c6 = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(psin_2d, density_c6_psi), inside_lcfs))
density_d = lambda x, y, z: electron_density(x, y, z) - 6 * density_c6(x, y, z)

d_distribution = Maxwellian(density_d, ion_temperature, flow_velocity, deuterium.atomic_weight * atomic_mass)
c6_distribution = Maxwellian(density_c6, ion_temperature, flow_velocity, carbon.atomic_weight * atomic_mass)
e_distribution = Maxwellian(electron_density, ion_temperature, flow_velocity, electron_mass)

d_species = Species(deuterium, 1, d_distribution)
c6_species = Species(carbon, 6, c6_distribution)

#define plasma parameters - electron distribution, impurity composition and B field from EFIT
plasma.electron_distribution = e_distribution
plasma.composition = [d_species, c6_species]
plasma.b_field = equil_time_slice.b_field

sigma = 0.25
integration_step = 0.02

plasma.geometry = Sphere(sigma * 2.0)
plasma.geometry_transform = None

plasma.integrator.step = integration_step
plasma.integrator.min_samples = 1000
plasma.atomic_data = adas

# Setup elements.deuterium lines
d_alpha = Line(elements.deuterium, 0, (3, 2))
d_beta = Line(elements.deuterium, 0, (4, 2))
d_gamma = Line(elements.deuterium, 0, (5, 2))
d_delta = Line(elements.deuterium, 0, (6, 2))
d_epsilon = Line(elements.deuterium, 0, (7, 2))

plasma.models = [
    # Bremsstrahlung()
    ExcitationLine(d_alpha),
    ExcitationLine(d_beta),
    ExcitationLine(d_gamma),
    ExcitationLine(d_delta),
    ExcitationLine(d_epsilon),
    RecombinationLine(d_alpha),
    RecombinationLine(d_beta),
    RecombinationLine(d_gamma),
    RecombinationLine(d_delta),
    RecombinationLine(d_epsilon)
]

# ########################### NBI CONFIGURATION ############################# #

# BEAM ------------------------------------------------------------------------

beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 65000
beam.power = 3e6
beam.element = elements.deuterium
beam.sigma = 0.025
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [
    BeamCXLine(Line(elements.helium, 1, (4, 3))),
    BeamCXLine(Line(elements.helium, 1, (6, 4))),
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
    BeamCXLine(Line(elements.carbon, 5, (9, 8))),
    BeamCXLine(Line(elements.carbon, 5, (10, 8))),
    BeamCXLine(Line(elements.neon, 9, (11, 10))),
    BeamCXLine(Line(elements.neon, 9, (12, 11))),
]
beam.integrator.step = integration_step
beam.integrator.min_samples = 10

beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 65000 / 2
beam.power = 3e6
beam.element = elements.deuterium
beam.sigma = 0.025
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [
    BeamCXLine(Line(elements.helium, 1, (4, 3))),
    BeamCXLine(Line(elements.helium, 1, (6, 4))),
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
    BeamCXLine(Line(elements.carbon, 5, (9, 8))),
    BeamCXLine(Line(elements.carbon, 5, (10, 8))),
    BeamCXLine(Line(elements.neon, 9, (11, 10))),
    BeamCXLine(Line(elements.neon, 9, (12, 11))),
]
beam.integrator.step = integration_step
beam.integrator.min_samples = 10

beam = Beam(parent=world, transform=translate(1.0, 0.0, 0) * rotate(90, 0, 0))
beam.plasma = plasma
beam.atomic_data = adas
beam.energy = 65000 / 3
beam.power = 3e6
beam.element = elements.deuterium
beam.sigma = 0.025
beam.divergence_x = 0.5
beam.divergence_y = 0.5
beam.length = 3.0
beam.attenuator = SingleRayAttenuator(clamp_to_zero=True)
beam.models = [
    BeamCXLine(Line(elements.helium, 1, (4, 3))),
    BeamCXLine(Line(elements.helium, 1, (6, 4))),
    BeamCXLine(Line(elements.carbon, 5, (8, 7))),
    BeamCXLine(Line(elements.carbon, 5, (9, 8))),
    BeamCXLine(Line(elements.carbon, 5, (10, 8))),
    BeamCXLine(Line(elements.neon, 9, (11, 10))),
    BeamCXLine(Line(elements.neon, 9, (12, 11))),
]
beam.integrator.step = integration_step
beam.integrator.min_samples = 10


# ############################### OBSERVATION ############################### #
print('Observation')

los = Point3D(-1.742, 1.564, 0.179)
direction = Vector3D(0.919, -0.389, -0.057).normalise()
los = los + direction * 0.9
up = Vector3D(1, 1, 0)

camera = PinholeCamera((512, 512), fov=52.9, parent=world, transform=translate(los.x, los.y, los.z) * rotate_basis(direction, up))
camera.pixel_samples = 50
camera.spectral_bins = 15

camera.observe()

