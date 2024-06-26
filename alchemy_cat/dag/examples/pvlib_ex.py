""" Bigger example on solar PV modeling using PVLib

https://github.com/pvlib/pvlib-python

Example: https://github.com/pvlib/pvlib-python/blob/master/docs/tutorials/tmy_to_power.ipynb
"""

import inspect
import os
from pprint import pprint
import warnings
try:
    import pvlib
except ImportError:
    raise ImportError('This example require pvlib')
from pvlib import pvsystem
from alchemy_cat.dag.core import Graph


warnings.filterwarnings("ignore")


pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))
datapath = os.path.join(pvlib_abspath, 'data', '703165TY.csv')
tmy_data, meta = pvlib.tmy.readtmy3(datapath, coerce_year=2015)
tmy_data.index.name = 'Time'
tmy_data = tmy_data.shift(freq='-30Min')
sandia_modules = pvlib.pvsystem.retrieve_sam(name='SandiaMod')
cec_modules = pvlib.pvsystem.retrieve_sam(name='CECMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('sandiainverter')
sapm_inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

data = {
    'index': tmy_data.index,
    'surface_tilt': 30,
    'surface_azimuth': 180,
    'DHI': tmy_data['DHI'],
    'DNI': tmy_data['DNI'],
    'GHI': tmy_data['GHI'],
    'Wspd': tmy_data['Wspd'],
    'DryBulb': tmy_data['DryBulb'],
    'albedo': 0.2,
    'latitude': meta['latitude'],
    'longitude': meta['longitude'],
    'sandia_module': sandia_modules.Canadian_Solar_CS5P_220M___2009_,
    'cec_module': cec_modules.Canadian_Solar_CS5P_220M,
    'alpha_sc': cec_modules.Canadian_Solar_CS5P_220M['alpha_sc'],
    'EgRef': 1.121,
    'dEgdT': -0.0002677,
    'inverter': sapm_inverter
}


def unpack_df(f):
    def wrap(*args, **kwargs):
        res = f(*args, **kwargs)
        return res.values.T
    return wrap


def unpack_dict(f):  # (ordered)
    def wrap(*args, **kwargs):
        res = f(*args, **kwargs)
        return [res[k] for k in res]
    return wrap


# parallelism not needed for this example
graph = Graph(parallel=False)

graph.add_node(
    unpack_df(pvlib.solarposition.get_solarposition),
    inputs=['index', 'latitude', 'longitude'],
    outputs=['apparent_elevation', 'apparent_zenith', 'azimuth',
             'elevation', 'equation_of_time', 'zenith']
)
graph.add_node(
    pvlib.irradiance.extraradiation,
    inputs=['index'],
    outputs=['dni_extra']
)
graph.add_node(
    pvlib.atmosphere.relativeairmass,
    inputs=['apparent_zenith'],
    outputs=['airmass']
)
graph.add_node(
    pvlib.irradiance.haydavies,
    inputs=['surface_tilt', 'surface_azimuth', 'DHI', 'DNI',
            'dni_extra', 'apparent_zenith', 'azimuth'],
    outputs=['poa_sky_diffuse']
)
graph.add_node(
    pvlib.irradiance.grounddiffuse,
    inputs=['surface_tilt', 'GHI', 'albedo'],
    outputs=['poa_ground_diffuse']
)
graph.add_node(
    pvlib.irradiance.aoi,
    inputs=['surface_tilt', 'surface_azimuth', 'apparent_zenith', 'azimuth'],
    outputs=['aoi']
)
graph.add_node(
    unpack_df(pvlib.irradiance.globalinplane),
    inputs=['aoi', 'DNI', 'poa_sky_diffuse', 'poa_ground_diffuse'],
    outputs=['poa_global', 'poa_direct', 'poa_diffuse']
)
graph.add_node(
    unpack_df(pvlib.pvsystem.sapm_celltemp),
    inputs=['poa_global', 'Wspd', 'DryBulb'],
    outputs=['temp_cell', 'temp_module']
)
graph.add_node(
    pvlib.pvsystem.sapm_effective_irradiance,
    inputs=['poa_direct', 'poa_diffuse', 'airmass', 'aoi', 'sandia_module'],
    outputs=['effective_irradiance'],
)
graph.add_node(
    unpack_dict(pvlib.pvsystem.sapm),
    inputs=['effective_irradiance', 'temp_cell', 'sandia_module'],
    outputs=['i_sc_sapm', 'i_mp_sapm', 'v_oc_sapm', 'v_mp_sapm', 'p_mp_sapm', 'i_x_sapm', 'i_xx_sapm']
)
graph.add_node(
    pvlib.pvsystem.calcparams_desoto,
    inputs=['poa_global', 'temp_cell', 'alpha_sc', 'cec_module', 'EgRef', 'dEgdT'],
    outputs=['photocurrent', 'saturation_current', 'resistance_series',
             'resistance_shunt', 'nNsVth']
)
graph.add_node(
    unpack_dict(pvlib.pvsystem.singlediode),
    inputs=['photocurrent', 'saturation_current', 'resistance_series', 'resistance_shunt', 'nNsVth'],
    outputs=['i_sc_sd', 'i_mp_sd', 'v_oc_sd', 'v_mp_sd', 'p_mp_sd', 'i_x_sd', 'i_xx_sd']
)
graph.add_node(
    pvlib.pvsystem.snlinverter,
    inputs=['v_mp_sapm', 'p_mp_sapm', 'inverter'],
    outputs=['pac_sapm']
)
graph.add_node(
    pvlib.pvsystem.snlinverter,
    inputs=['v_mp_sd', 'p_mp_sd', 'inverter'],
    outputs=['pac_sd']
)

pprint(graph.dag)
graph.calculate(data)
print(sum(graph.data['pac_sd']))
print(sum(graph.data['pac_sapm']))
