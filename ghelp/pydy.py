import numpy as np
from math import sin, cos, pi, fmod, sqrt

import os
from astropy.io import fits

class GengaOutData(object):
    """An object to read and store Genga output data.
    
    Parameters
        folder (str): The folder containing the Genga output fits files.
    
    Args:
        xyz (ndarray): The data load form all_xyz.fits, containing the possition and velocity of the particles.
        aei (ndarray): The data load form all_aei.fits, containing the kelpersian orbtial parameters of the particles.
        n_particle (int): The number of particles in the simulation.
        min_t (float): The minimum time of the simulation.
        max_t (float): The maximum time of the simulation.
        n_time (int): The number of time steps in the simulation.
        dt (float): The time step of the output.
        x (np.ndarray): The x position of the particles.
        y (np.ndarray): The y position of the particles.
        z (np.ndarray): The z position of the particles.
        vz (np.ndarray): The z velocity of the particles.
        vy (np.ndarray): The y velocity of the particles.
        vz (np.ndarray): The z velocity of the particles.
        a (np.ndarray): The semi-major axis of the particles.
        e (np.ndarray): The eccentricity of the particles.
        i (np.ndarray): The inclination of the particles.
        w (np.ndarray): The argument of periapsis of the particles.
        o (np.ndarray): The longitude of the ascending node of the particles.
        M (np.ndarray): The mean anomaly of the particles.
        mass (np.ndarray): The mass of the particles.
        radius (np.ndarray): The radius of the particles.

    Usage:
        One can use [particle_id, time_id] to access the data. Slice can also be used. If only one of the two is given, it means particle_id. 
        >>> genga_data = GengaOutData('folder')
        >>> genga_data[0, 0].x # The x position of the first particle at the first time step.
        >>> genga_data[0:5, 0].x # The x position of the first 5 particles at the first time step. ndarray with shape (5,).
        >>> genga_data[0, 0:10].x # The x position of the first particle at the first 5 time steps. ndarray with shape (10,).
        >>> genga_data[0:5, 0:10].x # The x position of the first 5 particles at the first 10 time steps. ndarray with shape (10, 5). Note that the shape is (10, 5) instead of (5, 10) because the first axis is the time axis.


    """
    def __init__(self, folder):
        self.xyz = fits.getdata(os.path.join(folder, 'all_xyz.fits'))
        self.aei = fits.getdata(os.path.join(folder, 'all_aei.fits'))

        self.n_particle = self.xyz.shape[1]
        self.n_time = self.xyz.shape[0]
        self.min_t = self.xyz[0, 0, 0]
        self.max_t = self.xyz[-1, 0, 0]
        self.dt = self.xyz[1, 0, 0] - self.xyz[0, 0, 0]

        self.i_slice = slice(None, None, None)
        self.t_slice = slice(None, None, None)
        self.init_obj = True

    @property
    def x(self):
        return self.xyz[self.t_slice, self.i_slice, 3]
    
    @property
    def y(self):
        return self.xyz[self.t_slice, self.i_slice, 4]
    
    @property
    def z(self):
        return self.xyz[self.t_slice, self.i_slice, 5]
    
    @property
    def vx(self):
        return self.xyz[self.t_slice, self.i_slice, 6]
    
    @property
    def vy(self):
        return self.xyz[self.t_slice, self.i_slice, 7]
    
    @property
    def vz(self):
        return self.xyz[self.t_slice, self.i_slice, 8]
    
    @property
    def a(self):
        return self.aei[self.t_slice, self.i_slice, 3]
    
    # e, i, w, o, M from now on for index 4 - 9
    @property
    def e(self):
        return self.aei[self.t_slice, self.i_slice, 4]
    
    @property
    def i(self):
        return self.aei[self.t_slice, self.i_slice, 5]
    
    @property
    def w(self):
        return self.aei[self.t_slice, self.i_slice, 7]
    
    @property
    def o(self):
        return self.aei[self.t_slice, self.i_slice, 6]
    
    @property
    def M(self):
        return self.aei[self.t_slice, self.i_slice, 8]
    
    @property
    def mass(self):
        return self.xyz[0, self.i_slice, 1]
    
    @property
    def radius(self):
        return self.xyz[0, self.i_slice, 2]
    
    @property
    def t(self):
        return self.xyz[self.t_slice, 0, 0]
    
    @classmethod
    def from_slice(cls, self, i_slice, t_slice):
        subobj = cls.__new__(cls)
        subobj.i_slice = i_slice
        subobj.t_slice = t_slice
        subobj.xyz = self.xyz
        subobj.aei = self.aei

        subobj.n_particle = self.xyz.shape[1]
        subobj.n_time = self.xyz.shape[0]
        subobj.min_t = self.xyz[0, 0, 0]
        subobj.max_t = self.xyz[-1, 0, 0]
        subobj.dt = self.xyz[1, 0, 0] - self.xyz[0, 0, 0]
        self.init_obj = False

        return subobj
    
    def __getitem__(self, key):
        # if not self.init_obj:
        #     raise ValueError('a sliced object cannot be sliced again')
        
        if isinstance(key, int):
            key = key,

        if len(key) == 1:
            i_slice = key[0]
            t_slice = slice(None, None, None)
        elif len(key) == 2:
            i_slice = key[0]
            t_slice = key[1]
        else:
            raise ValueError('only one(particle) or two(particle, time) indexs are allowed')
        
        sub = GengaOutData.from_slice(self, i_slice, t_slice)
        return sub


def elliptical_orbit(m_star, m_pl, a, ecc, inc, w, o, m0, t):
    """solving the x, y and z position of a planet by its orbital elements.
        Equations in this function are from 天体力学基础 by Jilin Zhou.

    Args:
        m_star (float): mass of star (unit in solar mass)
        m_pl (float): mass of planet (unit in solar mass)
        a (float): semi-majon axis (unit in AU)
        ecc (float): eccentricity
        inc (float): inclination (unit in rad)
        w (float): Argument of periapsis (unit in rad)
        o (float): Longitude of the ascending node (unit in rad)
        m0 (float): Mean anomaly at t=0 (unit in rad)
        t (float or np.array): the time point to solve the plsition of the planet (unit in day)

    Returns:
        (float, float, float): x-, y- and z-coordinates of the planet, refer to the star if input t is float
        (array, array, array) if input t is np.array
    """

    MIN_ERR = 1e-9
    MAX_ITER = 1000
    TPI = 2 * pi

    float_in = False
    t = t * 1.0
    if isinstance(t, float):
        t = [t]
        float_in = True

    t = np.array(t)

    n = np.sqrt((m_star + m_pl) / a**3)

    # Solving Keplerian equaiton using Newton method.

    m_anomaly = np.fmod(m0 + n * t / 365.25 * TPI, TPI)

    e_anomaly = m_anomaly.copy()
    de = 2
    iter = 0
    while np.any(np.abs(de) > MIN_ERR):
        iter += 1
        de = e_anomaly - ecc * np.sin(e_anomaly) - m_anomaly
        # print(de)
        e_anomaly -= de / (1 - ecc * np.cos(e_anomaly))
        if iter > MAX_ITER:
            break
    
    # print(e_anomaly)
    e_array = e_anomaly

    coso, sino, cosw, sinw = np.cos(o), np.sin(o), np.cos(w), np.sin(w)
    cosi, sini = np.cos(inc), np.sin(inc)

    px = coso * cosw - sino * sinw * cosi
    py = sino * cosw + coso * sinw * cosi
    pz = sinw * sini

    qx = -coso * sinw - sino * cosw * cosi
    qy = -sino * sinw + coso * cosw * cosi
    qz = cosw * sini

    coe1 = a * (np.cos(e_array) - ecc)
    coe2 = a * np.sqrt(1 - ecc * ecc) * np.sin(e_array)
    x, y, z = coe1 * px + coe2 * qx, coe1 * py + coe2 * qy, coe1 * pz + coe2 * qz
    r = np.sqrt(x*x + y*y + z*z)
    coe3 = -a * a * n / r * np.sin(e_array)
    coe4 = a * a * n / r * np.sqrt(1 - ecc * ecc) * np.cos(e_array)
    vx, vy, vz = coe3 * px + coe4 * qx, coe3 * py + coe4 * qy, coe3 * pz + coe4 * qz


    if float_in:
        return x[0], y[0], z[0], vx[0], vy[0], vz[0]
    else:
        return x, y, z, vx, vy, vz