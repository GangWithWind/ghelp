import os
import glob

import numpy as np
from astropy.io import fits
import matplotlib as plt
from time import sleep

import warnings
from tqdm import tqdm


def log(string):
    with open('auto_genga.log', 'a') as fid:
        fid.write(f'{string}\n')


def plot_resonant_pos(a_pl, jmax=5, mmax=6, a_min=8, axis=None, text_lvl=4):
    ratio = []
    nm = []
    for j in range(1, jmax):
        for m in range(1, mmax):
            n = m + j
            if np.all(abs(n/m - np.array(ratio)) > 1e-9):
                ratio.append(n/m)
                nm.append([n, m])
    if axis is None:
        axis = plt.gca()
        
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    # print(xlim, ylim)
    text_y = np.arange(0.9, 0, -0.1)

    argind = np.argsort(ratio)
    for j, i in enumerate(argind):
        r = ratio[i]
        (n, m) = nm[i] 
        a_r = a_pl * r**(2/3)
        if a_r < a_min:
            continue
        print(f'{m}/{n}:', r**(2/3))
        now = text_y[j % text_lvl]
        axis.arrow(a_r, ylim[1]*now, 0, -0.1*(ylim[1] - ylim[0]), linewidth=0.5, head_width=0.2, head_length=0.01) 
        axis.text(a_r, ylim[1]*now, f'{m}/{n}', fontsize=8, ha='center')
        
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)



TPI = 2 * np.pi
PI = np.pi

def aei(x, y, z, vx, vy, vz, mu):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rsq = x * x + y * y + z * z
        vsq = vx * vx + vy * vy + vz * vz
        u =  x * vx + y * vy + z * vz
        ir = 1.0 / np.sqrt(rsq)
        ia = 2.0 * ir - vsq / mu

        a = 1.0 / ia

        h3x = ( y * vz) - (z * vy)
        h3y = (-x * vz) + (z * vx)
        h3z = ( x * vy) - (y * vx)

        # inclination
        h2 = h3x * h3x + h3y * h3y + h3z * h3z
        h = np.sqrt(h2)

        inc = np.arctan2(np.sqrt(h3x*h3x+h3y*h3y), h3z)

        # plt.plot(inc, inc2, '.')
        # plt.show()

        # longitude of ascending node
        n = np.sqrt(h3x * h3x + h3y * h3y)
        Omega = np.arccos(-h3y / n)
        Omega[h3x < 0] = TPI - Omega[h3x < 0]
        Omega[(inc < 1.0e-10) | (n == 0)] = 0


        # argument of periapsis
        e3x = ( vy * h3z - vz * h3y) / mu - x * ir
        e3y = (-vx * h3z + vz * h3x) / mu - y * ir
        e3z = ( vx * h3y - vy * h3x) / mu - z * ir

        e = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        t = (-h3y * e3x + h3x * e3y) / (n * e)
        t = np.clip(t, -1.0, 1.0)
        w = np.arccos(t)
        w[e3z < 0] = TPI - w[e3z < 0]
        w[n == 0] = 0.0

        # True Anomaly
        t = (e3x * x + e3y * y + e3z * z) / e * ir
        t = np.clip(t, -1.0, 1.0)
        Theta = np.arccos(t)
        Theta[(u < 0) & (e < 1)] = TPI - Theta[(u < 0) & (e < 1)]
        Theta[(u < 0) & (e >= 1)] *= -1

        iscoplane = (e > 1.0e-10) & (inc < 1.0e-10)
        if iscoplane.any():
            Omega[iscoplane] = 0.0
            w[iscoplane] = np.arccos(e3x / e)[iscoplane]
            w[iscoplane & (e3y < 0.0)] = TPI - w[iscoplane & (e3y < 0.0)]

        w[e < 1.0e-10] = 0.0
        Omega[(e < 1.0e-10) & (inc < 1.0e-11)] = 0.0

        w0on = (w == 0.0) & (Omega != 0.0)
        t = (-h3y * x + h3x * y) / n * ir
        t = np.clip(t, -1.0, 1.0)
        Theta[w0on] = np.arccos(t[w0on])
        zon = (z < 0) & w0on
        Theta[zon & (e < 1 - 1e-10)] = TPI - Theta[zon & (e < 1 - 1e-10)]
        Theta[zon & (e >= 1 + 1e-10)] = -Theta[zon & (e > 1 + 1e-10)]

        w0o0 = (w == 0.0) & (Omega == 0.0)
        t = x * ir
        Theta[w0o0] = np.arccos(t[w0o0])
        zyn = (y < 0) & w0o0
        Theta[zyn & (e < 1 - 1e-10)] = TPI - Theta[zyn & (e < 1 - 1e-10)]
        Theta[zyn & (e >= 1 + 1e-10)] = -Theta[zyn & (e > 1 + 1e-10)]

        is_ellipsis = e < (1.0 - 1.0e-10)
        is_hyperbolic = e > (1.0 + 1.0e-10)
        is_parabolic = (~is_ellipsis) & (~is_hyperbolic)

        ecc_anomaly = Theta.copy()
        mean_anomaly = Omega.copy()

        if is_ellipsis.any():
            t2 = (e[is_ellipsis] + np.cos(Theta[is_ellipsis])) / (1.0 + e[is_ellipsis] * np.cos(Theta[is_ellipsis]))
            t2 = np.clip(t2, -1.0, 1.0)
            ecc_anomaly[is_ellipsis] = np.arccos(t2)
            ecc_anomaly[(Theta > PI) & (Theta < TPI)] = TPI - ecc_anomaly[(Theta > PI) & (Theta < TPI)]
            mean_anomaly = ecc_anomaly - e * np.sin(ecc_anomaly)

        if is_hyperbolic.any():
            t2 = np.cos(Theta[is_hyperbolic])
            t2 = (e[is_hyperbolic] + t2) / (1.0 + t2 * e[is_hyperbolic])
            ecc_anomaly[is_hyperbolic] = np.arccosh(t2)
            ecc_anomaly[(Theta < 0.0) & is_hyperbolic] *= -1
            mean_anomaly[is_hyperbolic] = e[is_hyperbolic] * np.sinh(ecc_anomaly[is_hyperbolic]) - ecc_anomaly[is_hyperbolic]

        if is_parabolic.any():
            ecc_anomaly[is_parabolic] = np.tan(Theta[is_parabolic] * 0.5)
            ecc_anomaly[(ecc_anomaly > np.pi) & is_parabolic] += TPI
            mean_anomaly[is_parabolic] = ecc_anomaly[is_ellipsis] + ecc_anomaly[is_ellipsis]**3 * 3
            a[is_parabolic] = h[is_parabolic]**2 / np.sqrt(mu[is_parabolic])

        return np.array([a, e, inc, Omega, w, Theta, ecc_anomaly, mean_anomaly]).T
    

type_dict = {
    "t": 'f8',
    "i": 'i4',
    "m": 'f8',
    "r": 'f8',
    "x": 'f8',
    "y": 'f8',
    "z": 'f8',
    "vx": 'f8',
    "vy": 'f8',
    "vz": 'f8',
    "Sx": 'f8',
    "Sy": 'f8',
    "Sz": 'f8',
    "amin": 'f4',
    "amax": 'f4',
    "emin": 'f4',
    "emax": 'f4',
    "k2": 'f8',
    "k2f": 'f8',
    "tau": 'f8',
    "Ic": 'f8',
    "aec": 'f4',
    "aecT": 'f4',
    "encc": 'u8',
    "Rc": 'f8',
    "test": 'f8',
}

from numpy.lib.recfunctions import structured_to_unstructured

def load_param(folder):
    with open(f'{folder}/param.dat', 'r') as fid:
            lines = fid.readlines()

    param = {}
    for line in lines:
        sep = ':'
        if '=' in line:
            sep = '='
        
        sub = line.split(sep)
        if len(sub) > 1:
            param[sub[0].strip()] = sub[1].strip()

    return param


def text2bin(folder):
    print(f'Transfer genga text output to bin output...')
    param = load_param(folder)

    interval = int(param['Coordinates output interval'])
    name = param['Output name']
    stepnumber = int(param['Integration steps'])
    columns = param['Output file Format'][2:-2].strip().split(' ')
    step = float(param['Time step in days'])
    
    dtype = np.dtype({
        'names': columns,
        'formats': [type_dict[col] for col in columns]
    })

    binfile = f'{folder}/Out{name}.bin'
    pbar = tqdm(total=stepnumber//interval)
    with open(binfile, 'wb') as fid:
        for ind, step in enumerate(range(0, stepnumber+1, interval)):
            filename = f'{folder}/Out{name}_{step:012d}.dat'
            try:
                allxy = np.loadtxt(filename, dtype=dtype)
                fid.write(allxy.tobytes())
            except OSError:
                break
            pbar.update(1)
    pbar.close()

def xv_func(structured_array, name=False, col=False, cmass=1):
    col_name = ['t', 'm', 'r', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    if name:
        return 'all_xyz'
    elif col:
        return col_name

    return structured_to_unstructured(structured_array[col_name])


def aei_func(structured_array, name=False, col=False, cmass=1):
    col_name = ['t', 'm', 'r', 'a', 'e', 'inc', 'w', 'Omega', 'Theta', 'E', 'M']
    if name:
        return 'all_aei'
    elif col:
        return col_name
    aei_res = aei(
        structured_array['x'],
        structured_array['y'], 
        structured_array['z'], 
        structured_array['vx'], 
        structured_array['vy'], 
        structured_array['vz'], 
        cmass + structured_array['m']
    )

    tmr = structured_to_unstructured(structured_array[['t', 'm', 'r']])
    return np.hstack([tmr, aei_res])


def bin2other(folder, func, patch=16383):
    
    print(f'Transfer bin output to {func(None, name=True)}.fits')
    param = load_param(folder)
    interval = int(param['Coordinates output interval'])
    name = param['Output name']
    stepnumber = int(param['Integration steps'])
    cmass = float(param['Central Mass'])
    columns = param['Output file Format'][2:-2].strip().split(' ')
    step = float(param['Time step in days'])
    interval_year = step * interval / 365.25

    inputfile = param['Input file']
    with open(f'{folder}/{inputfile}', 'r') as fid:
        lines = fid.readlines()
    nparticle = len(lines)

    dtype = np.dtype({
        'names': columns,
        'formats': [type_dict[col] for col in columns]
    })

    file = f'{folder}/Out{name}.bin'
    nstep = stepnumber//interval+1
    col = func(None, col=True)
    ndim = len(col)
    out_array = np.full([nstep+1, nparticle, ndim], np.nan)
    now = 0
    
    stats = os.stat(file)
    size = stats.st_size // dtype.itemsize

    pbar = tqdm(total=size//patch)
    while now < size:
        if now + patch > size:
            patch = size - now
        raw = np.fromfile(file, dtype=dtype, count=patch, offset=(now * dtype.itemsize))
        now = now + patch

        allt = np.unique(raw['t'])
        
        for t in np.array(allt):
            ind_f = t / interval_year
            ind = int(round(ind_f))
            if (ind_f - ind) > 0.01:
                break
            ind_max = ind
            array_t = raw[raw['t']==t]
            out_array[ind, array_t['i'], :] = func(array_t, cmass=cmass)
        pbar.update(1)
    pbar.close()
    out_name = func(None, name=True)    
    out_array = out_array[:ind_max+1, :, :]

    print(f'\nSaving to {out_name}.fits....')
    header = fits.Header()
    header['columns'] = ','.join(col)
    header['cmass'] = cmass
    fits.writeto(f'{folder}/{out_name}.fits', out_array, overwrite=True)



def clear_text(folder):
    print('Clearing text output of genga')
    param = load_param(folder)
    name = param['Output name']
    files = glob.glob(f'{folder}/Out{name}_*.dat')
    qbar = tqdm(total=len(files))
    for file in files:
        os.remove(file)
        qbar.update(1)
    qbar.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='transfer GENGA output to fits')
    parser.add_argument('folder', type=str, help='folder to process')
    parser.add_argument('-b', '--bin', action='store_true', help='transfer text output to bin', default=False)
    parser.add_argument('-x', '--xyz', action='store_true', help='transfer bin output to xyz fits', default=False)
    parser.add_argument('-a', '--aei', action='store_true', help='transfer bin output to keplerian fits', default=False)
    parser.add_argument('-c', '--clear', action='store_true', help='clear text output', default=False)
    parser.add_argument('--log', action='store_true', help='save log', default=False)


    args = parser.parse_args()

    if args.log:
        log(f'Processing {args.folder} to fits...')
    else:
        print(f'Processing {args.folder} to fits...')
        
    if args.bin:
        text2bin(args.folder)

    if args.xyz:
        bin2other(args.folder, xv_func)

    if args.aei:
        bin2other(args.folder, aei_func)
    
    if args.clear:
        clear_text(args.folder)

