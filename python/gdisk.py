import numpy as np
import yaml
import os
from g2fits import log

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Write parameter file for Genga')
    this_path = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument('output', type=str, help='name of project')
    parser.add_argument('-y', '--yaml', type=str, help='yaml config file', default=f"{this_path}/default_config.yaml")
    parser.add_argument('--center_mass', type=float, help='center mass [Msun]')
    parser.add_argument('--n_particle', type=int, help='number of particles')
    parser.add_argument('--disk_mass', type=float, help='disk mass [Mearth]')
    parser.add_argument('--disk_in', type=float, help='disk inner [AU]')
    parser.add_argument('--disk_out', type=float, help='disk outer [AU]')
    parser.add_argument('--disk_e', type=float, help='max eccentricity of particles')
    parser.add_argument('--disk_i', type=float, help='max inclination of particles [degree]')
    parser.add_argument('--disk_profile', type=float, help='disk surface density profile \Sigma * (r/r_out)^-p')
    parser.add_argument('--planet_mass', type=float, help='planet mass [Mjupiter]')
    parser.add_argument('--planet_a', type=float, help='planet semimajor axis [AU]')
    parser.add_argument('--planet_i', type=float, help='planet inclination [degree]')
    parser.add_argument('--planet_e', type=float, help='planet inclination [degree]')
    parser.add_argument('--t_end', type=float, help='end of simulation [kyr]')
    parser.add_argument('--step_ratio', type=float, help='time step ratio, time_step = planet period / timestep_ratio')
    parser.add_argument('--interval', type=float, help='output interval [yr]')
    parser.add_argument('--log', action='store_true', help='save log', default=False)

    args = parser.parse_args()

    if args.log:
        print = log

    in_args = vars(args)
    
    # new dict only for item with value "is not None"
    args = {k: v for k, v in in_args.items() if v is not None}
    
    with open(args['yaml']) as fid:
        default = yaml.load(fid, Loader=yaml.FullLoader)

    center_mass = args.get('center_mass', default['center_mass'])
    n_particle = args.get('n_particle', default['n_particle'])
    disk_mass = args.get('disk_mass', default['disk_mass'])
    disk_in = args.get('disk_in', default['disk_in'])
    disk_out = args.get('disk_out', default['disk_out'])
    disk_e = args.get('disk_e', default['disk_e'])
    disk_i = args.get('disk_i', default['disk_i'])
    disk_profile = args.get('disk_profile', default['disk_profile'])

    sigma0 = 2 * np.pi / (2 - disk_profile) \
        * (1 - (disk_out / disk_in)**(disk_profile - 2))
    
    pl_mass = args.get('planet_mass', default['planet_mass'])
    pl_a = args.get('planet_a', default['planet_a'])
    pl_i = args.get('planet_i', default['planet_i'])
    pl_e = args.get('planet_e', default['planet_e'])

    default_name = f'm{pl_mass:.1f}a{pl_a:.1f}i{pl_i:.0f}d{disk_mass:.0f}'
    default_name = default_name.replace('.0', '')
    default_name = default_name.replace('.', '_')
    output = args.get('output', default['name'])
    output = output.replace('$D', default_name)
    print(f'Writing config file {output}....')
    

    aout = disk_out * 1.5e13 #AU to cm
    dmass = disk_mass * 6.0e27 #earth mass to g
    sigma0 =  dmass / sigma0 / aout**2
    print(f'disk surface mass {sigma0:.2e} g/cm^2, {disk_mass/sigma0/disk_out**2:.2e}Me/AU^2')


    endtime = args.get('t_end', default['t_end'])
    timestep_ratio = args.get('step_ratio', default['step_ratio'])
    interval_year = args.get('interval', default['interval'])

    max_e = 0.99
    min_a = np.minimum(disk_in, pl_a)
    max_a = np.maximum(disk_out, pl_a)

    period = np.sqrt(min_a**3 / center_mass)  #year
    step = period / timestep_ratio
    if step > 0:
        step = int(step * 10)/10
    step = abs(step * 365.25) #day

    interval = int(interval_year * 365.25 / step)
    end_step = int(endtime * 365.25 * 1e3 / step)

    print(f'period_min {period:.2f}yr, step {step/365.25:.2f}yr, output every {interval}step/{interval_year:.1f}yr, end at {end_step}step/{endtime:.2f}Myr')

    with open(f'{this_path}/param_tamplate.dat', 'r') as fid:
        param = fid.read()

    inner_trunc = min_a * (1 - max_e)
    outer_trunc = max_a * (1 + max_e)

    param = param.format(
        step=step, 
        interval=interval, 
        nstep=end_step, 
        center=center_mass, 
        inner_trun=inner_trunc, 
        outer_trun=outer_trunc
    )

    if os.path.exists(output):
        print(f'Project {output} already exists')
        raise FileExistsError(f'Project {output} already exists')
    
    os.makedirs(output)
    with open(f'{output}/param.dat', 'w') as fid:
        fid.write(param)

    with open(f'{output}/particle.dat', 'w') as fid:
        for i_particle in range(n_particle + 1):
            a = np.random.rand() * (disk_out - disk_in) + disk_in
            e = np.random.rand() * disk_e
            inc = np.random.rand() * disk_i * np.pi / 180
            w = np.random.rand() * 2 * np.pi
            omega = np.random.rand() * 2 * np.pi
            orbit_m = np.random.rand() * 2 * np.pi
            mass = disk_mass / n_particle / 330000

            if i_particle == 0:
                mass = pl_mass * 1e-3
                a = pl_a
                inc = pl_i * np.pi / 180
                e = pl_e 

            fid.write(f'{mass} {a} {e} {inc} {omega} {w} {orbit_m} \n')