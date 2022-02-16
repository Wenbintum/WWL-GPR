import numpy as np
import pickle
from scipy.integrate import simps
from ase.units import Ry



def WorkFunction(outputfile, E_fermi):
    with open(outputfile, 'r') as infile:
        lines=infile.readlines()
    for i, line in enumerate(lines):
        if line.startswith(' CRYSTAL'):
            lines=lines[i:]
            xsf_start = i
        if line.startswith('DATAGRID_3D_UNKNOWN') or line.startswith('BEGIN_DATAGRID_3D_UNKNOWN'):
            number1 = i - xsf_start + 1
            number2 = number1 + 5
    nn = lines[number1].strip().split(' ')
    nx,ny,nz=[int(x) for x in nn if x != '']
    dx=float([x for x in lines[2].split(' ') if x!=''][0])/nx
    x_values = []
    for i in range(nx):
        x_values.append(i*dx)
    dy = float([x for x in lines[3].split(' ') if x!=''][1])/ny
    y_values = []
    for i in range(ny):
        y_values.append(i*dy)
    dz = float([x for x in lines[4].split(' ') if x!=''][2])/nz
    z_values = []
    for i in range(nz):
        z_values.append(i*dz)
    pot = []
    for line in lines[number2:-2]:
        values = line.strip().split(' ')
        values = [float(x)*Ry for x in values if x != '']
        pot += values
    av_pot = []
    n=0
    for i in range(0,nx*ny*nz,nx*ny):
        av_pot.append(sum(pot[n*nx*ny:(n+1)*nx*ny])/(nx*ny))
        n+=1
    #!2 anstr. as a criteria to fit pot value (convergence test should be done)
    d1=2
    d2=z_values[-1] - d1
    z_fit_values = []
    pot_fit_values = []
    for z,p in zip(z_values,av_pot):
        if (z<d1) or (z>d2):
            z_fit_values.append(z)
            pot_fit_values.append(p)
    Ev = max(pot_fit_values)
    Wf = Ev - E_fermi
    return Wf

def d_band_moment(atoms, outputfile):
    #slab feature
    d_band_center_list = []
    d_band_width_list  = []
    d_band_skewness_list = []
    d_band_kurtosis_list = []
    d_band_filling_list  = []
    with open(outputfile, 'rb') as input_file:
        dos_energies, dos_total, pdos = pickle.load(input_file)
    for atom_id in range(len(atoms)):
        states = 'd'
        symbols = [atom.symbol for atom in atoms]
        if 'Fe' in symbols or 'Co' in symbols or 'Ni' in symbols:
            summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
        else:
            summed_pdos = pdos[atom_id][states][0]
        dbc = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
        d_band_center_list.append(dbc)
        d_band_mom2 = simps(summed_pdos*np.power(dos_energies - dbc,2) ,dos_energies) / simps(summed_pdos,dos_energies)
        dbw = np.sqrt(d_band_mom2)
        d_band_width_list.append(dbw)
        d_band_mom3 = simps(summed_pdos*np.power(dos_energies - dbc,3) ,dos_energies) / simps(summed_pdos,dos_energies)
        dbs = d_band_mom3/np.power(dbw,3)
        d_band_skewness_list.append(dbs)
        d_band_mom4 = simps(summed_pdos*np.power(dos_energies - dbc,4) ,dos_energies) / simps(summed_pdos,dos_energies)
        dbk = d_band_mom4/np.power(dbw,4)
        d_band_kurtosis_list.append(dbk)
        filled_pdos = []
        filled_dos_energies = []
        for d,e in zip(summed_pdos,dos_energies):
            if e < 0:
                filled_pdos.append(d)
                filled_dos_energies.append(e)
        dbf = simps(filled_pdos,filled_dos_energies)
        d_band_filling_list.append(dbf)
    return d_band_filling_list, d_band_center_list, d_band_width_list, d_band_skewness_list, d_band_kurtosis_list

