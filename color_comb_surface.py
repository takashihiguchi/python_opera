# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.tri as tri 
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

import pandas as pd
import numpy as np
import os
import sys

from color_sub import * 

identifier = str(sys.argv[1])
## e.g. -1000A

# dir_dip =  'superB_1E5A_000A_fine/'
# plot_dir = 'plots_B_1E5A_000A_fine/'
dir_dip='D_dipole_000A_000A_mu_1E5_MB_R_LE_500_LES_510_SHELL/'
dir_coil='D_zero_%s_%s_mu_1E5_MB_R_LE_500_LES_510_SHELL/' %(identifier, identifier)
suptitles = ''
# dir_dip =  'dipole3_coils_%s/' %identifier
plot_dir = 'plots_cm_%s/' %identifier

texts = dir_dip.split('_')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## max value is taken from 5*sigma 
SD = 10


pl_pz1p = pd.read_csv(dir_dip + 'plane_pz+.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz1m = pd.read_csv(dir_dip + 'plane_pz1-.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz2p = pd.read_csv(dir_dip + 'plane_pz2+.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz2m = pd.read_csv(dir_dip + 'plane_pz2-.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz3p = pd.read_csv(dir_dip + 'plane_pz3+.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz3m = pd.read_csv(dir_dip + 'plane_pz3-.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 

pl_pz1p_c = pd.read_csv(dir_coil + 'plane_pz+.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz1m_c = pd.read_csv(dir_coil + 'plane_pz1-.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz2p_c = pd.read_csv(dir_coil + 'plane_pz2+.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz2m_c = pd.read_csv(dir_coil + 'plane_pz2-.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz3p_c = pd.read_csv(dir_coil + 'plane_pz3+.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 
pl_pz3m_c = pd.read_csv(dir_coil + 'plane_pz3-.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi']) 

# pl_0x = pd.read_csv(dir_dip + 'plane_0x.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_px = pd.read_csv(dir_dip + 'plane_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_mx = pd.read_csv(dir_dip + 'plane_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_0y = pd.read_csv(dir_dip + 'plane_0y.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_py = pd.read_csv(dir_dip + 'plane_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_my = pd.read_csv(dir_dip + 'plane_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_0z = pd.read_csv(dir_dip + 'plane_0z.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_pz = pd.read_csv(dir_dip + 'plane_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_mz = pd.read_csv(dir_dip + 'plane_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

# gr_px = pd.read_csv(dir_dip + 'grids_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_mx = pd.read_csv(dir_dip + 'grids_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_py = pd.read_csv(dir_dip + 'grids_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_my = pd.read_csv(dir_dip + 'grids_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_pz = pd.read_csv(dir_dip + 'grids_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_mz = pd.read_csv(dir_dip + 'grids_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

# pl_0x_c = pd.read_csv(dir_coil + 'plane_0x.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_px_c = pd.read_csv(dir_coil + 'plane_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_mx_c = pd.read_csv(dir_coil + 'plane_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_0y_c = pd.read_csv(dir_coil + 'plane_0y.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_py_c = pd.read_csv(dir_coil + 'plane_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_my_c = pd.read_csv(dir_coil + 'plane_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_0z_c = pd.read_csv(dir_coil + 'plane_0z.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_pz_c = pd.read_csv(dir_coil + 'plane_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# pl_mz_c = pd.read_csv(dir_coil + 'plane_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

# gr_px_c = pd.read_csv(dir_coil + 'grids_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_mx_c = pd.read_csv(dir_coil+ 'grids_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_py_c = pd.read_csv(dir_coil + 'grids_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_my_c = pd.read_csv(dir_coil  + 'grids_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_pz_c = pd.read_csv(dir_coil + 'grids_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
# gr_mz_c = pd.read_csv(dir_coil + 'grids_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])


# pl_0x['B'] = np.sqrt(pl_0x.Bx**2+pl_0x.By**2+pl_0x.Bz**2)
# pl_px['B'] = np.sqrt(pl_px.Bx**2+pl_px.By**2+pl_px.Bz**2)
# pl_mx['B'] = np.sqrt(pl_mx.Bx**2+pl_mx.By**2+pl_mx.Bz**2)
# pl_0y['B'] = np.sqrt(pl_0y.Bx**2+pl_0y.By**2+pl_0y.Bz**2)
# pl_py['B'] = np.sqrt(pl_py.Bx**2+pl_py.By**2+pl_py.Bz**2)
# pl_my['B'] = np.sqrt(pl_my.Bx**2+pl_my.By**2+pl_my.Bz**2)
# pl_0z['B'] = np.sqrt(pl_0z.Bx**2+pl_0z.By**2+pl_0z.Bz**2)
# pl_pz['B'] = np.sqrt(pl_pz.Bx**2+pl_pz.By**2+pl_pz.Bz**2)
# pl_mz['B'] = np.sqrt(pl_mz.Bx**2+pl_mz.By**2+pl_mz.Bz**2)

# gr_px['B'] = np.sqrt(gr_px.Bx**2+gr_px.By**2+gr_px.Bz**2)
# gr_mx['B'] = np.sqrt(gr_mx.Bx**2+gr_mx.By**2+gr_mx.Bz**2)
# gr_py['B'] = np.sqrt(gr_py.Bx**2+gr_py.By**2+gr_py.Bz**2) 
# gr_my['B'] = np.sqrt(gr_my.Bx**2+gr_my.By**2+gr_my.Bz**2)
# gr_pz['B'] = np.sqrt(gr_pz.Bx**2+gr_pz.By**2+gr_pz.Bz**2)
# gr_mz['B'] = np.sqrt(gr_mz.Bx**2+gr_mz.By**2+gr_mz.Bz**2)

pl_pz_cm = pl_pz[['Bx','By','Bz','Phi']].add(pl_pz_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz_cm[['x','y','z']]=  pl_pz[['x','y','z']]
pl_pz_cm['B'] = np.sqrt(pl_pz_cm.Bx**2+pl_pz_cm.By**2+pl_pz_cm.Bz**2)

pl_pz1p_cm = pl_pz1p[['Bx','By','Bz','Phi']].add(pl_pz1p_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz1p_cm[['x','y','z']]=  pl_pz1p[['x','y','z']]
pl_pz1p_cm['B'] = np.sqrt(pl_pz1p_cm.Bx**2+pl_pz1p_cm.By**2+pl_pz1p_cm.Bz**2)

pl_pz1m_cm = pl_pz1m[['Bx','By','Bz','Phi']].add(pl_pz1m_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz1m_cm[['x','y','z']]=  pl_pz1m[['x','y','z']]
pl_pz1m_cm['B'] = np.sqrt(pl_pz1m_cm.Bx**2+pl_pz1m_cm.By**2+pl_pz1m_cm.Bz**2)

pl_pz2p_cm = pl_pz1p[['Bx','By','Bz','Phi']].add(pl_pz1p_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz2p_cm[['x','y','z']]=  pl_pz1p[['x','y','z']]
pl_pz2p_cm['B'] = np.sqrt(pl_pz1p_cm.Bx**2+pl_pz1p_cm.By**2+pl_pz1p_cm.Bz**2)

pl_pz2m_cm = pl_pz2m[['Bx','By','Bz','Phi']].add(pl_pz2m_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz2m_cm[['x','y','z']]=  pl_pz2m[['x','y','z']]
pl_pz2m_cm['B'] = np.sqrt(pl_pz2m_cm.Bx**2+pl_pz2m_cm.By**2+pl_pz2m_cm.Bz**2)

pl_pz3p_cm = pl_pz3p[['Bx','By','Bz','Phi']].add(pl_pz3p_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz3p_cm[['x','y','z']]=  pl_pz3p[['x','y','z']]
pl_pz3p_cm['B'] = np.sqrt(pl_pz3p_cm.Bx**2+pl_pz3p_cm.By**2+pl_pz3p_cm.Bz**2)

pl_pz3m_cm = pl_pz3m[['Bx','By','Bz','Phi']].add(pl_pz3m_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz3m_cm[['x','y','z']]=  pl_pz3m[['x','y','z']]
pl_pz3m_cm['B'] = np.sqrt(pl_pz3m_cm.Bx**2+pl_pz3m_cm.By**2+pl_pz3m_cm.Bz**2)



# Bx_max0 = np.max([np.max(sig_cut(pl_0x_cm.Bx, SD)),
#                 np.max(sig_cut(pl_0y_cm.Bx, SD)),
#                 np.max(sig_cut(pl_0z_cm.Bx, SD))])
# Bx_min0 = np.min([np.min(sig_cut(pl_0x_cm.Bx, SD)),
#                 np.min(sig_cut(pl_0y_cm.Bx, SD)),
#                 np.min(sig_cut(pl_0z_cm.Bx, SD))])

# By_max0 = np.max([np.max(sig_cut(pl_0x_cm.By, SD)),
#                 np.max(sig_cut(pl_0y_cm.By, SD)),
#                 np.max(sig_cut(pl_0z_cm.By, SD))])
# By_min0 = np.min([np.min(sig_cut(pl_0x_cm.By, SD)),
#                 np.min(sig_cut(pl_0y_cm.By, SD)),
#                 np.min(sig_cut(pl_0z_cm.By, SD))])

# Bz_max0 = np.max([np.max(sig_cut(pl_0x_cm.Bz, SD)),
#                 np.max(sig_cut(pl_0y_cm.Bz, SD)),
#                 np.max(sig_cut(pl_0z_cm.Bz, SD))])
# Bz_min0 = np.min([np.min(sig_cut(pl_0x_cm.Bz, SD)),
#                 np.min(sig_cut(pl_0y_cm.Bz, SD)),
#                 np.min(sig_cut(pl_0z_cm.Bz, SD))])
# B_max0 = np.max([np.max(sig_cut(pl_0x_cm.B, SD)),
#                 np.max(sig_cut(pl_0y_cm.B, SD)),
#                 np.max(sig_cut(pl_0z_cm.B, SD))])
# B_min0 = np.min([np.min(sig_cut(pl_0x_cm.B, SD)),
#                 np.min(sig_cut(pl_0y_cm.B, SD)),
#                 np.min(sig_cut(pl_0z_cm.B, SD))])


# Bx_maxpm = np.max([np.max(sig_cut(pl_px_cm.Bx, SD)),np.max(sig_cut(pl_mx_cm.Bx, SD)),
#                 np.max(sig_cut(pl_py_cm.Bx, SD)),np.max(sig_cut(pl_my_cm.Bx, SD)),
#                 np.max(sig_cut(pl_pz_cm.Bx, SD)),np.max(sig_cut(pl_mz_cm.Bx, SD))])
# Bx_minpm = np.min([np.min(sig_cut(pl_px_cm.Bx, SD)),np.min(sig_cut(pl_mx_cm.Bx, SD)),
#                 np.min(sig_cut(pl_py_cm.Bx, SD)),np.min(sig_cut(pl_my_cm.Bx, SD)),
#                 np.min(sig_cut(pl_pz_cm.Bx, SD)),np.min(sig_cut(pl_mz_cm.Bx, SD))])

# By_maxpm = np.max([np.max(sig_cut(pl_px_cm.By, SD)),np.max(sig_cut(pl_mx_cm.By, SD)),
#                   np.max(sig_cut(pl_py_cm.By, SD)),np.max(sig_cut(pl_my_cm.By, SD)),
#                   np.max(sig_cut(pl_pz_cm.By, SD)),np.max(sig_cut(pl_mz_cm.By, SD))])
# By_minpm = np.min([np.min(sig_cut(pl_px_cm.By, SD)),np.min(sig_cut(pl_mx_cm.By, SD)),
#                   np.min(sig_cut(pl_py_cm.By, SD)),np.min(sig_cut(pl_my_cm.By, SD)),
#                   np.min(sig_cut(pl_pz_cm.By, SD)),np.min(sig_cut(pl_mz_cm.By, SD))])

# Bz_maxpm = np.max([np.max(sig_cut(pl_px_cm.Bz, SD)),np.max(sig_cut(pl_mx_cm.Bz, SD)),
#                   np.max(sig_cut(pl_py_cm.Bz, SD)),np.max(sig_cut(pl_my_cm.Bz, SD)),
#                   np.max(sig_cut(pl_pz_cm.Bz, SD)),np.max(sig_cut(pl_mz_cm.Bz, SD))])
# Bz_minpm = np.min([np.min(sig_cut(pl_px_cm.Bz, SD)),np.min(sig_cut(pl_mx_cm.Bz, SD)),
#                    np.min(sig_cut(pl_py_cm.Bz, SD)),np.min(sig_cut(pl_my_cm.Bz, SD)),
#                    np.min(sig_cut(pl_pz_cm.Bz, SD)),np.min(sig_cut(pl_mz_cm.Bz, SD))])
# B_maxpm = np.max([np.max(sig_cut(pl_px_cm.B, SD)),np.max(sig_cut(pl_mx_cm.B, SD)),
#                 np.max(sig_cut(pl_py_cm.B, SD)),np.max(sig_cut(pl_my_cm.B, SD)),
#                 np.max(sig_cut(pl_pz_cm.B, SD)),np.max(sig_cut(pl_mz_cm.B, SD))])
# B_minpm = np.min([np.min(sig_cut(pl_px_cm.B, SD)),np.min(sig_cut(pl_mx_cm.B, SD)),
#                 np.min(sig_cut(pl_py_cm.B, SD)),np.min(sig_cut(pl_my_cm.B, SD)),
#                 np.min(sig_cut(pl_pz_cm.B, SD)),np.min(sig_cut(pl_mz_cm.B, SD))])

def contour_dice(ax, lab_tgt, lab_ind, zmin, zmax):
    cont10 = tricont_sub(ax[0,1],  pl_pz_cm.y, pl_pz_cm.x, pl_pz_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont11 = tricont_sub(ax[1,0],  pl_my_cm.x, pl_my_cm.z, pl_my_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont12 = tricont_sub(ax[1,1],  pl_px_cm.y, pl_px_cm.z, pl_px_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont13 = tricont_sub(ax[1,2],  pl_py_cm.x, pl_py_cm.z, pl_py_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont14 = tricont_sub(ax[1,3],  pl_mx_cm.y, pl_mx_cm.z, pl_mx_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont15 = tricont_sub(ax[2,1],  pl_mz_cm.y, pl_mz_cm.x, pl_mz_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    return cont10 

def sct_dice(ax, lab_tgt, lab_ind, zmin, zmax):
    sct10 = sct_sub(ax[0,1],  pl_pz_cm.y, pl_pz_cm.x, pl_pz_cm[lab_tgt], zmin, zmax)
    sct11 = sct_sub(ax[1,0],  pl_my_cm.x, pl_my_cm.z, pl_my_cm[lab_tgt], zmin, zmax)
    sct12 = sct_sub(ax[1,1],  pl_px_cm.y, pl_px_cm.z, pl_px_cm[lab_tgt], zmin, zmax)
    sct13 = sct_sub(ax[1,2],  pl_py_cm.x, pl_py_cm.z, pl_py_cm[lab_tgt], zmin, zmax)
    sct14 = sct_sub(ax[1,3],  pl_mx_cm.y, pl_mx_cm.z, pl_mx_cm[lab_tgt], zmin, zmax)
    sct15 = sct_sub(ax[2,1],  pl_mz_cm.y, pl_mz_cm.x, pl_mz_cm[lab_tgt], zmin, zmax)
    return sct10



def cbar_dice(fig, ax, cont, clims, Nc):
    Nc = 2*(Nc//2) +1
    fig.subplots_adjust(wspace = .7, hspace = .4)
    # cticks = [clims[0] + (clims[1]-clims[0])/Nc *i for i in range(Nc)]
    cticks = np.linspace(clims[0], clims[1], Nc)
    cbar0 = fig.colorbar(cont, ax=ax.ravel().tolist(), ticks=cticks, format=ticker.FuncFormatter(fmt),aspect=40)
    cbar0.ax.get_yaxis().labelpad = 15
    cbar0.ax.set_ylabel(lab_ind, rotation=270, horizontalalignment='right')
    cbar0.set_clim(clims)
    return cbar0 

# def contour_cut(fig, ax, lab_tgt, lab_ind, zmin, zmax):
#     cont20, clim20 = tricont_sub(ax[0],pl_0x_cm.y, pl_0x_cm.z, pl_0x[lab_tgt], zmin, zmax, 31, lab_ind)
#     cont21, clim21 = tricont_sub(ax[1],pl_0y_cm.x, pl_0y_cm.z, pl_0y[lab_tgt], zmin, zmax, 31, lab_ind)
#     cont22, clim22 = tricont_sub(ax[2],pl_0z_cm.x, pl_0z_cm.y, pl_0z[lab_tgt], zmin, zmax, 31, lab_ind)
#     ax[0].set_title('cut at x=0')
#     ax[1].set_title('cut at y=0')
#     ax[2].set_title('cut at z=0')
    
#     for i in range(3):
#         ax[i].set_aspect('equal')
#         ax[i].set_xlim(-300,300)
#         ax[i].set_ylim(-300,300)
#         ax[i].xaxis.set_minor_locator(MultipleLocator(50))
#         ax[i].xaxis.set_major_locator(MultipleLocator(100))
#         ax[i].xaxis.set_major_formatter(FormatStrFormatter("%d"))
#         ax[i].yaxis.set_minor_locator(MultipleLocator(50))
#         ax[i].yaxis.set_major_locator(MultipleLocator(100))
#         ax[i].yaxis.set_major_formatter(FormatStrFormatter("%d"))
    
#     ax[0].set_xlabel('y (cm)')
#     ax[0].set_ylabel('z (cm)')
#     ax[1].set_xlabel('x (cm)')
#     ax[1].set_ylabel('z (cm)')
#     ax[2].set_xlabel('x (cm)')
#     ax[2].set_ylabel('y (cm)')
#     cbar2 = fig.colorbar(cont20, ax=ax.ravel().tolist(), format=ticker.FuncFormatter(fmt),aspect=40)
#     cbar2.ax.get_yaxis().labelpad = 15
#     cbar2.ax.set_ylabel(lab_ind, rotation=270)
#     fig.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
#     return 0

def contour_surface(fig, ax, lab_tgt, lab_ind, zmin, zmax):
    Nc =  31

    cont20, clim20 = tricont_sub(ax[0],pl_pz3m_cm.x, pl_pz3m_cm.y, pl_pz3m_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont21, clim21 = tricont_sub(ax[1],pl_pz2m_cm.x, pl_pz2m_cm.y, pl_pz2m_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    cont22, clim22 = tricont_sub(ax[2],pl_pz1m_cm.x, pl_pz1m_cm.y, pl_pz1m_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    # cont23, clim23 = tricont_sub(ax[3],pl_pz_cm.x, pl_pz_cm.y, pl_pz_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    # cont24, clim24 = tricont_sub(ax[4],pl_pz1p_cm.x, pl_pz1p_cm.y, pl_pz1p_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    # cont25, clim25 = tricont_sub(ax[5],pl_pz2p_cm.x, pl_pz2p_cm.y, pl_pz2p_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    # cont26, clim26 = tricont_sub(ax[6],pl_pz3p_cm.x, pl_pz3p_cm.y, pl_pz3p_cm[lab_tgt], zmin, zmax, 31, lab_ind)
    ax[0].set_title('cut at z=%.2f cm' %(pl_pz3m_cm.z[0]))
    ax[1].set_title('cut at z=%.2f cm' %(pl_pz2m_cm.z[0]))
    ax[2].set_title('cut at z=%.2f cm' %(pl_pz1m_cm.z[0]))
    ax[3].set_title('cut at z=%.2f cm' %(pl_pz_cm.z[0]))
    ax[4].set_title('cut at z=%.2f cm' %(pl_pz1p_cm.z[0]))
    ax[5].set_title('cut at z=%.2f cm' %(pl_pz2p_cm.z[0]))
    ax[6].set_title('cut at z=%.2f cm' %(pl_pz3p_cm.z[0]))

    
    for i in range(7):
        ax[i].set_aspect('equal')
        ax[i].set_xlim(-200,200)
        ax[i].set_ylim(-200,200)
        ax[i].xaxis.set_minor_locator(MultipleLocator(50))
        ax[i].xaxis.set_major_locator(MultipleLocator(100))
        ax[i].xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax[i].yaxis.set_minor_locator(MultipleLocator(50))
        ax[i].yaxis.set_major_locator(MultipleLocator(100))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax[i].set_xlabel('x (cm)')
        ax[i].set_xlabel('y (cm)')
    # clim = [min(clim20[0], clim21[0], clim21[0], clim22[0], clim23[0], clim24[0], clim25[0], clim26[0]), max(clim20[1], clim21[1], clim21[1], clim22[1], clim23[1], clim24[1], clim25[1], clim26[1])]
    clim = [min(clim20[0], clim21[0], clim22[0],  ), max(clim20[1], clim21[1],  clim22[1])]

    cticks2 = np.linspace(clim[0], clim[1], Nc)


    cbar2 = fig.colorbar(cont20, ax=ax.ravel().tolist(), ticks=cticks2, format=ticker.FuncFormatter(fmt),aspect=40)
    cbar2.ax.get_yaxis().labelpad = 15
    cbar2.set_clim(clim)
    cbar2.ax.set_ylabel(lab_ind, rotation=270)
    fig.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
    return 0


lab_tgt = 'Bx'
lab_ind = '$B_x$ (G)'
Bx_maxpm = 3e3
Bx_minpm = -3e3
fig2x, ax2x = plt.subplots(1,7, figsize=(19,4.5), constrained_layout=True)
contour_surface(fig2x, ax2x, lab_tgt, lab_ind, Bx_minpm, Bx_maxpm)
fig2x.savefig(plot_dir + 'surface_%s_' %lab_tgt + dir_dip[:-1])

lab_tgt = 'By'
lab_ind = '$B_y$ (G)'
By_maxpm = 3e3
By_minpm = -3e3
fig2y, ax2y = plt.subplots(1,7, figsize=(19,4.5), constrained_layout=True)
contour_surface(fig2y, ax2y, lab_tgt, lab_ind, By_minpm, By_maxpm)
fig2y.savefig(plot_dir + 'surface_%s_' %lab_tgt + dir_dip[:-1])

lab_tgt = 'Bz'
lab_ind = '$B_z$ (G)'
Bz_maxpm = 3e3
Bz_minpm = -3e3
fig2z, ax2z = plt.subplots(1,7, figsize=(19,4.5), constrained_layout=True)
contour_surface(fig2z, ax2z, lab_tgt, lab_ind, Bz_minpm, Bz_maxpm)
fig2z.savefig(plot_dir + 'surface_%s_' %lab_tgt + dir_dip[:-1])


lab_tgt = 'B'
lab_ind = '$|B|$ (G)'
B_maxpm = 3e3
B_minpm = -3e3
fig2n, ax2n = plt.subplots(1,7, figsize=(19,4.5), constrained_layout=True)
contour_surface(fig2n, ax2n, lab_tgt, lab_ind, B_minpm, B_maxpm)
fig2z.savefig(plot_dir + 'surface_%s_' %lab_tgt + dir_dip[:-1])
