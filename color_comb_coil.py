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
#plot_dir = 'plots_cm_%s/' %identifier
plot_dir = 'plots_c_%s/' %identifier

texts = dir_dip.split('_')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

## max value is taken from 5*sigma 
SD = 10


pl_0x = pd.read_csv(dir_dip + 'plane_0x.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_px = pd.read_csv(dir_dip + 'plane_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_mx = pd.read_csv(dir_dip + 'plane_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_0y = pd.read_csv(dir_dip + 'plane_0y.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_py = pd.read_csv(dir_dip + 'plane_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_my = pd.read_csv(dir_dip + 'plane_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_0z = pd.read_csv(dir_dip + 'plane_0z.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_pz = pd.read_csv(dir_dip + 'plane_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_mz = pd.read_csv(dir_dip + 'plane_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

gr_px = pd.read_csv(dir_dip + 'grids_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_mx = pd.read_csv(dir_dip + 'grids_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_py = pd.read_csv(dir_dip + 'grids_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_my = pd.read_csv(dir_dip + 'grids_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_pz = pd.read_csv(dir_dip + 'grids_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_mz = pd.read_csv(dir_dip + 'grids_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

pl_0x_c = pd.read_csv(dir_coil + 'plane_0x.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_px_c = pd.read_csv(dir_coil + 'plane_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_mx_c = pd.read_csv(dir_coil + 'plane_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_0y_c = pd.read_csv(dir_coil + 'plane_0y.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_py_c = pd.read_csv(dir_coil + 'plane_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_my_c = pd.read_csv(dir_coil + 'plane_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_0z_c = pd.read_csv(dir_coil + 'plane_0z.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_pz_c = pd.read_csv(dir_coil + 'plane_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_mz_c = pd.read_csv(dir_coil + 'plane_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

gr_px_c = pd.read_csv(dir_coil + 'grids_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_mx_c = pd.read_csv(dir_coil+ 'grids_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_py_c = pd.read_csv(dir_coil + 'grids_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_my_c = pd.read_csv(dir_coil  + 'grids_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_pz_c = pd.read_csv(dir_coil + 'grids_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_mz_c = pd.read_csv(dir_coil + 'grids_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])


pl_0x['B'] = np.sqrt(pl_0x.Bx**2+pl_0x.By**2+pl_0x.Bz**2)
pl_px['B'] = np.sqrt(pl_px.Bx**2+pl_px.By**2+pl_px.Bz**2)
pl_mx['B'] = np.sqrt(pl_mx.Bx**2+pl_mx.By**2+pl_mx.Bz**2)
pl_0y['B'] = np.sqrt(pl_0y.Bx**2+pl_0y.By**2+pl_0y.Bz**2)
pl_py['B'] = np.sqrt(pl_py.Bx**2+pl_py.By**2+pl_py.Bz**2)
pl_my['B'] = np.sqrt(pl_my.Bx**2+pl_my.By**2+pl_my.Bz**2)
pl_0z['B'] = np.sqrt(pl_0z.Bx**2+pl_0z.By**2+pl_0z.Bz**2)
pl_pz['B'] = np.sqrt(pl_pz.Bx**2+pl_pz.By**2+pl_pz.Bz**2)
pl_mz['B'] = np.sqrt(pl_mz.Bx**2+pl_mz.By**2+pl_mz.Bz**2)

gr_px['B'] = np.sqrt(gr_px.Bx**2+gr_px.By**2+gr_px.Bz**2)
gr_mx['B'] = np.sqrt(gr_mx.Bx**2+gr_mx.By**2+gr_mx.Bz**2)
gr_py['B'] = np.sqrt(gr_py.Bx**2+gr_py.By**2+gr_py.Bz**2)
gr_my['B'] = np.sqrt(gr_my.Bx**2+gr_my.By**2+gr_my.Bz**2)
gr_pz['B'] = np.sqrt(gr_pz.Bx**2+gr_pz.By**2+gr_pz.Bz**2)
gr_mz['B'] = np.sqrt(gr_mz.Bx**2+gr_mz.By**2+gr_mz.Bz**2)

""" 
pl_0x_cm = pl_0x[['Bx','By','Bz','Phi']].add(pl_0x_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_0x_cm[['x','y','z','Phi']]=  pl_0x[['x','y','z','Phi']]
pl_0x_cm['B'] = np.sqrt(pl_0x_cm.Bx**2+pl_0x_cm.By**2+pl_0x_cm.Bz**2)
pl_px_cm = pl_px[['Bx','By','Bz','Phi']].add(pl_px_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_px_cm[['x','y','z','Phi']]=  pl_px[['x','y','z','Phi']]
pl_px_cm['B'] = np.sqrt(pl_px_cm.Bx**2+pl_px_cm.By**2+pl_px_cm.Bz**2)
pl_mx_cm = pl_mx[['Bx','By','Bz','Phi']].add(pl_mx_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_mx_cm[['x','y','z','Phi']]=  pl_mx[['x','y','z','Phi']]
pl_mx_cm['B'] = np.sqrt(pl_mx_cm.Bx**2+pl_mx_cm.By**2+pl_mx_cm.Bz**2)

pl_0y_cm = pl_0y[['Bx','By','Bz','Phi']].add(pl_0y_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_0y_cm[['x','y','z','Phi']]=  pl_0y[['x','y','z','Phi']]
pl_0y_cm['B'] = np.sqrt(pl_0y_cm.Bx**2+pl_0y_cm.By**2+pl_0y_cm.Bz**2)
pl_py_cm = pl_py[['Bx','By','Bz','Phi']].add(pl_py_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_py_cm[['x','y','z','Phi']]=  pl_py[['x','y','z','Phi']]
pl_py_cm['B'] = np.sqrt(pl_py_cm.Bx**2+pl_py_cm.By**2+pl_py_cm.Bz**2)
pl_my_cm = pl_my[['Bx','By','Bz','Phi']].add(pl_my_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_my_cm[['x','y','z','Phi']]=  pl_my[['x','y','z','Phi']]
pl_my_cm['B'] = np.sqrt(pl_my_cm.Bx**2+pl_my_cm.By**2+pl_my_cm.Bz**2)

pl_0z_cm = pl_0z[['Bx','By','Bz','Phi']].add(pl_0z_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_0z_cm[['x','y','z','Phi']]=  pl_0z[['x','y','z','Phi']]
pl_0z_cm['B'] = np.sqrt(pl_0z_cm.Bx**2+pl_0z_cm.By**2+pl_0z_cm.Bz**2)
pl_pz_cm = pl_pz[['Bx','By','Bz','Phi']].add(pl_pz_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_pz_cm[['x','y','z','Phi']]=  pl_pz[['x','y','z','Phi']]
pl_pz_cm['B'] = np.sqrt(pl_pz_cm.Bx**2+pl_pz_cm.By**2+pl_pz_cm.Bz**2)
pl_mz_cm = pl_mz[['Bx','By','Bz','Phi']].add(pl_mz_c[['Bx','By','Bz','Phi']], fill_value=0)
pl_mz_cm[['x','y','z','Phi']]=  pl_mz[['x','y','z','Phi']]
pl_mz_cm['B'] = np.sqrt(pl_mz_cm.Bx**2+pl_mz_cm.By**2+pl_mz_cm.Bz**2)
 """
pl_0x_cm = pl_0x_c
pl_px_cm = pl_px_c
pl_mx_cm = pl_mx_c
pl_0x_cm['B'] = np.sqrt(pl_0x_cm.Bx**2+pl_0x_cm.By**2+pl_0x_cm.Bz**2)
pl_px_cm['B'] = np.sqrt(pl_px_cm.Bx**2+pl_px_cm.By**2+pl_px_cm.Bz**2)
pl_mx_cm['B'] = np.sqrt(pl_mx_cm.Bx**2+pl_mx_cm.By**2+pl_mx_cm.Bz**2)

pl_0y_cm = pl_0y_c
pl_py_cm = pl_py_c
pl_my_cm = pl_my_c
pl_0y_cm['B'] = np.sqrt(pl_0y_cm.Bx**2+pl_0y_cm.By**2+pl_0y_cm.Bz**2)
pl_py_cm['B'] = np.sqrt(pl_py_cm.Bx**2+pl_py_cm.By**2+pl_py_cm.Bz**2)
pl_my_cm['B'] = np.sqrt(pl_my_cm.Bx**2+pl_my_cm.By**2+pl_my_cm.Bz**2)

pl_0z_cm = pl_0z_c
pl_pz_cm = pl_pz_c
pl_mz_cm = pl_mz_c

pl_0z_cm['B'] = np.sqrt(pl_0z_cm.Bx**2+pl_0z_cm.By**2+pl_0z_cm.Bz**2)
pl_pz_cm['B'] = np.sqrt(pl_pz_cm.Bx**2+pl_pz_cm.By**2+pl_pz_cm.Bz**2)
pl_mz_cm['B'] = np.sqrt(pl_mz_cm.Bx**2+pl_mz_cm.By**2+pl_mz_cm.Bz**2)

Bx_max0 = np.max([np.max(sig_cut(pl_0x_cm.Bx, SD)),
                np.max(sig_cut(pl_0y_cm.Bx, SD)),
                np.max(sig_cut(pl_0z_cm.Bx, SD))])
Bx_min0 = np.min([np.min(sig_cut(pl_0x_cm.Bx, SD)),
                np.min(sig_cut(pl_0y_cm.Bx, SD)),
                np.min(sig_cut(pl_0z_cm.Bx, SD))])

By_max0 = np.max([np.max(sig_cut(pl_0x_cm.By, SD)),
                np.max(sig_cut(pl_0y_cm.By, SD)),
                np.max(sig_cut(pl_0z_cm.By, SD))])
By_min0 = np.min([np.min(sig_cut(pl_0x_cm.By, SD)),
                np.min(sig_cut(pl_0y_cm.By, SD)),
                np.min(sig_cut(pl_0z_cm.By, SD))])

Bz_max0 = np.max([np.max(sig_cut(pl_0x_cm.Bz, SD)),
                np.max(sig_cut(pl_0y_cm.Bz, SD)),
                np.max(sig_cut(pl_0z_cm.Bz, SD))])
Bz_min0 = np.min([np.min(sig_cut(pl_0x_cm.Bz, SD)),
                np.min(sig_cut(pl_0y_cm.Bz, SD)),
                np.min(sig_cut(pl_0z_cm.Bz, SD))])
B_max0 = np.max([np.max(sig_cut(pl_0x_cm.B, SD)),
                np.max(sig_cut(pl_0y_cm.B, SD)),
                np.max(sig_cut(pl_0z_cm.B, SD))])
B_min0 = np.min([np.min(sig_cut(pl_0x_cm.B, SD)),
                np.min(sig_cut(pl_0y_cm.B, SD)),
                np.min(sig_cut(pl_0z_cm.B, SD))])


Bx_maxpm = np.max([np.max(sig_cut(pl_px_cm.Bx, SD)),np.max(sig_cut(pl_mx_cm.Bx, SD)),
                np.max(sig_cut(pl_py_cm.Bx, SD)),np.max(sig_cut(pl_my_cm.Bx, SD)),
                np.max(sig_cut(pl_pz_cm.Bx, SD)),np.max(sig_cut(pl_mz_cm.Bx, SD))])
Bx_minpm = np.min([np.min(sig_cut(pl_px_cm.Bx, SD)),np.min(sig_cut(pl_mx_cm.Bx, SD)),
                np.min(sig_cut(pl_py_cm.Bx, SD)),np.min(sig_cut(pl_my_cm.Bx, SD)),
                np.min(sig_cut(pl_pz_cm.Bx, SD)),np.min(sig_cut(pl_mz_cm.Bx, SD))])

By_maxpm = np.max([np.max(sig_cut(pl_px_cm.By, SD)),np.max(sig_cut(pl_mx_cm.By, SD)),
                  np.max(sig_cut(pl_py_cm.By, SD)),np.max(sig_cut(pl_my_cm.By, SD)),
                  np.max(sig_cut(pl_pz_cm.By, SD)),np.max(sig_cut(pl_mz_cm.By, SD))])
By_minpm = np.min([np.min(sig_cut(pl_px_cm.By, SD)),np.min(sig_cut(pl_mx_cm.By, SD)),
                  np.min(sig_cut(pl_py_cm.By, SD)),np.min(sig_cut(pl_my_cm.By, SD)),
                  np.min(sig_cut(pl_pz_cm.By, SD)),np.min(sig_cut(pl_mz_cm.By, SD))])

Bz_maxpm = np.max([np.max(sig_cut(pl_px_cm.Bz, SD)),np.max(sig_cut(pl_mx_cm.Bz, SD)),
                  np.max(sig_cut(pl_py_cm.Bz, SD)),np.max(sig_cut(pl_my_cm.Bz, SD)),
                  np.max(sig_cut(pl_pz_cm.Bz, SD)),np.max(sig_cut(pl_mz_cm.Bz, SD))])
Bz_minpm = np.min([np.min(sig_cut(pl_px_cm.Bz, SD)),np.min(sig_cut(pl_mx_cm.Bz, SD)),
                   np.min(sig_cut(pl_py_cm.Bz, SD)),np.min(sig_cut(pl_my_cm.Bz, SD)),
                   np.min(sig_cut(pl_pz_cm.Bz, SD)),np.min(sig_cut(pl_mz_cm.Bz, SD))])
B_maxpm = np.max([np.max(sig_cut(pl_px_cm.B, SD)),np.max(sig_cut(pl_mx_cm.B, SD)),
                np.max(sig_cut(pl_py_cm.B, SD)),np.max(sig_cut(pl_my_cm.B, SD)),
                np.max(sig_cut(pl_pz_cm.B, SD)),np.max(sig_cut(pl_mz_cm.B, SD))])
B_minpm = np.min([np.min(sig_cut(pl_px_cm.B, SD)),np.min(sig_cut(pl_mx_cm.B, SD)),
                np.min(sig_cut(pl_py_cm.B, SD)),np.min(sig_cut(pl_my_cm.B, SD)),
                np.min(sig_cut(pl_pz_cm.B, SD)),np.min(sig_cut(pl_mz_cm.B, SD))])


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

def contour_cut(fig, ax, lab_tgt, lab_ind, zmin, zmax):
    cont20, clim20 = tricont_sub(ax[0],pl_0x_cm.y, pl_0x_cm.z, pl_0x[lab_tgt], zmin, zmax, 31, lab_ind)
    cont21, clim21 = tricont_sub(ax[1],pl_0y_cm.x, pl_0y_cm.z, pl_0y[lab_tgt], zmin, zmax, 31, lab_ind)
    cont22, clim22 = tricont_sub(ax[2],pl_0z_cm.x, pl_0z_cm.y, pl_0z[lab_tgt], zmin, zmax, 31, lab_ind)
    ax[0].set_title('cut at x=0')
    ax[1].set_title('cut at y=0')
    ax[2].set_title('cut at z=0')
    
    for i in range(3):
        ax[2].set_aspect('equal')
        ax[i].set_xlim(-300,300)
        ax[i].set_ylim(-300,300)
        ax[i].xaxis.set_minor_locator(MultipleLocator(50))
        ax[i].xaxis.set_major_locator(MultipleLocator(100))
        ax[i].xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax[i].yaxis.set_minor_locator(MultipleLocator(50))
        ax[i].yaxis.set_major_locator(MultipleLocator(100))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter("%d"))
    
    ax[0].set_xlabel('y (cm)')
    ax[0].set_ylabel('z (cm)')
    ax[1].set_xlabel('x (cm)')
    ax[1].set_ylabel('z (cm)')
    ax[2].set_xlabel('x (cm)')
    ax[2].set_ylabel('y (cm)')
    cbar2 = fig.colorbar(cont20, ax=ax.ravel().tolist(), format=ticker.FuncFormatter(fmt),aspect=40)
    cbar2.ax.get_yaxis().labelpad = 15
    cbar2.ax.set_ylabel(lab_ind, rotation=270)
    fig.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
    return 0


lab_tgt = 'Bx'
lab_ind = '$B_x$ (G)'
# Bx_maxpm = 3e3
# Bx_minpm = -3e3
fig1x, ax1x = plt.subplots(3,4, figsize=(14,8))
cont1x, clim1xc = contour_dice(ax1x, lab_tgt, lab_ind, Bx_minpm, Bx_maxpm)
#cont1x.set_clim(Bx_minpm, Bx_maxpm)
cbar1x = cbar_dice(fig1x, ax1x, cont1x, clim1xc, 10) 
dice(ax1x)
# fig1x.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1x.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1x.savefig(plot_dir+'dice_c_%s_' %lab_tgt + dir_dip[:-1])

fig1xs, ax1xs = plt.subplots(3,4, figsize=(14,8))
sct1x, clim1xs = sct_dice(ax1xs, lab_tgt, lab_ind, Bx_minpm, Bx_maxpm)
#sct1x.set_clim(Bx_minpm, Bx_maxpm)
cbar1xs = cbar_dice(fig1xs, ax1xs, sct1x, clim1xs, 10) 
dice(ax1xs)
fig1xs.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1xs.savefig(plot_dir+'dice_s_%s_' %lab_tgt + dir_dip[:-1])

fig2x, ax2x = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2x, ax2x, lab_tgt, lab_ind, Bx_min0, Bx_max0)
fig2x.savefig(plot_dir + 'cut_%s_' %lab_tgt + dir_dip[:-1])


lab_tgt = 'By'
lab_ind = '$B_y$ (G)'
# By_maxpm = 3e3
# By_minpm = -3e3

fig1y, ax1y = plt.subplots(3,4, figsize=(14,8))
cont1y, clim1yc = contour_dice(ax1y, lab_tgt, lab_ind, By_minpm, By_maxpm)
#cont1y.set_clim(By_minpm, By_maxpm)
cbar1y = cbar_dice(fig1y, ax1y, cont1y, clim1yc, 10 )
dice(ax1y)
# fig1y.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1y.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1y.savefig(plot_dir+'dice_c_%s_' %lab_tgt + dir_dip[:-1])

fig1ys, ax1ys = plt.subplots(3,4, figsize=(14,8))
sct1y, clim1ys = sct_dice(ax1ys, lab_tgt, lab_ind, By_minpm, By_maxpm)
#sct1y.set_clim(By_minpm, By_maxpm)
cbar1ys = cbar_dice(fig1ys, ax1ys, sct1y, clim1ys, 10)
dice(ax1ys)
fig1ys.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1ys.savefig(plot_dir+'dice_s_%s_' %lab_tgt + dir_dip[:-1])

fig2y, ax2y = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2y, ax2y, lab_tgt, lab_ind, By_min0, By_max0)
fig2y.savefig(plot_dir+'cut_%s_' %lab_tgt + dir_dip[:-1])

lab_tgt = 'Bz'
lab_ind = '$B_z$ (G)'
# Bz_maxpm = 5.5e3
# Bz_minpm = -5.5e3
fig1z, ax1z = plt.subplots(3,4, figsize=(14,8))
cont1z, clim1zc = contour_dice(ax1z, lab_tgt, lab_ind, Bz_minpm, Bz_maxpm)
#cont1z.set_clim(Bz_minpm, Bz_maxpm)
cbar1z = cbar_dice(fig1z, ax1z, cont1z, clim1zc, 10)
dice(ax1z)
# fig1z.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1z.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1z.savefig(plot_dir+'dice_c_%s_' %lab_tgt + dir_dip[:-1])

fig1zs, ax1zs = plt.subplots(3,4, figsize=(14,8))
sct1z, clim1zs = sct_dice(ax1zs, lab_tgt, lab_ind, Bz_minpm, Bz_maxpm)
#sct1z.set_clim(Bz_minpm, Bz_maxpm)
cbar1zs = cbar_dice(fig1zs, ax1zs, sct1z, clim1zs, 10)
dice(ax1zs)
fig1zs.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1zs.savefig(plot_dir+'dice_s_%s_' %lab_tgt + dir_dip[:-1])

fig2z, ax2z = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2z, ax2z, lab_tgt, lab_ind, Bz_min0, Bz_max0)
fig2z.savefig(plot_dir+'cut_%s_' %lab_tgt + dir_dip[:-1])

lab_tgt = 'B'
lab_ind = '$|B|$ (G)'
# B_maxpm = 6e3
# B_minpm = -6e3
fig1n, ax1n = plt.subplots(3,4, figsize=(14,8))
cont1n, clim1nc = contour_dice(ax1n, lab_tgt, lab_ind, B_minpm, B_maxpm)
#cont1n.set_clim(B_minpm, B_maxpm)
cbar1n = cbar_dice(fig1n, ax1n, cont1n, clim1nc, 10)
dice(ax1n)
# fig1n.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1n.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1n.savefig(plot_dir+'dice_c_%s_' %lab_tgt + dir_dip[:-1])

fig1ns, ax1ns = plt.subplots(3,4, figsize=(14,8))
sct1n, clim1ns = sct_dice(ax1ns, lab_tgt, lab_ind, B_minpm, B_maxpm)
#sct1n.set_clim(B_minpm, B_maxpm)
cbar1ns = cbar_dice(fig1ns, ax1ns, sct1n,clim1ns, 10)
dice(ax1ns)
fig1ns.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], identifier, identifier))
fig1ns.savefig(plot_dir+'dice_s_%s_' %lab_tgt + dir_dip[:-1])


fig2n, ax2n = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2n, ax2n, lab_tgt, lab_ind, B_min0, B_max0)
fig2n.savefig(plot_dir+'cut_%s_' %lab_tgt + dir_dip[:-1])


