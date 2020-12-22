# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.tri as tri 
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.cbook as cbook

import pandas as pd
import numpy as np
import os
import sys

from color_sub import * 

identifier = str(sys.argv[1])

# dir_name =  'superB_1E5A_000A_fine/'
# plot_dir = 'plots_B_1E5A_000A_fine/'
dir_name='D_dipole_000A_000A_mu_1E5_MB_R_LE_500_LES_510_SHELL/'
# dir_name =  'dipole3_coils_%s/' %identifier
plot_dir = 'plots3_%s/' %identifier

texts = dir_name.split('_')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)




pl_0x = pd.read_csv(dir_name + 'plane_0x.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_px = pd.read_csv(dir_name + 'plane_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_mx = pd.read_csv(dir_name + 'plane_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_0y = pd.read_csv(dir_name + 'plane_0y.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_py = pd.read_csv(dir_name + 'plane_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_my = pd.read_csv(dir_name + 'plane_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_0z = pd.read_csv(dir_name + 'plane_0z.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_pz = pd.read_csv(dir_name + 'plane_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
pl_mz = pd.read_csv(dir_name + 'plane_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

gr_px = pd.read_csv(dir_name + 'grids_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_mx = pd.read_csv(dir_name + 'grids_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_py = pd.read_csv(dir_name + 'grids_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_my = pd.read_csv(dir_name + 'grids_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_pz = pd.read_csv(dir_name + 'grids_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
gr_mz = pd.read_csv(dir_name + 'grids_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])

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

## max value is taken from 5*sigma 
SD = 7

Bx_max0 = np.max([np.max(sig_cut(pl_0x.Bx, SD)),
                np.max(sig_cut(pl_0y.Bx, SD)),
                np.max(sig_cut(pl_0z.Bx, SD))])
Bx_min0 = np.min([np.min(sig_cut(pl_0x.Bx, SD)),
                np.min(sig_cut(pl_0y.Bx, SD)),
                np.min(sig_cut(pl_0z.Bx, SD))])

By_max0 = np.max([np.max(sig_cut(pl_0x.By, SD)),
                np.max(sig_cut(pl_0y.By, SD)),
                np.max(sig_cut(pl_0z.By, SD))])
By_min0 = np.min([np.min(sig_cut(pl_0x.By, SD)),
                np.min(sig_cut(pl_0y.By, SD)),
                np.min(sig_cut(pl_0z.By, SD))])

Bz_max0 = np.max([np.max(sig_cut(pl_0x.Bz, SD)),
                np.max(sig_cut(pl_0y.Bz, SD)),
                np.max(sig_cut(pl_0z.Bz, SD))])
Bz_min0 = np.min([np.min(sig_cut(pl_0x.Bz, SD)),
                np.min(sig_cut(pl_0y.Bz, SD)),
                np.min(sig_cut(pl_0z.Bz, SD))])
B_max0 = np.max([np.max(sig_cut(pl_0x.B, SD)),
                np.max(sig_cut(pl_0y.B, SD)),
                np.max(sig_cut(pl_0z.B, SD))])
B_min0 = np.min([np.min(sig_cut(pl_0x.B, SD)),
                np.min(sig_cut(pl_0y.B, SD)),
                np.min(sig_cut(pl_0z.B, SD))])


Bx_maxpm = np.max([np.max(sig_cut(pl_px.Bx, SD)),np.max(sig_cut(pl_mx.Bx, SD)),
                np.max(sig_cut(pl_py.Bx, SD)),np.max(sig_cut(pl_my.Bx, SD)),
                np.max(sig_cut(pl_pz.Bx, SD)),np.max(sig_cut(pl_mz.Bx, SD))])
Bx_minpm = np.min([np.min(sig_cut(pl_px.Bx, SD)),np.min(sig_cut(pl_mx.Bx, SD)),
                np.min(sig_cut(pl_py.Bx, SD)),np.min(sig_cut(pl_my.Bx, SD)),
                np.min(sig_cut(pl_pz.Bx, SD)),np.min(sig_cut(pl_mz.Bx, SD))])

By_maxpm = np.max([np.max(sig_cut(pl_px.By, SD)),np.max(sig_cut(pl_mx.By, SD)),
                  np.max(sig_cut(pl_py.By, SD)),np.max(sig_cut(pl_my.By, SD)),
                  np.max(sig_cut(pl_pz.By, SD)),np.max(sig_cut(pl_mz.By, SD))])
By_minpm = np.min([np.min(sig_cut(pl_px.By, SD)),np.min(sig_cut(pl_mx.By, SD)),
                  np.min(sig_cut(pl_py.By, SD)),np.min(sig_cut(pl_my.By, SD)),
                  np.min(sig_cut(pl_pz.By, SD)),np.min(sig_cut(pl_mz.By, SD))])

Bz_maxpm = np.max([np.max(sig_cut(pl_px.Bz, SD)),np.max(sig_cut(pl_mx.Bz, SD)),
                  np.max(sig_cut(pl_py.Bz, SD)),np.max(sig_cut(pl_my.Bz, SD)),
                  np.max(sig_cut(pl_pz.Bz, SD)),np.max(sig_cut(pl_mz.Bz, SD))])
Bz_minpm = np.min([np.min(sig_cut(pl_px.Bz, SD)),np.min(sig_cut(pl_mx.Bz, SD)),
                   np.min(sig_cut(pl_py.Bz, SD)),np.min(sig_cut(pl_my.Bz, SD)),
                   np.min(sig_cut(pl_pz.Bz, SD)),np.min(sig_cut(pl_mz.Bz, SD))])
B_maxpm = np.max([np.max(sig_cut(pl_px.B, SD)),np.max(sig_cut(pl_mx.B, SD)),
                np.max(sig_cut(pl_py.B, SD)),np.max(sig_cut(pl_my.B, SD)),
                np.max(sig_cut(pl_pz.B, SD)),np.max(sig_cut(pl_mz.B, SD))])
B_minpm = np.min([np.min(sig_cut(pl_px.B, SD)),np.min(sig_cut(pl_mx.B, SD)),
                np.min(sig_cut(pl_py.B, SD)),np.min(sig_cut(pl_my.B, SD)),
                np.min(sig_cut(pl_pz.B, SD)),np.min(sig_cut(pl_mz.B, SD))])






def contour_dice(ax, lab_tgt, lab_ind, zmin, zmax):
    cont10 = tricont_sub(ax[0,1],  pl_pz.y, pl_pz.x, pl_pz[lab_tgt], zmin, zmax, 31, lab_ind)
    cont11 = tricont_sub(ax[1,0],  pl_my.x, pl_my.z, pl_my[lab_tgt], zmin, zmax, 31, lab_ind)
    cont12 = tricont_sub(ax[1,1],  pl_px.y, pl_px.z, pl_px[lab_tgt], zmin, zmax, 31, lab_ind)
    cont13 = tricont_sub(ax[1,2],  pl_py.x, pl_py.z, pl_py[lab_tgt], zmin, zmax, 31, lab_ind)
    cont14 = tricont_sub(ax[1,3],  pl_mx.y, pl_mx.z, pl_mx[lab_tgt], zmin, zmax, 31, lab_ind)
    cont15 = tricont_sub(ax[2,1],  pl_mz.y, pl_mz.x, pl_mz[lab_tgt], zmin, zmax, 31, lab_ind)
    return cont10 

def cbar_dice(fig, ax, cont, lab_ind):
    fig.subplots_adjust(wspace = .7, hspace = .4)
    cbar0 = fig.colorbar(cont, ax=ax.ravel().tolist(), format=ticker.FuncFormatter(fmt),aspect=40)
    cbar0.ax.get_yaxis().labelpad = 15
    cbar0.ax.set_ylabel(lab_ind, rotation=270)
    return cbar0 

def contour_cut(fig, ax, lab_tgt, lab_ind, zmin, zmax):
    cont20 = tricont_sub(ax[0],pl_0x.y, pl_0x.z, pl_0x[lab_tgt], zmin, zmax, 31, lab_ind)
    cont21 = tricont_sub(ax[1],pl_0y.x, pl_0y.z, pl_0y[lab_tgt], zmin, zmax, 31, lab_ind)
    cont22 = tricont_sub(ax[2],pl_0z.x, pl_0z.y, pl_0z[lab_tgt], zmin, zmax, 31, lab_ind)
    ax[0].set_title('cut at x=0')
    ax[1].set_title('cut at y=0')
    ax[2].set_title('cut at z=0')
    for i in range(3):
        ax[i].set_xlim(-300, 300)
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

fig1x, ax1x = plt.subplots(3,4, figsize=(14,8))

cont1x = contour_dice(ax1x, lab_tgt, lab_ind, Bx_minpm, Bx_maxpm)
cbar1x = cbar_dice(fig1x, ax1x, cont1x, lab_ind )
dice(ax1x)
fig1x.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1x.savefig(plot_dir+'dice_%s_' %lab_tgt + dir_name[:-1])

fig2x, ax2x = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2x, ax2x, lab_tgt, lab_ind, Bx_min0, Bx_max0)
fig2x.savefig(plot_dir + 'cut_%s_' %lab_tgt + dir_name[:-1])


lab_tgt = 'By'
lab_ind = '$B_y$ (G)'

fig1y, ax1y = plt.subplots(3,4, figsize=(14,8))

cont1y = contour_dice(ax1y, lab_tgt, lab_ind, By_minpm, By_maxpm)
cbar1y = cbar_dice(fig1y, ax1y, cont1y, lab_ind )
dice(ax1y)
fig1y.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1y.savefig(plot_dir+'dice_%s_' %lab_tgt + dir_name[:-1])

fig2y, ax2y = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2y, ax2y, lab_tgt, lab_ind, By_min0, By_max0)
fig2y.savefig(plot_dir+'cut_%s_' %lab_tgt + dir_name[:-1])

lab_tgt = 'Bz'
lab_ind = '$B_z$ (G)'

fig1z, ax1z = plt.subplots(3,4, figsize=(14,8))

cont1z = contour_dice(ax1z, lab_tgt, lab_ind, Bz_minpm, Bz_maxpm)
cbar1z = cbar_dice(fig1z, ax1z, cont1z, lab_ind )
dice(ax1z)
fig1z.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1z.savefig(plot_dir+'dice_%s_' %lab_tgt + dir_name[:-1])

fig2z, ax2z = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2z, ax2z, lab_tgt, lab_ind, Bz_min0, Bz_max0)
fig2z.savefig(plot_dir+'cut_%s_' %lab_tgt + dir_name[:-1])

lab_tgt = 'B'
lab_ind = '$|B|$ (G)'

fig1n, ax1n = plt.subplots(3,4, figsize=(14,8))

cont1n = contour_dice(ax1n, lab_tgt, lab_ind, B_minpm, B_maxpm)
cbar1n = cbar_dice(fig1n, ax1n, cont1n, lab_ind )
dice(ax1n)
fig1n.suptitle('%s, Geometry:%s, Coil1:%s, Coil2:%s' %(lab_ind, texts[0][-1], texts[1], texts[2][:-1]))
fig1n.savefig(plot_dir+'dice_%s_' %lab_tgt + dir_name[:-1])

fig2n, ax2n = plt.subplots(1,3, figsize=(11,4.5), constrained_layout=True)
contour_cut(fig2n, ax2n, lab_tgt, lab_ind, B_min0, B_max0)
fig2n.savefig(plot_dir+'cut_%s_' %lab_tgt + dir_name[:-1])


