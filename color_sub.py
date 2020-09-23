from matplotlib import pyplot as plt
import matplotlib.tri as tri 
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.cbook as cbook

import pandas as pd
import numpy as np


def fmt(x, pos):
    a, b = '{:.5e}'.format(x).split('e')
    b = int(b)
    return  r'${} \times 10^{{{}}}$'.format(a,b)


def sig_cut(a, SD):
    a = np.array(a)
    for i in range(100):
        sig = np.std(a)
        mu = np.mean(a)
        a_max = np.max(a)
        a_min = np.min(a)
        
        if (a_max > mu + sig*SD) or (a_min < mu - sig*SD):
            a = a[np.where((a < mu + sig*SD) & (a > mu - sig*SD))]
        else:
            a=a
    return a

def tricont_sub(ax, x, y, z, z_min, z_max, level_dv, cbar_label):
    zabsmax = max(abs(z_min),abs(z_max))
    ax.set_aspect('equal')
    ax.get_xaxis().labelpad = 1
    ax.get_yaxis().labelpad = 1
    levels = np.linspace(-zabsmax, zabsmax, level_dv)
    xy_tri = tri.Triangulation(x,y)
    cont_tri = ax.tricontourf(x, y, z, xy_tri, levels=levels, vmin=-zabsmax, vmax=zabsmax, cmap=  'RdYlBu_r')
    return cont_tri
#def tricont_sub(ax, x, y, z, z_min, z_max, level_dv, cbar_label):
#    ax.set_aspect('equal')
#    ax.get_xaxis().labelpad = 1
#    ax.get_yaxis().labelpad = 1
#    levels = np.linspace(z_min, z_max, level_dv)
#    xy_tri = tri.Triangulation(x,y)
#    cont_tri = ax.tricontourf(x, y, z, xy_tri, levels=levels, vmin=z_min, vmax=z_max, cmap=  'RdYlBu_r')
#    return cont_tri


def dice(ax):
    for i in [0,2,3]:
        ax[0,i].set_visible(False)
        ax[2,i].set_visible(False)

    ax[0,1].set_xlabel('y (cm)')
    ax[0,1].set_ylabel('x (cm)')
    ax[0,1].set_ylim(175,-175)

    ax[1,0].set_xlabel('x (cm)')
    ax[1,0].set_ylabel('z (cm)')
    # ax[1,0].set_xlim(175,-175)

    ax[1,1].set_xlabel('y (cm)')
    ax[1,1].set_ylabel('z (cm)')

    ax[1,2].set_xlabel('x (cm)')
    ax[1,2].set_ylabel('z (cm)')
    ax[1,2].set_xlim(175,-175)

    ax[1,3].set_xlabel('y (cm)')
    ax[1,3].set_ylabel('z (cm)')
    ax[1,3].set_xlim(175,-175)

    ax[2,1].set_xlabel('y (cm)')
    ax[2,1].set_ylabel('x (cm)')
    
    return 0
