from matplotlib import pyplot as plt
import matplotlib.tri as tri 
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.cbook as cbook

import pandas as pd
import numpy as np

def round5(x):
    n = int(np.log10(x))
    f = x / 10**n
    f5 = 0.5 * (f//0.5)+1
    return f5* 10**n

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
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
    zabsmax_r = round5(zabsmax)
    ax.set_aspect('equal')
    ax.get_xaxis().labelpad = 1
    ax.get_yaxis().labelpad = 1
    Ndev = level_dv
    if (Ndev//2-Ndev/2.)==0:
        Ndev = Ndev+1
    levels = np.linspace(-zabsmax_r, zabsmax_r, Ndev)
    xy_tri = tri.Triangulation(x,y)
    # cont_tri = ax.tricontourf(x, y, z, xy_tri, levels=levels, vmin=-zabsmax, vmax=zabsmax, cmap=  'RdYlBu_r', locator=ticker.LogLocator())
    cont_tri = ax.tricontourf(x, y, z, xy_tri, levels=levels, vmin=-zabsmax_r, vmax=zabsmax_r, cmap=  'RdYlBu_r')
    #cont_tri.set_clim(-zabsmax_r, zabsmax_r)
    #cont_tri.set_clim(-zabsmax_r, zabsmax_r)
    #cont_tri.set_ticks(levels)
    #cont_tri.draw_all()
    return cont_tri, [-zabsmax_r, zabsmax_r]

#def tricont_sub(ax, x, y, z, z_min, z_max, level_dv, cbar_label):
#    ax.set_aspect('equal')
#    ax.get_xaxis().labelpad = 1
#    ax.get_yaxis().labelpad = 1
#    levels = np.linspace(z_min, z_max, level_dv)
#    xy_tri = tri.Triangulation(x,y)
#    cont_tri = ax.tricontourf(x, y, z, xy_tri, levels=levels, vmin=z_min, vmax=z_max, cmap=  'RdYlBu_r')
#    return cont_tri

def sct_sub(ax, x, y, z, z_min, z_max):
    zabsmax = max(abs(z_min),abs(z_max))
    zabsmax_r = round5(zabsmax)
    ax.set_aspect('equal')
    ax.get_xaxis().labelpad = 1
    ax.get_yaxis().labelpad = 1

    scat = ax.scatter(x, y, c=z, s=1.5, vmin=-zabsmax_r, vmax=zabsmax_r, cmap='RdYlBu_r')
    #scat.set_clim(-zabsmax_r, zabsmax)
    #sact.set_clim(-zabsmax_r, zabsmax)
    return scat, [-zabsmax_r, zabsmax_r]

def dice(ax):
    for i in [0,2,3]:
        ax[0,i].set_visible(False)
        ax[2,i].set_visible(False)
    
    ax[0,1].set_xlabel('y (cm)')
    ax[0,1].set_ylabel('x (cm)')
    ax[0,1].set_aspect('equal')
    ax[0,1].set_xlim(-175,175)
    ax[0,1].set_ylim(175,-175)
    

    ax[1,0].set_xlabel('x (cm)')
    ax[1,0].set_ylabel('z (cm)')
    ax[1,0].set_aspect('equal')
    # ax[1,0].set_xlim(175,-175)
    ax[1,0].set_xlim(-175,175)
    ax[1,0].set_ylim(-175,175)

    ax[1,1].set_xlabel('y (cm)')
    ax[1,1].set_ylabel('z (cm)')
    ax[1,1].set_xlim(-175,175)
    ax[1,1].set_ylim(-175,175)
    ax[1,1].set_aspect('equal')
    
    ax[1,2].set_xlabel('x (cm)')
    ax[1,2].set_ylabel('z (cm)')
    ax[1,2].set_xlim(175,-175)
    ax[1,2].set_ylim(-175,175)
    ax[1,2].set_aspect('equal')

    ax[1,3].set_xlabel('y (cm)')
    ax[1,3].set_ylabel('z (cm)')
    ax[1,3].set_xlim(175,-175)
    ax[1,3].set_ylim(-175,175)
    ax[1,3].set_aspect('equal')

    ax[2,1].set_xlabel('y (cm)')
    ax[2,1].set_ylabel('x (cm)')
    ax[2,1].set_xlim(-175,175)
    ax[2,1].set_ylim(-175,175)
    ax[2,1].set_aspect('equal')
    
    return 0
