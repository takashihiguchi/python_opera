# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.tri as tri 
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

import pandas as pd
import numpy as np

def fmt(x, pos):
    a, b = '{:.5e}'.format(x).split('e')
    b = int(b)
    return  r'${} \times 10^{{{}}}$'.format(a,b)

from glob import glob


flist = glob("superD*_000A*")
ident = 'typeD_lin'

Imin = 90
Imax = 1200
Bzmin, Bzmax = -3200, 0
Bxmin, Bxmax = -30, 0
Bymin, Bymax = -200, 0

N =len(flist)
i_list = []

df_Bx_px = pd.DataFrame([])
df_Bx_mx = pd.DataFrame([])
df_Bx_py = pd.DataFrame([])
df_Bx_my = pd.DataFrame([])
df_Bx_pz = pd.DataFrame([])
df_Bx_mz = pd.DataFrame([])

df_By_px = pd.DataFrame([])
df_By_mx = pd.DataFrame([])
df_By_py = pd.DataFrame([])
df_By_my = pd.DataFrame([])
df_By_pz = pd.DataFrame([])
df_By_mz = pd.DataFrame([])

df_Bz_px = pd.DataFrame([])
df_Bz_mx = pd.DataFrame([])
df_Bz_py = pd.DataFrame([])
df_Bz_my = pd.DataFrame([])
df_Bz_pz = pd.DataFrame([])
df_Bz_mz = pd.DataFrame([])

for i in range(len(flist)):
    f = flist[i]
    i = float(flist[i].split('_')[1][:-1])
    print (i)
    i_list.append(i)
    sr_i = pd.Series([i])

    dir_name= f +'/'
    
    gr_px = pd.read_csv(dir_name + 'grids_px.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
    gr_mx = pd.read_csv(dir_name + 'grids_mx.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
    gr_py = pd.read_csv(dir_name + 'grids_py.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
    gr_my = pd.read_csv(dir_name + 'grids_my.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
    gr_pz = pd.read_csv(dir_name + 'grids_pz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
    gr_mz = pd.read_csv(dir_name + 'grids_mz.table', skiprows=9, sep='\s+', names=['x','y','z','Bx','By','Bz','Phi'])
    
    sr_Bx_px = pd.Series((gr_px[gr_px.y==gr_px.z].Bx))
    sr_Bx_mx = pd.Series((gr_mx[gr_mx.y==gr_mx.z].Bx))
    sr_Bx_py = pd.Series((gr_py[gr_py.x==gr_py.z].Bx))
    sr_Bx_my = pd.Series((gr_my[gr_my.x==gr_my.z].Bx))
    sr_Bx_pz = pd.Series((gr_pz[gr_pz.x==gr_pz.y].Bx))
    sr_Bx_mz = pd.Series((gr_mz[gr_mz.x==gr_mz.y].Bx))

    sr_By_px = pd.Series((gr_px[gr_px.y==gr_px.z].By))
    sr_By_mx = pd.Series((gr_mx[gr_mx.y==gr_mx.z].By))
    sr_By_py = pd.Series((gr_py[gr_py.x==gr_py.z].By))
    sr_By_my = pd.Series((gr_my[gr_my.x==gr_my.z].By))
    sr_By_pz = pd.Series((gr_pz[gr_pz.x==gr_pz.y].By))
    sr_By_mz = pd.Series((gr_mz[gr_mz.x==gr_mz.y].By))

    sr_Bz_px = pd.Series((gr_px[gr_px.y==gr_px.z].Bz))
    sr_Bz_mx = pd.Series((gr_mx[gr_mx.y==gr_mx.z].Bz))
    sr_Bz_py = pd.Series((gr_py[gr_py.x==gr_py.z].Bz))
    sr_Bz_my = pd.Series((gr_my[gr_my.x==gr_my.z].Bz))
    sr_Bz_pz = pd.Series((gr_pz[gr_pz.x==gr_pz.y].Bz))
    sr_Bz_mz = pd.Series((gr_mz[gr_mz.x==gr_mz.y].Bz))

    df_Bx_px_i = pd.concat([sr_i, sr_Bx_px], ignore_index=True)
    df_Bx_px = df_Bx_px.append(df_Bx_px_i, ignore_index=True)
    df_Bx_mx_i = pd.concat([sr_i, sr_Bx_mx], ignore_index=True)
    df_Bx_mx = df_Bx_mx.append(df_Bx_mx_i, ignore_index=True)
    df_Bx_py_i = pd.concat([sr_i, sr_Bx_py], ignore_index=True)
    df_Bx_py = df_Bx_py.append(df_Bx_py_i, ignore_index=True)
    df_Bx_my_i = pd.concat([sr_i, sr_Bx_my], ignore_index=True)
    df_Bx_my = df_Bx_my.append(df_Bx_my_i, ignore_index=True)
    df_Bx_pz_i = pd.concat([sr_i, sr_Bx_pz], ignore_index=True)
    df_Bx_pz = df_Bx_pz.append(df_Bx_pz_i, ignore_index=True)
    df_Bx_mz_i = pd.concat([sr_i, sr_Bx_mz], ignore_index=True)
    df_Bx_mz = df_Bx_mz.append(df_Bx_mz_i, ignore_index=True) 

    df_By_px_i = pd.concat([sr_i, sr_By_px], ignore_index=True)
    df_By_px = df_By_px.append(df_By_px_i, ignore_index=True)
    df_By_mx_i = pd.concat([sr_i, sr_By_mx], ignore_index=True)
    df_By_mx = df_By_mx.append(df_By_mx_i, ignore_index=True)
    df_By_py_i = pd.concat([sr_i, sr_By_py], ignore_index=True)
    df_By_py = df_By_py.append(df_By_py_i, ignore_index=True)
    df_By_my_i = pd.concat([sr_i, sr_By_my], ignore_index=True)
    df_By_my = df_By_my.append(df_By_my_i, ignore_index=True)
    df_By_pz_i = pd.concat([sr_i, sr_By_pz], ignore_index=True)
    df_By_pz = df_By_pz.append(df_By_pz_i, ignore_index=True)
    df_By_mz_i = pd.concat([sr_i, sr_By_mz], ignore_index=True)
    df_By_mz = df_By_mz.append(df_By_mz_i, ignore_index=True)
                               
    df_Bz_px_i = pd.concat([sr_i, sr_Bz_px], ignore_index=True)
    df_Bz_px = df_Bz_px.append(df_Bz_px_i, ignore_index=True)
    df_Bz_mx_i = pd.concat([sr_i, sr_Bz_mx], ignore_index=True)
    df_Bz_mx = df_Bz_mx.append(df_Bz_mx_i, ignore_index=True)
    df_Bz_py_i = pd.concat([sr_i, sr_Bz_py], ignore_index=True)
    df_Bz_py = df_Bz_py.append(df_Bz_py_i, ignore_index=True)
    df_Bz_my_i = pd.concat([sr_i, sr_Bz_my], ignore_index=True)
    df_Bz_my = df_Bz_my.append(df_Bz_my_i, ignore_index=True)
    df_Bz_pz_i = pd.concat([sr_i, sr_Bz_pz], ignore_index=True)
    df_Bz_pz = df_Bz_pz.append(df_Bz_pz_i, ignore_index=True)
    df_Bz_mz_i = pd.concat([sr_i, sr_Bz_mz], ignore_index=True)
    df_Bz_mz = df_Bz_mz.append(df_Bz_mz_i, ignore_index=True)


 

df_Bx_px.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bx_mx.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bx_py.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bx_my.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bx_pz.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bx_mz.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']

df_By_px.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_By_mx.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_By_py.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_By_my.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_By_pz.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_By_mz.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']

df_Bz_px.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bz_mx.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bz_py.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bz_my.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bz_pz.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']
df_Bz_mz.columns = ['I', 'P0','P1','P2','P3','P4','P5','P6','P7','P8']


figX, axX = plt.subplots(2,3, figsize=(10,4))
axX[0,0].set_title('+x plane')
axX[0,1].set_title('+y plane')
axX[0,2].set_title('+z plane')
axX[1,0].set_title('-x plane')
axX[1,1].set_title('-y plane')
axX[1,2].set_title('-z plane')

for i in range(3):
    for j in range(2):
#        axX[j,i].set_xscale('log')
        axX[j,i].yaxis.set_minor_locator(MultipleLocator(5))
        axX[j,i].set_xlabel('$I$ (A)')
        axX[j,i].set_ylabel('$B_x$ (G)')
        axX[j,i].set_ylim(Bxmin,Bxmax)
        axX[j,i].set_xlim(Imin,Imax)
        
for Pk in [ 'P0','P1','P2','P3','P4','P5','P6','P7','P8']:
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,0].plot(df_Bx_px.sort_values('I').I,df_Bx_px.sort_values('I')[Pk], 's-', markersize=2.)

    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,1].plot(df_Bx_py.sort_values('I').I,df_Bx_py.sort_values('I')[Pk], 's-', markersize=2.)

    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[0,2].plot(df_Bx_pz.sort_values('I').I,df_Bx_pz.sort_values('I')[Pk], 's-', markersize=2.)
    
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,0].plot(df_Bx_mx.sort_values('I').I,df_Bx_mx.sort_values('I')[Pk], 's-', markersize=2.)

    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,1].plot(df_Bx_my.sort_values('I').I,df_Bx_my.sort_values('I')[Pk], 's-', markersize=2.)

    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axX[1,2].plot(df_Bx_mz.sort_values('I').I,df_Bx_mz.sort_values('I')[Pk], 's-', markersize=2.)

  
figX.tight_layout()
figX.savefig(ident + '_scaling_Bx')



figY, axY = plt.subplots(2,3, figsize=(10,4))
axY[0,0].set_title('+x plane')
axY[0,1].set_title('+y plane')
axY[0,2].set_title('+z plane')
axY[1,0].set_title('-x plane')
axY[1,1].set_title('-y plane')
axY[1,2].set_title('-z plane')

for i in range(3):
    for j in range(2):
#        axY[j,i].set_xscale('log')
        axY[j,i].set_xlabel('$I$ (A)')
        axY[j,i].set_ylabel('$B_x$ (G)')
        axY[j,i].set_ylim(Bymin,Bymax)
        axY[j,i].set_xlim(Imin,Imax)
        
for Pk in [ 'P0','P1','P2','P3','P4','P5','P6','P7','P8']:
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,0].plot(df_By_px.sort_values('I').I,df_By_px.sort_values('I')[Pk], 's-', markersize=2.)

    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,1].plot(df_By_py.sort_values('I').I,df_By_py.sort_values('I')[Pk], 's-', markersize=2.)

    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[0,2].plot(df_By_pz.sort_values('I').I,df_By_pz.sort_values('I')[Pk], 's-', markersize=2.)
    
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,0].plot(df_By_mx.sort_values('I').I,df_By_mx.sort_values('I')[Pk], 's-', markersize=2.)

    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,1].plot(df_By_my.sort_values('I').I,df_By_my.sort_values('I')[Pk], 's-', markersize=2.)

    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axY[1,2].plot(df_By_mz.sort_values('I').I,df_By_mz.sort_values('I')[Pk], 's-', markersize=2.)

  
figY.tight_layout()
figY.savefig(ident+'_scaling_By')



figZ, axZ = plt.subplots(2,3, figsize=(10,4))
axZ[0,0].set_title('+x plane')
axZ[0,1].set_title('+y plane')
axZ[0,2].set_title('+z plane')
axZ[1,0].set_title('-x plane')
axZ[1,1].set_title('-y plane')
axZ[1,2].set_title('-z plane')

for i in range(3):
    for j in range(2):
#        axZ[j,i].set_xscale('log')
        axZ[j,i].set_xlabel('$I$ (A)')
        axZ[j,i].set_ylabel('$B_x$ (G)')
        axZ[j,i].set_ylim(Bzmin,Bzmax)
        axZ[j,i].set_xlim(Imin,Imax)
        
for Pk in [ 'P0','P1','P2','P3','P4','P5','P6','P7','P8']:
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,0].plot(df_Bz_px.sort_values('I').I,df_Bz_px.sort_values('I')[Pk], 's-', markersize=2.)

    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,1].plot(df_Bz_py.sort_values('I').I,df_Bz_py.sort_values('I')[Pk], 's-', markersize=2.)

    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[0,2].plot(df_Bz_pz.sort_values('I').I,df_Bz_pz.sort_values('I')[Pk], 's-', markersize=2.)
    
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,0].plot(df_Bz_mx.sort_values('I').I,df_Bz_mx.sort_values('I')[Pk], 's-', markersize=2.)

    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,1].plot(df_Bz_my.sort_values('I').I,df_Bz_my.sort_values('I')[Pk], 's-', markersize=2.)

    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)
    axZ[1,2].plot(df_Bz_mz.sort_values('I').I,df_Bz_mz.sort_values('I')[Pk], 's-', markersize=2.)

  
figZ.tight_layout()
figZ.savefig(ident+'_scaling_Bz')



