#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

dirt  = 'D_dip_000A_000A_mu_1_MB_T_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/'
LE = 500
LES = 510

import os
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots/'+dirt):
    os.makedirs('plots/'+dirt)

def Bx(x,y,z,M,x0,y0,z0):
    r2 = (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    return -2*M/3*(z-z0)*(x-x0)/(r2**2)
def By(x,y,z,M,x0,y0,z0):
    r2 = (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    return -2*M/3*(z-z0)*(y-y0)/(r2**2)
def Bz(x,y,z,M,x0,y0,z0):
    r2 = (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    return M/(3*r2)-2*M/3*(z-z0)*(z-z0)/(r2**2)

M = 2.83621926E7 #G*cm^3
x0 = -1666.67 #cm
y0 = -999.026
z0 = 140.78


# df_x = pd.read_csv('D_dip_000A_000A_mu_1_MB_R_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/line_x.table', skiprows=9,
#                 names=['x','y','z', 'Bx','By','Bz','Phi'], sep='\s+')
# df_y = pd.read_csv('D_dip_000A_000A_mu_1_MB_R_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/line_y.table', skiprows=9,
#                 names=['x','y','z', 'Bx','By','Bz','Phi'], sep='\s+')
# df_z = pd.read_csv('D_dip_000A_000A_mu_1_MB_R_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/line_z.table', skiprows=9,
#                 names=['x','y','z', 'Bx','By','Bz','Phi'], sep='\s+')
df_x = pd.read_csv(dirt+'line_x.table', skiprows=9,
                names=['x','y','z', 'Bx','By','Bz','Phi'], sep='\s+')
df_y = pd.read_csv(dirt+'line_y.table', skiprows=9,
                names=['x','y','z', 'Bx','By','Bz','Phi'], sep='\s+')
df_z = pd.read_csv(dirt+'line_z.table', skiprows=9,
                names=['x','y','z', 'Bx','By','Bz','Phi'], sep='\s+')


x_lins = np.array(df_x.x)
Bx_xaxis = [Bx(xi,0,0,M,x0,y0,z0) for xi in x_lins]
By_xaxis = [By(xi,0,0,M,x0,y0,z0) for xi in x_lins]
Bz_xaxis = [Bz(xi,0,0,M,x0,y0,z0) for xi in x_lins]

y_lins = np.array(df_y.y)
Bx_yaxis = [Bx(0,yi,0,M,x0,y0,z0) for yi in y_lins]
By_yaxis = [By(0,yi,0,M,x0,y0,z0) for yi in y_lins]
Bz_yaxis = [Bz(0,yi,0,M,x0,y0,z0) for yi in y_lins]

z_lins = np.array(df_z.z)
Bx_zaxis = [Bx(0,0,zi, M, x0, y0, z0) for zi in z_lins]
By_zaxis = [By(0,0,zi, M, x0, y0, z0) for zi in z_lins]
Bz_zaxis = [Bz(0,0,zi, M, x0, y0, z0) for zi in z_lins]


fig0, ax0 = plt.subplots(2, 3, figsize=(10, 5))

ax0[0, 0].set_ylabel('$B_x$ (G)')
ax0[0, 1].set_ylabel('$B_y$ (G)')
ax0[0, 2].set_ylabel('$B_z$ (G)')

ax0[1, 0].set_ylabel('$B_{x,anal}-B_{x,opera}$ (G)')
ax0[1, 1].set_ylabel('$B_{y,anal}-B_{y,opera}$ (G)')
ax0[1, 2].set_ylabel('$B_{z,anal}-B_{z,opera}$ (G)')

ax0[0, 0].plot(x_lins, Bx_xaxis, label='analytic')
ax0[0, 1].plot(x_lins, By_xaxis, label='analytic')
ax0[0, 2].plot(x_lins, Bz_xaxis, label='analytic')
ax0[0, 0].plot(df_x.x, df_x.Bx, '--', label='OPERA')
ax0[0, 1].plot(df_x.x, df_x.By, '--', label='OPERA')
ax0[0, 2].plot(df_x.x, df_x.Bz, '--', label='OPERA')
ax0[1, 0].plot(x_lins, Bx_xaxis - np.array(df_x.Bx))
ax0[1, 1].plot(x_lins, By_xaxis - np.array(df_x.By))
ax0[1, 2].plot(x_lins, Bz_xaxis - np.array(df_x.Bz))
for i in range(3):
    ax0[0, i].set_xlabel('$x$ (cm)')
    ax0[0, i].legend()
    ax0[1, i].set_xlabel('$x$ (cm)')

fig0.suptitle(
    'Dipole field, currents 0A (LE: %d, LES: %d), data along x axis' %(LE, LES))
fig0.tight_layout(rect=[0, 0.03, 1, 0.95])
fig0.savefig('plots/' +dirt+ 'comp_dip_xaxis')


fig1, ax1 = plt.subplots(2, 3, figsize=(10, 5))

ax1[0, 0].set_ylabel('$B_x$ (G)')
ax1[0, 1].set_ylabel('$B_y$ (G)')
ax1[0, 2].set_ylabel('$B_z$ (G)')

ax1[1, 0].set_ylabel('$B_{x,anal}-B_{x,opera}$ (G)')
ax1[1, 1].set_ylabel('$B_{y,anal}-B_{y,opera}$ (G)')
ax1[1, 2].set_ylabel('$B_{z,anal}-B_{z,opera}$ (G)')

ax1[0, 0].plot(y_lins, Bx_yaxis, label='analytic')
ax1[0, 1].plot(y_lins, By_yaxis, label='analytic')
ax1[0, 2].plot(y_lins, Bz_yaxis, label='analytic')
ax1[0, 0].plot(df_y.y, df_y.Bx, '--', label='OPERA')
ax1[0, 1].plot(df_y.y, df_y.By, '--', label='OPERA')
ax1[0, 2].plot(df_y.y, df_y.Bz, '--', label='OPERA')
ax1[1, 0].plot(y_lins, Bx_yaxis - np.array(df_y.Bx))
ax1[1, 1].plot(y_lins, By_yaxis - np.array(df_y.By))
ax1[1, 2].plot(y_lins, Bz_yaxis - np.array(df_y.Bz))
for i in range(3):
    ax1[0, i].set_xlabel('$y$ (cm)')
    ax1[0, i].legend()
    ax1[1, i].set_xlabel('$y$ (cm)')

fig1.suptitle(
    'Dipole field, currents 0A (LE: %d, LES: %d), data along y axis' %(LE, LES))
fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1.savefig('plots/' +dirt+ 'comp_dip_yaxis')

fig2, ax2 = plt.subplots(2, 3, figsize=(10, 5))

ax2[0, 0].set_ylabel('$B_x$ (G)')
ax2[0, 1].set_ylabel('$B_y$ (G)')
ax2[0, 2].set_ylabel('$B_z$ (G)')

ax2[1, 0].set_ylabel('$B_{x,anal}-B_{x,opera}$ (G)')
ax2[1, 1].set_ylabel('$B_{y,anal}-B_{y,opera}$ (G)')
ax2[1, 2].set_ylabel('$B_{z,anal}-B_{z,opera}$ (G)')

ax2[0, 0].plot(z_lins, Bx_zaxis, label='analytic')
ax2[0, 1].plot(z_lins, By_zaxis, label='analytic')
ax2[0, 2].plot(z_lins, Bz_zaxis, label='analytic')
ax2[0, 0].plot(df_z.z, df_z.Bx, '--', label='OPERA')
ax2[0, 1].plot(df_z.z, df_z.By, '--', label='OPERA')
ax2[0, 2].plot(df_z.z, df_z.Bz, '--', label='OPERA')
ax2[1, 0].plot(z_lins, Bx_zaxis - np.array(df_z.Bx))
ax2[1, 1].plot(z_lins, By_zaxis - np.array(df_z.By))
ax2[1, 2].plot(z_lins, Bz_zaxis - np.array(df_z.Bz))
for i in range(3):
    ax2[0, i].set_xlabel('$z$ (cm)')
    ax2[0, i].legend()
    ax2[1, i].set_xlabel('$z$ (cm)')

fig2.suptitle(
    'Dipole field, currents 0A (LE: %d, LES: %d), data along z axis' %(LE, LES))
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig('plots/' +dirt+ 'comp_dip_zaxis')




