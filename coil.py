#!/usr/bin/env python
# coding: utf-8

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
# from rect_coil import *
from rect_coil_misakian import *
#os.listdir('./')


a = 450e-2/2
h = 175e-2
I =-100
# xi = np.array(df.x*1e-2)
# Bx(xi, 0, 0, a, a, h, 100)*1e4

df_cx = pd.read_csv('D_zero_100A_100A_mu_1_MB_T_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/line_x.table', skiprows=9,
                   names=['x', 'y', 'z', 'Bx', 'By', 'Bz', 'Phi'], sep='\s+')
df_cy = pd.read_csv('D_zero_100A_100A_mu_1_MB_T_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/line_y.table', skiprows=9,
                    names=['x', 'y', 'z', 'Bx', 'By', 'Bz', 'Phi'], sep='\s+')
df_cz = pd.read_csv('D_zero_100A_100A_mu_1_MB_T_LE_500_LES_510_tol_1E-7_fine_Rfine10-8/line_z.table', skiprows=9,
                    names=['x', 'y', 'z', 'Bx', 'By', 'Bz', 'Phi'], sep='\s+')

x_lins = np.array(df_cx.x)
# Bx_xaxis = [Bx(xi*1e-2, 0, 0, a, a, h, I)*1e4 +
#             Bx(xi*1e-2, 0, 0, a, a, -h, I)*1e4 for xi in x_lins]
# By_xaxis = [By(xi*1e-2, 0, 0, a, a, h, I)*1e4 +
#             By(xi*1e-2, 0, 0, a, a, -h, I)*1e4 for xi in x_lins]
# Bz_xaxis = [Bz(xi*1e-2, 0, 0, a, a, h, I)*1e4 +
#             Bz(xi*1e-2, 0, 0, a, a, -h, I)*1e4 for xi in x_lins]
Bx_xaxis = [b_z(xi*1e-2, 0, 0, a, a, h, I)[0]*1e4 +
            b_z(xi*1e-2, 0, 0, a, a, -h, I)[0]*1e4 for xi in x_lins]
By_xaxis = [b_z(xi*1e-2, 0, 0, a, a, h, I)[1]*1e4 +
            b_z(xi*1e-2, 0, 0, a, a, -h, I)[1]*1e4 for xi in x_lins]
Bz_xaxis = [b_z(xi*1e-2, 0, 0, a, a, h, I)[2]*1e4 +
            b_z(xi*1e-2, 0, 0, a, a, -h, I)[2]*1e4 for xi in x_lins]

y_lins = np.array(df_cy.y)
# Bx_yaxis = [Bx(0, yi*1e-2, 0, a, a, h, I)*1e4 + Bx(0, yi*1e-2, 0, a, a, -h, I)*1e4
#             for yi in y_lins]
# By_yaxis = [By(0, yi*1e-2, 0, a, a, h, I)*1e4 + By(0, yi*1e-2, 0, a, a, -h, I)*1e4
#             for yi in y_lins]
# Bz_yaxis = [Bz(0, yi*1e-2, 0, a, a, h, I)*1e4 + Bz(0, yi*1e-2, 0, a, a, -h, I)*1e4
#             for yi in y_lins]
Bx_yaxis = [b_z(0, yi*1e-2, 0, a, a, h, I)[0]*1e4 + b_z(0, yi*1e-2, 0, a, a, -h, I)[0]*1e4
            for yi in y_lins]
By_yaxis = [b_z(0, yi*1e-2, 0, a, a, h, I)[1]*1e4 + b_z(0, yi*1e-2, 0, a, a, -h, I)[1]*1e4
            for yi in y_lins]
Bz_yaxis = [b_z(0, yi*1e-2, 0, a, a, h, I)[2]*1e4 + b_z(0, yi*1e-2, 0, a, a, -h, I)[2]*1e4
            for yi in y_lins]

z_lins = np.array(df_cz.z)
# Bx_zaxis = [Bx(0, 0, zi*1e-2, a, a, h, I)*1e4 + Bx(0, 0, zi*1e-2, a, a, -h, I)*1e4
#             for zi in z_lins]
# By_zaxis = [By(0, 0, zi*1e-2, a, a, h, I)*1e4 + By(0, 0, zi*1e-2, a, a, -h, I)*1e4
#             for zi in z_lins]
# Bz_zaxis = [Bz(0, 0, zi*1e-2, a, a, h, I)*1e4 + By(0, 0, zi*1e-2, a, a, -h, I)*1e4
#             for zi in z_lins]
Bx_zaxis = [b_z(0, 0, zi*1e-2, a, a, h, I)[0]*1e4 + b_z(0, 0, zi*1e-2, a, a, -h, I)[0]*1e4
            for zi in z_lins]
By_zaxis = [b_z(0, 0, zi*1e-2, a, a, h, I)[1]*1e4 + b_z(0, 0, zi*1e-2, a, a, -h, I)[1]*1e4
            for zi in z_lins]
Bz_zaxis = [b_z(0, 0, zi*1e-2, a, a, h, I)[2]*1e4 + b_z(0, 0, zi*1e-2, a, a, -h, I)[2]*1e4
            for zi in z_lins]

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
ax0[0, 0].plot(df_cx.x, df_cx.Bx, '--', label='OPERA')
ax0[0, 1].plot(df_cx.x, df_cx.By, '--', label='OPERA')
ax0[0, 2].plot(df_cx.x, df_cx.Bz, '--', label='OPERA')
ax0[1, 0].plot(x_lins, Bx_xaxis - np.array(df_cx.Bx))
ax0[1, 1].plot(x_lins, By_xaxis - np.array(df_cx.By))
ax0[1, 2].plot(x_lins, Bz_xaxis - np.array(df_cx.Bz))
for i in range(3):
    ax0[0, i].set_xlabel('$x$ (cm)')
    ax0[0, i].legend()
    ax0[1, i].set_xlabel('$x$ (cm)')

fig0.suptitle(
    'Zero field, currents 100A (LE: 500, LES: 510), data along x axis')
fig0.tight_layout(rect=[0, 0.03, 1, 0.95])
fig0.savefig('plots/comp_zero_100A_T_xaxis')
fig0.savefig('plots/comp_zero_100A_T_xaxis.pdf')


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
ax1[0, 0].plot(df_cy.y, df_cy.Bx, '--', label='OPERA')
ax1[0, 1].plot(df_cy.y, df_cy.By, '--', label='OPERA')
ax1[0, 2].plot(df_cy.y, df_cy.Bz, '--', label='OPERA')
ax1[1, 0].plot(y_lins, Bx_yaxis - np.array(df_cy.Bx))
ax1[1, 1].plot(y_lins, By_yaxis - np.array(df_cy.By))
ax1[1, 2].plot(y_lins, Bz_yaxis - np.array(df_cy.Bz))
for i in range(3):
    ax1[0, i].set_xlabel('$y$ (cm)')
    ax1[0, i].legend()
    ax1[1, i].set_xlabel('$y$ (cm)')

fig1.suptitle(
    'Zero field, currents 100A (LE: 500, LES: 510), data along y axis')
fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1.savefig('plots/comp_zero_100A_T_yaxis')
fig1.savefig('plots/comp_zero_100A_T_yaxis.pdf')

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
ax2[0, 0].plot(df_cz.z, df_cz.Bx, '--', label='OPERA')
ax2[0, 1].plot(df_cz.z, df_cz.By, '--', label='OPERA')
ax2[0, 2].plot(df_cz.z, df_cz.Bz, '--', label='OPERA')
ax2[1, 0].plot(z_lins, Bx_zaxis - np.array(df_cz.Bx))
ax2[1, 1].plot(z_lins, By_zaxis - np.array(df_cz.By))
ax2[1, 2].plot(z_lins, Bz_zaxis - np.array(df_cz.Bz))
for i in range(3):
    ax2[0, i].set_xlabel('$z$ (cm)')
    ax2[0, i].legend()
    ax2[1, i].set_xlabel('$z$ (cm)')

fig2.suptitle(
    'Zero field, currents 100A (LE: 500, LES: 510), data along z axis')
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig('plots/comp_zero_100A_T_zaxis')
fig2.savefig('plots/comp_zero_100A_T_zaxis.pdf')
