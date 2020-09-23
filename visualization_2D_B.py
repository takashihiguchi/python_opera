import pandas as pd
import numpy as np
import os
# %matplotlib inline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import FormatStrFormatter

import scipy
import scipy.interpolate as interp

## User controls 
fpath = 'data_export/map_export_[-141.8,98.2]_[-327.2,192.8]_[-100.0,100.0].csv'
ppath = 'planer_plots/'

NL = 100




##
if not os.path.exists(ppath):
    os.makedirs(ppath)


df = pd.read_csv(fpath)
x_all = df.x.unique() 
y_all = df.y.unique()
z_all = df.z.unique()


print(x_all)
print(y_all)
print(z_all)


print(x_all.size)
print(y_all.size)
print(z_all.size)

def B_d(x,y,z, MV, x0, y0, z0, MV1):
    r = np.sqrt((x-x0)**2+ (y-y0)**2 + (z-z0)**2)
#     Bx = 5/3.*MV*(z-z0)*(x-x0)*r**(-7)
#     By = 5/3.*MV*(z-z0)*(y-y0)*r**(-7)
#     Bz = -1/3.*MV*r**(-5) + 5/3.*MV*(z-z0)**2*r**(-7)
    Bx = MV*(z-z0)*(x-x0)*r**(-5)
    By = MV*(z-z0)*(y-y0)*r**(-5)
    Bz = -1/3.*MV*r**(-3) + MV*(z-z0)**2*r**(-5)
    return np.array([Bx, By, Bz])


for k in range(x_all.size):
    x_k = x_all[k]
    

    print ('%d_%f' %(k, x_k))
    df_all_sub = df[df.x==x_k]
    y_min = np.min(df_all_sub.y)
    y_max = np.max(df_all_sub.y)
    z_min = np.min(df_all_sub.z)
    z_max = np.max(df_all_sub.z)
    
    if (y_min!=y_max) & (z_min!=z_max):

        y_dense, z_dense = np.meshgrid(np.linspace(y_min,y_max, NL),np.linspace(z_min, z_max, NL))

        Bx_rbf = interp.Rbf(df_all_sub.y, df_all_sub.z, df_all_sub.B_x, function='cubic', smooth=0)  # default smooth=0 for interpolation
        Bx_dense = Bx_rbf(y_dense, z_dense)  # not really a function, but a callable class instance
        By_rbf = interp.Rbf(df_all_sub.y, df_all_sub.z, df_all_sub.B_y, function='cubic', smooth=0)  # default smooth=0 for interpolation
        By_dense = By_rbf(y_dense, z_dense)  # not really a function, but a callable class instance
        Bz_rbf = interp.Rbf(df_all_sub.y, df_all_sub.z, df_all_sub.B_z, function='cubic', smooth=0)  # default smooth=0 for interpolation
        Bz_dense = Bz_rbf(y_dense, z_dense)  # not really a function, but a callable class instance

        fig0=plt.figure(figsize=(12,8))
        ax0 = fig0.add_subplot(221)
        str0 = ax0.streamplot(y_dense, z_dense, By_dense, Bz_dense, linewidth=1,density=1.8,color='white')
        # cp = ax.contourf(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.autumn)
        cp0 = ax0.contourf(y_dense,z_dense, np.sqrt(By_dense**2+Bz_dense**2)*100, cmap=cm.autumn)
        # cp = ax.contour(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.binary)
        cb0 = fig0.colorbar(cp0, ax=ax0,label='$|\mathbf{B}|=\sqrt{B_y^2+B_z^2}\,\,\mathsf{(\mu T)}$')

        # lab = cb.set_label('$|\mathbf{B}|=\sqrt{B_x^2+B_y^2+B_z^2}\,\mathsf{(\mu T)}$')
        ax0.set_title('$\mathsf{(B_y, B_z)}$ at $\mathsf{x=%.2f\,cm}$' %x_k)
        ax0.set_xlabel('y (cm)')
        ax0.set_ylabel('z (cm)')

        ax1 = fig0.add_subplot(222)
        str1 = ax1.streamplot(y_dense, z_dense, Bx_dense, Bz_dense, linewidth=1,density=1.8,color='white')
        # cp = ax.contourf(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.autumn)
        cp1 = ax1.contourf(y_dense,z_dense, np.sqrt(Bx_dense**2+Bz_dense**2)*100, cmap=cm.autumn)
        # cp = ax.contour(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.binary)
        cb1 = fig0.colorbar(cp1, ax=ax1,label='$|\mathbf{B}|=\sqrt{B_x^2+B_z^2}\,\,\mathsf{(\mu T)}$')

        ax1.set_title('$\mathsf{(B_x, B_z)}$ at $\mathsf{x=%.2f\,cm}$' %x_k)
        ax1.set_xlabel('y (cm)')
        ax1.set_ylabel('z (cm)')
        
        ax2 = fig0.add_subplot(223)
        ax2.plot(np.linspace(y_min, y_max, NL), By_rbf(np.linspace(y_min, y_max,NL),np.zeros(NL))*100, label='$\mathsf{B_y}$')
        ax2.plot(np.linspace(y_min, y_max, NL), Bz_rbf(np.linspace(y_min, y_max,NL),np.zeros(NL))*100, label='$\mathsf{B_z}$')
        ax2.set_xlabel('$\mathsf{y\,\,(cm)}$')
        ax2.set_ylabel('$\mathsf{B_{y,z} (z=0\,cm)\,\,(\mu T)}$')
        ax2.legend()

        ax3 = fig0.add_subplot(224)
        ax3.plot(np.linspace(y_min, y_max, NL), Bx_rbf(np.linspace(y_min, y_max,NL),np.zeros(NL))*100, label='$\mathsf{B_x}$')
        ax3.plot(np.linspace(y_min, y_max, NL), Bz_rbf(np.linspace(y_min, y_max,NL),np.zeros(NL))*100, label='$\mathsf{B_z}$')
        ax3.set_xlabel('$\mathsf{y\,\,(cm)}$')
        ax3.set_ylabel('$\mathsf{B_{x,z} (z=0\,cm)\,\,(\mu T)}$')
        ax3.legend()

#         fig0=plt.figure(figsize=(12,4))
#         ax0 = fig0.add_subplot(121)
#         str0 = ax0.streamplot(y_dense, z_dense, By_dense, Bz_dense, linewidth=1,density=1.5,color='white')
#         # cp = ax.contourf(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.autumn)
#         cp0 = ax0.contourf(y_dense,z_dense, np.sqrt(By_dense**2+Bz_dense**2)*100, cmap=cm.autumn)
#         # cp = ax.contour(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.binary)
#         cb0 = fig0.colorbar(cp0, ax=ax0,label='$|\mathbf{B}|=\sqrt{B_y^2+B_z^2}\,\,\mathsf{(\mu T)}$')

#         # lab = cb.set_label('$|\mathbf{B}|=\sqrt{B_x^2+B_y^2+B_z^2}\,\mathsf{(\mu T)}$')
#         ax0.set_title('$\mathsf{(B_y, B_z)}$ at $\mathsf{x=%.3f\,cm}$' %x_k)
#         ax0.set_xlabel('y (cm)')
#         ax0.set_ylabel('z (cm)')

#         ax1 = fig0.add_subplot(122)
#         str1 = ax1.streamplot(y_dense, z_dense, Bx_dense, Bz_dense, linewidth=1,density=1.5,color='white')
#         # cp = ax.contourf(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.autumn)
#         cp1 = ax1.contourf(y_dense,z_dense, np.sqrt(Bx_dense**2+Bz_dense**2)*100, cmap=cm.autumn)
#         # cp = ax.contour(z_dense,y_dense, np.sqrt(Bx_dense**2+By_dense**2)*100, cmap=cm.binary)
#         cb1 = fig0.colorbar(cp1, ax=ax1,label='$|\mathbf{B}|=\sqrt{B_x^2+B_z^2}\,\,\mathsf{(\mu T)}$')

#         ax1.set_title('$\mathsf{(B_x, B_z)}$ at $\mathsf{x=%.3f\,cm}$' %x_k)
#         ax1.set_xlabel('y (cm)')
#         ax1.set_ylabel('z (cm)')



        fig0.tight_layout()
        fig0.savefig(ppath +'plane_k_%d.png' %k)
