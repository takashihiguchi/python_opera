import numpy as np

# Calculation of the magnetic field produced around a rectangular coil whose axis oriented along the z-axis 
# based on: A. Azpurua, A SEMI-ANALYTICAL METHOD FOR THE DESIGN OF COIL-SYSTEMS FOR HOMOGENEOUS MAGNETOSTATIC FIELD GENERATION
# Progress In Electromagnetics Research B, Vol. 37, 171-189, 2012
# http://www.jpier.org/PIERB/pierb37/09.11102606.pdf
#
# The shape and the position of the coil is  defined by a set of parameters (a,b,h,I), where
# 2*a : the length along the x-axis 
# 2*b : the length along the y axis 
# h: the z-coordinate of the center of the coil
# I: the current
#
#
# (x,y,z) represents the coordinate of the measurement point P(x,y,z)
# Thus the field at a general position (x,y,z) is expressed by Bx(x,y,z,a,b,h,I), By(...), Bz(...)
#
mu0 = 4e-7*np.pi

def Bx(x,y,z,a,b,h,I):
    a = np.abs(a)
    b = np.abs(b)

    Bx0_1 = (z-h)/((x-a)**2+(z-h)**2)*(y+b)/np.sqrt((x-a)**2+(y+b)**2+(z-h)**2)
    Bx0_2 = (z-h)/((x-a)**2+(z-h)**2)*(-y+b)/np.sqrt((x-a)**2+(y-b)**2+(z-h)**2)
    Bx0_3 = (z-h)/((x+a)**2+(z-h)**2)*(y-b)/np.sqrt((x+a)**2+(y-b)**2+(z-h)**2)
    Bx0_4 = (z-h)/((x+a)**2+(z-h)**2)*(-y-b)/np.sqrt((x+a)**2+(y+b)**2+(z-h)**2)
    return mu0*I/(4*np.pi)*(Bx0_1+Bx0_2+Bx0_3+Bx0_4)

def By(x,y,z,a,b,h,I):
    a = np.abs(a)
    b = np.abs(b)

    By0_1 = (z-h)/((y-b)**2+(z-h)**2)*(x+a)/np.sqrt((x+a)**2+(y-b)**2+(z-h)**2)
    By0_2 = (z-h)/((y-b)**2+(z-h)**2)*(x-a)/np.sqrt((x-a)**2+(y-b)**2+(z-h)**2)
    By0_3 = (z-h)/((y+b)**2+(z-h)**2)*(x-a)/np.sqrt((x-a)**2+(y+b)**2+(z-h)**2)
    By0_4 = (z-h)/((y+b)**2+(z-h)**2)*(x+a)/np.sqrt((x+a)**2+(y+b)**2+(z-h)**2)
    return mu0*I/(4*np.pi)*(By0_1+By0_2+By0_3+By0_4)

def Bz(x,y,z,a,b,h,I):
    a = np.abs(a)
    b = np.abs(b)

    b1 = 1/((y-b)**2+(z-h)**2)
    b2 = 1/((x-a)**2+(z-h)**2)
    b10 = 1/((y+b)**2+(z-h)**2)
    b20 = 1/((x+a)**2+(z-h)**2)

    Bz0_1 = (x-a)*(y-b)/np.sqrt((x-a)**2+(y-b)**2+(z-h)**2)*(b1+b2)
    Bz0_2 = (x-a)*(y+b)/np.sqrt((x-a)**2+(y+b)**2+(z-h)**2)*(b10+b2)
    Bz0_3 = (x+a)*(y-b)/np.sqrt((x+a)**2+(y-b)**2+(z-h)**2)*(b1+b20)
    Bz0_4 = (x+a)*(y+b)/np.sqrt((x+a)**2+(y+b)**2+(z-h)**2)*(b10+b20)
    return mu0*I/(4*np.pi)*(Bz0_1+Bz0_2+Bz0_3+Bz0_4)
