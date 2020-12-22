# Martin Misakian, J. Res. Natl. Inst. Stand. Technol. 105, 557 (2000)
# Equations for the Magnetic Field Produced by One or More Rectangular Loops of Wire in the Same Plane
# 2*ax : the edge length along the x-axis 
# 2*ay : the edge length along the y-axis
# h : the z-coordinate of the coil center 


import numpy as np

mu0 = 4e-7*np.pi

def b_z(x,y,z,ax,ay,h,I):
    # P and Pc should be 3-dim array each
    # x, y, z = P[0], P[1], P[2]
    x1, y1, z1 = 0, 0, h
    dx = x-x1
    dy = y-y1
    dz = z-z1

    c1 = ax + dx
    c2 = ax - dx
    c3 = -c2
    c4 = -c1

    d1 = dy + ay
    d2 = d1
    d3 = dy - ay
    d4 = d3

    r1 = np.sqrt((ax+dx)**2 + (dy+ay)**2 + dz**2)
    r2 = np.sqrt((ax-dx)**2 + (dy+ay)**2 + dz**2)
    r3 = np.sqrt((ax-dx)**2 + (dy-ay)**2 + dz**2)
    r4 = np.sqrt((ax+dx)**2 + (dy-ay)**2 + dz**2)

    B1x = mu0*I/(4*np.pi)*(dz/(r1*(r1+d1)) - dz/(r2*(r2+d2)) +
                           dz/(r3*(r3+d3)) - dz/(r4*(r4+d4)))
    B1y = mu0*I/(4*np.pi)*(dz/(r1*(r1+c1)) - dz/(r2*(r2-c2)) +
                           dz/(r3*(r3+c3)) - dz/(r4*(r4-c4)))
    B1z = mu0*I/(4*np.pi)*((-d1/(r1*(r1+c1))-c1/(r1*(r1+d1))) + (d2/(r2*(r2-c2))-c2/(r2*(r2+d2))) +
                           (-d3/(r3*(r3+c3))-c3/(r3*(r3+d3))) + (d4/(r4*(r4-c4))-c4/(r4*(r4+d4))))

    B1 = np.array([B1x, B1y, B1z])
    return B1


def b_x(x, y, z, ay, az, h, I):
    # P and Pc should be 3-dim array each
    # x, y, z = P[0], P[1], P[2]
    x1, y1, z1 = h, 0, 0
    dx = x-x1
    dy = y-y1
    dz = z-z1

    c1 = ay + dy
    c2 = ay - dy
    c3 = -c2
    c4 = -c1

    d1 = dz + az
    d2 = d1
    d3 = dz - az
    d4 = d3

    r1 = np.sqrt((ay+dy)**2 + (dz+az)**2 + dx**2)
    r2 = np.sqrt((ay-dy)**2 + (dz+az)**2 + dx**2)
    r3 = np.sqrt((ay-dy)**2 + (dz-az)**2 + dx**2)
    r4 = np.sqrt((ay+dy)**2 + (dz-az)**2 + dx**2)

    B1y = mu0*I/(4*np.pi)*(dx/(r1*(r1+d1)) - dx/(r2*(r2+d2)) +
                           dx/(r3*(r3+d3)) - dx/(r4*(r4+d4)))
    B1z = mu0*I/(4*np.pi)*(dx/(r1*(r1+c1)) - dx/(r2*(r2-c2)) +
                           dx/(r3*(r3+c3)) - dx/(r4*(r4-c4)))
    B1x = mu0*I/(4*np.pi)*((-d1/(r1*(r1+c1))-c1/(r1*(r1+d1))) + (d2/(r2*(r2-c2))-c2/(r2*(r2+d2))) +
                           (-d3/(r3*(r3+c3))-c3/(r3*(r3+d3))) + (d4/(r4*(r4-c4))-c4/(r4*(r4+d4))))

    B1 = np.array([B1x, B1y, B1z])
    return B1


def b_y(x, y, z, az, ax, h, I):
    # P and Pc should be 3-dim array each
    # x, y, z = P[0], P[1], P[2]
    x1, y1, z1 = 0, h, 0
    dx = x-x1
    dy = y-y1
    dz = z-z1

    c1 = az + dz
    c2 = az - dz
    c3 = -c2
    c4 = -c1

    d1 = dx + ax
    d2 = d1
    d3 = dx - ax
    d4 = d3

    r1 = np.sqrt((az+dz)**2 + (dx+ax)**2 + dy**2)
    r2 = np.sqrt((az-dz)**2 + (dx+ax)**2 + dy**2)
    r3 = np.sqrt((az-dz)**2 + (dx-ax)**2 + dy**2)
    r4 = np.sqrt((az+dz)**2 + (dx-ax)**2 + dy**2)

    B1z = mu0*I/(4*np.pi)*(dy/(r1*(r1+d1)) - dy/(r2*(r2+d2)) +
                           dy/(r3*(r3+d3)) - dy/(r4*(r4+d4)))
    B1x = mu0*I/(4*np.pi)*(dy/(r1*(r1+c1)) - dy/(r2*(r2-c2)) +
                           dy/(r3*(r3+c3)) - dy/(r4*(r4-c4)))
    B1y = mu0*I/(4*np.pi)*((-d1/(r1*(r1+c1))-c1/(r1*(r1+d1))) + (d2/(r2*(r2-c2))-c2/(r2*(r2+d2))) +
                           (-d3/(r3*(r3+c3))-c3/(r3*(r3+d3))) + (d4/(r4*(r4-c4))-c4/(r4*(r4+d4))))

    B1 = np.array([B1x, B1y, B1z])
    return B1
