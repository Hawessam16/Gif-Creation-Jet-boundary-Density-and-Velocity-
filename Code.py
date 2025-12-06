# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 21:48:23 2023

@author: cjrl@kent.ac.uk
"""
# import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matplotlib.colors as colors

plt.rcParams['figure.dpi'] = 600

# ------------------ begin function definitions --------------------------

def sign(n): 
  if n<0: return -1 
  elif n>=0: return 1
  else: return 1
  
def overlap (ishp, rad, x0, y0, z0, xin, yin, zin, xout, yout, zout):

#  overlap of region over Cartesian zone
#  written by: David Clarke
#  april, 1990

#  modified 1: Chris Lynch
#  november, 2023

#  PURPOSE:  Determines the fraction of a Cartesian zone that overlaps
#  the specified geometrical region (sphere or right cylinder).  This
#  is done by dividing the zone into 20**3 "subzones", and finding the
#  fraction of subzone centres lying inside the surface of the region.

#  INPUT VARIABLES:
#
#    ishp            =1 => sphere
#                    =2 => right cylinder
#    rad             radius of region
#    x0,y0,z0        coordinates of centre of curvature.
#    xin,yin,zin     coordinates of zone corner known to lie inside
#                    region.
#    xout,yout,zout  coordinates of zone corner diametrically opposed to
#                    zone corner known to lie inside region.


#  Number of subzones in the x-direction is "nx", etc. for "ny" and
#  "nz".  Increment between subzones in x-direction is "delx", etc. for
#  "dely" and "delz".

    nx   = 20
    ny   = 20
    nz   = 20
    
    xsq = np.zeros(nx)
    ysq = np.zeros(ny)
    zsq = np.zeros(nz)
    
    counter = np.zeros(nx)
    
    delx = ( xout - xin ) /  nx
    dely = ( yout - yin ) /  ny
    delz = ( zout - zin ) /  nz

#     Set up subgrid inside zone.

    for i in range(nx):
        xsq[i] = ( xin + ( 0.5 + i ) * delx - x0 )**2
    for j in range(ny):
        ysq[j] = ( yin + ( 0.5 + j ) * dely - y0 )**2
    for k in range(nz):
        zsq[k] = ( zin + ( 0.5 + k ) * delz - z0 )**2

#  Count the number of subzones lying inside the surface of the
#  region which passes through the zone.

    fact  = 1.0
    if (ishp == 2): 
        fact = 0.0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                r = np.sqrt ( fact * xsq[i] + ysq[j] + zsq[k] )
                if (r <= rad):
                    counter[i] = counter[i] + 1.0
    scount = 0.0
    for i in range(nx):
        scount = scount + counter[i]

    scount =   max ( 1, scount )

#    Set the fraction of the zone which overlaps the region.

    return scount/(nx*ny*nz)

# ------------------ end function definitions --------------------------

#      Default values
#

# initialise grid

# Grid zones in y and z dimensions
jzones = 345
kzones = 345

# Extents of problem domain in y and z dimensions (in cm)
x2min=-1.725e15
x2max=1.725e15
x3min=-1.725e15
x3max=1.725e15

# Arrays of grid coordinate values in y and z dimensions
x2a = np.linspace(x2min, x2max, num=jzones)
x3a = np.linspace(x3min, x3max, num=kzones)
dx2 = x2a[1] - x2a[0]
dx3 = x3a[1] - x3a[0]
x2b = x2a + (dx2/2) # midpoint y-coordinates
x3b = x3a + (dx3/2) # midpoint z-coordinates
jin = np.zeros(jzones, dtype=int)
jout = np.zeros(jzones, dtype=int)
kin = np.zeros(kzones, dtype=int)
kout = np.zeros(kzones, dtype=int)


# Set up boundary arrays, initialise to zero
denbound = np.zeros((jzones,kzones), dtype=float) # density boundary array
genbound = np.zeros((jzones,kzones), dtype=float) # gas energy boundary array
vel1bound = np.zeros((jzones,kzones), dtype=float) # x-velocity boundary array
vel2bound = np.zeros((jzones,kzones), dtype=float) # y-velocity boundary array
vel3bound = np.zeros((jzones,kzones), dtype=float) # z-velocity boundary array

# define some constants (in CGS unit system)

cmau = 1.49597e13
msol = 1.989e33
guniv = 6.6720e-8
yrsecs = 3.154e7

#determines 2 pi to the limit of the computer's accuracy
twopi    = np.arctan(1.0) * 8.0 

#  Parameters for the jet material:
#
#  hydrodynamical variables:
#    rjet      radius of jet (in coordinate units, default = 1.0)
#    drat      ratio between jet density and ambient medium density
#              effective at the in-flow boundary (default = 0.1)
#    prat      ratio between jet pressure and ambient medium pressure
#              effective at the in-flow boundary (default = 1.0)
#    mjet      Mach number of the jet with respect to its own sound
#              speed at the jet inlet (default = 6.0)

#  Parameters for the ambient medium:
#
#    d0      density
#    e0      specific internal energy
#    p0      pressure (determined from e0 with equation of state)


#    x10     origin of x coordinate (default = 0.0)
#    x20     origin of y coordinate (default = 0.0)
#    x30     origin of z coordinate (default = 0.0)


rjet     = 2e14
drat     = 10.0
prat     = 1.0
mjet     = 6.0
d0       = 1.67e-20
e0       = 1.63e-10
x10      = 0.0
x20      = 0.0
x30      = 0.0
x20ell   = 0
x30ell   = 0
gamma    = 1.6666666
gamm1    = (gamma - 1)

# kepler parameters for an orbiting jet
# mass1 = mass of primary
# mass2 = mass of secondary
# corad = co-orbital radius
# keps1 = orbital eccentricty of primary
mass1    = 0.25
mass2    = 5.00
corad    = 50
keps1    = 0.5

# pressure & internal energy relationship (equation of state)
# determines pressure of ambient medium based on internal energy
p0 = e0 * gamm1

# Determine other jet parameters

djet    = drat * d0
pjet    = prat * p0
ejet    = prat * e0  
rsq   = rjet**2

# calculate speed of sound in jet material
# then speed of jet from Mach number (mjet)
cjet    = np.sqrt ( gamma * pjet / djet )
v1jet   = mjet * cjet

# components of jet velocity default to zero in y and z directions
v2jet = 0.00
v3jet = 0.00

# solid body rotation omega (not used at the moment)
ojet   = 0.0

# This is where the pulsation bits could go....
# define a time period variable

pulsation_period = 20 * yrsecs

# a pulse amplitude

pulsation_amplitude = (10 - 5) * (d0/2)

# and work out pulsation angular frequency from the time period

pulsation_angfreq = (2 * np.pi)/pulsation_period

# these will need to be referenced later in the code, though

# timesteps (bit redundant when the jet is stationary...
# update this to a bigger number when you get the jet moving)
# Maybe 10-20 for test runs.....50+ for a longer run
numsteps = 155

# Non time-dependent kepler bits for an orbiting jet
m1g      = mass1 * msol
m2g      = mass2 * msol
sma1     = corad * cmau * m2g / (m1g + m2g)
gm1pm2   = (m1g + m2g) * guniv
smilr1   = sma1 * (1-keps1**2)
orbp     = 2 * np.pi * np.sqrt(((corad * cmau)**3)/gm1pm2)

# initialise density boundary and gas energy boundary
denbound = denbound + d0
genbound = genbound + e0

fig, ax = plt.subplots(figsize=(6, 6))

extents = (x2min, x2max, x3min, x3max)
x2lims = (x2min, x2max)
x3lims = (x3min, x3max)


def jetinit():
    ax.set(xlim=x2lims, ylim=x3lims)
    
    
    # comment in/out to select density or velocity    
    #cax = ax.imshow(denbound, origin="lower", extent=extents, vmin=d0, vmax=djet)
    cax = ax.imshow(vel1bound, origin="lower", extent=extents, vmin=0, vmax=v1jet)

# ------------- START OF MAIN PROGRAM LOOP ------------------------


def jetboundary(step):

    tnow = yrsecs * step
    

#----- Newton-Raphson to solve Keppler's Equation --------------------

# ---- Work out Mean Anomaly (manom1) first -----------------
# ---- A fixed Mean Anomaly just means it will be stationary!  
# ---- This needs to be modified to update the Mean Anomaly as
# ---- a function of time (which will also involved the orbital
# ---- parameter orbp)
        
    manom1   = (twopi * tnow)/orbp
    
# ---- Initial Guess - let's guess the eccentric anomaly equals the mean anomaly ------
    
    eanom1    = manom1
    
# ---- an iterative solver is needed here to update the eccentric anomaly -------------
# ---- to make it closer and closer to its true value ---------------------------------
# ---- use Newton-Raphson's method to solve the equation ------------------------------
# ---- and use a sensible number of iterations ----------------------------------------
# ---- Hint: about 5 iterations should do it ------------------------------------------

# put your solver code in here

    iterations = 5

    for i in range(iterations):
        f_e = eanom1 - keps1 * np.sin(eanom1) - manom1
        f_prime_e = 1 - keps1 * np.cos(eanom1)
        eanom1 -= f_e/f_prime_e
        
        
    
# ---- End of iterative solver section ------------------------------------------------

    eanom1   = np.pi - eanom1
    ctanm1  = (np.cos(eanom1)-keps1)/(1-keps1*np.cos(eanom1))
    signge  = sign(eanom1)**int(manom1/np.pi)
    tanom1   = np.pi - (signge * np.arccos(ctanm1))
    rvctor  = smilr1/(1+keps1*np.cos(tanom1))
    x20ell = rvctor * np.cos(tanom1)
    x30ell = rvctor * np.sin(tanom1)
    mx0ell = sma1 * (1 + keps1)

    rjelli = rjet + mx0ell
    
 
#      Determine the extent of the jet in j and k.

    kjetmn = 0
    for k in range(kzones):
        if ( x3a[k] < -rjelli ):
            kjetmn = k
    
    kjetmx = kzones
    for k in range(kzones-1, 0, -1):
        if ( x3a[k] >=  rjelli ):
            kjetmx = k

    jjetmn = 0
    for j in range(jzones):
        if ( x2a[j] < -rjelli ):
            jjetmn = j

    jjetmx = jzones
    for j in range(jzones-1, 0, -1):
        if ( x2a[j] >= rjelli ):
            jjetmx = j


#  Determine the corner of each zone which lies closest to the
#  origin (jin,kin) and furthest from the origin (jout,kout).

    for j in range(jjetmn,jjetmx):
        jp1 = j + 1
        if ( abs(x2a[j]) < abs(x2a[jp1]) ):
            jin[j]  = j
            jout[j] = jp1
        else:
            jin[j] = jp1
            jout[j] = j


    for k in range(kjetmn,kjetmx):
        kp1 = k + 1
        if ( abs(x3a[k]) < abs(x3a[kp1]) ):
            kin[k]  = k
            kout[k] = kp1
        else:
            kin[k]  = kp1
            kout[k] = k


#-----------------------------------------------------------------------
#---------------- SET HYDRODYNAMICAL BOUNDARY VARIABLES ----------------
#-----------------------------------------------------------------------

    for k in range(kjetmn,kjetmx):
      x3ak  = x3a[k]
      x3bk = x3b[k]
      x3aki = x3a[kin[k]]
      x3ako = x3a[kout[k]]
      
      for j in range(jjetmn,jjetmx):
        x2aj  = x2a[j]
        x2bj = x2b[j]
        x2aji = x2a[jin[j]]
        x2ajo = x2a[jout[j]]
        

# -------- Transformed Coordinates Adjusting for Centre ------------------

        x2a0 = (x2aj - x20ell)
        x3a0 = (x3ak - x30ell)

        x2a0i = (x2aji - x20ell)
        x3a0i = (x3aki - x30ell)
        
        x2a0o = (x2ajo - x20ell)
        x3a0o = (x3ako - x30ell)
        
        
# -------- a few useful quantities ---------------        
        

        # mid-point y and z distance of the centre of
        # the currently indexed grid zone wrt the centre 
        # of the jet column; not currently used but may be
        # needed later...
        x2bmid = (x2bj - x20ell)
        x3bmid = (x3bk - x30ell)


        # these are the squares of the inner and outer 
        # radial distance of the currently indexed grid 
        # zone with respect to the midpoint of the jet
        rinsq  = x2a0i**2 + x3a0i**2
        routsq = x2a0o**2 + x3a0o**2

        # and this is the square of the radial distance
        # of the midpoint of the currently indexed zone
        # with respect to the midpoint of the jet
        # use this for any calculation involving radial
        # distance from the centre of the jet column
        # not currently used but may be useful....
        x2amid = sign(x2a0)*(x2a0i+x2a0o)/2
        x3amid = sign(x3a0)*(x3a0i+x3a0o)/2
        rpntsq = x2amid**2 + x3amid**2


# -------- final jet velocity and density component calculations


        pulsation = np.sin(pulsation_angfreq * tnow) 
        density_factor = (10 - 5) / 2 * pulsation + (10 + 5) / 2
        djet_pulsed = d0 * density_factor

    
        rpnt = np.sqrt(x2amid**2 + x3amid**2)
        velocity_profile_factor = 0.5 + 0.5*(rjet - rpnt)/rjet
        velocity_profile_factor = np.clip(velocity_profile_factor, 0.5, 1.0)
        



# -------- add in any more required components here 
# -------- or modify these expressions

        # velocity factors
        v1fact = 1.00
        v2fact = 1.00
        v3fact = 1.00

        # velocity additions
        v1add = 0.00
        v2add = 0.00
        v3add = 0.00

        # final modified velocity components
        v1mod = v1jet * v1fact + v1add
        v2mod = v2jet * v2fact + v2add
        v3mod = v3jet * v3fact + v3add
        
        # demsity factor and additive
        dfact = 1.00
        dadd = 0.00
        
        # final modified density
        
        djmod = djet_pulsed * dfact + dadd        


# -------- (re)set default boundary condition -------------

        vel1bound[j,k] = 0
        vel2bound[j,k] = 0
        vel3bound[j,k] = 0
        denbound[j,k]  = d0
        genbound[j,k]  = e0


#  Case:  Zones are completely contained by the inlet cross section.

        if (routsq <= rsq):

            vel1bound[j,k] = v1mod * velocity_profile_factor
            vel2bound[j,k] = v2mod * velocity_profile_factor
            vel3bound[j,k] = v3mod * velocity_profile_factor
            denbound[j,k] = djmod
            genbound[j,k] = ejet

#  Case:  Zones are partially contained by the inlet cross section .
#
        if ((rinsq < rsq) and (routsq > rsq)):
            frac = overlap (2, rjet, x10, x20, x30, 0, x2a0i, x3a0i, 0, x2a0o, x3a0o )
            cofrac       = 1.0 - frac
            dfrac        = djmod * frac
            dcofrc       = d0   * cofrac
            dtot         = dfrac + dcofrc
            dfrtot       = dfrac / dtot
            dcotot       = dcofrc / dtot
            denbound[j,k] = dtot
            vel1bound[j,k] = ( v1mod * dfrtot )
            vel2bound[j,k] = ( v2mod * dfrtot ) 
            vel3bound[j,k] = ( v3mod * dfrtot )
            genbound[j,k] = (ejet * dfrtot) + (e0 * dcotot)
            



    print("Step " + str(step))
    
    ax.set(xlim=x2lims, ylim=x3lims)
    

    # comment in/out to select density or velocity
    #cax = ax.imshow(denbound, origin="lower", extent=extents, vmin=d0, vmax=djet)
    cax = ax.imshow(vel1bound, origin="lower", extent=extents, vmin=0, vmax=v1jet)

    #cax = ax.imshow(denbound, origin="lower", extent=extents, vmin=0, vmax=1)
       
    # ------------- END OF MAIN PROGRAM LOOP ------------------------  


anim = animation.FuncAnimation(fig, jetboundary, interval=1000, init_func=jetinit(), frames=numsteps)

 
ax.set(xlim=x2lims, ylim=x3lims)
plt.draw()
plt.show()

writeani = animation.PillowWriter(fps=10,
                                metadata=dict(artist='Me'),
                                bitrate=1800)

anim.save('jet_boundary2.gif', writer=writeani)
