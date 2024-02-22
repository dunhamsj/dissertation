#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'publication.sty' )
from scipy.interpolate import interp1d
from gaussxw import gaussxw

def rhoExact(x):
    return 1.0 + 0.1 * np.sin( 2.0 * np.pi * x )

xL = 0.0
xH = 1.0
x = np.linspace( xL, xH, 10000 )

dx = 0.2
xI = np.arange( xL, xH + 0.1*dx, dx )
nX = xI.shape[0]-1

def plotSine( fig, ax ):

    ax.plot( x, rhoExact( x ), 'k-' )

    ax.set_xlabel( r'$x$' )
    ax.set_ylabel( r'$u$' )

    ax.set_xlim( -0.1, 1.1 )

# Bare sine
fig, ax = plt.subplots()
plotSine( fig, ax )
#plt.show()
plt.savefig( 'fig.sine.png', dpi = 300 )
plt.close()

# Sine with grid lines
fig, ax = plt.subplots()
plotSine( fig, ax )
for i in xI: ax.axvline( i )
#plt.show()
plt.savefig( 'fig.sineWithLines.png', dpi = 300 )
plt.close()

# Sine with grid lines and constant approximation

def Lagrange( x, xq, i ):

    L = 1.0

    for j in range( xq.shape[0] ):

        if j != i:

            L *= ( x - xq[j] ) / ( xq[i] - xq[j] )

    return L

def ComputeMassMatrix( nN, nQ ):

    xn, wn = gaussxw( nN  )
    xq, wq = gaussxw( nQ )

    M = np.zeros( (nN,nN), np.float64 )

    for i in range( nN ):
        for j in range( nN ):

            Li = np.array( [ Lagrange( xq[q], xn, i ) for q in range( nQ ) ] )
            Lj = np.array( [ Lagrange( xq[q], xn, j ) for q in range( nQ ) ] )

            M[i,j] = np.sum( wq * Li * Lj )

    return M

def rhoh( x, rho_q, xq ):

    rho = 0.0
    for i in range( xq.shape[0] ):
        rho += rho_q[i] * Lagrange( x, xq, i )

    return rho

def PlotDensity( nN, nQ, fig, ax, c ):

    etan, wn = gaussxw( nN )
    etaq, wq = gaussxw( nQ )

    M = ComputeMassMatrix( nN, nQ )

    intU = np.empty( nN, np.float64 )

    xl = xL
    for iX1 in range( nX ):

        xh = xl + dx
        xC = 0.5 * ( xl + xh )
        xx = np.linspace( xl, xh, 100 )

        xn = xC + dx * etan
        xq = xC + dx * etaq

        for i in range( nN ):
            intU[i] = np.sum( wq * rhoExact( xq ) * Lagrange( xq, xn, i ) )

        rho_q = np.dot( np.linalg.inv( M ), intU )

        rho = np.empty( xx.shape[0], np.float64 )
        for i in range( rho.shape[0] ):
            rho[i] = rhoh( xx[i], rho_q, xn )

        ax.plot( xn, rho_q, marker = '.', color = c )
        ax.plot( xx, rho, ls = '-', color = c )

        xl = xl + dx

    return

# Sine with grid lines and approximation
fig, ax = plt.subplots()
plotSine( fig, ax )
for i in xI: ax.axvline( i )
nN = 1
nQ = 10
PlotDensity( nN, nQ, fig, ax, c = 'r' )
#plt.show()
plt.savefig( 'fig.sineWithLines_DG0.png', dpi = 300 )
plt.close()

# Sine with grid lines and approximation
fig, ax = plt.subplots()
plotSine( fig, ax )
for i in xI: ax.axvline( i )
nN = 2
nQ = 10
PlotDensity( nN, nQ, fig, ax, c = 'm' )
#plt.show()
plt.savefig( 'fig.sineWithLines_DG1.png', dpi = 300 )
plt.close()

# Sine with grid lines and approximation
fig, ax = plt.subplots()
plotSine( fig, ax )
for i in xI: ax.axvline( i )
nN = 3
nQ = 10
PlotDensity( nN, nQ, fig, ax, c = 'b' )
#plt.show()
plt.savefig( 'fig.sineWithLines_DG2.png', dpi = 300 )
plt.close()
