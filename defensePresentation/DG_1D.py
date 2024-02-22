#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'publication.sty' )
from scipy.interpolate import interp1d

from gaussxw import gaussxw

NN = 10

x = np.linspace( -0.5, +0.5, 100 )
xL = 0.6
xH = 0.8
xC = 0.5 * ( xL + xH )
dX = xH - xL

x = np.linspace( xL, xH, 100 )

def Lagrange( x, xq, i ):

    L = 1.0

    for j in range( xq.shape[0] ):

        if j != i:

            L *= ( x - xq[j] ) / ( xq[i] - xq[j] )

    return L

def rhoh( x, rho_q, xq ):

    N = xq.shape[0]

    rho = 0.0
    for i in range( N ):
        rho += rho_q[i] * Lagrange( x, xq, i )

    return rho

def rhoExact( x ):
    return 1.0 + 0.1 * np.sin( 2.0 * np.pi * x )
    #return 1.0 + x + x**2 + x**3 + x**4 + x**5

def CreateElement():

    fig, ax = plt.subplots( 1, 1 )

    ax.set_xticks( [ xL, xC, xH ], \
                   [ r'$x_{\mathrm{L}}$', \
                     r'$x_{\mathrm{C}}$', \
                     r'$x_{\mathrm{H}}$' ] )
    ax.axvline( xL )
    ax.axvline( xH )

    return fig, ax

def ComputeMassMatrix( N ):

    xqN, wqN = gaussxw( N  )
    xq , wq  = gaussxw( NN )

    M = np.zeros( (N,N), np.float64 )

    for i in range( N ):
        for j in range( N ):

            Li = np.array( [ Lagrange( xq[q], xqN, i ) for q in range( NN ) ] )
            Lj = np.array( [ Lagrange( xq[q], xqN, j ) for q in range( NN ) ] )

            M[i,j] = np.sum( wq * Li * Lj )

    return M

def AddQuadraturePoints( ax, N, c ):

    xq, wq = gaussxw( N )

    xq = xC + dX * xq

    ms = 5
    if N == 1: ms = 10
    rhoE = rhoExact( x )
    a = 0.5 * ( rhoE.min() + rhoE.max() )
    for i in range( N ):
        ax.plot( xq[i], a, c + '.', ms = ms )

    return

def AddFacePoints( ax ):

    rhoE = rhoExact( x )
    a = 0.5 * ( rhoE.min() + rhoE.max() )
    ax.plot( xL, a, 'ks', mfc = 'None' )
    ax.plot( xH, a, 'ks', mfc = 'None' )

def PlotExact( fig, ax ):

    rho = np.empty( (x.shape[0]), np.float64 )
    for i in range( rho.shape[0] ):
        rho[i] = rhoExact( x[i] )

    vmin = rho.min()
    vmax = rho.max()

    ax.plot( x, rho, 'k-', label = 'Exact' )

    return vmin, vmax

def PlotDensity( N, fig, ax, vmin, vmax, c ):

    rhoE = interp1d( x, rhoExact( x ) )

    xqN, wqN = gaussxw( N  )
    xq , wq  = gaussxw( NN )

    xqN = xC + dX * xqN
    xq  = xC + dX * xq

    intU = np.empty( N, np.float64 )

    for i in range( N ):
        intU[i] = np.sum( wq * rhoE( xq ) * Lagrange( xq, xqN, i ) )

    M = ComputeMassMatrix( N )

    rho_q = np.dot( np.linalg.inv( M ), intU )

    rho = np.empty( (x.shape[0]), np.float64 )
    for i in range( rho.shape[0] ):
        rho[i] = rhoh( x[i], rho_q, xqN )

    ax.plot( x, rho, c + '--', label = r'$N={:d}$'.format( N ) )
    ax.plot( xqN, rho_q, c + 'x' )

    return

if __name__ == '__main__':

    fig, ax = CreateElement()
    vmin, vmax = PlotExact( fig, ax )
    #AddFacePoints( ax )

    N = 1
    c = 'r'
    PlotDensity( N, fig, ax, vmin, vmax, c )
#    AddQuadraturePoints( ax, N, c )

    N = 2
    c = 'm'
    PlotDensity( N, fig, ax, vmin, vmax, c )
#    AddQuadraturePoints( ax, N, c )

    N = 3
    c = 'b'
    PlotDensity( N, fig, ax, vmin, vmax, c )
#    AddQuadraturePoints( ax, N, c )

    ax.legend( loc = 'upper center' )

    #plt.show()

    figName = 'fig.DG_1D.png'
    plt.savefig( figName, dpi = 300 )
    print( '\n  Saved {:}'.format( figName ) )

    plt.close()
