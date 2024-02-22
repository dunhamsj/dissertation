#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'publication.sty' )

from gaussxw import gaussxw

nN = [ '02', '03' ]
nX = [ '0016', '0032', '0064', '0128', '0256', '0512', '1024', '2048' ]

wq2 = np.array( [ 1.0, 1.0 ], np.float64 ) / 2.0
wq3 = np.array( [ 5.0, 8.0, 5.0 ], np.float64 ) / 18.0

L1    = np.zeros( (len(nN),len(nX)), np.float64 )
L2    = np.zeros( (len(nN),len(nX)), np.float64 )
norm1 = np.zeros( (len(nN),len(nX)), np.float64 )
norm2 = np.zeros( (len(nN),len(nX)), np.float64 )
nDOF  = np.zeros( (len(nN),len(nX)), np.float64 )

dN = 2

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

for i in range( len( nN ) ):

    N = np.int64( nN[i] )

    for j in range( len( nX ) ):

        NX = np.int64( nX[j] )

        fileName \
          = 'convergenceRates/Advection1D_SineWaveX1_nN{:}_RK{:}_nX{:}' \
             .format( nN[i], nN[i], nX[j] )

        init = np.loadtxt( fileName + '_init.dat' )
        finl = np.loadtxt( fileName + '_finl.dat' )

        i_q = init[:,-N:]
        f_q = finl[:,-N:]

        i_qq = np.empty( (NX,N+dN), np.float64 )
        f_qq = np.empty( (NX,N+dN), np.float64 )

        xq , wq  = gaussxw( N   )
        xqq, wqq = gaussxw( N+dN )

        for k in range( NX ):
            for ell in range( N+dN ):
                i_qq[k,ell] = rhoh( xqq[ell], i_q[k], xq )
                f_qq[k,ell] = rhoh( xqq[ell], f_q[k], xq )

        dx = init[0,4]

#        x_pp = np.empty( (NX,N+dN), np.float64 )
#        xL = np.array( [ k*dx for k in range( NX ) ], np.float64 )
#        xR = xL + dx
#        xC = 0.5 * ( xL + xR )
#
#        for k in range( NX ):
#            x_pp[k] = xC[k] + xqq * dx
#        x_p = init[:,7:7+N]
#
#        plt.plot( x_pp.flatten(), i_qq.flatten(), label = 'N+dN' )
#        plt.plot( x_p .flatten(), i_q .flatten(), label = 'N' )
#        plt.legend()
#        plt.show()
#        exit()

        d = f_qq - i_qq

        nDOF[i,j] = N * NX

        for k in range( d.shape[0] ):
            L1   [i,j] += dx * np.sum( wqq * np.abs( d[k] ) )
            L2   [i,j] += dx * np.sum( wqq * d[k]**2 )
            norm1[i,j] += dx * np.sum( wqq * np.abs( f_qq[k] ) )
            norm2[i,j] += dx * np.sum( wqq * f_qq[k]**2 )

fig, ax = plt.subplots( 1, 1 )

c = [ 'b', 'r' ]

for i in range( len( nN ) ):

    N = np.int64( nN[i] )

    x = nDOF[i]
    y1 = ( L1[i] / norm1[i] )**( 1.0 )
    y2 = ( L2[i] / norm2[i] )**( 0.5 )

    ax.loglog( x, y1, c[i] + '.' )
    #ax.loglog( x, y2, c[i] + 'x', label = r'$L_{2}$' )
    ax.loglog( x, y1[0] * 10**( - N * np.log10( x / x[0] ) ), c[i] + '-', \
                label = r'$N={:d}$'.format( N ) )

ax.legend()

ax.set_title( 'Convergence Rates for\nSine Wave Advection (1D)' )

ax.grid( which = 'both' )
ax.xaxis.set_ticks_position( "both" )
ax.yaxis.set_ticks_position( "both" )

ax.set_xlabel( r'$\mathrm{nDOF}:=N\times\mathrm{nElements}$' )

yLabel \
  = r'$\int_{\mathcal{D}}\left|u_{\mathrm{final}}-u_{\mathrm{initial}}\right|$' \
      + r'$/\int_{\mathcal{D}}\left|u_{\mathrm{final}}\right|$'
ax.set_ylabel( yLabel )

#plt.show()

figName = 'fig.ConvergenceRates.png'
plt.savefig( figName, dpi = 300 )
print( '\n  Saved {:}'.format( figName ) )
