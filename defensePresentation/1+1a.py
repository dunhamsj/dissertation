#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use( 'publication.sty' )
plt.rcParams.update \
  ( { 'text.latex.preamble': r'\usepackage{amsfonts}' \
                               + r'\usepackage{amsmath}' } )

style = 'Simple, tail_width = 0.5, head_width = 4, head_length = 8'
kw = dict( arrowstyle = style )

fig, ax = plt.subplots( 1, 1 )
ax.axis( 'off' )

xM = [ -1.0, -0.1 ]
yM = [ 0.0, 0.0 ]
ax.plot( [ xM[0], xM[1] ], [ yM[0], yM[0] ], 'k-' )
ax.plot( [ xM[1], xM[1] ], [ yM[0], yM[1] ], 'k-' )
ax.plot( [ xM[1], xM[0] ], [ yM[1], yM[1] ], 'k-' )
ax.plot( [ xM[0], xM[0] ], [ yM[1], yM[0] ], 'k-' )

xM = [ 0.0, 1.0 ]
yM = [ 0.0, 1.0 ]
ax.plot( [ xM[0], xM[1] ], [ yM[0], yM[0] ], 'k-' )
ax.plot( [ xM[1], xM[1] ], [ yM[0], yM[1] ], 'k-' )
ax.plot( [ xM[1], xM[0] ], [ yM[1], yM[1] ], 'k-' )
ax.plot( [ xM[0], xM[0] ], [ yM[1], yM[0] ], 'k-' )

ax.text( 0.44, 0.925, r'$\mathcal{M}\cong\Sigma\times\mathbb{R}^{+}$', \
         fontsize = 16 )

x = np.linspace( xM[0]+0.01, xM[1]-0.01, 100 )

cs1 = 'r'
def y1( x ):
    return 0.2 + ( 0.07 * np.sin( 5.0 * 2.0 * np.pi * x ) \
                    + 0.07 * np.sin( 2.0 * 2.0 * np.pi * x ) )
ax.plot( x, y1(x), cs1 + '-' )
x1 = x[45]
ax.plot( x1, y1( x1 ), cs1 + 'o', ms = 8 )
ax.text( 0.7, 0.05, r'$\Sigma_{t}$', c = cs1, fontsize = 16 )
ax.text( 0.05, 0.35, r'$\gamma_{ij}\left(t,x^{i}\right)$', \
         c = cs1, fontsize = 12 )

cs2 = 'b'
def y2( x ):
    return 0.75 + ( 0.07 * np.sin( 3.0 * 2.0 * np.pi * x ) \
                      - 0.03 * np.sin( 5.0 * 2.0 * np.pi * x ) )
    #return y1( x ) + 0.5
ax.plot( x, y2(x), cs2 + '-' )
x2 = x[70]
ax.plot( x2, y2( x2 ), cs2 + 'o', ms = 8 )
ax.text( 0.7, y2( 0.85 ) - 0.1, r'$\Sigma_{t+dt}$', c = cs2, fontsize = 16 )
ax.text( 0.05, 0.85, r'$\gamma_{ij}\left(t+dt,x^{i}\right)$', \
         c = cs2, fontsize = 12 )

n12 = patches.FancyArrowPatch \
        ( ( x1, y1( x1 ) ), ( x1, y2( x1 ) ), \
          color = 'm', **kw )
plt.gca().add_patch( n12 )
ax.text( x1-0.17, 0.5 * ( y1( x1 ) + y2( x1 ) ), \
         r'$\alpha\,\overline{n}$', fontsize = 16, color = 'm' )

ax.plot( -0.5, 0.0, 'ko', ms = 8 )
ax.text( -0.5, 0.04, r'$p$', fontsize = 15 )

ax.text( -0.7, -0.07, r'$\Sigma$', fontsize = 17 )
e1 = patches.FancyArrowPatch \
        ( ( -0.5, 0.0 ), ( x1, y1( x1 ) ), \
          connectionstyle="arc3,rad=0.3", \
          color = cs1, **kw )
plt.gca().add_patch( e1 )
ax.text( -0.05, -0.07, r'$e_{t}\left(p\right)$', \
         c = cs1, fontsize = 13 )

e2 = patches.FancyArrowPatch \
        ( ( -0.5, 0.0 ), ( 0.7, y2( 0.7 ) ), \
          connectionstyle="arc3,rad=0.3", \
          color = cs2, **kw )
plt.gca().add_patch( e2 )
ax.text( -0.4, 0.1, r'$e_{t+dt}\left(p\right)$', \
         c = cs2, fontsize = 13 )

#t = patches.FancyArrowPatch \
#        ( ( x1, y1( x1 ) ), ( x2, y2( x2 ) ), \
#          color = 'm', **kw )
#plt.gca().add_patch( t )
#ax.text( 0.5 * ( x1 + x2 )+0.05, 0.5 * ( y1( x1 ) + y2( x2 ) ), \
#         r'$\overline{t}$', \
#         c = 'm', fontsize = 16 )

b = patches.FancyArrowPatch \
        ( ( x1, y2( x1 ) ), ( x2, y2( x2 ) ), \
          color = 'm', **kw )
plt.gca().add_patch( b )
ax.text( x1+0.1, y2( x2 )+0.01, \
         r'$\overline{\beta}$', \
         c = 'm', fontsize = 16 )

#ind1 = 60
#ind2 = 70
#u12 = patches.FancyArrowPatch \
#        ( ( x[ind1], y1( x[ind1] ) ), ( x[ind2], y2( x[ind2] ) ), \
#          color = 'b', **kw )
#plt.gca().add_patch( u12 )
#ax.text( 0.5 * ( x[ind1] + x[ind2] ) + 0.02, \
#         0.5 * ( y1( x[ind1] ) + y2( x[ind2] ) ) - 0.01, \
#         r'$\overline{u}$', fontsize = 16, color = 'b' )

#plt.show()

figName = 'fig.1p1.png'
plt.savefig( figName, dpi = 300 )
print( '\n  Saved {:}'.format( figName ) )
