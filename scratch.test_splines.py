from scipy.interpolate import UnivariateSpline
from cmcl.io import mods_outputs
from utils.fitting import lsq_fit_cubic,turning_points
import matplotlib.pyplot as plt
import numpy as np
import pdb

#=================================================================================================#
def bspleval(x, knots, coeffs, order, debug=False):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = alen(t)
    npts = alen(x)
    B = zeros((m-1,k+1,npts))

    # if debug:
        # print('k=%i, m=%i, npts=%i' % (k, m, npts))
        # print('t=', t)
        # print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = float64(logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    # if debug:
        # plt.figure()
        # for i in range(m-1):
            # plt.plot(x, B[i,k,:])
        # plt.title('B-spline basis functions')

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]

    # if debug:
        # plt.figure()
        # plt.plot(x, y)
        # plt.title('spline curve')
        # plt.show()

    return(y)
#=================================================================================================#    

#=================================================================================================#
def fit_and_plot_mpl_bspline(x,y):
# Fit B-spline with matplotlib and plot it
    SPLINE_ORDER = 3
    spl = UnivariateSpline(x,y,k=SPLINE_ORDER)
    plt.plot(x,y,'r')
    plt.plot(x,spl(x),'g')

    knots  = spl.get_knots()
    coeffs = spl.get_coeffs()
    yr = plt.ylim()

    for knot in knots:
        plt.axvline(knot,color='g')
    #for coeff in coeffs:
        #plt.text(???,(yr[0]+yr[1])/2,"%f" % coeff)
    plt.show()
#=================================================================================================#    

#=================================================================================================#
def get_mods_pressure_data():
    # Read amd return a MoDS pressure profile
    algorithm = "Levenberg_Marquardt_Alg_1"
    subtype   = "Pressure__bar_"
    root_dir  = "C:\Users\Owen\Documents\MoDS\CAT MoDs"
    
    mo   = mods_outputs(root_dir)
    prof = mo.get_subtype_data(algorithm,subtype, cases=(1), nbest=1)
    return prof['cad'],prof['profile']
#=================================================================================================#

#==============================================================================
def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d
#==============================================================================

#==============================================================================
def fit_segment(x,y,xmin,xmax):
    
    for ix in range(0,x.size):
        if x[ix] > xmin: break
        imin = ix
    for ix in range(x.size-1,0,-1):
        if x[ix] < xmax: break
        imax = ix
    
    xseg = x[imin:imax+1]
    yseg = y[imin:imax+1]
    a,b,c,d = lsq_fit_cubic(xseg,yseg)
    yseg_fit = cubic(xseg,a,b,c,d)
    
    return xseg,yseg_fit
#==============================================================================

x,y = get_mods_pressure_data()
#fit_and_plot_mpl_bspline(x,y)

tps = turning_points(x,y,dx_smooth=2)

plt.plot(tps['x'], tps['y'],'r',label=r'$y(x)$')

for p1,p2 in zip(tps['all'][:-1],tps['all'][1:]):
    xmin,y1 = p1
    xmax,y2 = p2
    xseg,yseg = fit_segment(x,y,xmin,xmax)
    plt.plot(xseg,yseg,'b:')
    #print('Segment from %f to %f' % (xmin,xmax) )

plt.show()
    
# for xmin,y in tps['minima']:
    # print("Minimum at %f" % xmin)
# for xmax,y in tps['maxima']:
    # print("Maximum at %f" % xmax)
# for xpoi,y in tps['pois']:
    # print("PoI at %f" % xpoi)



