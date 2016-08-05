import numpy as np

#=================================================================================================#
def lsq_fit_cubic(x,y):
    coeffs = np.polyfit(x,y,3)
    return coeffs
#=================================================================================================#

#=================================================================================================#
def _smooth(arr,Ns):
    sm_arr = np.array(arr)
    Narr = arr.size
    half_Ns = int(round(Ns/2))
    for ii in range(Narr):
        imin = max(0,ii-half_Ns)
        imax = min(ii+half_Ns,Narr-1)
        sm_arr[ii] = np.mean(arr[imin:imax+1])
    return sm_arr
#=================================================================================================#

#=================================================================================================#
def _deriv_2d(x,y):
    deriv = np.array(y)
    for ii in range(1,len(y)-1):
        dy = y[ii+1]-y[ii-1]
        dx = x[ii+1]-x[ii-1]
        deriv[ii] = dy/dx
    # Handle first and last points
    deriv[0]  = (y[1]-y[0]) / (x[1]-x[0])
    deriv[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
    
    return deriv
#=================================================================================================#

#=================================================================================================#
def turning_points(x,y,Nsmooth=None,dx_smooth=None):
    """
    Find all local minima, maxima and points of inflection in 2d profile.
    *** x is assumed to be monotonically increasing/decreasing! ***
        Turning point criterion:
            MAX            : dydx_left: +ve; dydx_right: -ve
            MIN            : dydx_left: -ve; dydx_right: +ve
            POI            : d2ydx2_left/d2ydx2_right < 0
            STATIONARY POI : Is POI AND d2ydx2_left/d2ydx2_right < 0
        Parameters:
            dx_smooth : x-scale over which to smooth y(x), dy/dx before taking derivative
                      : N.B. x-scale is converted to a number of bins (by dividing by average binsize).
                      : (Smoothing over non-constant number of bins introduces noise!)
            Nsmooth   : Number of bins over which to smooth y(x), dy/dx before taking derivative
    """
    
    if Nsmooth is None:
        if dx_smooth is None:
            Nsmooth = 1
        else:
            Nsmooth = int(round(dx_smooth*x.size/(x[-1]-x[0])))
    
    # Store some properties of the profile
    tp = {}
    tp['x']      = x
    tp['y']      = y
    tp['N']      = y.size
    tp['Nsmooth'] = Nsmooth

    # Smooth, then compute first and second derivatives
    tp['y_sm']    = _smooth(y,Ns=Nsmooth)
    tp['dydx']    = _deriv_2d(x,tp['y_sm'])
    tp['dydx_sm'] = _smooth(tp['dydx'],Ns=Nsmooth)
    tp['d2ydx2']  = _deriv_2d(x,tp['dydx_sm'])
    tp['all']       = [] # List of (x,y) tuples identifying all turning point locations
    tp['maxima']    = [] # List of (x,y) tuples identifying maxima locations
    tp['minima']    = [] # List of (x,y) tuples identifying minima locations
    tp['pois']      = [] # List of (x,y) tuples identifying poi locations
    tp['stat_pois'] = [] # List of (x,y) tuples identifying stationary poi locations
    
    # Find turning points
    for ii in range(1,tp['N']):
        dxi = x[ii] - x[ii-1] 
        xi  = x[ii] + dxi/2
        dyi = y[ii] - y[ii-1] 
        yi  = y[ii] + dyi/2
        
        dydx_left    = tp['dydx'][ii-1]
        dydx_right   = tp['dydx'][ii] 
        d2ydx2_left  = tp['d2ydx2'][ii-1]
        d2ydx2_right = tp['d2ydx2'][ii]
        if (dydx_left > 0 and dydx_right < 0):
            tp['all'].append((xi,yi))
            tp['maxima'].append((xi,yi))
        elif (dydx_left < 0 and dydx_right > 0):
            tp['all'].append((xi,yi))
            tp['minima'].append((xi,yi))
        elif (d2ydx2_left/d2ydx2_right < 0):
            tp['all'].append((xi,yi))
            tp['pois'].append((xi,yi))
            if (dydx_left/dydx_right < 0):
                tp['stat_pois'].append((xi,yi))
    
    return tp
#=================================================================================================#