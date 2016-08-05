from utils.fitting import lsq_fit_cubic
import numpy as np
from matplotlib import pyplot as plt
#==============================================================================
def gen_data(a,b,c,d): 
    xmin = 0.
    xmax = 1.
    dx = 0.02
    x = np.arange(xmin,xmax,dx)
    y = cubic(x,a,b,c,d)
    # add noise
    dn = 0.05
    fnoise = (1-dn/2) + dn * np.random.random_sample(y.size)
    y *= fnoise
    return x,y
#==============================================================================

#==============================================================================
def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d
#==============================================================================

a_in = 2.35
b_in = -1.91
c_in = 0.4
d_in = 1.1

x,y = gen_data(a_in,b_in,c_in,d_in)

a,b,c,d = lsq_fit_cubic(x,y)

yfit = cubic(x,a,b,c,d)

plt.plot(x,y,'b.')
plt.plot(x,yfit,'r:')
plt.show()

print ("IN       : %f,%f,%f,%f" % (a_in,b_in,c_in,d_in))
print ("RECOVERED: %f,%f,%f,%f" % (a,b,c,d))
