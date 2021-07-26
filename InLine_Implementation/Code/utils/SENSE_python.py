# coding: utf-8

"""
This file is the result of running nbconvert on SENSE_python.ipynb
and manually replacing the `%pylab inline` magic with
the corresponding imports. I also removed the visualization
code.
"""

# In[1]:

from __future__ import division

from matplotlib import pylab, mlab, pyplot
import matplotlib
from numpy import *
import numpy
from pylab import *
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.special import iv as besseli


# Substitute for '%pylab inline' from
# https://ipython.org/ipython-doc/3/interactive/magics.html#magic-pylab
np = numpy
plt = pyplot
# from IPython.display import display
# from IPython.core.pylabtools import figsize, getfigs


# from forbiddenfruit import curse
# curse(np.ndarray,'H',property(fget=lambda A: A.T.conj()))


# In[2]:


def fftc(img):
    return ifftshift(fftn(fftshift(img)))

def ifftc(ksp):
    return fftshift(ifftn(ifftshift(ksp)))
    
# get data from http://hansenms.github.io/sunrise/sunrise2013/
HANSEN = loadmat("/media/helrewaidy/Data2/Playground/ismrm_sunrise_parallel/hansen_exercises.mat")
locals().update(HANSEN)

# In[5]:

x, y, ncoils = smaps.shape

# In[6]:

def NuFFT(image, coils, k_traj, w, n, osf, wg):    
    # pad image with zeros to the oversampling size
    pw = pad_width = int( n*(osf-1)/2 )
    image = pad(image, ([pw,pw], [pw,pw], [0,0]), 'constant' )

    # Kernel computation 
    kw = wg / osf
    kosf = floor(0.91 / (osf*1e-3))
    kwidth = osf*kw / 2
    beta = pi*sqrt((kw*(osf-0.5))**2-0.8)

    # compute kernel
    om = arange( floor(kosf*kwidth)+1 ) / floor(kosf*kwidth)
    p = besseli(0, beta*sqrt(1-om*om))
    p /= p[0]
    p[-1] = 0

    # deapodize
    x = arange(-osf*n/2, osf*n/2, dtype=complex) / n
    sqa = sqrt(pi*pi * kw*kw * x*x - beta*beta)
    dax = sin(sqa) / sqa
    dax /= dax[ int(osf*n/2) ]
    da = outer(dax.H, dax)
    image /= da[:,:,newaxis]
    
    m = np.zeros((len(k_traj), coils), dtype=complex) 
                      
    # convert k-space trajectory to matrix indices
    nx = n*osf/2 + osf*n*k_traj[:,0]
    ny = n*osf/2 + osf*n*k_traj[:,1]
    
    # Do the NuFFT for each coil
    for j in range(coils):
        # Compute FFT of each coil image
        kspace = fftc(image[:,:,j])

        # loop over samples in kernel at grid spacing
        for lx in arange(-kwidth, kwidth+1):
            for ly in arange(-kwidth, kwidth+1):                
                # find nearest samples
                nxt = around(nx+lx)
                nyt = around(ny+ly)
                
                # separable kernel value
                kkr = minimum(
                    np.round(kosf * sqrt(abs(nx-nxt)**2+abs(ny-nyt)**2)),
                    floor(kosf * kwidth)
                ).astype(int)
                kwr = p[kkr]

                # if data falls outside matrix, put it at the edge, zero out below
                nxt = nxt.clip(0,osf*n-1).astype(int)
                nyt = nyt.clip(0,osf*n-1).astype(int)
                vector = np.array([kspace[x,y] for x, y in zip(nxt, nyt)])
                
                # accumulate gridded data
                m[:,j] += vector * kwr
                
    return m * sqrt(w)


def NuFFT_adj(data, coils, k_traj, w, n, osf, wg):
    # width of the kernel on the original grid
    kw = wg / osf
    
    # preweight
    dw = data * sqrt(w)
    
    # compute kernel, assume e1 is 0.001, assuming nearest neighbor
    kosf = floor(0.91/(osf*1e-3))
    
    # half width in oversampled grid units
    kwidth = osf * kw / 2
    
    # beta from the Beatty paper
    beta = pi * sqrt((kw*(osf-0.5))**2 - 0.8)
    
    # compute kernel
    om = arange(floor(kosf*kwidth)+1, dtype=complex) / (kosf*kwidth)
    p = besseli(0, beta*sqrt(1-om*om))
    p /= p[0]
    p[-1] = 0

    # convert k-space trajectory to matrix indices
    nx = n*osf/2 + osf*n*k_traj[:,0]
    ny = n*osf/2 + osf*n*k_traj[:,1]
    
    im = zeros((osf*n, osf*n, coils), dtype=complex)
  
    for j in range(coils):    
        ksp = np.zeros_like(im[:,:,j])

        for lx in arange(-kwidth, kwidth+1):
            for ly in arange(-kwidth, kwidth+1):
                # find nearest samples
                nxt = around(nx + lx)
                nyt = around(ny + ly)
                
                # seperable kernel value
                kkr = minimum(
                        around(kosf * sqrt(abs(nx-nxt)**2 + abs(ny-nyt)**2)),
                        floor(kosf * kwidth)
                    ).astype(int)
                kwr = p[kkr]
                
                # if data falls outside matrix, put it at the edge, zero out below
                nxt = nxt.round().clip(0,osf*n-1).astype(int)
                nyt = nyt.round().clip(0,osf*n-1).astype(int)

                # accumulate gridded data
                ksp += coo_matrix((dw[:,j]*kwr.H, (nxt, nyt)), shape=(osf*n,osf*n))
                    
        # zero out data at edges, which is probably due to data outside mtx
        ksp[:,0] = ksp[:,-1] = ksp[0,:] = ksp[-1,:] = 0       
    
        # do the FFT and deapodize
        im[:,:,j] = ifftc( ksp )

    # deapodize
    x = arange(-osf*n/2, osf*n/2, dtype=complex) / n
    sqa = sqrt(pi*pi * kw*kw * x*x - beta**2)
    dax = sin(sqa) / sqa
    dax /= dax[ int(osf*n/2) ]
    da = outer(dax.H, dax)
    im /= da[:,:,newaxis] 
    
    # trim the images
    q = int((osf-1)*n/2)
    return im[q:n+q, q:n+q]     

print('ok')


# In[7]:

N = np.array( smaps.shape[:2] )
osf = 2
wg = 5
K = N*osf
n = N[0]
scale = sqrt(prod(K))
coils = smaps.shape[2]

assert all(N == [256, 256])
assert all(K == [512, 512])
assert n == 256
assert coils == 8
assert scale == 512


# In[8]:

def dotf(a, b):
    return dot(a.flat, b.flat)

def Eforward(image, csm, k_traj, weights, n, coils, wg, osf, scale):
    return NuFFT( image[:,:,newaxis] * csm, coils, k_traj, weights, n, osf, wg) / scale

def Etransp(data, csm, k_traj, weights, n, coils, wg, osf, scale):
    return (csm.conj() * NuFFT_adj( data, coils, k_traj, weights, n, osf, wg)).sum(-1) * scale

E_forward = lambda x: Eforward(x, smaps, k_spiral, w_spiral, n, coils, wg, osf, scale)
E_transp  = lambda y:  Etransp(y, smaps, k_spiral, w_spiral, n, coils, wg, osf, scale)   


# In[9]:

At  = lambda v: E_transp(v)
AtA = lambda v: E_transp( E_forward(v) )
coildata = data_spiral * sqrt(w_spiral)

# AtA*x = At*b
b = At(coildata)
x = zeros_like(smaps[:,:,0])
r = b - AtA(x)
p = copy(r)

for i in range(13):
    rsold = dotf(r, r)
    AtAp = AtA(p)
    alpha = rsold / dotf(p, AtAp)
    x += alpha * p
    r -= alpha * AtAp   
    rsnew = dotf(r,r)
    beta = rsnew / rsold
    p = r + beta * p
    
#imshow(abs(x), cmap='gray')
#show()
