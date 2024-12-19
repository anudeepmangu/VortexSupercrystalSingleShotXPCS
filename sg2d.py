import numpy as np
import scipy as sp 
import scipy.linalg
def sg2d(degree,n,m,deriv=0):
    ''' filter=sg2d(D,n,m,deriv=0)
        2D Savitzky-Golay filter for 2D polynomial of degree D and size (n,m).
        set deriv for 0=smooth, 1=y-deriv 2=x-deriv, 3=y^2, etc
	 (check normalization for deriv >1)
        #order is a0+a1*y+a2*x+a3*y^2+a4*xy+a5*x^2+a6*y^3+a7*y^2x + ...
        #IE sum over i of  a_i * x^p[i]*y^q[i]
    '''
    #indices of grid points
    x=np.tile(np.arange(-(n//2),(n//2) + 1,1),m)
    y=np.repeat(np.arange(-(m//2),(m//2)+1,1),n)
    #find powers p of x and q of y
    dp1=degree+1
    tmp=np.resize(np.arange(dp1),[dp1,dp1])
    p=tmp[np.tril_indices(dp1)]
    q=(np.arange(dp1).reshape(dp1,1)-tmp)[np.tril_indices(dp1)]
    #matrix for fit 
    M=x.reshape(1,-1)**p.reshape(-1,1)*y.reshape(1,-1)**q.reshape(-1,1)
    #check that x is row index and y is columnn index
    B=np.zeros(p.shape)
    B[deriv]=1. # smoothing coefficient
    f=sp.linalg.lstsq(M,B)[0].reshape((n,m))
    return f

# if __name__ == '__main__':
#     import timeit
#     print("function gives:  ", sg2d(3,5,5))
#     print('and should equal ', np.array([[-0.07428571,  0.01142857,  0.04      ,  0.01142857, -0.07428571],
#        [ 0.01142857,  0.09714286,  0.12571429,  0.09714286,  0.01142857],
#        [ 0.04      ,  0.12571429,  0.15428571,  0.12571429,  0.04      ],
#        [ 0.01142857,  0.09714286,  0.12571429,  0.09714286,  0.01142857],
#        [-0.07428571,  0.01142857,  0.04      ,  0.01142857, -0.07428571]]))
#     print(timeit.timeit('sg2d(3,11,11)',setup='from __main__ import sg2d',number=1000)/1000.)
