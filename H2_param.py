# Load Packages
from enr.H2 import *
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from enr.DDG import computeBoundary
from torch.autograd import grad
import utils.utils as io
import torch
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32


def H2Midpoint(geod,newN,F_init,a0,a1,b1,c1,d1,a2,param):   
    max_iter=param['max_iter']
    
    N=geod.shape[0]

    # Convert Data to Pytorch
    left = torch.from_numpy(np.array([geod[0]])).to(dtype=torchdtype, device=torchdeviceId)
    right = torch.from_numpy(np.array([geod[N-1]])).to(dtype=torchdtype, device=torchdeviceId)
    
    
    xp=np.linspace(0,1,N,endpoint=True)
    x=np.linspace(0,1,newN,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,geod,axis=0)
    midpoints=f(x)
    
    midpoint = torch.from_numpy(midpoints[1:newN-1]).to(dtype=torchdtype, device=torchdeviceId)
    F_sol= torch.from_numpy(F_init).to(dtype=torch.long, device=torchdeviceId)    

    n=midpoint.shape[1]
    # Define Energy and set parameters
    energy = enr_param_H2(left,right, F_sol,a0,a1,b1,c1,d1,a2)
    
    def gradE(midpoint):
        qmidpoint = midpoint.clone().requires_grad_(True)
        return grad(energy(qmidpoint), qmidpoint, create_graph=True)
    

    def funopt(midpoint):
        midpoint=torch.from_numpy(midpoint.reshape(newN-2,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(midpoint).detach().cpu().numpy())

    def dfunopt(midpoint):
        midpoint = torch.from_numpy(midpoint.reshape(newN-2,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        [Gmidpoint] = gradE(midpoint)
        Gmidpoint = Gmidpoint.detach().cpu().numpy().flatten().astype('float64')
        return Gmidpoint

    out,fopt,Dic=fmin_l_bfgs_b(funopt, midpoint.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 0, maxls=20,maxfun=150000)
    out=out.reshape(newN-2,n,3)
    ngeod=np.concatenate((np.array([geod[0]]),out,np.array([geod[N-1]])), axis=0)
    geod= np.array(ngeod)
    return geod

def H2Parameterized(source,target,a0,a1,b1,c1,d1,a2,paramlist,rotate = False):
    F0=source[1]
    geod=np.array([source[0],target[0]])
    for param in paramlist:
        newN= param['time_steps']
        geod=H2Midpoint(geod,newN,F0,a0,a1,b1,c1,d1,a2,param)
        print(geod.shape)
    return geod, F0
