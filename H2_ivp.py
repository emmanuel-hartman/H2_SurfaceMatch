# Load Packages
from enr.H2 import *
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from enr.DDG import computeBoundary, getLaplacian
from torch.autograd import grad
import utils.utils as io
import torch
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

def stepforward(V0,V1,a0,a1,b1,c1,d1,a2,F_init):
    N=V0.shape[0] 
    energy = enr_step_forward(V0,V1,a0,a1,b1,c1,d1,a2,F_init)    
    def gradE(V2):
        qV2=V2.clone().requires_grad_(True)
        return grad(energy(qV2), qV2, create_graph=True)
    def funopt(V2):
        V2=torch.from_numpy(V2.reshape(N,3)).to(dtype=torchdtype, device=torchdeviceId)
        return energy(V2).detach().cpu().numpy().flatten()
    def dfunopt(V2):
        V2 = torch.from_numpy(V2.reshape(N,3)).to(dtype=torchdtype, device=torchdeviceId)
        [GV2] = gradE(V2)        
        GV2 = GV2.detach().cpu().numpy().flatten().astype('float64')
        return GV2
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, (2*V1-V0).cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=2000, disp =1, maxls=20,maxfun=150000)
    V2=xopt.reshape(N,3)
    return V2,fopt

def H2InitialValueProblem(V0,h,steps,a0,a1,b1,c1,d1,a2,F_init,total_steps=None):
    if total_steps is None:
        total_steps=steps
    V0=torch.from_numpy(V0).to(dtype=torchdtype, device=torchdeviceId)
    h=torch.from_numpy(h).to(dtype=torchdtype, device=torchdeviceId)
    F_init=torch.from_numpy(F_init).to(dtype=torch.long, device=torchdeviceId)
    h0=h/steps
    V1=V0+h0
    ivp=[V0.cpu().numpy(),V1.cpu().numpy()]
    for i in range(2,total_steps):
        V2,fopt=stepforward(V0,V1,a0,a1,b1,c1,d1,a2,F_init)
        V2=torch.from_numpy(V2).to(dtype=torchdtype, device=torchdeviceId)
        ivp.append(V2.cpu().numpy())
        V0=V1
        V1=V2
    return np.asarray(ivp),F_init


def smooth_stepforward(V0,V1,a0,a1,b1,c1,d1,a2,F_init):
    N=V0.shape[0] 
    energy = enr_step_forward(V0,V1,a0,a1,b1,c1,d1,a2,F_init)    
    def gradE(V2):
        qV2=V2.clone().requires_grad_(True)
        return grad(energy(qV2), qV2, create_graph=True)
    def funopt(V2):
        V2=torch.from_numpy(V2.reshape(N,3)).to(dtype=torchdtype, device=torchdeviceId)
        return energy(V2).detach().cpu().numpy().flatten()
    def dfunopt(V2):
        V2 = torch.from_numpy(V2.reshape(N,3)).to(dtype=torchdtype, device=torchdeviceId)
        [GV2] = gradE(V2)
        
        L=getLaplacian(V2,F_init)
        GV2=L(GV2)
        
        GV2 = GV2.detach().cpu().numpy().flatten().astype('float64')
        return GV2
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, (2*(V1-V0)+V0).cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-15, epsilon=1e-15, maxiter=200000, disp =1, maxls=20,maxfun=1500000000)
    V2=xopt.reshape(N,3)
    return V2,fopt