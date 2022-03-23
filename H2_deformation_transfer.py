# Load Packages
from enr.H2 import *
from H2_ivp import stepforward, H2InitialValueProblem
from H2_param import H2Parameterized
from H2_match import H2StandardIterative
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

def H2ParamDeformationTransfer(VS, VT, VNS, F,a0,a1,b1,c1,d1,a2, geod1_params, geod2_params):
    print("Calculating Deformation Geodesic")
    geod1,F0=H2Parameterized([VS,F],[VT,F],a0,a1,b1,c1,d1,a2,geod1_params)
    print("Calculating Transfer Geodesic")
    geod2,F0=H2Parameterized([VS,F],[VNS,F],a0,a1,b1,c1,d1,a2,geod2_params)
    
    print("Performing Parallel Transport Using Schild's Ladder")
    F_init=torch.from_numpy(F).to(dtype=torch.long, device=torchdeviceId)
    X0=torch.from_numpy(geod1[1]).to(dtype=torchdtype, device=torchdeviceId)
    for i in range(0,geod2.shape[0]-1):
        A0=torch.from_numpy(geod2[i]).to(dtype=torchdtype, device=torchdeviceId)
        A1=torch.from_numpy(geod2[i+1]).to(dtype=torchdtype, device=torchdeviceId)
        P1=(A1+X0)/2
        X1,cost1=stepforward(A0,P1,a0,a1,b1,c1,d1,a2,F_init)
        X0=torch.from_numpy(X1).to(dtype=torchdtype, device=torchdeviceId)
    
    N=geod1.shape[0]
    print("Calculating Transfered Deformation")
    Ngeod1,F_init=H2InitialValueProblem(VNS,(N-1)*(X0.cpu().numpy()-VNS),N,a0,a1,b1,c1,d1,a2,F0)
    
    return geod1,geod2,Ngeod1,F

def H2UnparamDeformationTransfer(source, target, new_source, a0,a1,b1,c1,d1,a2, geod1_params,geod2_params, parameterizedDeform=False, parameterizedTransfer=False, match_newsource=False):
    print("Calculating Deformation Geodesic")
    if(parameterizedDeform):
        geod1,F0=H2Parameterized(source,target,a0,a1,b1,c1,d1,a2,geod1_params)
    else:
        geod1,F0=H2StandardIterative(source,target,a0,a1,b1,c1,d1,a2,geod1_params)
        
    print("Calculating Transfer Geodesic")
    if(parameterizedTransfer):
        geod2,F0=H2Parameterized(source,new_source,a0,a1,b1,c1,d1,a2,geod2_params)
    else:
        geod2,F0=H2StandardIterative(source,new_source,a0,a1,b1,c1,d1,a2,geod2_params)
    
    print("Performing Parallel Transport Using Schild's Ladder")
    F_init=torch.from_numpy(F0).to(dtype=torch.long, device=torchdeviceId)
    X0=torch.from_numpy(geod1[1]).to(dtype=torchdtype, device=torchdeviceId)
    for i in range(0,geod2.shape[0]-1):
        A0=torch.from_numpy(geod2[i]).to(dtype=torchdtype, device=torchdeviceId)
        A1=torch.from_numpy(geod2[i+1]).to(dtype=torchdtype, device=torchdeviceId)
        P1=(A1+X0)/2
        X1,cost1=stepforward(A0,P1,a0,a1,b1,c1,d1,a2,F_init)
        X0=torch.from_numpy(X1).to(dtype=torchdtype, device=torchdeviceId)
    
    N=geod1.shape[0]
    N1=geod2.shape[0]
    print("Calculating Transfered Deformation")
    Ngeod1,F_1=H2InitialValueProblem(geod2[N1-1], N*(X0.cpu().numpy()-geod2[N1-1]),N,a0,a1,b1,c1,d1,a2,F0)    
    
    if match_newsource:
        print("Calculating Transfered Deformation on New Source mesh structure")
        Ngeod2,F_2=H2StandardIterative(new_source,[Ngeod1[N-1],F_1.cpu().numpy()],a0,a1,b1,c1,d1,a2,geod1_params)
        return geod1,geod2,Ngeod1,Ngeod2,F_1.cpu().numpy(),F_2
    
    return geod1,geod2,Ngeod1,F_1.cpu().numpy()