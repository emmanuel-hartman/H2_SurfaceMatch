# Load Packages
from enr.H2 import *
from H2_param import H2Midpoint
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




def SymmetricMatching_coeff(a0,a1,b1,c1,d1,a2,param, source, target, chemin, faces, basis):
    sig_geom = param['sig_geom']

    if ('sig_grass' not in param):
        sig_grass = 1
    else:
        sig_grass = param['sig_grass']

    if ('kernel_geom' not in param):
        kernel_geom = 'gaussian'
    else:
        kernel_geom = param['kernel_geom']

    if ('kernel_grass' not in param):
        kernel_grass = 'binet'
    else:
        kernel_grass = param['kernel_grass']

    if ('kernel_fun' not in param):
        kernel_fun = 'constant'
    else:
        kernel_fun = param['kernel_fun']

    if 'sig_fun' not in param:
        sig_fun = 1
    else:
        sig_fun = param['sig_fun']

    weight_coef_dist = param['weight_coef_dist']
    max_iter = param['max_iter']
    
    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    F_sol = faces

    N = chemin.shape[0]
    n = chemin.shape[1]

    FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    Fun_sol = torch.from_numpy(np.zeros((int(chemin.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)

    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)

    #chemin = torch.from_numpy(chemin).to(dtype=torchdtype, device=torchdeviceId)
    chemin = chemin.to(dtype=torchdtype, device=torchdeviceId)
    # Define Energy and set parameters

    energy = enr_match_H2_sym_coeff(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, geod=chemin, basis=basis, weight_coef_dist_T=weight_coef_dist, weight_coef_dist_S=weight_coef_dist, kernel_geom=kernel_geom, kernel_grass=kernel_grass, kernel_fun=kernel_fun, sig_geom=sig_geom, sig_grass=sig_grass, sig_fun=sig_fun, a0=a0, a1=a1, b1=b1, c1=c1, d1=d1, a2=a2)
    
    tm = chemin.shape[0]

    def gradE(X):
        qX = X.clone().requires_grad_(True)
        return grad(energy(qX), qX, create_graph=True)
    

    def funopt(X):
        X=torch.from_numpy(X.reshape(tm,-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(X).detach().cpu().numpy())

    def dfunopt(X):
        X = torch.from_numpy(X.reshape(tm,-1)).to(dtype=torchdtype, device=torchdeviceId)
        [GX] = gradE(X)
        GX = GX.detach().cpu().numpy().flatten().astype('float64')
        return GX
    
    
    X0 = np.zeros((basis.shape[0]*tm))
    xopt, fopt, Dic = fmin_l_bfgs_b(funopt, X0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08,
                                    maxiter=max_iter, iprint=1, maxls=20, maxfun=150000,factr=weight_coef_dist)
    X_t = torch.from_numpy(xopt.reshape((tm, -1))).to(dtype=torchdtype, device=torchdeviceId)
    chemin_exp = chemin + torch.einsum("ij, jkl-> ikl", X_t, basis)
    return chemin_exp, X_t,fopt, Dic

def StandardMatching_coeff(a0,a1,b1,c1,d1,a2,param, source, target, chemin, faces, basis):    
    sig_geom=param['sig_geom']
    
    if ('sig_grass'not in param):
        sig_grass = 1
    else:
        sig_grass=param['sig_grass']
    
    if ('kernel_geom' not in param):
        kernel_geom = 'gaussian'
    else:
        kernel_geom = param['kernel_geom']
        
    if ('kernel_grass' not in param):
        kernel_grass = 'binet'
    else:
        kernel_grass = param['kernel_grass']
    
    if ('kernel_fun' not in param):
        kernel_fun = 'constant'
    else:
        kernel_fun = param['kernel_fun']
        
    if 'sig_fun' not in param:
        sig_fun = 1
    else:
        sig_fun=param['sig_fun']
        
    weight_coef_dist_T=param['weight_coef_dist']
    max_iter=param['max_iter']
    
    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    F_sol = faces

    N = chemin.shape[0]
    n = chemin.shape[1]

    FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    Fun_sol = torch.from_numpy(np.zeros((int(chemin.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)

    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)

    #chemin = torch.from_numpy(chemin).to(dtype=torchdtype, device=torchdeviceId)
    chemin = chemin.to(dtype=torchdtype, device=torchdeviceId)
    # Define Energy and set parameters

    energy = enr_match_H2_coeff(VS, VT, FT, FunT, F_sol, Fun_sol, geod=chemin, basis=basis, weight_coef_dist_T=weight_coef_dist_T, kernel_geom=kernel_geom, kernel_grass=kernel_grass, kernel_fun=kernel_fun, sig_geom=sig_geom, sig_grass=sig_grass, sig_fun=sig_fun, a0=a0, a1=a1, b1=b1, c1=c1, d1=d1, a2=a2)
    
    tm = chemin.shape[0]

    def gradE(X):
        qX = X.clone().requires_grad_(True)
        return grad(energy(qX), qX, create_graph=True)
    

    def funopt(X):
        X=torch.from_numpy(X.reshape((tm-1),-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(X).detach().cpu().numpy())

    def dfunopt(X):
        X = torch.from_numpy(X.reshape((tm-1),-1)).to(dtype=torchdtype, device=torchdeviceId)
        [GX] = gradE(X)
        GX = GX.detach().cpu().numpy().flatten().astype('float64')
        return GX
    
    
    X0 = np.zeros((basis.shape[0]*(tm-1)))
    
    xopt, fopt, Dic = fmin_l_bfgs_b(funopt, X0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08,
                                    maxiter=max_iter, iprint=1, maxls=20, maxfun=150000,factr=weight_coef_dist)
    X_t = torch.from_numpy(xopt.reshape(((tm-1), -1))).to(dtype=torchdtype, device=torchdeviceId)
    X_t=torch.cat((torch.unsqueeze(torch.zeros((X_t.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),X_t),dim=0)
    chemin_exp = chemin + torch.einsum("ij, jkl-> ikl", X_t, basis)
    return chemin_exp, X_t, fopt, Dic


def H2Midpoint_coeff(geod,faces,newN,a0,a1,b1,c1,d1,a2,param, basis):   
    max_iter=param['max_iter']    
    N=geod.shape[0]

    # Convert Data to Pytorch
    
    if torch.is_tensor(geod):
        geod=geod.cpu().numpy()
    
    xp=np.linspace(0,1,N,endpoint=True)
    x=np.linspace(0,1,newN,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,geod,axis=0)
    geod=f(x)
    
    geod=torch.from_numpy(geod).to(dtype=torchdtype, device=torchdeviceId)
    if not torch.is_tensor(faces):
        F_sol= torch.from_numpy(faces).to(dtype=torch.long, device=torchdeviceId)    
    else:
        F_sol=faces
    n=geod.shape[1]
    
    energy = enr_param_H2_coeff(F_sol,geod,basis,a0,a1,b1,c1,d1,a2)
    
    tm = geod.shape[0]
    
    def gradE(X):
        qX = X.clone().requires_grad_(True)
        return grad(energy(qX), qX, create_graph=True)

    def funopt(X):
        X=torch.from_numpy(X.reshape(tm-2,-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(X).detach().cpu().numpy())

    def dfunopt(X):
        X = torch.from_numpy(X.reshape(tm-2,-1)).to(dtype=torchdtype, device=torchdeviceId)
        [GX] = gradE(X)
        GX = GX.detach().cpu().numpy().flatten().astype('float64')
        return GX

    X0 = np.zeros((basis.shape[0]*(tm-2)))

    out,fopt,Dic=fmin_l_bfgs_b(funopt, X0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000,factr=1e5)
    out=torch.from_numpy(out.reshape(((tm-2), -1))).to(dtype=torchdtype, device=torchdeviceId)
    out=torch.cat((torch.unsqueeze(torch.zeros((out.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),out,torch.unsqueeze(torch.zeros((out.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0)),dim=0)
    geod = geod + torch.einsum("ij, jkl-> ikl", out, basis)
    return geod, out


def H2Parameterized_coeff(source,target,a0,a1,b1,c1,d1,a2,paramlist,basis):
    F0=source[1]
    geod=np.array([source[0],target[0]])
    for param in paramlist:
        newN= param['time_steps']
        geod,X=H2Midpoint_coeff(geod,F0,newN,a0,a1,b1,c1,d1,a2,param, basis)
        print(geod.shape)
    return geod,X, F0


def H2MultiRes_sym_coeff(a0,a1,b1,c1,d1,a2,paramlist, source, target, chemin, faces, basis):

    total_X= torch.zeros((chemin.shape[0],basis.shape[0])).to(dtype=torchdtype, device=torchdeviceId)
    
    iterations=len(paramlist)
    for j in range(0,iterations):
        params=paramlist[j]
        time_steps= params['time_steps']
        [N,n,three]=chemin.shape
        chemin,X,ener,Dic=SymmetricMatching_coeff(a0,a1,b1,c1,d1,a2,params, source, target, chemin, faces, basis)        
        total_X+=X
        if time_steps>2:
            xp=np.linspace(0,1,total_X.shape[0],endpoint=True)
            x=np.linspace(0,1,time_steps,endpoint=True)    
            f=scipy.interpolate.interp1d(xp,total_X.cpu().numpy(),axis=0)
            total_X=f(x)
            total_X=torch.from_numpy(total_X).to(dtype=torchdtype, device=torchdeviceId)
            
            chemin,X=H2Midpoint_coeff(chemin,faces,time_steps,a0,a1,b1,c1,d1,a2,params, basis)           
            total_X+=X      
    return chemin, total_X


def H2MultiRes_coeff(a0,a1,b1,c1,d1,a2,paramlist, source, target, chemin, faces, basis):
    
    total_X= torch.zeros((chemin.shape[0],basis.shape[0])).to(dtype=torchdtype, device=torchdeviceId)
    
    iterations=len(paramlist)
    for j in range(0,iterations):
        params=paramlist[j]
        time_steps= params['time_steps']
        [N,n,three]=chemin.shape
        chemin,X,ener,Dic=StandardMatching_coeff(a0,a1,b1,c1,d1,a2,params, source, target, chemin, faces, basis)        
        total_X+=X
        if time_steps>2:
                
            xp=np.linspace(0,1,total_X.shape[0],endpoint=True)
            x=np.linspace(0,1,time_steps,endpoint=True)    
            f=scipy.interpolate.interp1d(xp,total_X.cpu().numpy(),axis=0)
            total_X=f(x)
            total_X=torch.from_numpy(total_X).to(dtype=torchdtype, device=torchdeviceId)
            
            chemin,X=H2Midpoint_coeff(chemin,faces,time_steps,a0,a1,b1,c1,d1,a2,params, basis)           
            total_X+=X
                      
    return chemin,total_X
