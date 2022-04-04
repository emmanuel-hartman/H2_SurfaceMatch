# Load Packages
from enr.H2 import *
from SRNF_match import computeBoundary
from H2_match import SymmetricH2Matching,H2Midpoint
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from torch.autograd import grad
import utils.utils as io
import torch
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32


def SymmetricH2Matching_W(source,target,geod,F_init,Rho_init,a0,a1,b1,c1,d1,a2,param):    
    # Read out parameters
    sig_geom=param['sig_geom']
    
    if ('sig_grass'not in param):
        sig_grass = 1
    else:
        sig_grass=param['sig_grass']
    
    weight_coef_dist_S=param['weight_coef_dist_S']
    weight_coef_dist_T=param['weight_coef_dist_T']
    max_iter=param['max_iter']
    
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

    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    geod = torch.from_numpy(geod).to(dtype=torchdtype, device=torchdeviceId)
    F_sol= torch.from_numpy(F_init).to(dtype=torch.long, device=torchdeviceId)
    Rho_sol= torch.from_numpy(Rho_init).to(dtype=torchdtype, device=torchdeviceId)
    B_sol = torch.from_numpy(computeBoundary(F_init)).to(dtype=torch.bool, device=torchdeviceId)
    
    N=geod.shape[0]
    n=geod.shape[1]

    FunS = torch.from_numpy(np.ones((int(np.size(source[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    FunT = torch.from_numpy(np.ones((int(np.size(target[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    
    Fun_sol = torch.from_numpy(np.ones((int(geod.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)


    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)


    # Define Energy and set parameters
    energy = enr_match_H2_sym_w(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol,Rho_sol, B_sol, weight_coef_dist_S=weight_coef_dist_S ,weight_coef_dist_T=weight_coef_dist_T, kernel_geom = kernel_geom, kernel_grass = kernel_grass, kernel_fun = kernel_fun, sig_geom = sig_geom, sig_grass = sig_grass, sig_fun = sig_fun,a0=a0,a1=a1,b1=b1,c1=c1,d1=d1,a2=a2)
    
    
    


    def gradE(geod,Rho):
        qgeod = geod.clone().requires_grad_(True)
        qRho = Rho.clone().requires_grad_(True)
        return grad(energy(qgeod,qRho), [qgeod,qRho], create_graph=True)
    

    def funopt(geod):
        ngeod=geod[0:n*N*3]
        Rho=geod[n*N*3:]
        
        geod=torch.from_numpy(ngeod.reshape(N,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        Rho=torch.from_numpy(Rho).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(geod,Rho).detach().cpu().numpy())

    def dfunopt(geod):
        ngeod=geod[0:n*N*3]
        Rho=geod[n*N*3:]
        
        geod=torch.from_numpy(ngeod.reshape(N,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        Rho=torch.from_numpy(Rho).to(dtype=torchdtype, device=torchdeviceId)
        [Ggeod,GRho] = gradE(geod,Rho)
        Ggeod = np.concatenate((Ggeod.detach().cpu().numpy().flatten().astype('float64'),GRho.detach().cpu().numpy().flatten().astype('float64')))
        return Ggeod
    
    inp_vect=np.concatenate((geod.cpu().numpy().flatten(),Rho_sol.cpu().numpy().flatten()))
    
    
    bounds=([(-np.inf,np.inf)]*n*N*3).append([(0,1)]*n)
    
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, inp_vect, fprime=dfunopt, bounds=bounds,pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 0, maxls=20,maxfun=150000)
    geod = xopt[0:int(n*N*3)]
    Rho = xopt[int(n*N*3):]
    return geod,Rho,fopt,Dic

def WeightUpdate(source,target,geod,F_init,Rho_init,param):    
    # Read out parameters
    sig_geom=param['sig_geom']
    
    if ('sig_grass'not in param):
        sig_grass = 1
    else:
        sig_grass=param['sig_grass']
    
    weight_coef_dist_S=param['weight_coef_dist_S']
    weight_coef_dist_T=param['weight_coef_dist_T']
    max_iter=param['max_iter']
    
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

    # Convert Data to Pytorch
    N=geod.shape[0]
    n=geod.shape[1]

    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    V_sol = torch.from_numpy(geod[N-1]).to(dtype=torchdtype, device=torchdeviceId)
    F_sol= torch.from_numpy(F_init).to(dtype=torch.long, device=torchdeviceId)
    Rho_sol= torch.from_numpy(Rho_init).to(dtype=torchdtype, device=torchdeviceId)
    B_sol = torch.from_numpy(computeBoundary(F_init)).to(dtype=torch.bool, device=torchdeviceId)
    
    FunS = torch.from_numpy(np.ones((int(np.size(source[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    FunT = torch.from_numpy(np.ones((int(np.size(target[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    
    Fun_sol = torch.from_numpy(np.ones((int(geod.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)


    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)


    # Define Energy and set parameters
    energy = enr_match_weight(VT,FT,FunT,V_sol,F_sol, Fun_sol, B_sol,weight_coef_dist_T=weight_coef_dist_T, kernel_geom = kernel_geom, kernel_grass = kernel_grass, kernel_fun = kernel_fun, sig_geom = sig_geom, sig_grass = sig_grass, sig_fun = sig_fun)

    def gradE(Rho):
        qRho = Rho.clone().requires_grad_(True)
        return grad(energy(qRho), [qRho], create_graph=True)
    

    def funopt(Rho):
        Rho=torch.from_numpy(Rho).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(Rho).detach().cpu().numpy())

    def dfunopt(Rho):
        Rho=torch.from_numpy(Rho).to(dtype=torchdtype, device=torchdeviceId)
        [GRho] = gradE(Rho)
        Ggeod = GRho.detach().cpu().numpy().flatten().astype('float64')
        return Ggeod
    
    inp_vect=Rho_sol.cpu().numpy().flatten()
    
    bounds=([(-np.inf,np.inf)]*n*N*3).append([(0,1)]*n)
    
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, inp_vect, fprime=dfunopt, bounds=bounds ,pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    Rho = xopt
    return Rho



def H2MultiRes(source,target,a0,a1,b1,c1,d1,a2,resolutions,paramlist,start=None, rotate = False,Rho0=None):
    N=2
    [VS,FS]=source
    [VT,FT]=target
    
    print(FS.shape)        
    
    sources = [[VS,FS]]
    targets = [[VT,FT]]
    
    for i in range(0,resolutions):
        [VS,FS]=io.decimate_mesh(VS,FS,int(FS.shape[0]/4))
        sources = [[VS,FS]]+sources
        [VT,FT]=io.decimate_mesh(VT,FT,int(FT.shape[0]/4))
        targets = [[VT,FT]]+targets
        print(FS.shape)
        
    source_init = sources[0]
    target_init = sources[0]
    
    if start is not None:
        geod=np.zeros((N,start[0].shape[1],start[0].shape[2]))
        F0=start[1]
        for i in range(0,N):
            geod[i,:,:]=start[0][0]
    else:
        geod=np.zeros((N,source_init[0].shape[0],source_init[0].shape[1]))
        F0=source_init[1]
        for i in range(0,N):
            geod[i,:,:]=source_init[0]
            
    if Rho0 is None:         
        Rho0 = np.ones((int(np.size(source_init[0])/3),1))

    
    iterations=len(paramlist)
    for j in range(0,iterations):
        params=paramlist[j]
        time_steps= params['time_steps']
        tri_upsample= params['tri_unsample']
        index= params['index']
        if 'partial' not in params:
            partial=True
        else:
            partial=params['partial']
            
        if 'weight_only' not in params:
            weight_only=False
        else:
            weight_only=params['weight_only']
            
        geod=np.array(geod)
        [N,n,three]=geod.shape
        
        if partial:
            geod,Rho0,ener,Dic=SymmetricH2Matching_W(sources[index],targets[index],geod,F0,Rho0,a0,a1,b1,c1,d1,a2,params)
        else:
            geod,ener,Dic=SymmetricH2Matching(sources[index],targets[index],geod,F0,a0,a1,b1,c1,d1,a2,params)
            
        geod=geod.reshape(N,n,3)
        print(F0.shape)

        if time_steps>2:
            geod=H2Midpoint(geod,time_steps,F0,a0,a1,b1,c1,d1,a2,params)
            
        [N,n,three]=geod.shape
        if tri_upsample:
            geod_sub=[]
            F_Sub=[]
            
            geod_subi,F_Subi,RhoSub=io.subdivide_mesh(geod[0],F0,Rho=Rho0,order=1)
            Rho0=RhoSub
                
            F_Sub.append(F_Subi)
            geod_sub.append(geod_subi)
            for i in range(1,N):
                geod_subi,F_Subi=io.subdivide_mesh(geod[i],F0,order=1)
                F_Sub.append(F_Subi)
                geod_sub.append(geod_subi)
            geod=geod_sub
            F0=F_Sub[0]
        
        if weight_only:
            Rho0=WeightUpdate(source,target,np.array(geod),F0,Rho0,params)
        
    print(F0.shape)
    return geod,Rho0,F0