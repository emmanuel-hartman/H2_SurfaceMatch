import torch
import numpy as np
from enr.DDG import *
from enr.varifold import *
from enr.regularizers import *
from torch.autograd import grad

use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

##############################################################################################################################
#H2 Helper Functions
##############################################################################################################################


def getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol,stepwise=False):
    b1=(b1-a1)/8
    N=geod.shape[0]        
    diff=(geod[1:,:,:] - geod[:-1,:,:])    
    midpoints=geod[0:N-1,:,:]+diff/2
    #diff=diff*N
    enr=0
    step_enr=torch.zeros((N-1,1),dtype=torchdtype)
    alpha0=getMeshOneForms(geod[0],F_sol)
    g0=getSurfMetric(geod[0],F_sol)
    n0=getNormal(F_sol, geod[0])    
    for i in range(0,N-1):   
        dv=diff[i]        
        if a2>0 or a0>0:
            M=getVertAreas(geod[i],F_sol)
        if a2>0:
            L=getLaplacian(midpoints[i],F_sol) 
            L=L(dv)
            NL=batchDot(L,L)/M            
            enr+=a2*torch.sum(NL)*N
        if a1>0 or b1>0 or c1>0 or d1>0:
            alpha1=getMeshOneForms(geod[i+1],F_sol)
            g1=getSurfMetric(geod[i+1],F_sol)
            n1=getNormal(F_sol, geod[i+1])      
            xi=alpha1-alpha0
            dg=(g1-g0)
            dn=(n1-n0)
            enr+=getGabNorm(getMeshOneForms(midpoints[i],F_sol),xi,getSurfMetric(midpoints[i],F_sol),dg,dn,a1,b1,c1,d1)*N
            g0=g1
            n0=n1
            alpha0=alpha1
        if a0>0:
            Ndv=M*batchDot(dv,dv)
            enr+=a0*torch.sum(Ndv)*N          
        if stepwise:
            if i==0:
                step_enr[0]=enr
            else:
                step_enr[i]=enr-torch.sum(step_enr[0:i])
    if stepwise:
        return enr,step_enr    
    return enr

def getH2Norm(M,dv,a0,a1,b1,c1,d1,a2,F_sol):    
    b1=(b1-a1)/8
    M1=M+dv
    enr=0
    if a2>0 or a0>0:
        A=getVertAreas(M,F_sol)
    if a2>0:      
        L=getLaplacian(M,F_sol) 
        L=L(dv)
        NL=batchDot(L,L)/A
        enr+=a2*torch.sum(NL)
    if a1>0 or b1>0 or c1>0:
        alpha0=getMeshOneForms(M,F_sol)
        g0=getSurfMetric(M,F_sol)
        n0=getNormal(F_sol, M)
        alpha1=getMeshOneForms(M1,F_sol)
        g1=getSurfMetric(M1,F_sol)
        n1=getNormal(F_sol, M1)
        xi=alpha1-alpha0
        dg=(g1-g0)
        dn=(n1-n0)
        enr+=getGabNorm(alpha0,xi,g0,dg,dn,a1,b1,c1,d1)
    if a0>0:
        Ndv=A*batchDot(dv,dv)
        enr+=a0*torch.sum(Ndv)
    return enr


def getH2Metric(M,dv1,dv2,a0,a1,b1,c1,d1,a2,F_sol):    
    b1=(b1-a1)/8
    M1=M+dv1   
    M2=M+dv2   
    enr=0
    if a2>0 or a0>0:
        A=getVertAreas(M,F_sol)
    if a2>0:      
        L=getLaplacian(M,F_sol) 
        NL=batchDot(L(dv1),L(dv2))/A
        enr+=a2*torch.sum(NL)
    if a1>0 or b1>0 or c1>0:
        
        alpha0=getMeshOneForms(M,F_sol)
        g0=getSurfMetric(M,F_sol)
        n0=getNormal(F_sol, M)  
        
        
        alpha1=getMeshOneForms(M1,F_sol)
        g1=getSurfMetric(M1,F_sol)
        n1=getNormal(F_sol, M1)      
        dg1=(g1-g0)
        dn1=(n1-n0)
        xi1=alpha1-alpha0
        
        
        alpha2=getMeshOneForms(M2,F_sol)
        g2=getSurfMetric(M2,F_sol)
        n2=getNormal(F_sol, M2)      
        dg2=(g2-g0)
        dn2=(n2-n0)
        xi2=alpha2-alpha0
        
        enr+=getGabMetric(alpha0,xi1,xi2,g0,dg1,dg2,dn1,dn2,a1,b1,c1,d1)
    if a0>0:
        Ndv=A*batchDot(dv1,dv2)
        enr+=a0*torch.sum(Ndv)
    return enr

def getGabNorm(alpha,xi,g,dg,dn,a,b,c,d):
    n=g.shape[0]
    areas=torch.sqrt(torch.det(g)).to(dtype=torchdtype, device=torchdeviceId)
    ginv=torch.inverse(g)
    ginvdg=torch.matmul(ginv,dg)    
    A=0
    B=0
    C=0
    D=0
    if a>0:
        afunc = torch.einsum('bii->b', torch.matmul(ginvdg,ginvdg))
        A=a*torch.sum(afunc*areas)
    if b>0:        
        bfunc = torch.einsum('bii->b', ginvdg)
        bfunc = bfunc*bfunc
        B=b*torch.sum(bfunc*areas)
    if c>0:        
        cfunc = torch.einsum('bi,bi->b', dn,dn)
        C=c*torch.sum(cfunc*areas)    
    if d>0:
        xi_0=torch.matmul(torch.matmul(alpha,ginv),torch.matmul(xi.transpose(1,2),alpha)-torch.matmul(alpha.transpose(1,2),xi))
        dfunc = torch.einsum('bii->b',torch.matmul(xi_0,torch.matmul(ginv,xi_0.transpose(1,2))))      
        D=d*torch.sum(dfunc*areas)    
    return (A+B+C+D)    
        
    
def getGabMetric(alpha,xi1,xi2,g,dg1,dg2,dn1,dn2,a,b,c,d):
    n=g.shape[0]
    areas=torch.sqrt(torch.det(g)).to(dtype=torchdtype, device=torchdeviceId)
    ginv=torch.inverse(g)
    ginvdg1=torch.matmul(ginv,dg1) 
    ginvdg2=torch.matmul(ginv,dg2)    
    A=0
    B=0
    C=0
    D=0
    if a>0:
        afunc = torch.einsum('bii->b', torch.matmul(ginvdg1,ginvdg2))
        A=a*torch.sum(afunc*areas)
    if b>0:        
        bfunc1 = torch.einsum('bii->b', ginvdg1)
        bfunc2 = torch.einsum('bii->b', ginvdg2)
        bfunc = bfunc1*bfunc2
        B=b*torch.sum(bfunc*areas)
    if c>0:        
        cfunc = torch.einsum('bi,bi->b', dn1,dn2)
        C=c*torch.sum(cfunc*areas)        
    if d>0:
        xi1_0=torch.matmul(torch.matmul(alpha,ginv),torch.matmul(xi1.transpose(1,2),alpha)-torch.matmul(alpha.transpose(1,2),xi1))
        xi2_0=torch.matmul(torch.matmul(alpha,ginv),torch.matmul(xi2.transpose(1,2),alpha)-torch.matmul(alpha.transpose(1,2),xi2))
        dfunc = torch.einsum('bii->b',torch.matmul(xi1_0,torch.matmul(ginv,xi2_0.transpose(1,2))))      
        D=d*torch.sum(dfunc*areas) 
    return (A+B+C+D)    

##############################################################################################################################
#H2_Matching_Energies
##############################################################################################################################

def enr_match_H2_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod):
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1])
        return E
    return energy

def enr_match_H2(VS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_T=1, weight_Gab=1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)   
    def energy(geod):
        geod=torch.cat((torch.unsqueeze(VS,dim=0),geod),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_T*dataloss_T(geod[N-1])
        return E
    return energy



def enr_match_H2_sym_w(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, Rho, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_Gab = 1,a0=1,a1=1,b1=1,c1=1,d1=1,a2=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol,VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf_Weighted(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(geod,Rho):
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)
        N=geod.shape[0]    
        E=weight_Gab*enr + weight_coef_dist_S*dataloss_S(geod[0]) + weight_coef_dist_T*dataloss_T(geod[N-1],Rho)#torch.clamp(Rho,-.25,1.25)+.01*penalty(geod[N-1],F_sol, Rho)
        return E
    return energy


def enr_match_weight(VT,FT,FunT,V_sol,F_sol, Fun_sol, B_sol,weight_coef_dist_T=1, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_T = lossVarifoldSurf_Weighted(F_sol, Fun_sol, VT, FT, FunT, K)
    
    
    def energy(Rho):
        E=weight_coef_dist_T*dataloss_T(V_sol,Rho)#torch.clamp(Rho,-.25,1.25)
        return E
    return energy



def enr_param_H2(left,right, F_sol,a0,a1,b1,c1,d1,a2):    
    def energy(mid):
        geod=torch.cat((left, mid,right),dim=0).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
        enr=getPathEnergyH2(geod,a0,a1,b1,c1,d1,a2,F_sol)        
        return enr 
    return energy

def enr_param_H2Kmean(samples,F_sol,a0,a1,b1,c1,d1,a2):   
    N=samples.shape[0]
    n=samples.shape[1]
    def energy(mu,mid):
        geods=torch.cat((mu.repeat((N,1,1,1)), mid,samples.unsqueeze(dim=1)),dim=1).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)        
        enr=0
        for i in range(0,N):
            enr+=getPathEnergyH2(geods[i],a0,a1,b1,c1,d1,a2,F_sol)        
        return enr 
    return energy

def enr_unparam_H2Kmean(samples,Sample_Funs,F_Sol,Fun_Sol,a0,a1,b1,c1,d1,a2,weight_coef_dist_T=1, **objfun):   
    N=len(samples)
    K=VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    
    
    dataloss_Ts=[lossVarifoldSurf(F_Sol, Fun_Sol, sample[0],sample[1], Sample_Funs[ind], K) for ind,sample in enumerate(samples)]
    
    
    def energy(mu,mid):
        geods=torch.cat((mu.repeat((N,1,1,1)), mid),dim=1).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)     
        enr=0
        n=geods.shape[1]
        for i in range(0,N):
            enr+=getPathEnergyH2(geods[i],a0,a1,b1,c1,d1,a2,F_Sol)+weight_coef_dist_T*dataloss_Ts[i](geods[i,n-1])  
        return enr 
    return energy

def enr_step_forward(M0,M1,a0,a1,b1,c1,d1,a2,F):
    def energy(M2):
        M1dot=M1-M0
        M2dot=M2-M1
        qM1 = M1.clone().requires_grad_(True)
        sys=2*getFlatMap(M0,M1-M0,F,a0,a1,b1,c1,d1,a2)-2*getFlatMap(M1,M2-M1,F,a0,a1,b1,c1,d1,a2)+grad(getH2Norm(qM1,M2dot,a0,a1,b1,c1,d1,a2,F), qM1, create_graph=True)[0]
        return (sys**2).sum()
    return energy
    
def getFlatMap(M,V,F,a0,a1,b1,c1,d1,a2):
    B=torch.zeros((M.shape[0],3)).to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    return grad(getH2Metric(M,V,B,a0,a1,b1,c1,d1,a2,F), B, create_graph=True)[0] 

