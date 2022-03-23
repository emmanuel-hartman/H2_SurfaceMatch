# Load Packages
from enr.H2 import *
import numpy as np
from numpy import cov,linalg,mean
from H2_ivp import H2InitialValueProblem
from H2_match import H2StandardIterative
from H2_param import H2Parameterized
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from enr.DDG import computeBoundary
from torch.autograd import grad
import utils.utils as io
import torch
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32


def H2KMean(samples,F0,a0,a1,b1,c1,d1,a2,paramlist):
    N=len(samples)
    samples=torch.from_numpy(samples).to(dtype=torchdtype, device=torchdeviceId)
    mu = torch.mean(samples, axis=0)
    geods=torch.cat((mu.repeat((N,1,1,1)),samples.unsqueeze(dim=1)),dim=1).cpu().numpy()
    print(geods.shape)
    for param in paramlist:
        newk= param['time_steps']
        geods,mu,F0 =KMeanIteration(mu,geods,samples,F0,newk,a0,a1,b1,c1,d1,a2,param)
        print(geods.shape)
    return geods,mu,F0

def KMeanIteration(mu,geods,samples,F,newk,a0,a1,b1,c1,d1,a2,param):
    max_iter=param['max_iter']     
    [N,k,n,dum]=geods.shape
    
    xp=np.linspace(0,1,k,endpoint=True)
    x=np.linspace(0,1,newk,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,geods,axis=1)
    midpoints=f(x)
    
    midpoint = torch.from_numpy(midpoints[:,1:newk-1,:,:]).to(dtype=torchdtype, device=torchdeviceId)
    F_sol= torch.from_numpy(F).to(dtype=torch.long, device=torchdeviceId) 
    
    # Define Energy and set parameters
    energy = enr_param_H2Kmean(samples, F_sol,a0,a1,b1,c1,d1,a2)
    
    def gradE(mu,midpoint):
        qmidpoint = midpoint.clone().requires_grad_(True)
        qmu = mu.clone().requires_grad_(True)
        return grad(energy(qmu,qmidpoint), [qmu,qmidpoint], create_graph=True)
    

    def funopt(midpoint):
        mu=torch.from_numpy(midpoint[0:3*n]).to(dtype=torchdtype, device=torchdeviceId).reshape(n,3)
        midpoint=torch.from_numpy(midpoint[3*n:].reshape(N,newk-2,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(mu,midpoint).detach().cpu().numpy())

    def dfunopt(midpoint):
        mu=torch.from_numpy(midpoint[0:3*n]).to(dtype=torchdtype, device=torchdeviceId).reshape(n,3)
        midpoint=torch.from_numpy(midpoint[3*n:].reshape(N,newk-2,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        [Gmidpoint,Gmu] = gradE(mu,midpoint)
        Gmidpoint = np.concatenate((Gmidpoint.detach().cpu().numpy().flatten().astype('float64'),Gmu.detach().cpu().numpy().flatten().astype('float64')))
        return Gmidpoint

    
    inp_vect=np.concatenate((mu.cpu().numpy().flatten(),midpoint.cpu().numpy().flatten()))
    
    out,fopt,Dic=fmin_l_bfgs_b(funopt, inp_vect, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    mu=out[0:n*3].reshape(n,3)
    ngeods=np.concatenate((np.tile(mu,(N,1,1,1)),out[n*3:].reshape(N,newk-2,n,3),np.expand_dims(samples.cpu().numpy(),1)), axis=1)
    geods= np.array(ngeods)
    return geods,torch.from_numpy(mu).to(dtype=torchdtype, device=torchdeviceId),F



def H2UnparamKMean(samples,template,a0,a1,b1,c1,d1,a2,paramlist, N=None, geodesics=False):
    if N is None:
        N=len(samples)
        
    mu= template[0]
    F0= template[1]
    k=2
    for i in range(0,N):
        sample=samples[np.random.randint(0,len(samples))]
        print('Computing Step {}/{}'.format(i+1,N))
        geod,F0=H2StandardIterative([mu,F0],sample,a0,a1,b1,c1,d1,a2,paramlist)
        k=geod.shape[0]
        xp=np.linspace(0,1,geod.shape[0],endpoint=True)
        x=np.linspace(0,1,i+2,endpoint=True)    
        f=scipy.interpolate.interp1d(xp,geod,axis=0)
        midpoints=f(x)
        print(midpoints.shape)
        mu=midpoints[1]
        print(mu.shape)  
        
    if geodesics:
        geods=np.zeros((len(samples),k,mu.shape[0],3)) 
        for i in range(0,len(samples)):
            print('Computing geodesic {}/{}'.format(i+1,len(samples)))
            geod,F0=H2StandardIterative([mu,F0],[samples[i][0],samples[i][1]],a0,a1,b1,c1,d1,a2,paramlist)
            geods[i]=geod
        return geods,mu,F0
    return mu,F0



def H2_UnparamPCA(V0,samples,F0,a0,a1,b1,c1,d1,a2,paramlist,components=1,tol=None, geods=None):
    N=len(samples)        
    T=np.zeros((len(samples), V0.shape[0]*3))
    if geods is None:
        for i in range(0,len(samples)):
            print('Computing Tangent Vector in the direction of sample {}/{}'.format(i+1,N))
            geod,F0=H2StandardIterative([V0,F0],[samples[i][0],samples[i][1]],a0,a1,b1,c1,d1,a2,paramlist)
            k=geod.shape[0]
            T[i,:]=(k-1)*(geod[1]-geod[0]).flatten()
    else:
        for i in range(0,len(samples)):
            geod=geods[i]
            k=geod.shape[0]
            T[i,:]=(k-1)*(geod[1]-geod[0]).flatten()
        samples=geods[N-1]
    M = (T-mean(T.T,axis=1)).T
    [evalue,evector] = linalg.eig(cov(M))
    perm=(-1*evalue).argsort()
    evalue=np.real(evalue)[perm]
    evector=np.real(evector.T).reshape((-1,V0.shape[0],3))[perm]
    
    print(evalue/evalue.sum())
    
    PCs=[]
    if tol is not None:
        for i in range(0,coeff.shape[0]):
            if evalue[i]>tol:
                PC1p,F=H2InitialValueProblem(V0,evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
                PC1n,F=H2InitialValueProblem(V0,-evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
                PC1=np.concatenate((np.flip(PC1n,axis=0),PC1p[1:]),axis=0)
                PCs+=[PC1]
            return evalue,evector,PCs
    else:
        for i in range(0,components):
            PC1p,F=H2InitialValueProblem(V0,evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
            PC1n,F=H2InitialValueProblem(V0,-evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
            PC1=np.concatenate((np.flip(PC1n,axis=0),PC1p[1:]),axis=0)
            PCs+=[PC1]
        return evalue,evector,PCs
    
def H2PCA(V0,samples,F0,a0,a1,b1,c1,d1,a2,paramlist,components=1,tol=None, geods=None):
    N=len(samples)        
    T=np.zeros((len(samples), V0.shape[0]*3))
    if geods is None:
        for i in range(0,len(samples)):
            print('Computing Tangent Vector in the direction of sample {}/{}'.format(i+1,N))
            geod,F0=H2Parameterized([V0,F0],[samples[i],F0],a0,a1,b1,c1,d1,a2,paramlist)
            k=geod.shape[0]
            T[i,:]=(k-1)*(geod[1]-geod[0]).flatten()
    else:
        for i in range(0,len(samples)):
            geod=geods[i]
            k=geod.shape[0]
            T[i,:]=(k-1)*(geod[1]-geod[0]).flatten()
        samples=geods[N-1]
    M = (T-mean(T.T,axis=1)).T
    [evalue,evector] = linalg.eig(cov(M))
    perm=(-1*evalue).argsort()
    evalue=np.real(evalue)[perm]
    evector=np.real(evector.T).reshape((-1,V0.shape[0],3))[perm]
    
    
    PCs=[]
    if tol is not None:
        for i in range(0,coeff.shape[0]):
            if evalue[i]>tol:
                PC1p,F=H2InitialValueProblem(V0,evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
                PC1n,F=H2InitialValueProblem(V0,-evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
                PC1=np.concatenate((np.flip(PC1n,axis=0),PC1p[1:]),axis=0)
                PCs+=[PC1]
            return evalue,evector,PCs
    else:
        for i in range(0,components):
            PC1p,F=H2InitialValueProblem(V0,evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
            PC1n,F=H2InitialValueProblem(V0,-evalue[i]*evector[i],k,a0,a1,b1,c1,d1,a2,F0)
            PC1=np.concatenate((np.flip(PC1n,axis=0),PC1p[1:]),axis=0)
            PCs+=[PC1]
        return evalue,evector,PCs