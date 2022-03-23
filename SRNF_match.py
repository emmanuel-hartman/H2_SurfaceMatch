# Load Packages
from enr.SRNF import *
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from torch.autograd import grad
from enr.DDG import computeBoundary
import utils.utils as io
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32


def StandardMatching(source,target,target_init,param):
# Wrapper function that performs the standard (non symmetric) matching algorithm
# Input: source,target...vertices, faces and textures of source and target
#        target_init...vertices, faces and tetures of initial guess
#        param...list with all parameters for optimization and energy functional:
#                  'weight_MCV'...Curvature weight
#                  'weight_coef_dist_T'... Varifold distance to target weight
#                  'kernel_geom'... Type of geom kernel for varifold metric
#                  'sig_geom'...Kernel size for geom kernel
#                  'kernel_grass'... Type of grass kernel for varifold metric
#                  'sig_grass'...Kernel size for grassmannian kernel
#                  'kernel_fun'... Type of functional kernel for varifold metric
#                  'sig_fun'...Kernel size for functional kernel
#                  'max_iter'...Maximum iterations for BFGS method
#
# Output: optimal vertices f0 and f1.
    # Read out parameters
    sig_geom=param['sig_geom']
    sig_grass=param['sig_grass']
    weight_coef_dist_T=param['weight_coef_dist_T']
    weight_MCV=param['weight_MCV']
    max_iter=param['max_iter']
    kernel_geom = param['kernel_geom']
    kernel_grass = param['kernel_grass']

    use_fundata= param['use_fundata']

    if (use_fundata==0) or ('kernel_fun' not in param):
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
    BS = torch.from_numpy(computeBoundary(source[1])).to(dtype=torch.bool, device=torchdeviceId)

    if len(source)<3 or use_fundata==0:
        FunS = torch.from_numpy(np.zeros((int(np.size(source[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    else:
        FunS = torch.from_numpy(source[2]).to(dtype=torchdtype, device=torchdeviceId)

    if len(target)<3 or use_fundata==0:
        FunT = torch.from_numpy(np.zeros((int(np.size(target[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    else:
        FunT = torch.from_numpy(target[2]).to(dtype=torchdtype, device=torchdeviceId)


    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)


    # Define Energy and set parameters
    energy = enr_match_SRNF(VS,FS,FunS,BS,VT,FT,FunT,weight_coef_dist=weight_coef_dist_T, weight_MCV =weight_MCV, kernel_geom = kernel_geom, kernel_grass = kernel_grass, kernel_fun = kernel_fun, sig_geom = sig_geom, sig_grass = sig_grass, sig_fun = sig_fun)


    def gradE(x):
        qt = x.clone().requires_grad_(True)
        Gx = grad(energy(qt), [qt], create_graph=True)
        return Gx

    def funopt(z):
        n = int(len(z)/3)
        x = z.reshape(n,3)
        x = torch.from_numpy(x).to(dtype=torchdtype, device=torchdeviceId)
        E = float(energy(x).detach().cpu().numpy())
        return E

    def dfunopt(z):
        n = int(len(z)/3)
        x = z.reshape(n,3)
        x = torch.from_numpy(x).to(dtype=torchdtype, device=torchdeviceId)
        [Gx] = gradE(x)
        G=Gx.detach().cpu().numpy().flatten().astype('float64')
        return G



    z=target_init[0].flatten() # initialization for the estimated target
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, z, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1,disp =1, maxls=20,maxfun=150000)
    f1 = xopt.reshape(int(len(xopt)/3),3)
    return f1,fopt,Dic



def SymmetricMatching(source,target,source_init,target_init,param):
# Wrapper function that performs the symmetric matching algorithm
# Input: source,target...vertices, faces and textures of source and target
#        source_init,target_init...vertices, faces and textures of initial guess (need to have the
#                                                                       same connectivity struct)
#        param...list with all parameters for optimization and energy functional:
#                  'weight_MCV'...Curvature weight
#                  'weight_coef_dist_S'... Varifold distance to source weight
#                  'weight_coef_dist_T'... Varifold distance to target weight
#                  'kernel_geom'... Type of geom kernel for varifold metric
#                  'sig_geom'...Kernel size for geom kernel
#                  'kernel_grass'... Type of grass kernel for varifold metric
#                  'sig_grass'...Kernel size for grassmannian kernel
#                  'kernel_fun'... Type of functional kernel for varifold metric
#                  'sig_fun'...Kernel size for functional kernel
#                  'max_iter'...Maximum iterations for BFGS method
#
# Output: optimal vertices f0 and f1.

    # Read out parameters
    sig_geom=param['sig_geom']
    sig_grass=param['sig_grass']
    weight_coef_dist_S=param['weight_coef_dist_S']
    weight_coef_dist_T=param['weight_coef_dist_T']
    weight_MCV=param['weight_MCV']
    max_iter=param['max_iter']
    kernel_geom = param['kernel_geom']
    kernel_grass = param['kernel_grass']

    use_fundata= param['use_fundata']

    if (use_fundata==0) or ('kernel_fun' not in param):
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
    F_sol= torch.from_numpy(source_init[1]).to(dtype=torch.long, device=torchdeviceId)
    B_sol = torch.from_numpy(computeBoundary(source_init[1])).to(dtype=torch.bool, device=torchdeviceId)

    if len(source)<3 or use_fundata==0:
        FunS = torch.from_numpy(np.zeros((int(np.size(source[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    else:
        FunS = torch.from_numpy(source[2]).to(dtype=torchdtype, device=torchdeviceId)

    if len(target)<3 or use_fundata==0:
        FunT = torch.from_numpy(np.zeros((int(np.size(target[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    else:
        FunT = torch.from_numpy(target[2]).to(dtype=torchdtype, device=torchdeviceId)

    if len(source_init)<3 or use_fundata==0:
        Fun_sol = torch.from_numpy(np.zeros((int(np.size(source_init[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
    else:
        Fun_sol= torch.from_numpy(source_init[2]).to(dtype=torchdtype, device=torchdeviceId)

    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)


    # Define Energy and set parameters
    energy = enr_match_SRNF_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_S=weight_coef_dist_S ,weight_coef_dist_T=weight_coef_dist_T, weight_MCV =weight_MCV, kernel_geom = kernel_geom, kernel_grass = kernel_grass, kernel_fun = kernel_fun, sig_geom = sig_geom, sig_grass = sig_grass, sig_fun = sig_fun)




    def gradE(x,y):
        qs = x.clone().requires_grad_(True)
        qt = y.clone().requires_grad_(True)
        [Gx,Gy] = grad(energy(qs,qt), [qs,qt], create_graph=True)
        return Gx,Gy

    def funopt(z):
        n = int(len(z)/6)
        #print(x[:,np.newaxis].shape)
        x = z[:(3*n),np.newaxis].reshape(n,3)
        y = z[3*n:,np.newaxis].reshape(n,3)
        x = torch.from_numpy(x).to(dtype=torchdtype, device=torchdeviceId)
        y = torch.from_numpy(y).to(dtype=torchdtype, device=torchdeviceId)
        E = float(energy(x,y).detach().cpu().numpy())
        #print(E)
        return E

    def dfunopt(z):
        n = int(len(z)/6)
        x = z[:(3*n),np.newaxis].reshape(n,3)
        y = z[3*n:,np.newaxis].reshape(n,3)
        x = torch.from_numpy(x).to(dtype=torchdtype, device=torchdeviceId)
        y = torch.from_numpy(y).to(dtype=torchdtype, device=torchdeviceId)
        [Gx,Gy] = gradE(x,y)
        Gx=Gx.detach().cpu().numpy().flatten().astype('float64')
        Gy=Gy.detach().cpu().numpy().flatten().astype('float64')
        G=np.concatenate((Gx,Gy),axis=0)
        return G


    z0=source_init[0].flatten() #initialization for the estimated source
    z1=target_init[0].flatten() # initialization for the estimated target
    z=np.concatenate((z0,z1),axis=0)
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, z, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1,disp =1, maxls=20,maxfun=150000)
    f0 = xopt[0:int(len(xopt)/2)]
    f0 = f0[:,np.newaxis].reshape(int(len(f0)/3),3)
    f1 = xopt[int(len(xopt)/2):]
    f1 = f1[:,np.newaxis].reshape(int(len(f1)/3),3)
    return f0,f1,fopt,Dic


def MultiResMatching(source,target,source_init,target_init,paramlist):
# performs the multiRes symmetric matching algorithm
# Input: source,target...vertices, faces and textures of source and target
#        source_init,target_init...vertices, faces and textures of initial guess (need to have the
#                                                                       same connectivity struct)
#        paramlist...list of parameters-lists for each of the runs
# Output: list of optimal vertices f0 and f1 (after each run).
    VS = source[0]
    FS = source[1]
    use_fundata=paramlist[0]['use_fundata']
    if len(source)<3 or use_fundata==0:
        FunS = np.zeros((int(np.size(VS)/3),))
    else:
        FunS = source[2]

    f0={}
    f1={}
    En = {}
    Dic={}
    Tri={}
    Funct={}
    Tri[0] = source_init[1]

    if len(source_init)<3 or use_fundata==0:
        Funct[0] = np.zeros((int(np.size(source_init[0])/3),))
        source_init=[source_init[0], source_init[1], Funct[0]]
    else:
        Funct[0] = source_init[2]

    target_init=[target_init[0], target_init[1], source_init[2]]
    NoRuns=len(paramlist)
    # make for only until NoRuns-1 and do the last run after. Right now the last subsampling + smoothening
    # is unnecessary computational cost
    for i in range(0, NoRuns):
        print("Iteration: ",i)
        param=paramlist[i]
        f0[i],f1[i],En[i],Dic[i] = SymmetricMatching(source,target,source_init,target_init,param)


# Subdivide mesh with vtk. Warning: only applies to manifold meshes.
        VS_sub, FS_sub = io.subdivide_mesh(f0[i],Tri[i],order=1)
        VT_sub, FT_sub = io.subdivide_mesh(f1[i],Tri[i],order=1)
# Project texture maps
        if param['use_fundata']==1:
            Fun_sub=project_function(VS,FS,FunS,VS_sub)
        else:
            Fun_sub=np.zeros((int(np.size(VS_sub)/3),))
        #set initial guess for next run
        [source_init[0], source_init[1], source_init[2]] = [VS_sub,FS_sub,Fun_sub]
        [target_init[0], target_init[1], target_init[2]] = [VT_sub,FT_sub,Fun_sub]
        Tri[i+1]=source_init[1]
        Funct[i+1] = source_init[2]
    return f0,f1,Tri,Funct,En,Dic



def SRNF_map(source):
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    nX = getNormal(FS, VS)
    normX = torch.norm(nX,p=2,dim=1).view(nX.shape[0],1)
    nX=nX/normX.sqrt()
    nX=nX.detach().cpu().numpy().astype('float64')
    return nX



def SRNF_dist_square(source,target):
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    nX = getNormal(FS, VS)
    nY = getNormal(FT, VT)
    dist= SRNF_cost(nX,nY)
    dist=dist.detach().cpu().numpy().flatten().astype('float64')
    return dist


def SRCF_dist_square(source,target):
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    BS = torch.from_numpy(computeBoundary(source[1])).to(dtype=torch.bool, device=torchdeviceId)
    BT = torch.from_numpy(computeBoundary(target[1])).to(dtype=torch.bool, device=torchdeviceId)
    HS = SRCF(FS,BS,VS)
    mc_S = (HS**2).sum(axis=1).sqrt()
    HT = SRCF(FT,BT,VT)
    mc_T = (HT**2).sum(axis=1).sqrt()
    dist= ((mc_S-mc_T)**2).sum()
    dist=dist.detach().cpu().numpy().flatten().astype('float64')
    return dist


def project_function(V,F,Fun,V_sub):
    [n,d]=list(V.shape)
    [m,d]=list(V_sub.shape)
    Fun_sub=np.zeros((m,))
    nb_average=3

    for i in range(0, m-1):
        vi=V_sub[i]
        Vec_dist=((V-np.tile(vi,(n,1)))**2).sum(axis=1)
        Min_ind=Vec_dist.argsort()[:nb_average]
        Fun_sub[i]=np.mean(Fun[Min_ind])

    return Fun_sub



def SRNF_inversion(Q,source_init,max_iter):
# Wrapper function that performs the SRVF inversion
# Input: Q... SRVF function to be inverted
#        source_init,target_init...vertices and faces of initial guess (need to have the
#                                                                       same mess struct as Q)
# Output: estimated vertices V for such that SRVF(V) aprox Q.
    # Read out parameters

    # Convert Data to Pytorch
    F_init = torch.from_numpy(source_init[1]).to(dtype=torch.long, device=torchdeviceId)
    Q = torch.from_numpy(Q).to(dtype=torchdtype, device=torchdeviceId)

    # Define Energy and set parameters
    energy = enr_invert_SRNF(F_init,Q)

    def gradE(x):
        qs = x.clone().requires_grad_(True)
        Gx = grad(energy(qs), qs, create_graph=True)
        return Gx

    def funopt(z):
        n = int(len(z)/3)
        x = z.reshape(n,3)
        x = torch.from_numpy(x).to(dtype=torchdtype, device=torchdeviceId)
        E = float(energy(x).detach().cpu().numpy())
        return E

    def dfunopt(z):
        n = int(len(z)/3)
        x = z.reshape(n,3)
        x = torch.from_numpy(x).to(dtype=torchdtype, device=torchdeviceId)
        [G] = gradE(x)
        G=G.detach().cpu().numpy().flatten().astype('float64')
        return G


    z0=source_init[0].flatten() #initialization for the estimated source
    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, z0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1,disp =1, maxls=20,maxfun=150000)
    f0 = xopt.reshape(int(len(xopt)/3),3)
    return f0,fopt,Dic


def SRNF_geodesic(matched_source,matched_target,TimePoints,max_iter):
# Wrapper function that calculates the SRNF geodesic by approx. invert the linear path in SRNF space
# Input: matched_source,matched_target ... Registered Boundary Surfaces
#        TimePoints ... number of timepoints for the geodesic (including the boundary surfaces)
#        max_iter ... Maximal number of iteratiosn for minimization
# Output: Vertices of estimated  geodesic at time points t_i, i=1...TimePoints
    F=matched_source[1]
    Q0 = SRNF_map(matched_source)
    Q1 = SRNF_map(matched_target)
    t= np.linspace(0.0, 1.0, TimePoints)
    SRNF_path = []
    for i in range(TimePoints):
        SRNF_path.append((1-t[i])*Q0 + t[i]*Q1)
    Sol= []
    Sol.append(matched_source[0])
    for i in range(TimePoints-2):
        f0,fopt,Dic = SRNF_inversion(SRNF_path[i+1],[Sol[i],F],max_iter)
        Sol.append(f0)
    Sol.append(matched_target[0])
    return Sol
