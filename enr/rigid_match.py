# Load Packages
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from scipy.linalg import expm
#from enr.DDG import computeBoundary
from enr.varifold import *
from torch.autograd import grad
#import utils.utils as io
import torch
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32


def RigidVarifoldMatch(source, target, param):
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)

    # Align mass centers
    [VS, mcS] = mass_centering(VS, FS)
    [VT, mcT] = mass_centering(VT, FT)


    # Set parameters
    if ('scaling' not in param):
        scaling_flag=0
    else:
        scaling_flag=param['scaling']

    if ('loss' not in param):
        param['loss'] = 'ener_kernel'

    if param['loss']=="pd_kernel":

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

        if ('sig_geom' not in param):
            diam=compute_diameter(VS)
            sig_geom = diam/3
        else:
            sig_geom = param['sig_geom']

        if ('sig_grass' not in param):
            sig_grass = 1.5
        else:
            sig_grass = param['sig_grass']

        if ('sig_fun' not in param):
            sig_fun = 1
        else:
            sig_fun = param['sig_fun']

        sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
        sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
        sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)
        FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
        FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)

        K=VKerenl(kernel_geom, kernel_grass, kernel_fun, sig_geom, sig_grass, sig_fun)
        E = lossVarifoldProd(FS,FunS, VT, FT, FunT, K, scaling_flag)

    else:
        if ('clamp_min' not in param):
            cmin = 1e-5
        else:
            cmin = param['clamp_min']

        if ('clamp_max' not in param):
            cmax = 1e10
        else:
            cmax = param['clamp_max']

        cmin = torch.tensor([cmin], dtype=torchdtype, device=torchdeviceId)
        cmax = torch.tensor([cmax], dtype=torchdtype, device=torchdeviceId)
        K=Energy_Kernel(cmin,cmax)
        E=lossEnerdProdScaled(FS, VT, FT, K, scaling_flag)


    if ('max_iter' not in param):
        max_iter = 20
    else:
        max_iter=param['max_iter']

    if ('nb_seeds' not in param):
        nb_seeds = 10
    else:
        nb_seeds=param['nb_seeds']


    #Generate seed rotations
    sphere_points = fibonacci_sphere(nb_seeds)
    nb_angles=5
    angle_vec = np.linspace(0, 2 * np.pi, nb_angles, endpoint=False).reshape(-1, 1)
    sphere_points = np.repeat(sphere_points, nb_angles - 1, axis=0)  # (50 * 8, 3)
    seed_rots = np.concatenate(
        ([[0., 0., 0.]], np.multiply(sphere_points, np.tile(angle_vec[1:], [nb_seeds, 3]))), axis=0)


    # Main optimization loop
    if scaling_flag == 1:
        areaS=compute_total_area(VS,FS)
        areaT=compute_total_area(VT,FT)
        init_lam = torch.sqrt(areaT/areaS).detach().cpu().numpy()
        tcoeffs = torch.tensor(np.array([0., 0., 0., 0., 0., 0., init_lam]), dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    else:
        tcoeffs = torch.tensor(np.array([0., 0., 0., 0., 0., 0.]), dtype=torchdtype, device=torchdeviceId, requires_grad=True)

    dmin=np.Inf
    loss_min = np.inf
    for i in range(seed_rots.shape[0]):
        #print("Performing optimization of rotation with initial direction and angle", i, ":")

        roti = torch.tensor(seed_rots[i, :]).to(dtype=torchdtype, device=torchdeviceId)
        Mroti = rotation_matrix(roti)
        V0 = torch.matmul(VS, torch.t(Mroti))

        dataloss = rigid_loss(E,V0,scaling_flag)
        # r_para=torch.tensor([0., 0., 0., 5., -1., 3.5],dtype=torchdtype, device=torchdeviceId, requires_grad=True)


        optimizer = torch.optim.LBFGS([tcoeffs], max_eval=10, max_iter=10, line_search_fn='strong_wolfe')
        loss_list = []

        #print(dataloss(tcoeffs))

        def closure():
            optimizer.zero_grad()
            L = dataloss(tcoeffs)
            L1 = L.detach().cpu().numpy()
            #print("loss", L1)
            loss_list.append(L1)

            L.backward()
            return L

        for k in range(max_iter):
            #print("it ", k, ": ", end="")
            optimizer.step(closure)

        if loss_list[-1] < loss_min:
            loss_min = loss_list[-1]
            print("New optimal loss:", loss_min)
            Opt_rot = torch.matmul(rotation_matrix(tcoeffs[0:3]), Mroti)
            Opt_tra = tcoeffs[3:6]
            if scaling_flag==1:
                Opt_scal=tcoeffs[6]
            else:
                Opt_scal=torch.tensor(np.array([1.]), dtype=torchdtype, device=torchdeviceId)


    V1 = Opt_scal*torch.matmul(VS, torch.t(Opt_rot)) + Opt_tra
    V1 = V1.detach().cpu().numpy()
    t_source=[V1,source[1]]
    if len(source)>2:
        t_source.append(source[2])

    t_target=[VT.detach().cpu().numpy(),target[1]]
    if len(target)>2:
        t_target.append(target[2])

    rot=Opt_rot.detach().cpu().numpy()
    tra=Opt_tra.detach().cpu().numpy()

    if scaling_flag==1:
        scal=Opt_scal.detach().cpu().numpy()
    else:
        scal=1

    transf_param= {'tra':tra,'rot':rot,'scal':scal}

    return t_source, t_target, transf_param

def compute_diameter(V):
    n=V.size(0)
    V1=torch.stack((torch.reshape(V[:,0],(n,1)).repeat(1,n),torch.reshape(V[:,1],(n,1)).repeat(1,n),\
                    torch.reshape(V[:,2],(n,1)).repeat(1,n)),2)
    V2=torch.stack((torch.reshape(V[:,0],(1,n)).repeat(n,1),torch.reshape(V[:,1],(1,n)).repeat(n,1),\
                    torch.reshape(V[:,2],(1,n)).repeat(n,1)),2)
    M=torch.sum(torch.square(V2-V1),2)
    diam=M.max().sqrt().detach().cpu().numpy().item()

    return diam

def compute_total_area(V,F):
    V0, V1, V2 = (
        V.index_select(0, F[:, 0]),
        V.index_select(0, F[:, 1]),
        V.index_select(0, F[:, 2]),
    )
    normals = 0.5 * torch.cross(V1 - V0, V2 - V0)
    areas = (normals ** 2).sum(dim=1)[:, None].sqrt()
    t_area= areas.sum()
    return t_area


def mass_centering(V,F):
    def get_triangle_centers(V, F):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        areas = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, areas

    c,m = get_triangle_centers(V, F)
    mcenter = torch.sum(torch.mul(m.repeat(1,c.size(1)),c),0)/torch.sum(m)
    Vc=V-mcenter

    return Vc,mcenter


def fibonacci_sphere(sample_num):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))    # golden angle in radians

    for i in range(sample_num):
        y = 1 - (i / float(sample_num - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)    # radius at y

        theta = phi * i    # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points


def rotation_matrix(r_coeffs):
    A = torch.tensor([ [0, -r_coeffs[2], r_coeffs[1]], [r_coeffs[2], 0, -r_coeffs[0]], [-r_coeffs[1], r_coeffs[0], 0] ])\
        .to(dtype=torchdtype, device=torchdeviceId)


    matrix = torch.linalg.matrix_exp(A)

    return matrix


def rigid_loss(E,V0,scaling_flag):
    def loss(tr_coeffs):
        A = torch.tensor(
            [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.], [-1., 0., 0.], [0., -1., 0.],
             [1., 0., 0.], [0., 0., 0.]]).to(dtype=torchdtype, device=torchdeviceId)
        M = torch.reshape(torch.matmul(A, tr_coeffs[:3]), (3, 3))
        R = torch.linalg.matrix_exp(M)
        if scaling_flag==1:
            q=tr_coeffs[6]*torch.matmul(V0, torch.t(R)) + tr_coeffs[3:6]
        else:
            q = torch.matmul(V0, torch.t(R)) + tr_coeffs[3:]

        return (
            E(q)
        )

    return loss


def rand_rot():
    v=np.random.randn(3,)
    v=v/np.sqrt(np.sum(np.square(v)))
    ang=np.random.uniform(-np.pi,np.pi)
    M = np.array([ [0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0] ])


    R = expm(ang*M)

    return R


def RigidInertia_Align(source, target, param):
    VS=source[0]
    FS=source[1]
    VT=target[0]
    FT=target[1]

    if ('scaling' not in param):
        scaling_flag=0
    else:
        scaling_flag=param['scaling']

    [mcS, I_matS] = compute_inertia_matrix(VS, FS)
    [mcT, I_matT] = compute_inertia_matrix(VT, FT)

    VS=VS-mcS
    VT=VT-mcT
    tra=mcT-mcS

    if scaling_flag==1:
        TrS = np.trace(I_matS)
        TrT = np.trace(I_matT)
        scal= np.sqrt(TrT/TrS)
        VS=scal*VS
    else:
        scal=1

    U0, S0, Vh0 = np.linalg.svd(I_matS, full_matrices=True)
    U1, S1, Vh1 = np.linalg.svd(I_matT, full_matrices=True)

    if ('method' not in param) or param['method'] == 'basic':
        rot = np.matmul(U0, Vh1)

    else:
        cmin = torch.tensor([1e-5], dtype=torchdtype, device=torchdeviceId)
        cmax = torch.tensor([1e10], dtype=torchdtype, device=torchdeviceId)
        VT_torch = torch.from_numpy(VT).to(dtype=torchdtype, device=torchdeviceId)
        FS_torch = torch.from_numpy(FS).to(dtype=torch.long, device=torchdeviceId)
        FT_torch = torch.from_numpy(FT).to(dtype=torch.long, device=torchdeviceId)
        K = Energy_Kernel(cmin, cmax)
        E = lossEnerdProdScaled(FS_torch, VT_torch, FT_torch, K, scaling_flag)

        min_dist=np.Inf
        for i in range(0,2):
            Ui=U0
            Ui=-Ui
            Ui[:,i] = -Ui[:,i]
            roti = np.matmul(Ui,Vh1)
            VSi = np.matmul(VS,roti)
            VSi_torch = torch.from_numpy(VSi).to(dtype=torchdtype, device=torchdeviceId)
            Ei=E(VSi_torch)
            Ei=Ei.detach().cpu().numpy()

            if Ei < min_dist:
                min_dist=Ei
                rot=roti


    VS=np.matmul(VS,rot)

    t_source = [VS, source[1]]
    t_target = [VT, target[1]]


    transf_param= {'tra':tra,'rot':rot,'scal':scal}

    return t_source, t_target, transf_param


def compute_inertia_matrix(V,F):
    V0=V[F[:,0],:]
    V1=V[F[:,1],:]
    V2=V[F[:,2],:]
    c=(V0 + V1 + V2) / 3
    norm=np.cross(V1 - V0, V2 - V0)
    areas=np.sqrt(np.sum(norm**2,1))
    mcenter = np.sum(np.transpose(np.tile(areas,(3,1)))*c,0)/np.sum(areas)
    cc=c-mcenter
    M11=cc[:,0]*cc[:,0]
    M12=cc[:,0]*cc[:,1]
    M13=cc[:,0]*cc[:,2]
    M22=cc[:,1]*cc[:,1]
    M23=cc[:,1]*cc[:,2]
    M33=cc[:,2]*cc[:,2]

    I_mat=np.zeros((3,3))
    I_mat[0,0] = np.sum(areas*(M22+M33), 0)
    I_mat[0,1] = -np.sum(areas*M12, 0)
    I_mat[1,0] = I_mat[0,1]
    I_mat[0,2] = -np.sum(areas*M13, 0)
    I_mat[2,0] = I_mat[0,2]
    I_mat[1,1] = np.sum(areas*(M11+M33), 0)
    I_mat[1,2] = -np.sum(areas*M23, 0)
    I_mat[2,1] = I_mat[1,2]
    I_mat[2,2] = np.sum(areas*(M11+M22), 0)
    I_mat=I_mat/np.sum(areas)


    return mcenter,I_mat