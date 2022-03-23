import torch
import numpy as np
from enr.DDG import getNormal
from enr.varifold import *

use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32



##############################################################################################################################
#SRNF Helper Functions
##############################################################################################################################


def enr_invert_SRNF(F,Q):
# Input connectivity matrix F and target SRVF Q.
    def energy(x):
        nX= getNormal(F, x)
        normX = torch.norm(nX,p=2,dim=1).view(nX.shape[0],1)
        SRNF=nX/normX.sqrt()
        return ((SRNF - Q)**2).sum()
    return energy


def SRNF_cost(nX,nY):

    normX = torch.norm(nX,p=2,dim=1).view(nX.shape[0],1)
    normY = torch.norm(nY,p=2,dim=1).view(nY.shape[0],1)

    return ((nX/normX.sqrt() - nY/normY.sqrt())**2).sum()


def SRCF(F,B,V):
    Vno = V.shape[0]
    # compute the vector area of each face
    normalOfFace = getNormal(F, V)

    # compute the scalar area of each face by taking a norm
    AreaOfFace = torch.norm(normalOfFace,p=2,dim=1)

    # compute the unit normal vector of each face
    # the view command appends a singleton dimension
    UnitNormalVectorOfFace = normalOfFace / AreaOfFace.reshape(AreaOfFace.shape[0],1)

    # distribute the scalar area to vertices
    AreaOfVertex = torch.zeros(V.shape[0]).to(dtype=V.dtype, device=V.device)
    AreaOfVertex.index_add_(0,F[:,0],AreaOfFace)
    AreaOfVertex.index_add_(0,F[:,1],AreaOfFace)
    AreaOfVertex.index_add_(0,F[:,2],AreaOfFace)
    AreaOfVertex.mul_(1/3)

    # compute the vector-valued mean curvature at each vertex
    # in words: for each face, for each vertex of the face,
    # compute the cross product of the opposite edge and the vector area,
    # and sum up all of these terms individually for each vertex
    VectorMeanCurvatureOfVertex = torch.zeros(V.shape).to(dtype=V.dtype, device=V.device)
    VectorMeanCurvatureOfVertex.index_add_(0,F[:,0],torch.cross(V[F[:,2],:]-V[F[:,1],:], UnitNormalVectorOfFace))
    VectorMeanCurvatureOfVertex.index_add_(0,F[:,1],torch.cross(V[F[:,0],:]-V[F[:,2],:], UnitNormalVectorOfFace))
    VectorMeanCurvatureOfVertex.index_add_(0,F[:,2],torch.cross(V[F[:,1],:]-V[F[:,0],:], UnitNormalVectorOfFace))
    VectorMeanCurvatureOfVertex.mul_(1/(8*AreaOfVertex.sqrt().reshape(Vno,1)))

    #Set the curvature vectors at boundary vertices to be 0
    VectorMeanCurvatureOfVertex[B,:] = 0

    return VectorMeanCurvatureOfVertex


##############################################################################################################################
#SRNF_Matching_Energies
##############################################################################################################################


def enr_match_SRNF(VS, FS, FunS, BS, VT, FT, FunT, weight_coef_dist=1 ,weight_SRNF = 1, weight_MCV = 0, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss = lossVarifoldSurf(FS, FunS, VT, FT, FunT, K)
    nS = getNormal(FS, VS)
    if weight_MCV>0:
        HS = SRCF(FS,BS,VS)

    def energy(x):
        nx= getNormal(FS, x)
        if weight_MCV>0:
            Hx = SRCF(FS,BS,x)
            return weight_SRNF*SRNF_cost(nS,nx) + weight_coef_dist*dataloss(x) + weight_MCV*((Hx-HS)**2).sum(axis=1).sum()
        else:
            return weight_SRNF*SRNF_cost(nS,nx) + weight_coef_dist*dataloss(x)
          #  return weight_coef_dist*dataloss(x)
    return energy


def enr_match_SRNF_sym(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, B_sol, weight_coef_dist_S=1 ,weight_coef_dist_T=1, weight_SRNF = 1, weight_MCV = 0, **objfun):

    K = VKerenl(objfun['kernel_geom'],objfun['kernel_grass'],objfun['kernel_fun'],objfun['sig_geom'],objfun['sig_grass'],objfun['sig_fun'])
    dataloss_S = lossVarifoldSurf(F_sol, Fun_sol, VS, FS, FunS, K)
    dataloss_T = lossVarifoldSurf(F_sol, Fun_sol, VT, FT, FunT, K)
    def energy(x,y):
        nx= getNormal(F_sol, x)
        ny= getNormal(F_sol, y)
        if weight_MCV>0:
            Hx = SRCF(F_sol,B_sol,x)
            Hy = SRCF(F_sol,B_sol,y)
            return weight_SRNF*SRNF_cost(nx,ny)+ weight_MCV*((Hx-Hy)**2).sum(axis=1).sum() + weight_coef_dist_S*dataloss_S(x) + weight_coef_dist_T*dataloss_T(y)
        else:
            E=0
            E+=weight_coef_dist_S*dataloss_S(x) + weight_coef_dist_T*dataloss_T(y)
            E+=weight_SRNF*SRNF_cost(nx,ny)
            
            
            return E
    return energy