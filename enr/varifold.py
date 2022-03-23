from pykeops.torch import kernel_product, Genred
from pykeops.torch.kernel_product.formula import *

import torch
import numpy as np
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

##############################################################################################################################
#Varifold_Functions
##############################################################################################################################


def lossVarifoldSurf(FS,FunS, VT, FT, FunT, K):
    def CompCLNn(F, V, Fun):

        if F.shape[1] == 2:
            V0, V1 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1])
            Fun0, Fun1 = Fun.index_select(0, F[:, 0]), Fun.index_select(0, F[:, 1])
            C, N, Fun_F =  (V0 + V1)/2, V1 - V0, (Fun0 + Fun1)/2
        else:
            V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
            Fun0, Fun1, Fun2 = Fun.index_select(0, F[:, 0]), Fun.index_select(0, F[:, 1]), Fun.index_select(0, F[:, 2])
            C, N, Fun_F =  (V0 + V1 + V2)/3, .5 * torch.cross(V1 - V0, V2 - V0), (Fun0 + Fun1 + Fun2)/3

        L = (N ** 2).sum(dim=1)[:, None].clamp_(min=1e-6).sqrt()
        [n]=list(Fun_F.size())
        Fun_F=Fun_F.resize_((n,1))
        return C, L, N / L, Fun_F

    CT, LT, NTn, Fun_FT = CompCLNn(FT, VT, FunT)
    cst = K(CT, CT, NTn, NTn, Fun_FT, Fun_FT, LT, LT)
    def loss(VS):
        CS, LS, NSn, Fun_FS = CompCLNn(FS, VS, FunS)
        return cst + K(CS, CS, NSn, NSn, Fun_FS, Fun_FS, LS, LS) - 2 * K(CS, CT, NSn, NTn, Fun_FS, Fun_FT, LS, LT)
    return loss 


def lossVarifoldSurf_Weighted(FS, FunS, VT, FT, FunT, K):
    def CompCLNn(F, V, Fun):

        if F.shape[1] == 2:
            V0, V1 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1])
            Fun0, Fun1 = Fun.index_select(0, F[:, 0]), Fun.index_select(0, F[:, 1])
            C, N, Fun_F =  (V0 + V1)/2, V1 - V0, (Fun0 + Fun1)/2
        else:
            V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
            Fun0, Fun1, Fun2 = Fun.index_select(0, F[:, 0]), Fun.index_select(0, F[:, 1]), Fun.index_select(0, F[:, 2])
            C, N, Fun_F =  (V0 + V1 + V2)/3, .5 * torch.cross(V1 - V0, V2 - V0), (Fun0 + Fun1 + Fun2)/3

        L = (N ** 2).sum(dim=1)[:, None].clamp_(min=1e-6).sqrt()
        [n]=list(Fun_F.size())
        
        Fun_F=Fun_F.reshape((n,1))
            
        return C, L, N / L, Fun_F

    CT, LT, NTn, Fun_FT = CompCLNn(FT, VT, FunT)
    cst = K(CT, CT, NTn, NTn, Fun_FT, Fun_FT, LT, LT)

    def loss(VS,Rho):
        CS, LS, NSn, Fun_FS = CompCLNn(FS, VS, FunS)
        Rho0, Rho1, Rho2 = Rho.index_select(0, FS[:, 0]), Rho.index_select(0, FS[:, 1]), Rho.index_select(0, FS[:, 2])
        RhoF=(Rho0 + Rho1 + Rho2)/3
        LS=torch.unsqueeze(RhoF,1)*LS
        return cst + K(CS, CS, NSn, NSn, Fun_FS, Fun_FS, LS, LS) - 2 * K(CS, CT, NSn, NTn, Fun_FS, Fun_FT, LS, LT)
    
    return loss



def VKerenl(kernel_geom,kernel_grass,kernel_fun,sig_geom,sig_grass,sig_fun):
    #kernel on spatial domain
    if kernel_geom.lower() == "gaussian":
        expr_geom = 'Exp(-SqDist(x,y)*a)'
    elif kernel_geom.lower() == "cauchy":
        expr_geom = 'IntCst(1)/(IntCst(1)+SqDist(x,y)*a)'

    #kernel on Grassmanian
    if kernel_grass.lower() == 'constant':
            expr_grass = 'IntCst(1)'

    elif kernel_grass.lower() == 'linear':
            expr_grass = '(u|v)'

    elif kernel_grass.lower() == 'gaussian_oriented':
            expr_grass = 'Exp(IntCst(2)*b*((u|v)-IntCst(1)))'

    elif kernel_grass.lower() == 'binet':
            expr_grass = 'Square((u|v))'

    elif kernel_grass.lower() == 'gaussian_unoriented':
            expr_grass='Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))'

    #kernel on signal
    if kernel_fun.lower() == 'constant':
        expr_fun = 'IntCst(1)'
    elif kernel_fun.lower() == "gaussian":
        expr_fun = 'Exp(-SqDist(g,h)*c)'
    elif kernel_fun.lower() == "cauchy":
        expr_fun = 'IntCst(1)/(IntCst(1)+SqDist(g,h)*c)'



    def K(x, y, u, v, f, g, r1, r2):
        d = x.shape[1]
        pK = Genred(expr_geom + '*' + expr_grass + '*' + expr_fun +'*r',
            ['a=Pm(1)','b=Pm(1)','c=Pm(1)','x=Vi('+str(d)+')','y=Vj('+str(d)+')','u=Vi('+str(d)+')',
            'v=Vj('+str(d)+')','g=Vi(1)','h=Vj(1)','r=Vj(1)'],
            reduction_op='Sum',
            axis=1)
        return (pK(1/sig_geom**2,1/sig_grass**2,1/sig_fun**2,x,y,u,v,f,g,r2)*r1).sum()
    return K


    
    
    
    
    
    