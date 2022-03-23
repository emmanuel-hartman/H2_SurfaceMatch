import torch
import numpy as np
from enr.DDG import getVertAreas

use_cuda=1
torchdeviceId=torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype=torch.float32


##############################################################################################################################
# Regularizers for Partial Matching
##############################################################################################################################


def quartic_pen(m0=0., m1=1., h=0.5, clip_epsilon=0.25, clip_grad=0.5, normalization=True):
    '''Regularization of the weight function defined on a triangulated surface via the (clipped) quartic penalty.
    
    Input:
        - m0: first minimizer of the quartic penalty [default=0.0]
        - m1: second minimizer of the quartic penalty [default=1.0]
        - h: height of local maximum of quartic penalty [default=0.5]
        - clip_epsilon: clipping point for the quartic penalty [default=0.25]
        - clip_grad: gradient of linear segments of the clipped quartic penalty [default=0.5]
        - normalization: boolean variable for normalizing the quartic penalty by vertex areas [default=True]
        
    Output:
        - penalty: function that will evaluate the clipped quartic penalty at the weights defined on the triangulated surface [function]
    '''

    # Compute midpoint of minimizers (zeros) of the quartic penalty
    m = (m0 + m1)/2

    # Matrix of quartic polynomial expression and its first derivative evaluated at m0, m1, m
    X = np.array([m0**np.arange(6),
                  m1**np.arange(6),
                  m**np.arange(6),
                  [0, 1, 2*m0, 3*m0**2, 4*m0**3, 5*m0**4],
                  [0, 1, 2*m1, 3*m1**2, 4*m1**3, 5*m1**4],
                  [0, 1, 2*m, 3*m**2, 4*m**3, 5*m**4]])
                
    # Value of quartic penalty and its derivative at m0, m1, m
    y = np.array([[0], [0], [h], [0], [0], [0]])

    # Compute coefficients of quartic polynomial
    C = torch.from_numpy(np.linalg.solve(X,y)).to(dtype=torchdtype, device=torchdeviceId)
    deg = len(C) # get length of torch tensor C

    # Define function that will evaluate the clipped quartic penalty at the weights defined on the triangulated surface
    def penalty(V, F, Rho):

        # Get number of vertices
        nV = V.shape[0]

        # Preallocate tensor for (clipped) quartic polynomial evaluated at weights
        Q = torch.from_numpy(np.zeros(nV,1)).to(dtype=torchdtype, device=torchdeviceId)
        Rho=torch.unsqueeze(Rho,dim=1)

        # Find indices of weights that have to be clipped
        clip_left=Rho<m0-clip_epsilon
        clip_right=Rho>m1+clip_epsilon
        clip_mid= torch.from_numpy(np.logical_and((Rho<=m1+clip_epsilon).cpu().numpy() ,(Rho>=m0-clip_epsilon).cpu().numpy())).to(dtype=torch.bool, device=torchdeviceId)

        # Evaluate clipped quartic polynomial at the weights defined on the surface
        for p in range(deg):
            Q[clip_left] += torch.mul((m0-clip_epsilon)**p, C[p]) 
            Q[clip_right] += torch.mul((m1+clip_epsilon)**p, C[p]) 
            Q[clip_mid] += torch.mul(Rho[clip_mid]**p, C[p])
                
        Q[clip_left] += -clip_grad*(Rho[clip_left] - (m0-clip_epsilon))
        Q[clip_right] += clip_grad*(Rho[clip_right] - (m1+clip_epsilon))

        # Normalize quartic loss by vertex areas
        if normalization:
            A = getVertAreas(V, F)
            Q = Q*A
            quartic_loss = torch.sum(Q)

        else:
            quartic_loss = torch.sum(Q)

        return quartic_loss

    return penalty