import torch
import numpy as np
import scipy

use_cuda=1
torchdeviceId=torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype=torch.float32


##############################################################################################################################
# Discrete Differential Geometry Helper Functions
##############################################################################################################################


def batchDot(dv1,dv2):
    '''Parallel computation of batches of dot products.
    
    Input:
        - dv1 [Vxd torch tensor]
        - dv2 [Vxd torch tensor]
        
    Output:
        - tensor of dot products between corresponding rows of dv1 and dv2 [Vx1 torch tensor]
    '''

    return torch.einsum('bi,bi->b', dv1,dv2)


def getSurfMetric(V,F):
    '''Computation of the Riemannian metric evaluated at the faces of a triangulated surface.
    
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - g: Riemannian metric evaluated at each face of the triangulated surface [nFx2x2 torch tensor]
    '''

    # Number of faces
    nF = F.shape[0]

    # Preallocate tensor for Riemannian metric
    alpha = torch.zeros((nF,3,2)).to(dtype=torchdtype, device=torchdeviceId)   

    # Compute Riemannian metric at each face
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2]) 
    
    alpha[:,:,0]=V1-V0
    alpha[:,:,1]=V2-V0    

    return torch.matmul(alpha.transpose(1,2),alpha)

def getMeshOneForms(V,F):
    '''Computation of the Riemannian metric evaluated at the faces of a triangulated surface.
    
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - alpha: One form evaluated at each face of the triangulated surface [nFx3x2 torch tensor]
    '''

    # Number of faces
    nF = F.shape[0]

    # Preallocate tensor for Riemannian metric
    alpha = torch.zeros((nF,3,2)).to(dtype=torchdtype, device=torchdeviceId)   

    # Compute Riemannian metric at each face
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2]) 
    
    alpha[:,:,0]=V1-V0
    alpha[:,:,1]=V2-V0
    
    return alpha


def getLaplacian(V,F):   
    '''Computation of the mesh Laplacian operator of a triangulated surface evaluated at one of its tangent vectors h.
    
    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - L: function that will evaluate the mesh Laplacian operator at a tangent vector to the surface [function]
    ''' 

    # Number of vertices and faces
    nV, nF = V.shape[0], F.shape[0]

    # Get x,y,z coordinates of each face
    face_coordinates = V[F]
    v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]

    # Compute the area of each face using Heron's formula
    A = (v1 - v2).norm(dim=1) 
    B = (v0 - v2).norm(dim=1) # lengths of each side of the faces
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C) # semi-perimeter
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt() # Apply Heron's formula and clamp areas of small faces for numerical stability

    # Compute cotangent expressions for the mesh Laplacian operator
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 2.0

    # Find indices of adjacent vertices in the triangulated surface (i.e., edge list between vertices)
    ii = F[:, [1, 2, 0]]
    jj = F[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, nF * 3)

    # Define function that evaluates the mesh Laplacian operator at one of the surface's tangent vectors
    def L(h):
        '''Function that evaluates the mesh Laplacian operator at a tangent vector to the surface.

        Input:
            - h: tangent vector to the triangulated surface [nVx3 torch tensor]

        Output:
            - Lh: mesh Laplacian operator of the triangulated surface applied to one its tangent vectors h [nVx3 torch tensor]
        '''

        # Compute difference between tangent vectors at adjacent vertices of the surface
        hdiff = h[idx[0]]-h[idx[1]]

        # Evaluate mesh Laplacian operator by multiplying cotangent expressions of the mesh Laplacian with hdiff
        values = (torch.stack([cot.view(-1)]*3, dim=1)*hdiff)

        # Sum expression over adjacent vertices for each coordinate
        Lh = torch.zeros((nV,3)).to(dtype=torchdtype, device=torchdeviceId)  
        Lh[:,0]=Lh[:,0].scatter_add(0,idx[1,:],values[:,0])
        Lh[:,1]=Lh[:,1].scatter_add(0,idx[1,:],values[:,1])
        Lh[:,2]=Lh[:,2].scatter_add(0,idx[1,:],values[:,2])    

        return Lh

    return L


def getVertAreas(V,F):
    '''Computation of vertex areas for a triangulated surface.

    Input:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        
    Output:
        - VertAreas: vertex areas [nVx1 torch tensor]
    '''

    # Number of vertices
    nV = V.shape[0]
    
    # Get x,y,z coordinates of each face
    face_coordinates = V[F]
    v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]

    # Compute the area of each face using Heron's formula
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1) # lengths of each side of the faces
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C) # semi-perimeter
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt() # Apply Heron's formula and clamp areas of small faces for numerical stability

    # Compute the area of each vertex by averaging over the number of faces that it is incident to
    idx = F.view(-1)
    incident_areas = torch.zeros(nV, dtype=torch.float32, device=torchdeviceId)
    val = torch.stack([area] * 3, dim=1).view(-1)
    incident_areas.scatter_add_(0, idx, val)    
    vertAreas = 2*incident_areas/3.0+1e-24   

    return vertAreas


def getNormal(F, V):
    '''Computation of normals at each face of a triangulated surface.

    Input:
        - F: faces of the triangulated surface [nFx3 torch tensor]
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        
    Output:
        - N: vertex areas [nFx1 torch tensor]
    '''

    # Compute normals at each face by taking the cross product between edges of each face that are incident to its x-coordinate
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
    N = .5 * torch.cross(V1 - V0, V2 - V0)

    return N


def computeBoundary(F):
    '''Determining if a vertex is at the boundary of the mesh of a triagulated surface.

    Input:
        - F: faces of the triangulated surface [nFx3 ndarray]

    Output:
        - BoundaryIndicatorOfVertex: boolean vector indicating which vertices are at the boundary of the mesh [nVx1 boolean ndarray]

    Note: This is a CPU computation
    '''
    
    # Get number of vertices and faces
    nF = F.shape[0]
    nV = F.max()+1

    # Find whether vertex is at the boundary of the mesh
    Fnp = F # F.detach().cpu().numpy()
    rows = Fnp[:,[0,1,2]].reshape(3*nF)
    cols = Fnp[:,[1,2,0]].reshape(3*nF)
    vals = np.ones(3*nF,dtype=np.int)
    E = scipy.sparse.coo_matrix((vals,(rows,cols)),shape=(nV,nV))
    E -= E.transpose()
    i,j = E.nonzero()
    BoundaryIndicatorOfVertex = np.zeros(nV,dtype=np.bool)
    BoundaryIndicatorOfVertex[i] = True
    BoundaryIndicatorOfVertex[j] = True

    return BoundaryIndicatorOfVertex