import os
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import open3d as o3d
from utils.utils import *


##############################################################################################################################
# Input-Output (IO) and Plotting Functions
##############################################################################################################################


def loadData(file_name):    
    """Load mesh information from either a mat file or a ply file.

    Input: 
        - file_name [string]

    Output:
        - V: vertices of the triangulated surface [nVx3 torch tensor]
        - F: faces of the triangulated surface [nFx3 torch tensor]
        - Rho: weights defined on the vertices of the triangulated surface [nVx1 torch tensor]
    """

    # Determine whether to load a .mat or .ply file
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()                

    # Load .mat file
    if extension == ".mat":
        data = loadmat(file_name)
        V = data['V']
        F = data['F']-1
        F = F.astype('int64')

        if 'Fun' not in data:
            Rho = np.ones((int(np.size(V)/3),))
            print("No weights found: set to 1")
        else:
            Rho = data['Fun']
            Rho = np.reshape(Rho,(Rho.size,))

    # Load .ply file
    else:        
        mesh = o3d.io.read_triangle_mesh(file_name)
        V, F, Rho = getDataFromMesh(mesh)

    return V, F, Rho

    
def saveData(file_name,extension,V,F,Rho=None,color=None):
    """Save mesh information either as a mat file or ply file.
    
    Input:
        - file_name: specified path for saving mesh [string]
        - extension: extension for file_name, i.e., "mat" or "ply"
        - V: vertices of the triangulated surface [nVx3 numpy ndarray]
        - F: faces of the triangulated surface [nFx3 numpy ndarray]
        - Rho: weights defined on the vertices of the triangulated surface [nVx1 numpy ndarray, default=None]
        - color: colormap [nVx3 numpy ndarray of RGB triples]

    Output:
        - file_name.mat or file_name.ply file containing mesh information
    """

    # Save as .mat file
    if extension == 'mat':
        if Rho is None:
            savemat(file_name+".mat",{'V':V,'F':F+1}) 
        else:
            savemat(file_name+".mat",{'V':V,'F':F+1,'Rho':Rho}) 

    # Save as .ply file
    else:
        nV = V.shape[0]
        nF = F.shape[0]    
        file = open("{}.ply".format(file_name), "w")
        lines = ("ply","\n","format ascii 1.0","\n", "element vertex {}".format(nV),"\n","property float x","\n","property float y","\n","property float z","\n")
        
        if color is not None:
            lines += ("property uchar red","\n","property uchar green","\n","property uchar blue","\n")
            if Rho is not None:
                lines += ("property uchar alpha","\n")
        
        lines += ("element face {}".format(nF),"\n","property list uchar int vertex_index","\n","end_header","\n")

        file.writelines(lines)
        lines = []
        for i in range(0,nV):
            for j in range(0,3):
                lines.append(str(V[i][j]))
                lines.append(" ")
            if color is not None:
                for j in range(0,3):
                    lines.append(str(color[i][j]))
                    lines.append(" ")
                if Rho is not None:
                    lines.append(str(Rho[i]))
                    lines.append(" ")
                        
            lines.append("\n")
        for i in range(0,nF):
            l = len(F[i,:])
            lines.append(str(l))
            lines.append(" ")

            for j in range(0,l):
                lines.append(str(F[i,j]))
                lines.append(" ")
            lines.append("\n")

        file.writelines(lines)
        file.close()

        
def plotMesh(mesh):
    """Plot a given surface.
    
    Input:
        - mesh [tuple with fields mesh[0]=vertices, mesh[1]=faces]
    """        

    # Convert data to open3d mesh object for plotting the surface
    mesh = getMeshFromData(mesh)

    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    colors_np= np.asarray(mesh.vertex_normals)
    colors_np= (colors_np+1)/2
    mesh.vertex_colors =  o3d.utility.Vector3dVector(colors_np)
    o3d.visualization.draw_geometries([mesh])
        
        
def plotMatchingResult(source,matched_target,target,matching_type,matched_source=None):
    """Plot source, matched source, matched target and target after matching.
    
    Input:
        - source [tuple with source[0]=vertices, source[1]=faces]
        - matched_target [tuple with matched_target[0]=vertices, matched_target[1]=faces]
        - target [tuple with target[0]=vertices, target[1]=faces]
        - matching_type: "Symmetric" matching or otherwise [string]
        - matched_source [tuple with matched_source[0]=vertices, matched_source[1]=faces, default=None]

    Output:
        - Plot with source (left), matched target (middle) and target (right)
        Note: If the matching is symmetric, the matched source is displayed between the source and matched target.
    """   

    # Convert data to open3d mesh objects for generating plots after symmetric matching     
    if matching_type == "Symmetric" and (matched_source is not None):
        source = getMeshFromData(source).translate((0,0,0),relative=False)
        matched_source = getMeshFromData(matched_source).translate((5,0,0),relative=False)
        matched_target = getMeshFromData(matched_target).translate((10,0,0),relative=False)
        target = getMeshFromData(target).translate((15,0,0),relative=False)  
        o3d.visualization.draw_geometries([source,matched_source,matched_target,target])
        
    # Convert data to open3d mesh objects for generating plots after asymmetric matching  
    else:
        source = getMeshFromData(source).translate((0,0,0),relative=False)
        matched_target = getMeshFromData(matched_target).translate((5,0,0),relative=False)
        target = getMeshFromData(target).translate((10,0,0),relative=False)  
        o3d.visualization.draw_geometries([source,matched_source,matched_target,target])        
    
def plotGeodesic(geod,F,source=None,target=None,file_name=None,offsetstep=2.5,stepsize=2,axis=[0,0,1],angle=-1*np.pi/2):
    """Plot geodesic evolution after symmetric or asymmetric matching with the H2 metric and varifold relaxation.
    
    Input:
        - geod: geodesic path [tuple with tuple[k]=vertices of k^th surface in the geodesic stored as an nVx3 ndarray]
        - F: faces for the mesh structure of the surfaces on the geodesic path [nFx3 ndarray]
        - source [tuple with source[0]=vertices, source[1]=faces, default=None]
        - target [tuple with target[0]=vertices, target[1]=faces, default=None]
        - file_name: specified path for saving geodesic mesh [string, default=None]
        - offsetstep: spacing between different geodesics on the plot [default=2.5]
        - stepsize: spacing within a geodesic on the plot [default=2]
        - axis: axis of rotation for each individual surface in the geodesic [default=[0,0,1]]
        - angle: angle of rotation [default=-pi/2]

    Output:
        - Plot of geodesic with source (left), geodesic path (middle) and target (right) 
        - file_name.ply file containing geodesic mesh information (optional)
    """   

    # Convert data to open3d mesh objects for generating plots of the geodesic path
    ls = makeGeodMeshes(geod,F,source,target,\
                        offsetstep=offsetstep,stepsize=stepsize,axis=axis,angle=angle)
    o3d.visualization.draw_geometries(ls)
    
    # Save plots if specified by user
    if file_name != None:
        mesh = ls[0]    
        for i in range(1,len(ls)):
            mesh += ls[i]       
        V,F,Color = getDataFromMesh(mesh)
        if mesh.has_vertex_colors():
            saveData(file_name,"ply",V,F,color=Color)    
        else:
            saveData(file_name,"ply",V,F)        
      
    
def plotPartialGeodesic(geod,F,source=None,target=None,Rho=None,file_name=None,offsetstep=2.5,stepsize=2,axis=[0,0,1],angle=-1*np.pi/2):
    """Plot geodesic evolution after partial matching with the H2 metric and weighted varifold relaxation.
    
    Input:
        - geod: geodesic path [tuple with tuple[k]=vertices of k^th surface in the geodesic stored as an nVx3 ndarray]
        - F: faces for the mesh structure of the surfaces on the geodesic path [nFx3 ndarray]
        - source [tuple with source[0]=vertices, source[1]=faces, default=None]
        - target [tuple with target[0]=vertices, target[1]=faces, default=None]
        - Rho: weights defined on the endpoint of the geodesic [nVx1 numpy ndarray, default=None]
        - file_name: specified path for saving geodesic mesh [string, default=None]
        - offsetstep: spacing between different geodesics on the plot [default=2.5]
        - stepsize: spacing within a geodesic on the plot [default=2]
        - axis: axis of rotation for each individual surface in the geodesic [default=[0,0,1]]
        - angle: angle of rotation [default=-pi/2]

    Output:
        - Plot of geodesic with source (left), geodesic path (middle) and target (right) - with interpolated weights on the path
        - file_name.ply file containing geodesic mesh information (optional)
    """   

    # Convert data to open3d mesh objects for generating plots of the geodesic path
    if Rho is not None:
        ls,Rhon = makeGeodMeshes(geod,F,source,target,Rho=Rho,\
                                 offsetstep=offsetstep,stepsize=stepsize,axis=axis,angle=angle)
        Rhot = np.array(Rhon)
    else:
        ls = makeGeodMeshes(geod,F,source,target,\
                            offsetstep=offsetstep,stepsize=stepsize,axis=axis,angle=angle)
    o3d.visualization.draw_geometries(ls)
    
    # Save plots if specified by user
    if file_name != None:
        mesh = ls[0]    
        for i in range(1,len(ls)):
            mesh += ls[i]
        V,F,Color = getDataFromMesh(mesh)
        if mesh.has_vertex_colors():
            if Rho is not None:
                Rhot = np.asarray(255*Rhot, dtype=np.int)
                saveData(file_name,"ply",V,F,Rho=Rhot,color=Color)
            else:
                saveData(file_name,"ply",V,F,color=Color)
                
        else:
            saveData(file_name,"ply",V,F)