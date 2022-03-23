import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import os
import open3d as o3d
import copy


def decimate_mesh(V,F,target):    
    """
    Decimates mesh given by V,F to have number of faces approximatelyu equal to target 
    """
    mesh=getMeshFromData([V,F])
    mesh=mesh.simplify_quadric_decimation(target)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    return VS, FS    
    
def subdivide_mesh(V,F,Rho=None,order=1):    
    """
    Performs midpoint  subdivision. Order determines the number of iterations
    """
    mesh=getMeshFromData([V,F],Rho=Rho)
    mesh = mesh.subdivide_midpoint(number_of_iterations=order)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    if Rho is not None:
        RhoS = np.asarray(mesh.vertex_colors,np.float64)[:,0]
        return VS, FS, RhoS
   
    return VS, FS
    
def getDataClosed(V,F,d=9):    
    """
    Get vertex and
    """
    mesh = getMeshFromData([V,F])
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=d)
    V,F,color = getDataFromMesh(mesh)
    return V, F, color    

def getDataFromMesh(mesh):    
    """
    Get vertex and
    """
    V = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    F = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array  
    color=np.zeros((int(np.size(V)/3),0))
    if mesh.has_vertex_colors():
        color=np.asarray(255*np.asarray(mesh.vertex_colors,dtype=np.float64), dtype=np.int)
    return V, F, color
    
def getMeshFromData(mesh,Rho=None, color=None):    
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    V=mesh[0]
    F=mesh[1] 
    #mesh=o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(V),o3d.cpu.pybind.utility.Vector3iVector(F))
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F))
    
    if Rho is not None:
        Rho=np.squeeze(Rho)
        col=np.stack((Rho,Rho,Rho))
        mesh.vertex_colors =  o3d.utility.Vector3dVector(col.T)
        
    if color is not None:
        mesh.vertex_colors =  o3d.utility.Vector3dVector(color)   
    return mesh


def makeGeodMeshes(Geod,F,o_source=None,o_target=None,Rho=None, offset=0, offsetstep=2.5, stepsize=2, axis=[0,0,1], angle=-1*np.pi/2):
    """
    Constructs an open3d mesh for the geodesic
    """
    Nt=len(Geod)
    ls=[]
    ls_Rho=np.array([], np.float64)
    mesh=getMeshFromData([Geod[0],F]).translate((-stepsize,0,0), relative=False)
    R = mesh.get_rotation_matrix_from_axis_angle(np.array(axis)*angle)
    mesh.clear()
    
    if o_source!=None:
        N=o_source[0].shape[0]
        mesh=getMeshFromData(o_source)
        newmesh=copy.deepcopy(mesh)
        newmesh.compute_vertex_normals()
        newmesh.normalize_normals()
        colors_np= np.asarray(newmesh.vertex_normals)
        colors_np= (colors_np+1)/2
        mesh.vertex_colors =  o3d.utility.Vector3dVector(colors_np)
        if Rho is not None:
            ls_Rho=np.append(ls_Rho,np.ones((N,), np.float64))
        mesh.translate((-stepsize,0,offset*offsetstep), relative=False)
        mesh.rotate(R, center=(-stepsize,0,offset*offsetstep))
        ls.append(mesh)    
    for i in range(0,Nt):
        V=Geod[i]
        mesh=getMeshFromData([V,F])
        if Rho is not None:
            t=i/float(Nt-1)
            Rhot=t*Rho+1-t
        if i==0:
            newmesh=copy.deepcopy(mesh)
            newmesh.compute_vertex_normals()
            newmesh.normalize_normals()
            colors_np= np.asarray(newmesh.vertex_normals)
            colors_np= (colors_np+1)/2 
        if Rho is not None:
            mesh.vertex_colors =  o3d.utility.Vector3dVector(np.stack((Rhot,Rhot,Rhot),axis=1)*colors_np)
            ls_Rho=np.append(ls_Rho,Rhot)
        else:
            mesh.vertex_colors =  o3d.utility.Vector3dVector(colors_np)            
        mesh.translate((i*stepsize,0,offset*offsetstep), relative=False)    
        mesh.rotate(R, center=(i*stepsize,0,offset*offsetstep))  
        ls.append(mesh)    
    if o_target!=None:
        N=o_target[0].shape[0]
        mesh=getMeshFromData(o_target)
        newmesh=copy.deepcopy(mesh)
        newmesh.compute_vertex_normals()
        newmesh.normalize_normals()
        colors_np= np.asarray(newmesh.vertex_normals)
        colors_np= (colors_np+1)/2
        mesh.vertex_colors =  o3d.utility.Vector3dVector(colors_np)
        if Rho is not None:
            ls_Rho=np.append(ls_Rho,np.ones((N,), np.float64))
        mesh.translate((Nt*stepsize,0,offset*offsetstep), relative=False)    
        mesh.rotate(R, center=(Nt*stepsize,0,offset*offsetstep))
        ls.append(mesh)           
    if Rho is not None:
        return ls, ls_Rho
    return ls

def LinearInterpolation(source,target,steps):
    geod= [source,target]
    xp=np.linspace(0,1,2,endpoint=True)
    x=np.linspace(0,1,steps,endpoint=True)
    f=scipy.interpolate.interp1d(xp,geod,axis=0)
    midpoints=f(x)
    return midpoints
    