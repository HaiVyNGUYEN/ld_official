import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
from scipy.spatial import  distance_matrix
from scipy.sparse.csgraph import shortest_path
import random


def lens_depth_function(dist_mat):
    assert np.array_equal(np.array(dist_mat), np.array(dist_mat).T)
    N = np.shape(dist_mat)[0]
    norm_factor = N*(N-1)/2
    lens_depth = np.zeros(N)
    for j in range(N):
        for k in range(j):
            sphere_1 = dist_mat[j,:] - dist_mat[j,k]
            sphere_2 = dist_mat[k,:] - dist_mat[j,k]
            lens_depth += (sphere_1<=0).astype(int)*(sphere_2<=0).astype(int)
    lens_depth_norm = lens_depth/norm_factor
    return lens_depth_norm


def fermat_function(dist_mat, alpha=3,path_method='FW'):
    """
    This function is to calculate pair-wise fermat distance matrix for the inner points,
    from this pair-wise euclidean distance matrix.
    """
    assert np.array_equal(np.array(dist_mat), np.array(dist_mat).T)
    fermat_dist_exact = shortest_path(csgraph=np.matrix(np.power(dist_mat, alpha)),\
                                    method=path_method,directed=False)
    return fermat_dist_exact

def fermat_function_ext(fermat_dist_mat, dist_array,alpha=3): #
    """
    This function is to calculate fermat distance from an ext point to all the inner points.
    dist_array is the euclidean distance from the ext point to innerpoints.
    fermat_dist_mat is the pair-wise fermat distance matrix of the inner points.
    """
    assert np.array_equal(np.array(fermat_dist_mat), np.array(fermat_dist_mat).T)
    assert (np.shape(fermat_dist_mat)[0] == len(np.squeeze(dist_array)))
    N = np.shape(fermat_dist_mat)[0]
    dist_array_tile = np.tile(np.squeeze(dist_array).reshape(-1)**alpha,(N,1))
    D = fermat_dist_mat + dist_array_tile
    d = np.min(D,axis=1)
    return d

def fermat_function_ext_simultaneous(fermat_dist_mat, dist_array,alpha=3,device='gpu'):
    """"
    similar as the function fermat_function_ext above but for mutilple ext points simultaneously.
    dist_array of shape (p,N) p is No. of ext points, N inner points (euclidean distance).
    fermat_dist_mat is the pair-wise fermat distance matrix of the inner points.
    """
    if device =='gpu':
        fermat_dist_mat_gpu = torch.tensor(fermat_dist_mat)
        dist_array_gpu = torch.tensor(dist_array)
        assert torch.equal(fermat_dist_mat_gpu, fermat_dist_mat_gpu.T)
        assert (fermat_dist_mat_gpu.size(0) == dist_array_gpu.size(1))
        N = fermat_dist_mat_gpu.size(0)
        with torch.no_grad():
            dist_array_tile = torch.tile(dist_array_gpu**alpha,(N,1,1))
            dist_array_tile = dist_array_tile.permute(1, 0, 2)
            D = fermat_dist_mat_gpu + dist_array_tile
            d, _ = torch.min(D,dim=2)
        return d.cpu().numpy()
    else:  
        assert np.array_equal(np.array(fermat_dist_mat), np.array(fermat_dist_mat).T)
        assert (np.shape(fermat_dist_mat)[0] == np.shape(dist_array)[1])
        N = np.shape(fermat_dist_mat)[0]
        dist_array_tile = np.tile(dist_array**alpha,(N,1,1))
        dist_array_tile = dist_array_tile.transpose(1, 0, 2)
        D = fermat_dist_mat + dist_array_tile
        d = np.min(D,axis=2)
        return d
    
def lens_depth_ext_function(dist_mat, dist_ext):
    "This function is to calculate LD of an ext point w.r.t. a cluster of inner points"
    assert np.array_equal(np.array(dist_mat), np.array(dist_mat).T)
    assert (np.shape(dist_mat)[0] == len(dist_ext))
    N = np.shape(dist_mat)[0]
    norm_factor = N*(N-1)/2
    dist_ext_tile = np.tile(np.array(dist_ext).reshape(-1),(N,1))
    dist_mat_x1 = ((dist_mat - dist_ext_tile)>=0).astype(int)
    dist_mat_x2 = ((dist_mat - dist_ext_tile.T)>=0).astype(int)
    return 0.5*np.sum(dist_mat_x1*dist_mat_x2)/norm_factor

def lens_depth_ext_function_simultaneous(dist_mat, dist_ext,device='gpu'): 
    """"
    similar as the function lens_depth_ext_function above but for mutilple ext points simultaneously.
    shape dist_ext = (p,N) p is No. of ext points, N inner points
    dist_mat is the pair-wise distance matrix of the inner points.
    """
    if device == 'gpu':
        dist_mat_gpu = torch.tensor(dist_mat)
        dist_ext_gpu = torch.tensor(dist_ext)
        assert torch.equal(dist_mat_gpu, dist_mat_gpu.T)
        assert (dist_mat_gpu.size(0) == dist_ext_gpu.size(1))
        N = dist_mat_gpu.size(0)
        norm_factor = N*(N-1)/2
        with torch.no_grad():
            dist_ext_tile = torch.tile(dist_ext_gpu,(N,1,1))
            dist_ext_tile = dist_ext_tile.permute(1, 0, 2)
            dist_mat_x1 = ((dist_mat_gpu - dist_ext_tile)>=0).int()
            dist_mat_x2 = ((dist_mat_gpu - dist_ext_tile.permute(0, 2, 1))>=0).int()
        LD =  (0.5*torch.sum(dist_mat_x1*dist_mat_x2, dim=(1,2))/norm_factor).cpu().numpy()
        return LD
    else:
        assert np.array_equal(np.array(dist_mat), np.array(dist_mat).T)
        assert (np.shape(dist_mat)[0] == np.shape(dist_ext)[1])
        N = np.shape(dist_mat)[0]
        norm_factor = N*(N-1)/2
        dist_ext_tile = np.tile(dist_ext,(N,1,1))
        dist_ext_tile = dist_ext_tile.transpose(1, 0, 2)
        dist_mat_x1 = ((dist_mat - dist_ext_tile)>=0).astype(int)
        dist_mat_x2 = ((dist_mat - dist_ext_tile.transpose(0, 2, 1))>=0).astype(int)
        return 0.5*np.sum(dist_mat_x1*dist_mat_x2, axis=(1,2))/norm_factor
    
