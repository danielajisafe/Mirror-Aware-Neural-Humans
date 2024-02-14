import torch
import itertools
import numpy as np
dtype = torch.float64


def focal_equation_fsolve(vars):
    L, au, av, comb = global_data
    return L[0]*(L[3 + comb[0]]*au[comb[0]] - L[3 + comb[1]]*au[comb[1]]) + L[1]*(L[3 + comb[0]]*av[comb[0]] - L[3 + comb[1]]*av[comb[1]]) + (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))*vars[0]**2

def focal_equation(L, au, av, comb_array):
    focal = 0
    focal_denominator = 0
    for comb in comb_array:
        comb = list(comb)
        #print(comb, " comb!!!")
        focal = focal + (-L[0]*(L[3 + comb[0]]*au[comb[0]] - L[3 + comb[1]]*au[comb[1]]) - L[1]*(L[3 + comb[0]]*av[comb[0]] - L[3 + comb[1]]*av[comb[1]]))
        
        focal_denominator = focal_denominator + (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
        
    print(focal/focal_denominator, " focal before sqrt")
    return np.sqrt(np.absolute(focal/focal_denominator))

def focal_equation_average(L, au, av, comb_array):
    focal = 0
    focal_denominator = 0
    for comb in comb_array:
        comb = list(comb)
        #print(comb, " comb!!!")
        focal = focal + np.sqrt(np.absolute((-L[0]*(L[3 + comb[0]]*au[comb[0]] - L[3 + comb[1]]*au[comb[1]]) - L[1]*(L[3 + comb[0]]*av[comb[0]] - L[3 + comb[1]]*av[comb[1]]))/(L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))))
        
    return focal/len(comb_array)

def focal_equation_homogenous(vars ,*data):
    
    L, au, av, comb = data

    focal = (-L[0]*(L[3 + comb[0]]*au[comb[0]] - L[3 + comb[1]]*au[comb[1]]) - L[1]*(L[3 + comb[0]]*av[comb[0]] - L[3 + comb[1]]*av[comb[1]]))
        
    focal_denominator = (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
    return focal/focal_denominator - vars[0]**2

def focal_equation_homogenous_system(vars):
    
    L, au, av, comb_array = global_data
    focal = 0
    focal_denominator = 0
    focal_equation = []
    
    for comb in comb_array:
        comb = list(comb)
        #print(comb, " comb!!!")
        focal_equation.append(focal_equation_homogenous(vars , *(L, au, av, comb)))
        
        #focal_denominator = focal_denominator + (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
    return focal_equation
    
    
def focal_length_equation(vars, *data):
    L, au, av, comb = data
    return L[0]*(L[3 + comb[0]]*au[comb[0]] - L[3 + comb[1]]*au[comb[1]]) + L[1]*(L[3 + comb[0]]*av[comb[0]] - L[3 + comb[1]]*av[comb[1]]) + (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))*vars[0]**2

def focal_length_system(vars):
    L, au, av, comb_array = global_data
    
    equations = []
    
    for comb in comb_array:
        comb = list(comb)
        equations.append(focal_length_equation(vars, *(L, au, av, comb)))
    
    return equations

def focal_length_sum(vars):
    #print(list(data), " data")
    #print(global_data, " DATA !!!")
    L, au, av, comb_array = global_data
    
    equations = 0
    
    for comb in comb_array:
        comb = list(comb)
        equations = equations + (L[0]*(L[3 + comb[0]]*au[comb[0]] - L[3 + comb[1]]*au[comb[1]]) + L[1]*(L[3 + comb[0]]*av[comb[0]] - L[3 + comb[1]]*av[comb[1]]) + (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))*vars[0]**2)**2
    
    return equations

def calibration_focalpoint_lstq(num_points, hv, av, hu, au, h, t1, t2, true_cam=None, useGTfocal=False):
    C = np.zeros([2*num_points,3 + num_points], dtype = float)

    norm_array = []
    for ind in range(len(au)):
        norm_array.append(np.linalg.norm(np.array([hu[ind],hv[ind]]) - np.array([au[ind],av[ind]])))

    if num_points == 3:
        C[0] = [0, -1, hv[0], hv[0] - av[0],0,0]
        C[1] = [1, 0, -hu[0], au[0] - hu[0],0,0]
        C[2] = [0, -1, hv[1], 0, hv[1] - av[1], 0]
        C[3] = [1, 0, -hu[1], 0, au[1] - hu[1], 0]
        C[4] = [0, -1, hv[2], 0, 0, hv[2] - av[2]]
        C[5] = [1, 0, -hu[2], 0, 0, au[2] - hu[2]]
    else:
        for col in range(0, 2*num_points):
            if col % 2 == 0:
                #print(col)
                C[col][1] = -1
                C[col][2] = hv[int(col/2)]
                C[col][3 + int(col/2)] = hv[int(col/2)] - av[int(col/2)]

                C[col + 1][0] = 1
                C[col + 1][2] = -hu[int(col/2)]
                C[col + 1][3 + int(col/2)] = au[int(col/2)] - hu[int(col/2)]
    
    '''solution L from SVD - first three is the normal, last three is the depth'''
    U, S, Vh = np.linalg.svd(C)
    L = (Vh[-1, :])
    
    '''
    if L[1] < 0: # ENFORCING POSITIVE Y
        L[0] = -1*L[0]
        L[1] = -1*L[1]
        L[2] = -1*L[2]
    '''
    comb_array = []
    for comb in itertools.combinations(range(len(au)), 2):
        comb_array.append(comb)
    
    focal_num = 0
    focal_den = 0
    
    for comb in comb_array:
        comb = list(comb)
        focal_num = focal_num + -((L[0] - L[2]*t1)*(L[3 + comb[0]]*(au[comb[0]] - t1) - L[3 + comb[1]]*(au[comb[1]] - t1)) + (L[1] - L[2]*t2)*(L[3 + comb[0]]*(av[comb[0]] - t2) - L[3 + comb[1]]*(av[comb[1]] - t2)))*(L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))
        
    for comb in comb_array:
        comb = list(comb)
        focal_den = focal_den + (L[2]*(L[3 + comb[0]] - L[3 + comb[1]]))**2
        

    focal_predicted = np.sqrt(np.absolute(focal_num/focal_den))

    # to get nx, ny -> subtract (nz*center) and mulptiply by focal (check paper)
    L[0] = L[0] - t1*L[2]
    L[1] = L[1] - t2*L[2]
    L[2] = L[2]*focal_predicted

    normal = np.array([L[0], L[1], L[2]])
    normal = normal/np.linalg.norm(normal)
    
    if true_cam is not None and useGTfocal:
        true_fx = true_cam[0,0]
        true_fy = true_cam[1,1]
        c_predicted = np.array([[true_fx, 0, t1], [0, true_fy, t2], [0, 0, 1]])

        L[3] = L[3]*h*true_fx
        L[4] = L[4]*h*true_fy
        # use the average
        L[5] = L[5]*h*(true_fx+true_fy)/2
        
    else:
        c_predicted = np.array([[focal_predicted, 0, t1], [0, focal_predicted, t2], [0, 0, 1]])

        L[3] = L[3]*h*focal_predicted
        L[4] = L[4]*h*focal_predicted
        L[5] = L[5]*h*focal_predicted

    depth_Z = np.array([L[3], L[4], L[5]])/np.linalg.norm([L[0],L[1],L[2]])
    return normal, depth_Z, focal_predicted, c_predicted
