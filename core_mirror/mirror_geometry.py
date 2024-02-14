import torch
import numpy as np
import matplotlib.pyplot as plt
#from zmq import device

# avoid division by zero
eps = 1e-32 

def generate_p_data(normal=None)->float:
    '''Generate p and p_dash (ankle data) for simulating mirror calibration'''
    
    import random
    # hard code simulation (normal lies on the y-axis)
    n = torch.Tensor([0.5547, 0.0000, 0.8321])

    a,b= 0,9
    y = random.randint(a, b)
    p = torch.Tensor([random.randint(a, b), y, random.randint(a, b)])
    p_dash = torch.Tensor([random.randint(a, b), y , random.randint(a, b)])
    return p, p_dash


def mirror_calibrate_batch(p, p_dash)->float:
    '''Automatically calibrates the mirror scene given position of
    person and its mirrored conterpart
    
    args:
        p: (B, 3) ankle coordinates of real person in 3D space
        p_dash: (B, 3)  ankle coordinates of virtual person in 3D space
        
    return:
        a, n, c, N, d, D, c_dash: calibration outputs
    '''

    B = p.shape[0]
    
    a = (p - p_dash) # diff vec a is not relative to (0,0,0)
    a_norm = torch.norm(a, dim =1).view(-1,1)  + eps
    n_m = torch.div(a, a_norm) # mirror normal

    # position of real camera 
    c = torch.Tensor([[0,0,0]]).repeat(B, 1) # (B, 3)
    # mid-point of p and p_dash
    N = (p_dash+p)/2
    v_d =  N
                                                                     
    
    '''Batch-wise dot-product for scalar projection on to the normal '''
    assert v_d.shape[1] == n_m.shape[1], "the no of elements in both vectors are not the same"
    
    # scalar projection on to the normal 
    d = torch.bmm(v_d.view(B, 1, v_d.shape[1]), n_m.view(B, n_m.shape[1], 1)).view(B, -1) 
    # mirror position 
    D = (c+d*n_m)
    # position of virtual camera 
    c_dash = (c + 2*d*n_m)
    # Ground Plane (mirror vector)
    oth_vector = (N-D)
    otho_norm = torch.norm(oth_vector, dim =1).view(-1,1)  + eps
    n_oth = (oth_vector)/otho_norm

    # rough computation
    n_g_rough = np.cross(n_m, n_oth)
    return a, n_m, c, N, d, D, c_dash, n_g_rough



def mirror_operation_batch(point, n):
    '''Performs the mirror operation along the normal that 
    includes rotation and reflection
    
    args:
        point: (B, 3) any point on the plane to define the location of the plane (in this case, point D in our figure)
        n: normal of the mirror to define the orientation of the plane
        
    return:
        m_mat: (B, 4, 4) mirror matrix 
    
    '''
    
    B = point.shape[0]
    batch_zeros = torch.zeros(B)
    batch_ones = torch.ones(B)
    n1,n2,n3 = n[:,0], n[:,1], n[:,2]; 

    # plane distance d from origin
    # plane_d = -point.dot(n)
    
    assert point.shape[1] == point.shape[1], "the no of elements in both vectors are not the same"
    plane_d = -(torch.bmm(point.view(B, 1, point.shape[1]), n.view(B, n.shape[1], 1)).view(B)) 

    m_mat = torch.cat([
    torch.stack([(1 - (2*n1**2)), (-(2*n1*n2)), (-2*n1*n3), (-(2*n1*plane_d))], dim = 1).view(B, 1, 4), 
    torch.stack([-(2*n1*n2), (1 - (2*n2**2)), -(2*n2*n3), -(2*n2*plane_d)], dim = 1).view(B, 1, 4),
    torch.stack([-(2*n1*n3), -(2*n2*n3), (1 - (2*n3**2)), -(2*n3*plane_d)], dim = 1).view(B, 1, 4),
    torch.stack([batch_zeros, batch_zeros, batch_zeros, batch_ones], dim = 1).view(B, 1, 4),
    ], dim = 1)

    return m_mat, plane_d




def mirror_calibrate(p, p_dash)->float:
    '''Automatically calibrates the mirror scene given position of
    person and its mirrored conterpart
    
    args:
        p: ankle coordinates of real person in 3D space
        p_dash: ankle coordinates of virtual person in 3D space
        
    return:
        a, n, c, N, d, D, c_dash: calibration outputs
    
    '''

    a = (p - p_dash) # diff vec a is not relative to (0,0,0)
    n_m = a/(torch.norm(a) + eps) # mirror normal
    print("a, n_m", a, n_m)
    

    # position of real camera 
    c = torch.Tensor([0,0,0])

    # mid-point of p and p_dash
    N = (p_dash+p)/2
    v_d =  N
    
    # scalar projection on to the normal 
    d = v_d@n_m
    print("d", d)
    
    # mirror position 
    D = (c+d*n_m)
    
    # position of virtual camera 
    c_dash = (c + 2*d*n_m)
        
    # Ground Plane (mirror vector)
    mirr_vector = (N-D)
    n_oth = (mirr_vector)/(torch.norm(mirr_vector) + eps)
    
    # rough computation
    n_g_rough = np.cross(n_m, n_oth)
    
    print("a, n_m, c, N, d, D, c_dash, n_g_rough", a, n_m, c, N, d, D, c_dash, n_g_rough)
    return a, n_m, c, N, d, D, c_dash, n_g_rough



def mirror_operation(point, n):
    '''Performs the mirror operation along the normal that 
    includes rotation and reflection
    
    args:
        point: any point on the plane to define the location of the plane
        n: normal of the mirror to define the orientation of the plane
        
    return:
        m_mat: mirror matrix 
    
    '''
    
    n1,n2,n3 = n; #print("n1", n1)

    # plane distance d from origin
    plane_d = -point.dot(n)

    m_mat = torch.Tensor([[(1 - (2*n1**2)), -(2*n1*n2), -2*n1*n3,-(2*n1*plane_d)],
                          [-(2*n1*n2), (1 - (2*n2**2)), -(2*n2*n3), -(2*n2*plane_d)],
                          [-(2*n1*n3), -(2*n2*n3), (1 - (2*n3**2)), -(2*n3*plane_d)],
                          [0,0,0,1]])
    
    
    return m_mat, plane_d


def create_mirror_plane(point, n_m, length_x=10., length_y=5,
                 length_z=10, n_sample=50):
    # plane distance d from origin
    plane_d = -point.dot(n_m)

    # create y-z plane
    #y, z = np.meshgrid(np.linspace(-y_r, y_r, n_sample), np.linspace(-z_r, z_r, n_sample))
    xx, zz = np.meshgrid(range(n_sample), range(n_sample))
    # calculate corresponding z
    #yy_prime = (-n_m[0].item()*xx  -n_m[1].item()*zz  - plane_d.item()) * 1./(n_m[2].item() + eps)
    yy_prime = (-n_m[0].item()*xx  -n_m[1].item()*zz  - plane_d.item()) * 1./(n_m[2].item() + eps)
    #import ipdb; ipdb.set_trace()
    plane = np.stack([xx, yy_prime, zz], axis=-1)
    return plane.astype(np.float32)





def visualize_sim(params):
    '''Visualizes the symmetrical relationship between entities in the scene using a 
    simulated matplotlib environment'''
    
    p, p_dash, N, D, c, c_dash, n_m, rnd_obj, mirr_obj, plane_d, mirror = params
    

    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111, projection='3d')

    # set aspect ration to have same range for x,y,z axis
    #ax.set_box_aspect(aspect = (1,1,1))


    colors = ['k', 'k', 'g', 'g', 'g', 'g', 'b', 'k', 'k']

    #points
    param_a_to_viz = [p, p_dash, N, D, c, c_dash, n_m, rnd_obj, mirr_obj]
    #lines
    param_b = [p, p_dash, N, D, c, c_dash, n_m]


    # labels
    ax.text(p[0]+0.5, p[1],p[2],'p', size=20, color='black')
    ax.text(p_dash[0]+0.5, p_dash[1],p_dash[2],'p`', size=20, color='black')
    ax.text(N[0]+0.3, N[1],N[2]-0.5,'N', size=20, color='green')
    ax.text(D[0]-1.0, D[1],D[2],'D', size=20, color='green')
    ax.text(c[0]-1.0, c[1],c[2]-1,'c', size=25, color='purple')
    ax.text(c_dash[0]+0.5, c_dash[1],c_dash[2],'c`', size=25, color='purple')

    #ax.annotate('jcvhshch', (p[0], p[1], p[2]) )

    x,y,z = zip(*param_a_to_viz)
    x_,y_,z_ = zip(*param_b)


    # create x,y (me: plane)d
    xx, yy = np.meshgrid(range(10), range(10))

    '''Mirror Plane'''

    # calculate corresponding z
    z_plane = (-n_m[0].item()*xx  -n_m[1].item()*yy  - plane_d.item())
    z_plane = z_plane * 1./(n_m[2].item() + eps)
    print("z_plane", z_plane.shape)

    #plot the surface
    ax.plot_surface(xx, yy, z_plane, alpha=0.3)

    '''GroundPlane'''
    a_g = (N-D)
    n_g = (a_g)/(torch.norm(a_g) + eps)

    # ground plane distance d using plane equation
    ground_d = -p_dash.dot(n_g); print("ground_d",ground_d)

    # calculate corresponding z
    ground_z_plane = (-n_g[0].item()*xx  -n_g[1].item()*yy  - ground_d.item()) 
    ground_z_plane = ground_z_plane * 1./(n_g[2].item() + eps)
    print("ground_z_plane", ground_z_plane.shape)

    #plot the surface
    ax.plot_surface(xx, yy, ground_z_plane, alpha=0.5)


    # plot the orientation of the rnd_obj
    # rnd_obj is relative to c coordinate system
    ax.quiver([rnd_obj[0]], [rnd_obj[1]], [rnd_obj[2]], [rnd_obj[0]], [rnd_obj[1]], [rnd_obj[2]], linewidths = (1,), edgecolor="blue");

    print(mirr_obj, mirr_obj - rnd_obj)
    
    # plot the orientation of the mirr_obj1  
    # remember to reference the mirrored object relative to c_dash coordinate system
    ax.quiver([mirr_obj[0]], [mirr_obj[1]], [mirr_obj[2]], [mirr_obj[0] - c_dash[0]], [mirr_obj[1] - c_dash[1]], [mirr_obj[2] - c_dash[2]], linewidths = (1,), edgecolor="blue");

    diff_v = N-D
    # plot the diff vector from point to point
    ax.quiver([D[0]], [D[1]], [D[2]], [diff_v[0]], [diff_v[1]], [diff_v[2]], linewidths = (1,), edgecolor="blue");


    # Plot scatter of points
    ax.scatter3D(x,y,z, c=colors)
    ax.plot3D(x_,y_,z_)

    ax.set_xlim([-1, 25])
    ax.set_ylim([-1, 25])
    ax.set_zlim([-1, 25])
    
    # mirror midpoint for normal
    mid = (N+D)/2

    # plot the normal vector
    ax.quiver([mid[0]], [mid[1]], [mid[2]], [n_m[0]], [n_m[1]], [n_m[2]], linewidths = (3,), edgecolor="red");


    #plot the x,y,z axis of real cam
    x,y,z = torch.Tensor([1,0,0,1]), torch.Tensor([0,1,0,1]), torch.Tensor([0,0,1,1])

    ax.quiver([c[0]], [c[1]], [c[2]], [x[0]], [x[1]], [x[2]], linewidths = (1,), edgecolor="red");
    ax.quiver([c[0]], [c[1]], [c[2]], [y[0]], [y[1]], [y[2]], linewidths = (1,), edgecolor="green");
    ax.quiver([c[0]], [c[1]], [c[2]], [z[0]], [z[1]], [z[2]], linewidths = (1,), edgecolor="blue");

    # plot the orientation of virtual cam
    x_hat = mirror @ x
    y_hat = mirror @ y
    z_hat = mirror @ z

    # comment: x and z is oriented, this can be verified by looking at the values of the normal itself
    ax.quiver([c_dash[0]], [c_dash[1]], [c_dash[2]], [x_hat[0] - c_dash[0]], [x_hat[1] - c_dash[1]], [x_hat[2] - c_dash[2]], linewidths = (1,), edgecolor="red");
    ax.quiver([c_dash[0]], [c_dash[1]], [c_dash[2]], [y_hat[0] - c_dash[0]], [y_hat[1] - c_dash[1]], [y_hat[2] - c_dash[2]], linewidths = (1,), edgecolor="green");
    ax.quiver([c_dash[0]], [c_dash[1]], [c_dash[2]], [z_hat[0] - c_dash[0]], [z_hat[1] - c_dash[1]], [z_hat[2] - c_dash[2]], linewidths = (1,), edgecolor="blue");




    def axisEqual3D(ax):
        ''' Plot invincible points to solve the unequal aspect ratio '''
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
            
    
    ''' add aspect ratio'''
    #axisEqual3D(ax) 