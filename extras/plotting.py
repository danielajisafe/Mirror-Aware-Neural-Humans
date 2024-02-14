
import ipdb
import torch
from tkinter import E
import matplotlib.pyplot as plt


def get_2d_axis(joints_position3d, joints_orient, camera):
    N_J = joints_position3d.shape[0]
    
    
    x_vec =  joints_position3d + joints_orient[:,0]
    y_vec =  joints_position3d + joints_orient[:,1]
    z_vec =  joints_position3d + joints_orient[:,2]
    #import ipdb; ipdb.set_trace()
    xvec_2d = torch.bmm(camera.view(1,3,3).repeat(N_J,1,1), x_vec.view(N_J, 3, 1))
    xvec_2d = torch.div(xvec_2d, xvec_2d[:, 2:3])[:,0:2].view(N_J,-1)

    yvec_2d = torch.bmm(camera.view(1,3,3).repeat(N_J,1,1), y_vec.view(N_J, 3, 1))
    yvec_2d = torch.div(yvec_2d, yvec_2d[:, 2:3])[:,0:2].view(N_J,-1)
    
    zvec_2d = torch.bmm(camera.view(1,3,3).repeat(N_J,1,1), z_vec.view(N_J, 3, 1))
    zvec_2d = torch.div(zvec_2d, zvec_2d[:, 2:3])[:,0:2].view(N_J,-1)
    
    return xvec_2d, yvec_2d, zvec_2d

def plot_bone_orient(joint_position2d, xvec_2d, yvec_2d, zvec_2d, ax = plt,linewidth = 2, alpha = 1.0): 
    """Plot the orientation (x,y and z axis) of the bone
    args:
        joint_position2d: shape (N_J, 2) 
        xvec_2d: shape (N_J, 2)
        yvec_2d: shape (N_J, 2)
        zvec_2d: shape (N_J, 2)
    """
    print("plotting joints axis")
    for joint, xvec, yvec, zvec  in zip(joint_position2d, xvec_2d, yvec_2d, zvec_2d):
        ax.plot([joint[0], xvec[0]], [joint[1],xvec[1]], linewidth=linewidth, color="white")

def plotPoseOnImage(poses, img=None, ax = plt, title_text = None, color = 'r', alpha=1.0):
    kps= torch.cat([poses[:,0].view(1,-1), poses[:,1].view(1,-1)])

    ax.scatter(*kps.numpy(), c=color, alpha = alpha,s=10)
    try:
        ax.set_title(title_text)
    except:
        pass
    if img is not None:
        ax.imshow(img)
    

def plot_multiple_views(pose_pred, poses_gt, imgs, figsize = None):

    plt.figure()
    f, axarr = plt.subplots(2,4, figsize=figsize) 

    # use the created array to output your multiple images. 
    plotPoseOnImage(pose_pred[0].detach().cpu(), imgs[0].squeeze(0), ax = axarr[0,0], title_text= "View 0 (pred)")
    plotPoseOnImage(poses_gt[0].detach().cpu(), imgs[0].squeeze(0), ax = axarr[0,1], title_text= "View 0 (gt)")
    
    plotPoseOnImage(pose_pred[1].detach().cpu(), imgs[1].squeeze(0), ax = axarr[0,2], title_text= "View 1 (pred)")
    plotPoseOnImage(poses_gt[1].detach().cpu(), imgs[1].squeeze(0), ax = axarr[0,3], title_text= "View 1 (gt)")
    
    plotPoseOnImage(pose_pred[2].detach().cpu(), imgs[2].squeeze(0), ax = axarr[1,0], title_text= "View 2 (pred)")
    plotPoseOnImage(poses_gt[2].detach().cpu(), imgs[2].squeeze(0), ax = axarr[1,1], title_text= "View 2 (gt)")

    plotPoseOnImage(pose_pred[3].detach().cpu(), imgs[3].squeeze(0), ax = axarr[1,2], title_text= "View 3 (pred)")
    plotPoseOnImage(poses_gt[3].detach().cpu(), imgs[3].squeeze(0), ax = axarr[1,3], title_text= "View 3 (gt)")

    
def plot_raw_h36m(x,y,z, ax = plt, color = 'r', linestyle = "--", alpha = 1.0):
    
    ''' Bast - Plotting based on raw h36m joint ordering '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    
    '''x,y,z values'''
    
    ax.scatter3D(x, y, z,  c=['#ff0000'], alpha=alpha,s=10)
    
    for i in range(x.shape[0]):
        ax.text(x[i],y[i],z[i],str(i))

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    plt.xlabel('x')
    plt.ylabel('y')
    return plt

    
    
def plot15j_3d(x,y,z, ax = plt, color = 'r', linestyle = "--", alpha = 1.0):
    
    ''' Bast - Plotting based on DCPose joint ordering '''
    
    import matplotlib as mpl
    import numpy as np
    import matplotlib.animation as anim
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    
    '''x,y,z values'''
    ax.scatter3D(x, y, z,  c=['#ff0000'], alpha=alpha,s=10)

    # middle bones
    ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], linestyle=linestyle, color="black")
    ax.plot(x[([0, 2])], y[([0, 2])], z[([0, 2])], linestyle=linestyle, color="black")
    
    # left bones
    ax.plot(x[([3, 1])], y[([3, 1])], z[([3, 1])], linestyle=linestyle, color="green")
    ax.plot(x[([5, 3])], y[([5, 3])], z[([5, 3])], linestyle=linestyle, color="green")
    ax.plot(x[([7, 5])], y[([7, 5])], z[([7, 5])], linestyle=linestyle, color="green")
    ax.plot(x[([9, 3])], y[([9, 3])], z[([9, 3])], linestyle=linestyle, color="green")
    ax.plot(x[([11, 9])], y[([11, 9])], z[([11, 9])], linestyle=linestyle, color="green")
    ax.plot(x[([13, 11])], y[([13, 11])], z[([13, 11])], linestyle=linestyle, color="green")
    
    # right bones
    ax.plot(x[([4, 1])], y[([4, 1])], z[([4, 1])], linestyle=linestyle, color="red")
    ax.plot(x[([6, 4])], y[([6, 4])], z[([6, 4])], linestyle=linestyle, color="red")
    ax.plot(x[([8, 6])], y[([8, 6])], z[([8, 6])], linestyle=linestyle, color="red")
    ax.plot(x[([10, 4])], y[([10, 4])], z[([10, 4])], linestyle=linestyle, color="red")
    ax.plot(x[([12, 10])], y[([12, 10])], z[([12, 10])], linestyle=linestyle, color="red")
    ax.plot(x[([14, 12])], y[([14, 12])], z[([14, 12])], linestyle=linestyle, color="red")

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = 3 # np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.xlabel('x')
    plt.ylabel('y')

    return plt



def plot15j_2d(x,y, img, ax=plt, title = "", linestyle = "-", linewidth =2, alpha = 0.5, show=True, color=None, true=None, plot_true=False):

    '''x values and y values
    - Plotting based on DCPose joint ordering
    '''

    if color is not None:
        #middle bones
        ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle,linewidth=linewidth, color= color)
        ax.plot(x[([0, 2])], y[([0, 2])], linestyle=linestyle,linewidth=linewidth, color=color)
        # left bones
        ax.plot(x[([3, 1])], y[([3, 1])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([5, 3])], y[([5, 3])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([7, 5])], y[([7, 5])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([9, 3])], y[([9, 3])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([11, 9])], y[([11, 9])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([13, 11])], y[([13, 11])], linestyle=linestyle,linewidth=linewidth, color=color)
        # right bones
        ax.plot(x[([4, 1])], y[([4, 1])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([6, 4])], y[([6, 4])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([8, 6])], y[([8, 6])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([10, 4])], y[([10, 4])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([12, 10])], y[([12, 10])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([14, 12])], y[([14, 12])], linestyle=linestyle,linewidth=linewidth, color=color)#middle bones

    else:

        #middle bones
        ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle,linewidth=linewidth, color="black")
        ax.plot(x[([0, 2])], y[([0, 2])], linestyle=linestyle,linewidth=linewidth, color="black")
        
        
        # left bones
        ax.plot(x[([3, 1])], y[([3, 1])], linestyle=linestyle,linewidth=linewidth, color="green")
        ax.plot(x[([5, 3])], y[([5, 3])], linestyle=linestyle,linewidth=linewidth, color="green")
        ax.plot(x[([7, 5])], y[([7, 5])], linestyle=linestyle,linewidth=linewidth, color="green")
        ax.plot(x[([9, 3])], y[([9, 3])], linestyle=linestyle,linewidth=linewidth, color="green")
        ax.plot(x[([11, 9])], y[([11, 9])], linestyle=linestyle,linewidth=linewidth, color="green")
        ax.plot(x[([13, 11])], y[([13, 11])], linestyle=linestyle,linewidth=linewidth, color="green")
        
        # right bones
        ax.plot(x[([4, 1])], y[([4, 1])], linestyle=linestyle,linewidth=linewidth, color="red")
        ax.plot(x[([6, 4])], y[([6, 4])], linestyle=linestyle,linewidth=linewidth, color="red")
        ax.plot(x[([8, 6])], y[([8, 6])], linestyle=linestyle,linewidth=linewidth, color="red")
        ax.plot(x[([10, 4])], y[([10, 4])], linestyle=linestyle,linewidth=linewidth, color="red")
        ax.plot(x[([12, 10])], y[([12, 10])], linestyle=linestyle,linewidth=linewidth, color="red")
        ax.plot(x[([14, 12])], y[([14, 12])], linestyle=linestyle,linewidth=linewidth, color="red")


    try:
        ax.title.set_text(title)
    except:
        pass
    ax.imshow(img)


def plot2d_halpe26(x,y, img, ax=plt, title = "", linestyle = "-", linewidth=2, alpha = 0.5, show=True, color=None, true=None, plot_true=False):
    "plot alphapose in its raw form"

    if true is not None:
        true_x, true_y = true
        size =15
        colors=["red","green","teal"]
        ax.scatter([true_x[0:1]], true_y[0:1], c=colors[2], alpha=alpha,s=size)
        ax.scatter([true_x[1:2]], true_y[1:2], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[2:3]], true_y[2:3], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[3:4]], true_y[3:4], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[4:5]], true_y[4:5], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[5:6]], true_y[5:6], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[6:7]], true_y[6:7], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[7:8]], true_y[7:8], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[8:9]], true_y[8:9], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[9:10]], true_y[9:10], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[10:11]], true_y[10:11], c=colors[0], alpha=alpha,s=size)

        ax.scatter([true_x[11:12]], true_y[11:12], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[12:13]], true_y[12:13], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[13:14]], true_y[13:14], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[14:15]], true_y[14:15], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[15:16]], true_y[15:16], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[16:17]], true_y[16:17], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[17:18]], true_y[17:18], c=colors[2], alpha=alpha,s=size)
        ax.scatter([true_x[18:19]], true_y[18:19], c=colors[2], alpha=alpha,s=size)
        ax.scatter([true_x[19:20]], true_y[19:20], c=colors[2], alpha=alpha,s=size)
        ax.scatter([true_x[20:21]], true_y[20:21], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[21:22]], true_y[21:22], c=colors[0], alpha=alpha,s=size)

        ax.scatter([true_x[22:23]], true_y[22:23], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[23:24]], true_y[23:24], c=colors[0], alpha=alpha,s=size)
        ax.scatter([true_x[24:25]], true_y[24:25], c=colors[1], alpha=alpha,s=size)
        ax.scatter([true_x[25:26]], true_y[25:26], c=colors[0], alpha=alpha,s=size)
  
    def bones(x,y,colors,linewidth=linewidth):

        color = colors[0]
        ax.plot(x[([0, 2])], y[([0, 2])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([2, 4])], y[([2, 4])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([6, 18])], y[([6, 18])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([6, 8])], y[([6, 8])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([8, 10])], y[([8, 10])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([19, 12])], y[([19, 12])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([12, 14])], y[([12, 14])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([14, 16])], y[([14, 16])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([16, 25])], y[([16, 25])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([21, 25])], y[([21, 25])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([23, 25])], y[([23, 25])], linestyle=linestyle,linewidth=linewidth, color=color) 
        
        color = colors[1]
        ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle,linewidth=linewidth, color= color)
        ax.plot(x[([1, 3])], y[([1, 3])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([5, 18])], y[([5, 18])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([5, 7])], y[([5, 7])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([7, 9])], y[([7, 9])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([19, 11])], y[([19, 11])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([11, 13])], y[([11, 13])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([13, 15])], y[([13, 15])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([20, 24])], y[([20, 24])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([22, 24])], y[([22, 24])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([15, 24])], y[([15, 24])], linestyle=linestyle,linewidth=linewidth, color=color)
        
        color = colors[2]
        ax.plot(x[([17, 18])], y[([17, 18])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([18, 19])], y[([18, 19])], linestyle=linestyle,linewidth=linewidth, color=color)
        
    
    colors=["red","green","teal"]
    bones(x, y, colors)

    try:
        ax.title.set_text(title)
    except:
        pass

def plot2d_combset26_2D(x, y, img=None, bone_connect_idxs=None, ax=plt, title = "", linestyle = "-", 
                        linewidth=2, alpha = 0.5, show=True, colors=None, true=None, 
                        plot_true=False, yes=False, size=15):
    """plot combined GT2D+alphapose+head in raw format"""

    if colors is None:
            colors=["red","green","teal"]
    # Right
    color = colors[0]
    ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([1, 2])], y[([1, 2])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([2, 3])], y[([2, 3])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([3, 4])], y[([3, 4])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([4, 6])], y[([4, 6])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([4, 5])], y[([4, 5])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([13, 23])], y[([13, 23])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([23, 24])], y[([23, 24])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([24, 25])], y[([24, 25])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([14, 15])], y[([14, 15])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([15, 16])], y[([15, 16])], linestyle=linestyle,linewidth=linewidth, color=color)

    # Left
    color = colors[1]
    ax.plot(x[([0, 7])], y[([0, 7])], linestyle=linestyle,linewidth=linewidth, color= color)
    ax.plot(x[([7, 8])], y[([7, 8])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([8, 9])], y[([8, 9])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([9, 10])], y[([9, 10])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([10, 11])], y[([10, 11])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([10, 12])], y[([10, 12])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([14, 17])], y[([14, 17])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([17, 18])], y[([17, 18])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([13, 20])], y[([13, 20])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([20, 21])], y[([20, 21])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([21, 22])], y[([21, 22])], linestyle=linestyle,linewidth=linewidth, color=color)

    # Middle
    color = colors[2]
    ax.plot(x[([0, 13])], y[([0, 13])], linestyle=linestyle,linewidth=linewidth, color=color)
    ax.plot(x[([13, 19])], y[([13, 19])], linestyle=linestyle,linewidth=linewidth, color=color)

    if len(title) !=0:
        ax.title.set_text(title)


def plot2d_halpe26_mirror_common_2D(x=None,y=None, img=None, ax=plt, title = "", linestyle = "-", plot_detect=None, linewidth=2, alpha = 0.5, show=True, colors=None, true=None, plot_true=False, yes=False):
    """plot alphapose or mirror19 in common format"""
    
    size =15
    if colors is None:
            colors=["red","green","teal"]
    
    if true is not None:
        true_x, true_y = true
        
        
        ax.scatter(true_x[15:16], true_y[15:16], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[17:18], true_y[17:18], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[2:3], true_y[2:3], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[3:4], true_y[3:4], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[4:5], true_y[4:5], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[9:10], true_y[9:10], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[10:11], true_y[10:11], c=colors[0], alpha=alpha,s=size)
        ax.scatter(true_x[11:12], true_y[11:12], c=colors[0], alpha=alpha,s=size)
        try: # mirror19 common exception
            ax.scatter(true_x[23:24], true_y[23:24], c=colors[0], alpha=alpha,s=size)
            ax.scatter(true_x[22:23], true_y[22:23], c=colors[0], alpha=alpha,s=size)
            ax.scatter(true_x[24:25], true_y[24:25], c=colors[0], alpha=alpha,s=size)
        except:
            pass

        ax.scatter(true_x[16:17], true_y[16:17], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[18:19], true_y[18:19], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[5:6], true_y[5:6], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[6:7], true_y[6:7], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[7:8], true_y[7:8], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[12:13], true_y[12:13], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[13:14], true_y[13:14], c=colors[1], alpha=alpha,s=size)
        ax.scatter(true_x[14:15], true_y[14:15], c=colors[1], alpha=alpha,s=size)
        try: # mirror19 common exception
            ax.scatter(true_x[21:22], true_y[21:22], c=colors[1], alpha=alpha,s=size)
            ax.scatter(true_x[20:21], true_y[20:21], c=colors[1], alpha=alpha,s=size)
            ax.scatter(true_x[19:20], true_y[19:20], c=colors[1], alpha=alpha,s=size)
        except:
            pass
  

    def bones(x,y,colors,linewidth=linewidth):

        color = colors[0]

        ax.plot(x[([0, 15])], y[([0, 15])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([15, 17])], y[([15, 17])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([1, 2])], y[([1, 2])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([2, 3])], y[([2, 3])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([3, 4])], y[([3, 4])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([8, 9])], y[([8, 9])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([9, 10])], y[([9, 10])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([10, 11])], y[([10,11])], linestyle=linestyle,linewidth=linewidth, color=color)
        
        try:
            ax.plot(x[([11, 24])], y[([11,24])], linestyle=linestyle,linewidth=linewidth, color=color)
            ax.plot(x[([23, 24])], y[([23, 24])], linestyle=linestyle,linewidth=linewidth, color=color)
            ax.plot(x[([22, 24])], y[([22, 24])], linestyle=linestyle,linewidth=linewidth, color=color)
        except:
            pass
            
        color = colors[1]
        ax.plot(x[([0, 16])], y[([0, 16])], linestyle=linestyle,linewidth=linewidth, color= color)
        ax.plot(x[([16, 18])], y[([16, 18])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([1, 5])], y[([1, 5])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([5, 6])], y[([5, 6])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([6, 7])], y[([6, 7])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([8, 12])], y[([8, 12])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([12, 13])], y[([12, 13])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([13, 14])], y[([13, 14])], linestyle=linestyle,linewidth=linewidth, color=color)
        
        try:
            ax.plot(x[([14, 21])], y[([14, 21])], linestyle=linestyle,linewidth=linewidth, color=color)
            ax.plot(x[([20, 21])], y[([20, 21])], linestyle=linestyle,linewidth=linewidth, color=color)
            ax.plot(x[([19, 21])], y[([19, 21])], linestyle=linestyle,linewidth=linewidth, color=color)
        except:
            pass
        
        color = colors[2]
        ax.plot(x[([1, 8])], y[([1, 8])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([1, 0])], y[([1, 0])], linestyle=linestyle,linewidth=linewidth, color=color)

    if plot_detect:
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], c="red", alpha=alpha,s=size)

    if x !=None and y!=None:
        bones(x, y, colors)

    try:
        ax.title.set_text(title)
    except:
        pass


def plot2d_halpe26_mirror_common_3D(x,y,z, ax=plt, title = "", linestyle = "-", linewidth=2, alpha = 0.5, show=True, color=None, true=None, plot_true=False):
    """plot alphapose in mirror_common format"""
    import matplotlib as mpl
    import numpy as np
    import matplotlib.animation as anim
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    if true is not None:
        true_x, true_y, true_z = true
        size =15
        colors=["red","green","teal"]
        
        ax.scatter3D([true_x[15:16]], true_y[15:16], true_z[15:16], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[17:18]], true_y[17:18], true_z[17:18], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[2:3]], true_y[2:3], true_z[2:3], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[3:4]], true_y[3:4], true_z[3:4], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[4:5]], true_y[4:5], true_z[4:5], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[9:10]], true_y[9:10], true_z[9:10], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[10:11]], true_y[10:11], true_z[10:11], c=colors[0], alpha=alpha,s=size)
        ax.scatter3D([true_x[11:12]], true_y[11:12], true_z[11:12], c=colors[0], alpha=alpha,s=size)


        ax.scatter3D([true_x[16:17]], true_y[16:17], true_z[16:17], c=colors[1], alpha=alpha,s=size)
        ax.scatter3D([true_x[18:19]], true_y[18:19], true_z[18:19], c=colors[1], alpha=alpha,s=size)
        ax.scatter3D([true_x[5:6]], true_y[5:6], true_z[5:6], c=colors[1], alpha=alpha,s=size)
        ax.scatter3D([true_x[6:7]], true_y[6:7], true_z[6:7], c=colors[1], alpha=alpha,s=size)
        ax.scatter3D([true_x[12:13]], true_y[12:13], true_z[12:13], c=colors[1], alpha=alpha,s=size)
        ax.scatter3D([true_x[13:14]], true_y[13:14], true_z[13:14], c=colors[1], alpha=alpha,s=size)
        ax.scatter3D([true_x[14:15]], true_y[14:15], true_z[14:15], c=colors[1], alpha=alpha,s=size)
  
        ax.scatter3D([true_x[0:1]], true_y[0:1], true_z[0:1], c=colors[2], alpha=alpha,s=size)
        ax.scatter3D([true_x[8:9]], true_y[8:9], true_z[8:9], c=colors[2], alpha=alpha,s=size)
        ax.scatter3D([true_x[1:2]], true_y[1:2], true_z[1:2], c=colors[2], alpha=alpha,s=size)
  
    def bones(x,y,z,colors,linewidth=linewidth):

        color = colors[0]
        ax.plot(x[([0, 15])], y[([0, 15])], z[([0, 15])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([15, 17])], y[([15, 17])], z[([15, 17])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([10, 11])], y[([10,11])], z[([10,11])], linestyle=linestyle,linewidth=linewidth, color=color)

        color = colors[1]
        ax.plot(x[([0, 16])], y[([0, 16])], z[([0, 16])], linestyle=linestyle,linewidth=linewidth, color= color)
        ax.plot(x[([16, 18])], y[([16, 18])], z[([16, 18])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([1, 5])], y[([1, 5])], z[([1, 5])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([8, 12])], y[([8, 12])], z[([8, 12])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])], linestyle=linestyle,linewidth=linewidth, color=color)
 
        color = colors[2]
        ax.plot(x[([1, 8])], y[([1, 8])], z[([1, 8])], linestyle=linestyle,linewidth=linewidth, color=color)
        ax.plot(x[([1, 0])], y[([1, 0])], z[([1, 0])], linestyle=linestyle,linewidth=linewidth, color=color)
        
    
    colors=["red","green","teal"]
    bones(x, y, z, colors)

    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.numpy().max() + x.numpy().min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.numpy().max() + y.numpy().min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.numpy().max() + z.numpy().min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        ax.title.set_text(title)
    except:
        pass



def plot_dc_15j_2d(x,y, color, ax=plt, title = "", linestyle = "-", linewidth =3, alpha = 0.2):
    '''x values and y values
    - Plotting based on DCPose joint ordering
    '''
    
    ax.scatter(x, y, c="yellow", alpha=alpha,s=10) #c=['#ff0000']
    
    #middle bones
    ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle,linewidth=linewidth, color="black")
    ax.plot(x[([0, 2])], y[([0, 2])], linestyle=linestyle,linewidth=linewidth, color="black")
    
    # left bones
    ax.plot(x[([3, 1])], y[([3, 1])], linestyle=linestyle,linewidth=linewidth, color="green")
    ax.plot(x[([5, 3])], y[([5, 3])], linestyle=linestyle,linewidth=linewidth, color="green")
    ax.plot(x[([7, 5])], y[([7, 5])], linestyle=linestyle,linewidth=linewidth, color="green")
    ax.plot(x[([9, 3])], y[([9, 3])], linestyle=linestyle,linewidth=linewidth, color="green")
    ax.plot(x[([11, 9])], y[([11, 9])], linestyle=linestyle,linewidth=linewidth, color="green")
    ax.plot(x[([13, 11])], y[([13, 11])], linestyle=linestyle,linewidth=linewidth, color="green")
    
    # right bones
    ax.plot(x[([4, 1])], y[([4, 1])], linestyle=linestyle,linewidth=linewidth, color="red")
    ax.plot(x[([6, 4])], y[([6, 4])], linestyle=linestyle,linewidth=linewidth, color="red")
    ax.plot(x[([8, 6])], y[([8, 6])], linestyle=linestyle,linewidth=linewidth, color="red")
    ax.plot(x[([10, 4])], y[([10, 4])], linestyle=linestyle,linewidth=linewidth, color="red")
    ax.plot(x[([12, 10])], y[([12, 10])], linestyle=linestyle,linewidth=linewidth, color="red")
    ax.plot(x[([14, 12])], y[([14, 12])], linestyle=linestyle,linewidth=linewidth, color="red")
    
    try:
        ax.title.set_text(title)
    except:
        pass 

    
def plot_2d_grouped(x,y, bones_grouped, joint_color="yellow", colors=["red","green","teal"], ax=plt, title = "", linestyle = "-", linewidth =3, alpha = 0.2, true=None, plot_true=False):
    '''x values and y values
    - Plotting based on DCPose joint ordering
    '''

    if true is not None:
        true_x, true_y = true
        ax.scatter(true_x, true_y, c="yellow", alpha=alpha,s=10) 
    else:
        ax.scatter(x, y, c="yellow", alpha=alpha,s=10) 
    # middle bones
    for i, group in enumerate(bones_grouped):
        for bone in group:
            ax.plot(x[(bone)], y[(bone)], linestyle=linestyle, linewidth=linewidth, color=colors[i])
            
def plot_h36m_2d(x,y, color, ax=plt, title = "", linestyle = "-", linewidth =3, alpha = 0.2):
    '''x values and y values
    - Bast - Plotting based on DCPose joint ordering
    '''
    
    ax.scatter(x, y, c="yellow", alpha=alpha) 

    joint_names_H36M = ["pelvis","right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
                   "left_ankle", "spine1", "neck", "nose", "head", "left_shoulder", "left_elbow", 
                   "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]

    bones_h36m = [["pelvis","right_hip"],["right_hip", "right_knee"],["right_knee", "right_ankle"],
              ["pelvis","left_hip"],["left_hip", "left_knee"],["left_knee", "left_ankle"],
              ["pelvis","spine1"],["spine1","neck"],["nose","neck"],["nose","head"],
              ["neck","right_shoulder"],["right_shoulder","right_elbow"], ["right_elbow","right_wrist"],
              ["neck","left_shoulder"],["left_shoulder","left_elbow"], ["left_elbow","left_wrist"]]
    bones_h36m_indices = [[joint_names_H36M.index(n1),joint_names_H36M.index(n2)] for (n1,n2) in bones_h36m]

    #middle bones
    for bone in bones_h36m_indices:
        ax.plot(x[(bone)], y[(bone)], linestyle=linestyle,linewidth=linewidth, color="black")
    
    try:
        ax.title.set_text(title)
    except:
        pass 
        
    
def plot15j_2d_uniform(x,y, img, color, ax=plt, title = "", linestyle = "--", linewidth = 3,alpha = 0.2):
    
    '''x values and y values'''
    ax.scatter(x, y, c="yellow", alpha=alpha,s=10) 
    
    # uniform
    ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle, linewidth=linewidth, color="red")
    ax.plot(x[([1 ,2])], y[([1, 2])], linestyle=linestyle, linewidth=linewidth, color="red")
    ax.plot(x[([2, 3])], y[([2, 3])], linestyle=linestyle,  linewidth=linewidth,color="red")
    ax.plot(x[([7, 13])], y[([7, 13])], linestyle=linestyle,  linewidth=linewidth,color="red")
    ax.plot(x[([13, 14])], y[([13, 14])], linestyle=linestyle, linewidth=linewidth, color="red")
    ax.plot(x[([14, 15])], y[([14, 15])], linestyle=linestyle, linewidth=linewidth, color="red")
    
    ax.plot(x[([0, 4])], y[([0, 4])], linestyle=linestyle,  linewidth=linewidth,color="green")
    ax.plot(x[([4, 5])], y[([4, 5])], linestyle=linestyle, linewidth=linewidth, color="green")
    ax.plot(x[([5, 6])], y[([5, 6])], linestyle=linestyle, linewidth=linewidth,color="green")
    ax.plot(x[([7, 10])], y[([7, 10])], linestyle=linestyle, linewidth=linewidth, color="green")
    ax.plot(x[([10, 11])], y[([10, 11])], linestyle=linestyle, linewidth=linewidth, color="green")
    ax.plot(x[([11, 12])], y[([11, 12])], linestyle=linestyle, linewidth=linewidth, color="green")
    
    ax.plot(x[([0, 7])], y[([0, 7])], linestyle=linestyle, linewidth=linewidth, color="dodgerblue")
    ax.plot(x[([7, 8])], y[([7, 8])], linestyle=linestyle, linewidth=linewidth, color="dodgerblue")
    ax.plot(x[([8, 9])], y[([8, 9])], linestyle=linestyle, linewidth=linewidth, color="dodgerblue")

    try:
        ax.title.set_text(title)
    except:
        pass
    
    ax.imshow(img)
    
    
def plot15j_3d_uniform(x,y, img, color, ax=plt, title = "", linestyle = "--", alpha = 0.2):
    
    '''x values and y values
    '''
    ax.scatter(x, y, c="yellow", alpha=alpha,s=10)
    
    # uniform
    ax.plot(x[([0, 1])], y[([0, 1])], linestyle=linestyle, color="red")
    ax.plot(x[([1 ,2])], y[([1, 2])], linestyle=linestyle, color="red")
    ax.plot(x[([2, 3])], y[([2, 3])], linestyle=linestyle, color="red")
    ax.plot(x[([7, 13])], y[([7, 13])], linestyle=linestyle, color="red")
    ax.plot(x[([13, 14])], y[([13, 14])], linestyle=linestyle, color="red")
    ax.plot(x[([14, 15])], y[([14, 15])], linestyle=linestyle, color="red")
    
    ax.plot(x[([0, 4])], y[([0, 4])], linestyle=linestyle, color="green")
    ax.plot(x[([4, 5])], y[([4, 5])], linestyle=linestyle, color="green")
    ax.plot(x[([5, 6])], y[([5, 6])], linestyle=linestyle, color="green")
    ax.plot(x[([7, 10])], y[([7, 10])], linestyle=linestyle, color="green")
    ax.plot(x[([10, 11])], y[([10, 11])], linestyle=linestyle, color="green")
    ax.plot(x[([11, 12])], y[([11, 12])], linestyle=linestyle, color="green")
    
    ax.plot(x[([0, 7])], y[([0, 7])], linestyle=linestyle)
    ax.plot(x[([7, 8])], y[([7, 8])], linestyle=linestyle)
    ax.plot(x[([8, 9])], y[([8, 9])], linestyle=linestyle)

    try:
        ax.title.set_text(title)
    except:
        pass
    ax.imshow(img)
    


def plot15j_2d_no_image(x,y, color, ax=plt, title = ""):
    '''x values and y values
    - Bast
    '''
    
    ax.scatter(x, y, c=['#ff0000'], alpha=0.2,s=10)

    ax.plot(x[([0, 1])], y[([0, 1])], color=color)
    ax.plot(x[([0, 2])], y[([0, 2])], color=color)
    ax.plot(x[([3, 1])], y[([3, 1])], color=color)
    ax.plot(x[([4, 1])], y[([4, 1])], color=color)
    ax.plot(x[([6, 4])], y[([6, 4])], color=color)
    ax.plot(x[([8, 6])], y[([8, 6])],color=color)
    ax.plot(x[([5, 3])], y[([5, 3])], color=color)
    ax.plot(x[([7, 5])], y[([7, 5])], color=color)
    ax.plot(x[([9, 3])], y[([9, 3])], color=color)
    ax.plot(x[([10, 4])], y[([10, 4])], color=color)
    ax.plot(x[([12, 10])], y[([12, 10])], color=color)
    ax.plot(x[([11, 9])], y[([11, 9])], color=color)
    ax.plot(x[([14, 12])], y[([14, 12])], color=color)
    ax.plot(x[([13, 11])], y[([13, 11])], color=color)
    
    ax.title.set_text(title)

def add_bbox_in_image(image, bbox):
    """
    ref: https://github.com/Pose-Group/DCPose/blob/078f4495c654f7220fa599fda07a8eef5dc54f21/visualization/bbox_vis.py#L8
    :param image
    :param bbox   -  xyxy
    """

    color = (random() * 255, random() * 255, random() * 255)
    x1, y1, x2, y2 = map(int, bbox)
    image_with_bbox = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=6)
    return image_with_bbox