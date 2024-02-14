import xml.dom.minidom as minidom
import numpy as np
import torch 


def project_point_radial(P, f, c, k, p):
    """
    Args
    P: Nx3 points in now in camera coordinates (Dan)
    R: 3x3 Camera rotation matrix - EXCLUDED
    T: 3x1 Camera translation parameters - EXCLUDED
    f: 2x1 (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3
    
    X = P.T
    N = P.shape[0]
    
    # ------
    XX = X[:2, :] / X[2, :]  # 2x16
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2  # 16,
    
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))  # 16,
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]  # 16,

    tm = np.outer(np.array([p[1], p[0]]).reshape(-1), r2)  # 2x16
    XXX = XX * np.tile(radial + tan, (2, 1)) + tm  # 2x16
    Proj = (f * XXX) + c  # 2x16
    Proj = Proj.T

    D = X[2, ]
    return Proj, D, radial, tan, r2


def project_point_radial_torch(P, f, c, k, p):
    """
    project_point_radial in torch Tensors
    
    Args
    P: Nx3 points in now in camera coordinates (Dan)
    R: 3x3 Camera rotation matrix - EXCLUDED
    T: 3x1 Camera translation parameters - EXCLUDED
    f: 2x1 (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    if len(P.shape) == 2 and P.shape[1] == 3:

    
        X = P.T
        N = P.shape[0]

        # ------
        XX = X[:2, :] / X[2, :]  # 2x16
        r2 = (XX[0, :] ** 2 + XX[1, :] ** 2).view(1,-1)  # former: 16,, now: (1,16)
        radial = 1 + torch.einsum('ij,ij->j', (k.repeat(1, N)), torch.cat([r2, r2 ** 2, r2 ** 3]))  
        tan = p[0] * XX[1, :] + p[1] * XX[0, :]  # tan: 16,

        tm = torch.outer(torch.cat([p[1], p[0]]).view(-1), r2.view(-1))  # r2 now (16) again tm: 2x16
        #XXX = XX * np.tile(radial + tan, (2, 1)) + tm  # 2x16
        XXX = XX * (radial + tan).repeat(2, 1) + tm  # 2x16

        Proj = (f * XXX) + c  # 2x16
        Proj = Proj.T

        D = X[2, ]
        return Proj, D, radial, tan, r2

    # P is a matrix of 3-dimensional points
    if len(P.shape) == 3 and P.shape[2] == 3:

    
        X = P.permute(0,2,1)
        N = P.shape[1]
        B = P.shape[0]

        # ------
        XX = X[:, :2, :] / X[:, 2:3, :]  # Bx2x16
        r2 = (XX[:, 0:1, :] ** 2 + XX[:, 1:2, :] ** 2).view(B,1,-1)  # former: B,16, now: (B,1,16),
        radial = 1 + torch.einsum('bij,bij->bj', (k.repeat(1,1, N)), torch.cat([r2, r2 ** 2, r2 ** 3], dim=1))  # radial: B,16,
        tan = p[:, 0] * XX[:, 1, :] + p[:, 1] * XX[:, 0, :]  # tan: B,16,

        tm = torch.bmm(torch.cat([p[:,1], p[:, 0]], dim=1).view(B,-1, 1), r2.view(B,1,-1))  # r2 now (B,16) again tm: BX2x16

        #XXX = XX * np.tile(radial + tan, (2, 1)) + tm  # 2x16
        #print("XX, radial + tan", XX.shape, radial.shape, tan.shape);stop
        XXX = XX * (radial + tan).view(B,1,N).repeat(1, 2, 1) + tm  # Bx2x16

        Proj = (f * XXX) + c  # Bx2x16
        Proj = Proj.permute(0,2,1)

        D = X[:, 2,]
        return Proj, D, radial, tan, r2

def world_to_camera_frame(P, R, T):
    """
  Convert points from world to camera coordinates
  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
    """

    if len(P.shape) == 2 and P.shape[1] == 3:
        X_cam = R @ ( P.T - T ) # rotate and translate
        
        return X_cam.T

    # batch
    elif len(P.shape) == 3 and P.shape[2] == 3:
        X_cam = R @ (P.permute(0,2,1) - T)
        
        return X_cam.permute(0,2,1)


def camera_to_world_frame(P, R, T):
    """Inverse of world_to_camera_frame
  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

    if len(P.shape) == 2 and P.shape[1] == 3:
        X_cam = R.T @ (P.T) + T  # rotate and translate
        return X_cam.T

    # batch
    elif len(P.shape) == 3 and P.shape[2] == 3:
        X_cam = R.permute(0,2,1) @ (P.permute(0,2,1)) + T  # rotate and translate
        return X_cam.permute(0,2,1)

def camera_to_other_frame(P, R, T):
    """Same as camera_to_world_frame
  Args
    P: Nx3 points in camera coordinates
    R: 3x3 unknown Camera rotation matrix
    T: 3x1 unknown Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

    if len(P.shape) == 2 and P.shape[1] == 3:
        X_cam = (R.T @ (P.T)) + T  # rotate and translate
        return X_cam.T

    # batch
    elif len(P.shape) == 3 and P.shape[2] == 3:
        X_cam = R.permute(0,2,1) @ (P.permute(0,2,1)) + T  # rotate and translate
        return X_cam.permute(0,2,1)


CAMERA_ID_TO_NAME = {
  1: "54138969",
  2: "55011271",
  3: "58860488",
  4: "60457274",
}

def load_cameras(bpath, subjects=[1,5,6,7,8,9,11]):
  """Loads the cameras of h36m
  Args
    bpath: path to xml file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  xmldoc = minidom.parse(bpath)
  string_of_numbers = xmldoc.getElementsByTagName('w0')[0].firstChild.data[1:-1]

  # Parse into floats
  w0 = np.array(list(map(float, string_of_numbers.split(" "))))

  assert len(w0) == 300

  for s in subjects:
    for c in range(4): # There are 4 cameras in human3.6m
      rcams[(s, c+1)] = load_camera_params(w0, s, c+1)

  return rcams


def load_camera_params(w0, subject, camera):
  """Load h36m camera parameters
  Args
    w0: 300-long array read from XML metadata
    subect: int subject id
    camera: int camera id
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  # Get the 15 numbers for this subject and camera
  w1 = np.zeros(15)
  start = 6 * ((camera-1)*11 + (subject-1))
  w1[:6] = w0[start:start+6]
  w1[6:] = w0[(265+(camera-1)*9 - 1): (264+camera*9)]

  def rotationMatrix(r):    
    R1, R2, R3 = [np.zeros((3, 3)) for _ in range(3)]

    R1[0:] = [1, 0, 0]
    R1[1:] = [0, np.cos(r[0]), -np.sin(r[0])]
    R1[2:] = [0, np.sin(r[0]),  np.cos(r[0])]

    R2[0:] = [ np.cos(r[1]), 0, np.sin(r[1])]
    R2[1:] = [0, 1, 0]
    R2[2:] = [-np.sin(r[1]), 0, np.cos(r[1])]

    R3[0:] = [np.cos(r[2]), -np.sin(r[2]), 0]
    R3[1:] = [np.sin(r[2]),  np.cos(r[2]), 0]
    R3[2:] = [0, 0, 1]

    return (R1.dot(R2).dot(R3))
    
  R = rotationMatrix(w1)
  T = w1[3:6][:, np.newaxis]
  f = w1[6:8][:, np.newaxis]
  c = w1[8:10][:, np.newaxis]
  k = w1[10:13][:, np.newaxis]
  p = w1[13:15][:, np.newaxis]
  name = CAMERA_ID_TO_NAME[camera]

  return R, T, f, c, k, p, name



def project_to_cameras(poses_set, cams, ncams=4 ):
  """
  Project 3d poses using camera parameters
  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )
      sname = seqname[:-3]+ name + ".h5"  # e.g.: Waiting 1.58860488.h5
      t2d[ (subj, a, sname) ] = pts2d

  return t2d

# ref: https://github.com/una-dinosauria/3d-pose-baseline/blob/666080d86a96666d499300719053cc8af7ef51c8/src/cameras.py#L15