# standard libraries
import torch
import numpy as np

def MPJPE(ref_pose, fit_pose) -> float:
    ''' This calculates the Euclidean or L2 Distance
    between reference pose and fitted pose.
    The distance is averaged over the batch
    
    Args:
        ref_pose : (B, .., 3)
        fit_pose : (B, .., 3)
        where .. is the no of joints
      
    returns:
        MPJPE Error: Scalar
    '''
    
    #Calculates Euclidean or Distance. Sqrt((x2-x1)2 + (y2-y1)2 + ...); mpjpe is l2 distance but not mse or l2 squared distance
    dist = torch.sqrt(torch.sum((ref_pose - fit_pose)**2, dim=-1)) # sum over the joint (x,y,z) # Bastian
    #average over (the batch if applicable)
    error = torch.mean(torch.mean(dist)) # mean over skeleton, mean over batch    
    return error

def NMPJPE(label, pred):
    #print("pred", pred.shape)
    batch_size = label.size()[0]
    
    gt_vec = label.view(batch_size,-1)
    pred_vec = pred.view(batch_size,-1)
    
    
    dot_pose_gt   = torch.sum(torch.mul(pred_vec,gt_vec), dim=-1, keepdim=True); #print("dot_pose_gt", dot_pose_gt.shape)
    dot_pose_pose = torch.sum(torch.mul(pred_vec,pred_vec), dim=-1, keepdim=True); #print("dot_pose_pose", dot_pose_pose.shape)

    s_opt = dot_pose_gt / dot_pose_pose; #print("s_opt", s_opt.shape)

    if len(pred.size())==3:
        output = s_opt.unsqueeze(-1).expand_as(pred)*pred
    elif len(pred.size())==2:
        output = s_opt.expand_as(pred)*pred
        
    return MPJPE(label, output)

    
def calculate_batch_mpjpe(output, label):

    difference= output-label
    square_difference= torch.square(difference)
    sum_square_difference_per_point= torch.sum(square_difference, dim=2)
    euclidean_distance_per_point= torch.sqrt(sum_square_difference_per_point)
    mpjpe= torch.mean(euclidean_distance_per_point)

    return mpjpe


def MPJPE_PA(ref_pose, fit_pose) -> float:
    
    '''Compute MPJPE after Procrustes and also store similiarity matrices 
    that can be applied later to rotation matrices for MPJAE_PA'''
    
    pred3d_sym, R = compute_similarity_transform(fit_pose.detach().cpu().numpy(), ref_pose.detach().cpu().numpy())
    pa_error = np.sqrt(np.sum((ref_pose.detach().cpu() - pred3d_sym) ** 2, axis=1))
    
    return pa_error


#ref: https://github.com/aymenmir1/3dpw-eval/blob/2640f244898d5503a8e3ce9825da5af3c77edb33/evaluate.py#L78
def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Ensure that the first argument is the prediction
    Source: https://en.wikipedia.org/wiki/Kabsch_algorithm
    :param S1 predicted joint positions array 24 x 3
    :param S2 ground truth joint positions array 24 x 3
    :return S1_hat: the predicted joint positions after apply similarity transform
            R : the rotation matrix computed in procrustes analysis
    '''
    # If all the values in pred3d are zero then procrustes analysis produces nan values
    # Instead we assume the mean of the GT joint positions is the transformed joint value

    if not (np.sum(np.abs(S1)) == 0):
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat, R
    else:
        S1_hat = np.tile(np.mean(S2, axis=0), (SMPL_NR_JOINTS, 1))
        R = np.identity(3)

        return S1_hat, R


