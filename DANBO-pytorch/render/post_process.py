import cv2
import ipdb
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from core.utils.skeleton_utils import draw_skeleton2d, SMPLSkeleton


def cca_image(kps, img=None, acc_img=None, img_url=None, plot=False, x_margin=200, y_margin=150, chk_folder=None,
              white_bkgd=True):
    # ref: https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/ | https://stackoverflow.com/a/46442154/12761745 | https://stackoverflow.com/a/51524749/12761745 | https://stackoverflow.com/a/35854198/12761745

    if img_url is not None:
        BGR_img = cv2.imread(img_url)
    else:
        BGR_img = img.copy()
    
    H,W,C = BGR_img.shape
    
    kps = kps.astype(int)
    # add allowable margin
    min_y, max_y = kps[:,1].min(), kps[:,1].max()
    min_x, max_x = kps[:,0].min(), kps[:,0].max()
    
    min_y = max(0, min_y-y_margin)
    min_x = max(0, min_x-x_margin)
    max_y = min(H, max_y+y_margin)
    max_x = min(W, max_x+x_margin)

    cropped_img = BGR_img[min_y:max_y, min_x:max_x]
    try:
        gray_img = cv2.cvtColor(cropped_img , cv2.COLOR_BGR2GRAY)
    except:
        # gray level needs to be dtype "uint8"
        BGR_img = img.astype("uint8")
        cropped_img = BGR_img[min_y:max_y, min_x:max_x]

    if plot:
        plt.imshow(cropped_img); 
        plt.savefig(f"{chk_folder}/cropped_img.png")
    
    # 7x7 Gaussian Blur, threshold and component analysis function
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Apply the Component analysis function
    connectivity = 8
    analysis = cv2.connectedComponentsWithStats(threshold, connectivity, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # pre-compute kp mask
    kps_mask = np.ones((H,W), dtype=np.uint8)

    # out of bound threshold
    out_threshold = 10 # pixels
    out_whole_kps_bool = False
    out_count_H, out_count_W = 0, 0
    out_count_per_whole_kps = 0

    for kp in kps:
        if kp[1] >= H:
            kp_H = min(H, kp[1])-1
            out_count_H +=1
            out_whole_kps_bool = True
        else:
            kp_H = kp[1]

        if kp[0] >= W:    
            kp_W = min(kp[0], W)-1
            out_count_W +=1
            out_whole_kps_bool = True
        else:
            kp_W = kp[0]

        try:
            kps_mask[kp_H, kp_W] = 1
        except:
            ipdb.set_trace()

    if out_whole_kps_bool == True:
        out_count_per_whole_kps +=  1

    # store stats
    out_stats = (out_whole_kps_bool, out_count_per_whole_kps, out_count_H, out_count_W)
    
    # plug crop back into full image size
    label_ids_full = np.zeros((H,W), dtype=np.uint8) # start with 0s | usually stand for background
    label_ids_full[min_y:max_y, min_x:max_x] = label_ids 
    
    # get closest component
    composite = label_ids_full*kps_mask
    vals = composite[np.nonzero(composite)]
    component_label, freq = stats.mode(vals)
    # found_locs = np.transpose(np.nonzero(composite))
    
    # extract component mask
    mask = np.zeros((H,W), dtype=np.uint8)
    mask[label_ids_full == component_label] = 1
    
    if plot:
        plt.scatter(kps[:,1], kps[:,0], linewidth=0.5, color="r")
        plt.imshow(mask); plt.show()#; mask.min(); mask.max()
        plt.savefig(f"{chk_folder}/mask.png")

    # filter image with component mask
    # RGB_img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB) # faster if cv2 already imported
    filtered_rgb = BGR_img*mask[:,:,None]
    filtered_acc = acc_img*mask[:,:,None]
    
    
    # white background
    rgb_white_bkgd = np.zeros_like(filtered_rgb)
    skel_white_bkgd = np.zeros_like(filtered_rgb)
    if white_bkgd:
        bgd_pixel = 255
    else:
        bgd_pixel = 0
    rgb_white_bkgd[label_ids_full != component_label] = bgd_pixel
    # skel_white_bkgd[skel_img == 0] = 255
    # skel_white_bkgd[skel_img != 0] = 0

    rgb = rgb_white_bkgd + filtered_rgb
    skel_img = draw_skeleton2d(rgb, kps, skel_type=SMPLSkeleton, width=3, flip=False)
    # skel = skel_white_bkgd + skel_img
    return rgb, filtered_acc, skel_img, out_stats
    