import cv2
import glob
import numpy as np
from camera import Camera
import laneutils as lut

cam = Camera(nx=9, ny=6, calib_folder='../camera_cal/')
cam.calibrate()

warpsrc=np.float32([[584, 460],[202, 720],[1128, 720],[695, 460]])
warpdst=np.float32([[320, 0],[320, 720],[960, 720],[960, 0]])
cam.setupwarp(warpsrc, warpdst)

src=cv2.VideoCapture('../project_video.mp4')
dst=lut.get_dst(src,'../viz_average.mp4',(1,1))

ret,img=src.read()
prev=None
#test_images = ['../test_images/straight_lines1.jpg']#glob.glob('../test_images/*.jpg')
#newname = lambda n,app: n[:n.rfind('.')] + '_{}.jpg'.format(app)
while ret:
    #for path in test_images:
    #img=cv2.imread(path,1)
    undist = cam.undistort_image(img)
    
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[..., -1]
    ksize = 5  # Choose a larger odd number to smooth gradient measurements
    gradx = lut.abs_sobel_thresh(
        s_channel, orient='x', sobel_kernel=ksize, thresh=(50, 150))
    grady = lut.abs_sobel_thresh(
        s_channel, orient='y', sobel_kernel=ksize, thresh=(50, 150))
    mag_binary = lut.mag_thresh(s_channel, sobel_kernel=ksize, mag_t=(50, 150))
    dir_binary = lut.dir_threshold(
        s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros(dir_binary.shape, dtype=np.uint8)
    combined[(gradx & grady) | (mag_binary & dir_binary)] = 255
    warped = cam.warp_img(combined)
    #cv2.imwrite(newname(path,'original'),img)
    #cv2.imwrite(newname(path,'warp'),cam.warp_img(img))
    #quit()
    lane_mask,prev,roc,dist=lut.findanddraw(warped,prev)
    unwarped=cam.unwarp_img(lane_mask)
    annotated=cv2.addWeighted(undist, 0.8, unwarped, 0.2, 0.)
    print(dist)
    msg='Radius of curvature ={} m'.format(int(roc))
    cv2.putText(annotated, msg, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, [255, 255, 255], 2)
    msg='Distance from center= {:.2f} m'.format(dist)
    cv2.putText(annotated, msg, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, [255, 255, 255], 2)
    ccat = annotated#np.vstack((rgb,annotated))
    #cv2.imwrite(newname(path), ccat)
    dst.write(ccat)
    ret,img=src.read()


src.release()
dst.release()

#bottomrow = np.hstack((s_channel, combined))
#bottomrow = np.dstack([bottomrow] * 3)
#final = np.vstack((toprow, bottomrow))