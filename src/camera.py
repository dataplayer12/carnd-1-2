import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import pickle

class Camera(object):
    def __init__(self,nx,ny,calib_folder=None):
        self.nx=nx
        self.ny=ny
        self.calib_folder=calib_folder
        self.objpoints = []
        self.imgpoints = []
        self.mtx=None
        self.dist=None
        self.warpmat=None
        self.unwarpmat=None

    def calibrate(self,debug=False):
        if not self.calib_folder:
            print('Calibration folder not specified. Camera will not be calibrated')
            self.mtx=np.array([[100, 0, 0],[0, 100, 0],[0, 0, 1]])
            #camera matrix with 100px focal length and no off-center correction
            self.dist=np.array([[0.,0.,0.,0.,0.]])
            #distortion coefficients are assumed to be 0 (no distortion)
        else:
            print('Calibrating camera from images in {}'.format(self.calib_folder))
            self.mtx,self.dist=self.get_camera_mat(debug)
            print('Camera calibration complete')

    def get_camera_mat(self,debug=False):
        objp = np.zeros((self.nx*self.ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        #objp[:,3] is always 0 because all points lie in a plane

        images = glob.glob(os.path.join(self.calib_folder,'calibration*.jpg'))

        for idx, fname in enumerate(images):
            gray= cv2.imread(fname,0)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)
            # If found, add object points, image points
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                if debug:
                    img = cv2.imread(fname,1)
                    cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)
            if debug:
                cv2.destroyAllWindows()

        img_size=(gray.shape[1],gray.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)

        return mtx,dist  #return camera matrix and distortion coefficients

    def undistort_image(self,img):
        if self.mtx is None:
            print('Camera not calibrated!')
            return

        dst = cv2.undistort(img, self.mtx, self.dist)
        return dst

    def setupwarp(self,src,dst):
        self.warpmat = cv2.getPerspectiveTransform(src, dst)
        self.unwarpmat=cv2.getPerspectiveTransform(dst, src)

    def warp_img(self,img):
        if self.warpmat is None:
            print('Warp source and destination points not specified')
            return img
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.warpmat, img_size, flags=cv2.INTER_NEAREST)
        return warped

    def unwarp_img(self,img):
        if self.unwarpmat is None:
            print('Warp source and destination points not specified')
            return img
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.unwarpmat, img_size, flags=cv2.INTER_NEAREST)
        return unwarped