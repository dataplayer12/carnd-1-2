import numpy as np
import cv2
import os

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        gradabs = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel))
    else:
        gradabs = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel))

    gradmagscaled = np.uint8(255.0 * gradabs / np.max(gradabs))
    mask = np.zeros(gray.shape, dtype=np.bool)
    mask[(gradmagscaled >= thresh[0]) & (gradmagscaled <= thresh[1])] = 1
    return mask


def mag_thresh(gray, sobel_kernel=3, mag_t=(0, 255)):
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(gradx * gradx + grady * grady)
    gradmagscaled = np.uint8(255.0 * gradmag / np.max(gradmag))
    # plt.imshow(gradmagscaled,cmap='gray')
    # plt.show()
    mask = np.zeros(gray.shape, dtype=np.bool)
    mask[gradmagscaled >= mag_t[0]] = 1

    return mask


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gradx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel))
    grady = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel))
    gradir = np.arctan2(grady, gradx)
    mask = np.zeros(gray.shape, dtype=np.bool)
    mask[(gradir >= thresh[0]) & (gradir <= thresh[1])] = 1
    return mask

def findanddraw(warped,prev=None,minpoints=500):
    nwindows=9
    margin=100
    lidx,ridx,lpoints,rpoints=candidatepoints(warped,nwindows,margin,50,prev)
    rgb=np.dstack([warped]*3)
    win_height = np.int(warped.shape[0]//nwindows)
    # for ix,(l,r) in enumerate(zip(lidx,ridx)):
    #     lpt1=(l-margin//2,warped.shape[0]-(ix+1)*win_height)
    #     lpt2=(l+margin//2,warped.shape[0]-ix*win_height)

    #     rpt1=(r-margin//2,warped.shape[0]-(ix+1)*win_height)
    #     rpt2=(r+margin//2,warped.shape[0]-ix*win_height)

    #     cv2.rectangle(rgb,lpt1,lpt2,[0,0,255],2)
    #     cv2.rectangle(rgb,rpt1,rpt2,[255,0,0],2)

    if len(lpoints[1])<minpoints:
        left_fit=prev[0]
    else:
        left_fit =np.polyfit(np.array(lpoints[0]),np.array(lpoints[1]),2)

    if len(rpoints[1])<minpoints:
        right_fit=prev[1]
    else:
        right_fit=np.polyfit(np.array(rpoints[0]),np.array(rpoints[1]),2)

    if prev is not None:
        for idx in range(3):
            left_fit[idx]=0.05*left_fit[idx]+0.95*prev[0][idx]
            right_fit[idx]=0.05*right_fit[idx]+0.95*prev[1][idx]
            #smooth out values from previous frames
    
    ploty = np.arange(warped.shape[0])#np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fitx=np.minimum(np.array(left_fitx,dtype=np.int32),1279)
    right_fitx=np.minimum(np.array(right_fitx,dtype=np.int32),1279)
    Al,Bl,Cl=left_fit
    Ar,Br,Cr=right_fit

    leftxcenter =(Al*ploty**2+Bl*ploty+Cl).astype(np.int32)
    rightxcenter=(Ar*ploty**2+Br*ploty+Cr).astype(np.int32)
    ldesx=leftxcenter[:,None]
    truex=np.arange(warped.shape[1])[None,:]
    lmask=(ldesx-truex)<0
    rdesx=rightxcenter[:,None]
    rmask=(rdesx-truex)>0
    mask=np.bitwise_and(lmask,rmask)
    rgb[mask]=[0,255,0]
    #rgb[mask]=[0,255,0]
    mx=30.0/720
    my=3.7/700

    A,B,C=(Al+Ar)/2,(Bl+Br)/2,(Cl+Cr)/2
    A=(mx/my**2)*A
    B=(mx/my)*B
    roc=(1+(2*A*30.0+B)**2)**(1.5)/(2*abs(A))
    lane_mid=((Al*720**2+Bl*720+Cl)+(Ar*720**2+Br*720+Cr))/2
    image_mid=640
    distance=abs(image_mid-lane_mid)*my
    return rgb, (left_fit,right_fit),roc,distance #(lidx[0],ridx[0])

def candidatepoints(warpedimg,nwindows=9,margin=100,minpix=50, prev=None):
    """
    warpedimg: binary warped image for current frame
    nwindows: number of windows
    margin: max horizontal width of a window
    minpix: minimum number of pixels to update window positions
    prev: leftx and rightx positions of lowermost window in previous frame
    """
    win_height = np.int(warpedimg.shape[0]//nwindows)
    left_lane_inds = []
    right_lane_inds = []
    lpoints=[[],[]]
    rpoints=[[],[]]
    if prev is None:
        first=True
    else:#this branch will be used most of the time
        first=False
        Al,Bl,Cl=prev[0]
        Ar,Br,Cr=prev[1]
        ploty = np.arange(warpedimg.shape[0])#np.linspace(0, warped.shape[0]-1, warped.shape[0])
        leftxcenter =(Al*ploty**2+Bl*ploty+Cl).astype(np.int32)
        rightxcenter=(Ar*ploty**2+Br*ploty+Cr).astype(np.int32)
        ldesx=leftxcenter[:,None]
        truex=np.arange(warpedimg.shape[1])[None,:]
        lmask=abs(ldesx-truex)<margin//2
        rdesx=rightxcenter[:,None]
        rmask=abs(rdesx-truex)<margin//2
        #cv2.imwrite('rmask.jpg',255*rmask.astype(np.uint8))
        lregion=np.bitwise_and(warpedimg,255*lmask.astype(np.uint8))
        rregion=np.bitwise_and(warpedimg,255*rmask.astype(np.uint8))
        #cv2.imwrite('lregion.jpg',lregion)
        lnz=lregion.nonzero()
        rnz=rregion.nonzero()
        return None,None,lnz,rnz
        
    for win in range(nwindows,0,-1):
        if first: #used only for first window of first frame
            window=warpedimg[warpedimg.shape[0]//2:,:]
            leftx_current,rightx_current = findbase(window)
            left_lane_inds.append(leftx_current)
            right_lane_inds.append(leftx_current+625)
            first=False
            continue

        #this branch will be used for 2nd to nth window of first frame
        ymin=(win-1)*win_height
        ymax=win*win_height
        window=warpedimg[ymin:ymax,:]
        lxmin,lxmax = (leftx_current-margin//2,leftx_current+margin//2)
        rxmin,rxmax=(rightx_current-margin//2,rightx_current+margin//2)

        lwindow=window[:,lxmin:lxmax]
        rwindow=window[:,rxmin:rxmax]

        lnz=lwindow.nonzero()
        rnz=rwindow.nonzero()

        lnonzerox = lxmin+np.array(lnz[1])
        lnonzeroy = ymin+np.array(lnz[0])
        rnonzerox = rxmin+np.array(rnz[1])
        rnonzeroy = ymin+np.array(rnz[0])

        if lnonzerox.shape[0]>=minpix:
            leftx_current = lnonzerox.mean(dtype=np.int)
        if rnonzerox.shape[0]>=minpix:
            rightx_current =rnonzerox.mean(dtype=np.int)

        left_lane_inds.append(leftx_current)
        right_lane_inds.append(leftx_current+625)

        lpoints[0].extend(list(lnonzeroy))
        lpoints[1].extend(list(lnonzerox))
        rpoints[0].extend(list(rnonzeroy))
        rpoints[1].extend(list(rnonzerox))

    return left_lane_inds,right_lane_inds,lpoints,rpoints

def findbase(window,noise_level=0):
    histogram=np.sum(window,axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base,rightx_base

def get_dst(src,dstname,scale=(1,1)):
    fps=src.get(cv2.CAP_PROP_FPS)
    fourcc=int(src.get(cv2.CAP_PROP_FOURCC))
    size=(scale[0]*int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),scale[1]*int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if os.path.exists(dstname):
        os.remove(dstname) #opencv won't write if file already exists
    dst=cv2.VideoWriter(dstname,fourcc,fps,size)
    return dst