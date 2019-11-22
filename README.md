# carnd-1-2
Advanced lane line detection

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/test6_mask.jpg "Binary Example"
[image4]: ./output_images/straight_lines1_warp.jpg "Warp Example"
[image5]: ./output_images/windows.jpg "Fit Visual"
[image6]: ./output_images/test1_result.jpg "Output"
[imageo]: ./output_images/test6_original.jpg "Original"
[imageud]: ./output_images/test6_undist.jpg "Undistorted"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

While doing this project, my aim was to make the code modular so that it could be used in future projects. I defined a `Camera` class to handle the camera calibration, undistortion, image warping and unwarping (to plot the lane lines onto the original image).

The code for camera calibration is contained in lines 20-58 of `camera.py` located in `./src/camera.py`.

The procedure I used is exactly as we practised in the quizzes in the class. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Original image:
![alt text][imageo]

##### Results
Here is the result of undistorting the image from camera parameters:

Undistorted image:
![alt text][imageud]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All the functions for applying color transformations and gradient thresholds are defined in `./src/laneutils.py`.

- The function to apply thresholding to gradients along a direction `x` or `y` is defined in lines 5-14.
- The function to apply thresholding to the magnitude of gradients is defined in lines 17-27.
- A function to apply thresholding based on the direction of the gradients is defined in lines 30-36.

These functions are used in `./src/main.py` as follows: 
- After undistortion, the image is converted into HLS format and S-channel in grayscale format was chosen for further analysis. (line 29)
- A binary map based on x gradient of the grayscale image was calculated. (line 31)
- A binary map based on y gradient of the grayscale image was calculated. (line 33)
- A binary map based on the magnitude of gradients in both directions was calculated. (line 35)
- A binary map based on the direction of gradients was calculated (line 36). Here we take advantage of the fact that the lane lines are expected to be roughly vertical.
- The binary maps are combined to create a final map which highlights points where either both x and y gradients are within threshold bounds or both magnitude and direction of gradients are within threshold bounds. This results in a very clean binary mask in which the lane boundaries stand out.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

As explained earlier, the perspective transform of images is handled by the `Camera` class in `./src/camera.py`. The function `warp_img` to warp images is defined in lines 72-78 of this file.

Before applying perspective transform, we specify the source and destination points in `main.py` and call the `setupwarp` function of the camera. This function calculates and stopres the matrices for warping as well as unwarping images. 
The `warp_img` function takes as inputs an image (`img`) and outputs the warped image. I chose the hardcode the source and destination points in the following manner:

```python
warpsrc=np.float32([[584, 460],[202, 720],[1128, 720],[695, 460]])
warpdst=np.float32([[320, 0],[320, 720],[960, 720],[960, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 584, 460      | 320, 0        | 
| 202, 720      | 320, 720      |
| 1128, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I chose to warp the binary mask instead of warping the image and then creating mask. After warping the mask, the lane lines are approximately vertical, but there is some noise in images where there are other objects like cars or some features on the road with rough edges.  The pixels corresponding to lane lines were idenitifed as follows:

- The binary mask was divided into 9 windows.
- We initialized the `x` positions of left and right lane lines in the first window by looking at the histogram of the bottom half of the whole image and identifying the peaks of the histogram in the left and right halves of the image. This is performed by a function `findbase` defined in `./src/laneutils.py` (lines 177-182).
- For subsequent windows, we searched in the area nearby the previous window and shifted the location of the window if the number of pixels in the region of interest was more than 50.

![alt text][image5]

After identifying the pixels, I fit the pixels to a quadratic polynomial in lines 54-62 of `laneutils.py`.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This was done in the `findanddraw` function in `laneutils.py`. (lines 91-97)
First, we average the parameters of the two lane lines to reduce any errors resulting from the estimation of parameters of either line. The resulting parameters are scaled by factors `mx` and `my` given in the project instructions and are used for estimation of radius of curvature. The distance of the car (camera) from center of the lane is estimated by finding the difference of the center of the image from the center of the lane lines (line 97).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 43 through 45 in my code in `main.py`.  Here is an example of my result on a test image:

![alt text][image6]
The estimated radius of curvature and distance from the center are also shown in the image.
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline explained above is applied to the video with two major modifications.
- In a video, we can use the fact that lane lines do not change abruptly from frame to frame. The pixels corresponding to lane lines were identified based on the equations of the lane lines obtained in the previous frame to reduce the search space.

- If the number of points for fitting the poynomial for a given frame is too small, it is likely to result in a large error in the estimation of parameters of the quadratic. Thus, if the number of points in the entire lane is below 500, we skip curve fitting for that lane for the given frame and use the parameters of the previous frame (lines 54 and 59 of `laneutils.py`)

- Similarly, after estimation of the parameters for a given frame, we take a weighted average of the parameters from the previous frame and this frame. This helps in stabilizing the output when the lighting conditions change and there is a lot of noise in the frame.

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### Problems:
- The project video contained many different lighting conditions, resulting in either too many points for curve fitting (due to noise) or too few points if the lanes were far out and not prominently visible.
- The estimate of the radius of curvature was erroneous until I realized that averaging the parameters of the left and right lanes should help stabilize the estimation.

##### Limitations
This project involved a variety of techniques. Although the pipeline works well for the project video, here are some limitations of my approach:
- There are several parameters used in this project which are tuned to the lighting conditions of the video.
- This approach of using classical computer vision is not readily parallelizeable. The pipeline would need to run at >100fps in a real car but my results on my laptop were much more modest. In comparison, other approaches such as using deep learning are much more readily parallelizeable and there is custom hardware to accelerate those applications. Thus, these techniques could provide a greater throughput, even with their higher computing requirements.

##### Future work
- Motivated from the previous discussion, if I were to pursue this project further, I would use a CNN to segment out the lane lines as mentioned in one of the suggested readings.
- If it was necessary to use classical computer vision, I would use another camera mounted at the back of the vehicle (as is commonly found in vehicles for assisting in parking). The output from front facing and back facing cameras would provide additional error correction for the processing pipeline.

