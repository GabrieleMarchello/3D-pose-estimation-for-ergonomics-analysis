import argparse
import logging
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pyrealsense2 as rs

from tf_pose.estimator_3d_paper_thin import TfPoseEstimator
from tf_pose.networks import get_graph_path


##########################################################################################
# Init Posenet Estimation 
##########################################################################################

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

##########################################################################################
# Init RealSense
##########################################################################################

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
        
w, h = 640, 480

config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


##########################################################################################
# Real-time pose estimation
##########################################################################################

model = 'mobilenet_v2_large'
        
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=str2bool("False"))
            
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=10)
# ax.set_xlim([-1,1])
ax.set_xlim([-0.5,0.5])
ax.set_xticks(np.arange(-1, 1.4, step=0.4))
ax.set_xticklabels(['-1','-0.6','-0.2','0.2','0.4','1'])
# ax.set_ylim([-2,0])
ax.set_ylim([-1.5,0])
ax.set_yticks(np.arange(-2, 0.4, step=0.4))
ax.set_yticklabels(['-2','-1.6','-1.2','0.8','0.4','0'])
ax.set_zlim([-1,1])
ax.set_zticks(np.arange(-1, 1.4, step=0.4))
ax.set_zticklabels(['-1','-0.6','-0.2','0.2','0.4','1'])
# plt.ion()
fig.show()

fps_time = 0

temp_filter = rs.temporal_filter() 		# reduces temporal noise
dec_filter = rs.decimation_filter()		# reduces depth frame density

# Streaming loop
try:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    
    aligned_depth_frame = temp_filter.process(aligned_depth_frame)
    aligned_depth_frame = dec_filter.process(aligned_depth_frame)
    
    camera_info = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
	# print(camera_info)
    
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    cv2.imshow('RGB Image', color_image)
    
    humans = e.inference(color_image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    # bparts=TfPoseEstimator.draw_humans_3d(fig, ax, depth_image, depth_scale, rs, camera_info, humans, [], imgcopy=False)# body_dict, imgcopy=False)
    
    bparts_col = TfPoseEstimator.draw_humans_3d(fig, ax,depth_image, depth_scale, rs, camera_info, humans, [], imgcopy=False)# body_dict, imgcopy=False)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # img_out = TfPoseEstimator.draw_humans(color_image, humans, imgcopy=False)
    img_out = TfPoseEstimator.draw_humans(color_image, humans, bparts_col, imgcopy=False)
    cv2.imshow('OpenPose Result', img_out)
	
    key = cv2.waitKey(0)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        
finally:
    pipeline.stop()
