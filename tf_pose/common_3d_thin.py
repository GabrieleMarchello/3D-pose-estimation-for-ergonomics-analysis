from enum import Enum


from tf_pose.ergo_bparts_fn_thin import *

import tensorflow as tf
import numpy as np
import cv2


regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


class MPIIPart(Enum):
    RAnkle = 0
    RKnee = 1
    RHip = 2
    LHip = 3
    LKnee = 4
    LAnkle = 5
    RWrist = 6
    RElbow = 7
    RShoulder = 8
    LShoulder = 9
    LElbow = 10
    LWrist = 11
    Neck = 12
    Head = 13

    @staticmethod
    def from_coco(human):
        # t = {
        #     MPIIPart.RAnkle: CocoPart.RAnkle,
        #     MPIIPart.RKnee: CocoPart.RKnee,
        #     MPIIPart.RHip: CocoPart.RHip,
        #     MPIIPart.LHip: CocoPart.LHip,
        #     MPIIPart.LKnee: CocoPart.LKnee,
        #     MPIIPart.LAnkle: CocoPart.LAnkle,
        #     MPIIPart.RWrist: CocoPart.RWrist,
        #     MPIIPart.RElbow: CocoPart.RElbow,
        #     MPIIPart.RShoulder: CocoPart.RShoulder,
        #     MPIIPart.LShoulder: CocoPart.LShoulder,
        #     MPIIPart.LElbow: CocoPart.LElbow,
        #     MPIIPart.LWrist: CocoPart.LWrist,
        #     MPIIPart.Neck: CocoPart.Neck,
        #     MPIIPart.Nose: CocoPart.Nose,
        # }

        t = [
            (MPIIPart.Head, CocoPart.Nose),
            (MPIIPart.Neck, CocoPart.Neck),
            (MPIIPart.RShoulder, CocoPart.RShoulder),
            (MPIIPart.RElbow, CocoPart.RElbow),
            (MPIIPart.RWrist, CocoPart.RWrist),
            (MPIIPart.LShoulder, CocoPart.LShoulder),
            (MPIIPart.LElbow, CocoPart.LElbow),
            (MPIIPart.LWrist, CocoPart.LWrist),
            (MPIIPart.RHip, CocoPart.RHip),
            (MPIIPart.RKnee, CocoPart.RKnee),
            (MPIIPart.RAnkle, CocoPart.RAnkle),
            (MPIIPart.LHip, CocoPart.LHip),
            (MPIIPart.LKnee, CocoPart.LKnee),
            (MPIIPart.LAnkle, CocoPart.LAnkle),
        ]

        pose_2d_mpii = []
        visibilty = []
        for mpi, coco in t:
            if coco.value not in human.body_parts.keys():
                pose_2d_mpii.append((0, 0))
                visibilty.append(False)
                continue
            pose_2d_mpii.append((human.body_parts[coco.value].x, human.body_parts[coco.value].y))
            visibilty.append(True)
        return pose_2d_mpii, visibilty

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
# CocoPairsNetwork = [
#     (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
#     (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
#  ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

CocoColors_3d = [[0, 0, 255/255], [0, 85/255, 255/255], [0, 170/255, 255/255], [0, 255/255, 255/255], [0, 255/255, 170/255], [0, 255/255, 85/255], [0, 255/255, 0],
                 [85/255, 255/255, 0], [170/255, 255/255, 0], [255/255, 255/255, 0], [255/255, 170/255, 0], [255/255, 85/255, 0], [255/255, 0, 0], [255/255, 0, 85/255],
                 [255/255, 0, 170/255], [255/255, 0, 255/255], [170/255, 0, 255/255], [85/255, 0, 255/255]]


def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def get_sample_images(w, h):
    val_image = [
        read_imgfile('./images/p1.jpg', w, h),
        read_imgfile('./images/p2.jpg', w, h),
        read_imgfile('./images/p3.jpg', w, h),
        read_imgfile('./images/golf.jpg', w, h),
        read_imgfile('./images/hand1.jpg', w, h),
        read_imgfile('./images/hand2.jpg', w, h),
        read_imgfile('./images/apink1_crop.jpg', w, h),
        read_imgfile('./images/ski.jpg', w, h),
        read_imgfile('./images/apink2.jpg', w, h),
        read_imgfile('./images/apink3.jpg', w, h),
        read_imgfile('./images/handsup1.jpg', w, h),
        read_imgfile('./images/p3_dance.png', w, h),
    ]
    return val_image


def to_str(s):
    if not isinstance(s, str):
        return s.decode('utf-8')
    return s


def ergo_angle(sk_coord_mat):

	sk_coord_mat_noz=sk_coord_mat[np.all(sk_coord_mat[:,1:7]!=np.zeros((1,6)),axis=1)]
	
	# sk_coord_mat_noz[:,-1] = 1
    
	if 0 in sk_coord_mat_noz[:,0]:
		pos = 0
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		lat_bend_r_ang = np.rad2deg(np.arctan2(pair_f[1]-pair_i[1],pair_f[0]-pair_i[0]))
        
		if lat_bend_r_ang < 0:
			lat_bend_r_ang = -180 - lat_bend_r_ang
		else:
			lat_bend_r_ang = 180 - lat_bend_r_ang
		
		sk_coord_mat = lower_back_ang(pos,lat_bend_r_ang,sk_coord_mat)
		
		# print('Lower back R - Lateral bending: '+str(lat_bend_r_ang))
		
	if 1 in sk_coord_mat_noz[:,0]:
		pos = 1
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		lat_bend_l_ang = np.rad2deg(np.arctan2(pair_f[1]-pair_i[1],pair_f[0]-pair_i[0]))    
		
		sk_coord_mat = lower_back_ang(pos,lat_bend_l_ang,sk_coord_mat)
		
		# print('Lower back L - Lateral bending: '+str(lat_bend_l_ang))
		
	if 2 in sk_coord_mat_noz[:,0]:
		pos = 2
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		ext_ang = -np.rad2deg(np.arctan2(pair_f[1]-pair_i[1],pair_f[2]-pair_i[2]))-90
		abd_ang = -np.rad2deg(np.arctan2(pair_f[2]-pair_i[2],pair_f[0]-pair_i[0]))+90
        
		if abd_ang >= 180:
			abd_ang = abd_ang - 180
		else:
			abd_ang = 180 - abd_ang
        
		shoulder_r_ang = np.asarray([ext_ang,abd_ang])
        
		sk_coord_mat = shoulder_ang(pos,shoulder_r_ang,sk_coord_mat)
		
		print('Identified right arm at: '+str((pair_i+pair_f)/2))
		# print('Shoulder R - Extension: '+str(ext_ang))
		# print('Shoulder R - Adduction: '+str(abd_ang))
		
	if 3 in sk_coord_mat_noz[:,0]:
		pos = 3
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		
		print('Identified right fore arm at: '+str((pair_i+pair_f)/2))
	
	if 4 in sk_coord_mat_noz[:,0]:
		pos = 4
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
        
		ext_ang = -np.rad2deg(np.arctan2(pair_f[1]-pair_i[1],pair_f[2]-pair_i[2]))-90
		abd_ang = 90-np.rad2deg(np.arctan2(pair_f[2]-pair_i[2],pair_f[0]-pair_i[0]))
        
		if abd_ang >= 0:
			abd_ang = 180 - abd_ang
        
		shoulder_l_ang = np.asarray([ext_ang,abd_ang])
		
		sk_coord_mat = shoulder_ang(pos,shoulder_l_ang,sk_coord_mat)
		
		print('Identified left arm at: '+str((pair_i+pair_f)/2))
		# print('Shoulder L - Extension: '+str(ext_ang))
		# print('Shoulder L - Adduction: '+str(abd_ang))
		
	if 5 in sk_coord_mat_noz[:,0]:
		pos = 5
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		
		print('Identified left fore arm at: '+str((pair_i+pair_f)/2))
	
	if 6 in sk_coord_mat_noz[:,0]:
		pos = 6
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
        
		lower_back_r_ang = np.rad2deg(np.arctan2(pair_f[2]-pair_i[2],pair_f[0]-pair_i[0]))+125
		
		sk_coord_mat = lower_back_rot_ang(pos,lower_back_r_ang,sk_coord_mat)
		
		# print('Lower back R - Rotation: '+str(lower_back_r_ang))
	
	if 9 in sk_coord_mat_noz[:,0]:
		pos = 9
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
        
		lower_back_l_ang = np.rad2deg(np.arctan2(pair_f[2]-pair_i[2],pair_f[0]-pair_i[0]))+50
		
		sk_coord_mat = lower_back_rot_ang(pos,lower_back_l_ang,sk_coord_mat)
		
		# print('Lower back L - Rotation: '+str(lower_back_l_ang))
	
	if 12 in sk_coord_mat_noz[:,0]:
		pos = 12
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		lat_bend_ang = np.rad2deg(np.arctan2(pair_f[1]-pair_i[1],pair_f[0]-pair_i[0]))-90
		ext_ang = np.rad2deg(np.arctan2(pair_f[1]-pair_i[1],pair_f[2]-pair_i[2]))-90
        
		neck_v_ang = np.asarray([lat_bend_ang,ext_ang])
		
		sk_coord_mat = neck_ang(pos,neck_v_ang,sk_coord_mat)
		
		# print('Neck - Lateral bending: '+str(lat_bend_ang))
		# print('Neck - Extension: '+str(ext_ang))
	
	if 13 in sk_coord_mat_noz[:,0]:
		pos = 13
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		rot_ang = 180-np.rad2deg(np.arctan2(pair_f[2]-pair_i[2],pair_f[0]-pair_i[0]))
        		
		sk_coord_mat = neck_rot_ang(12,rot_ang,sk_coord_mat)
        
		# print('Neck - Rot ang 13: '+str(rot_ang))
	
	if 15 in sk_coord_mat_noz[:,0]:
		pos = 15
	
		pair_i = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,1:4][0]
		pair_f = sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,4:7][0]
		
		rot_ang=np.rad2deg(np.arctan2(pair_f[2]-pair_i[2],pair_f[0]-pair_i[0]))
        		
		sk_coord_mat = neck_rot_ang(12,rot_ang,sk_coord_mat)
        
		# print('Neck - Rot ang 15: '+str(rot_ang))
    
	#print('Neck - Extension:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa '+str(sk_coord_mat[12,-1]))    
	return sk_coord_mat
