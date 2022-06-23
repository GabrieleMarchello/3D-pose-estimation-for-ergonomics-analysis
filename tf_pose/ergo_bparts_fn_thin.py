import numpy as np


def shoulder_ang(pos,sh_ang,sk_coord_mat_noz):
    
	ext_ang = sh_ang[0]
	abd_ang = sh_ang[1]
    
	ext_flag = 1
	abd_flag = 1
    
	ext_0_min = -13
	ext_0_max = 33
    
	ext_1_min_p = ext_0_max
	ext_1_max_p = 75
    
	ext_2_min_p = ext_1_max_p
	ext_2_max_p = 112
    
	ext_1_min_n = -25
	ext_1_max_n = ext_0_min
    
	ext_2_min_n = -36
	ext_2_max_n = ext_1_min_n
	
	if ext_ang >= ext_0_min and ext_ang <= ext_0_max:
		ext_flag = 0
	elif (ext_ang >= ext_1_min_n and ext_ang < ext_1_max_n) or (ext_ang > ext_1_min_p and ext_ang <= ext_1_max_p):
		ext_flag = 1
	elif (ext_ang >= ext_2_min_n and ext_ang < ext_2_max_n) or (ext_ang > ext_2_min_p and ext_ang <= ext_2_max_p):
		ext_flag = 2
	else:
		ext_flag = 2
        
	abd_0_min = -5
	abd_0_max = 5
    
	abd_1_min_p = abd_0_max
	abd_1_max_p = 43
    
	abd_2_min_p = abd_1_max_p
	abd_2_max_p = 73
    
	abd_1_min_n = -17
	abd_1_max_n = ext_0_min
    
	abd_2_min_n = -25
	abd_2_max_n = abd_1_min_n
	
	if abd_ang >= abd_0_min and abd_ang <= abd_0_max:
		abd_flag = 0
	elif (abd_ang >= abd_1_min_n and abd_ang < abd_0_min) or (abd_ang > abd_1_min_p and abd_ang <= abd_1_max_p):
		abd_flag = 1
	elif (abd_ang >= abd_2_min_n and abd_ang < abd_2_max_n) or (abd_ang > abd_2_min_p and abd_ang <= abd_2_max_p):
		abd_flag = 2
	else:
		abd_flag = 2
        
	sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = np.maximum(ext_flag, abd_flag)
        

# 	print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))
	return sk_coord_mat_noz 
    

def lower_back_ang(pos,lat_bend_ang,sk_coord_mat_noz):
    
	lat_bend_0_min = -6
	lat_bend_0_max = 6
	
	lat_bend_1_min_p = lat_bend_0_max
	lat_bend_1_max_p = 12
	
	lat_bend_2_min_p = lat_bend_1_max_p
	lat_bend_2_max_p = 17
	
	lat_bend_1_min_n = -12
	lat_bend_1_max_n = lat_bend_0_min
	
	lat_bend_2_min_n = -17
	lat_bend_2_max_n = lat_bend_1_min_n
	
	if lat_bend_ang >= lat_bend_0_min and lat_bend_ang <= lat_bend_0_max:
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 0
	elif (lat_bend_ang >= lat_bend_1_min_n and lat_bend_ang < lat_bend_1_max_n) or (lat_bend_ang > lat_bend_1_min_p and lat_bend_ang <= lat_bend_1_max_p):
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 1
	elif (lat_bend_ang >= lat_bend_2_min_n and lat_bend_ang < lat_bend_2_max_n) or (lat_bend_ang > lat_bend_2_min_p and lat_bend_ang <= lat_bend_2_max_p):
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 2
	else:
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 2
    

# 	print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))
	return sk_coord_mat_noz
    

def lower_back_rot_ang(pos,low_back_ang,sk_coord_mat_noz):
    
	low_back_0_min = -5
	low_back_0_max = 5
	
	low_back_1_min_p = low_back_0_max
	low_back_1_max_p = 10
	
	low_back_2_min_p = low_back_1_max_p
	low_back_2_max_p = 20
	
	low_back_1_min_n = -10
	low_back_1_max_n = low_back_0_min
	
	low_back_2_min_n = -20
	low_back_2_max_n = low_back_1_min_n
	
	if low_back_ang >= low_back_0_min and low_back_ang <= low_back_0_max:
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 0
	elif (low_back_ang >= low_back_1_min_n and low_back_ang < low_back_0_min) or (low_back_ang > low_back_1_min_p and low_back_ang <= low_back_1_max_p):
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 1
	elif (low_back_ang >= low_back_2_min_n and low_back_ang < low_back_2_max_n) or (low_back_ang > low_back_2_min_p and low_back_ang <= low_back_2_max_p):
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 2
	else:
		sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = 2

# 	print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))
    
	return sk_coord_mat_noz


def neck_ang(pos,neck_ang,sk_coord_mat_noz):
    
	lat_bend_ang = neck_ang[0]
	ext_ang = neck_ang[1]
	
	lat_bend_flag = 1
	ext_flag = 1
    
	lat_bend_0_min = -28
	lat_bend_0_max = 28
	
	lat_bend_1_min_p = lat_bend_0_max
	lat_bend_1_max_p = 40
	
	lat_bend_2_min_p = lat_bend_1_max_p
	lat_bend_2_max_p = 48
	
	lat_bend_1_min_n = -40
	lat_bend_1_max_n = lat_bend_0_min
	
	lat_bend_2_min_n = -48
	lat_bend_2_max_n = lat_bend_1_min_n
	
	if lat_bend_ang >= lat_bend_0_min and lat_bend_ang <= lat_bend_0_max:
		lat_bend_flag = 0
	elif (lat_bend_ang >= lat_bend_1_min_n and lat_bend_ang < lat_bend_1_max_n) or (lat_bend_ang > lat_bend_1_min_p and lat_bend_ang <= lat_bend_1_max_p):
		lat_bend_flag = 1
	elif (lat_bend_ang >= lat_bend_2_min_n and lat_bend_ang < lat_bend_2_max_n) or (lat_bend_ang > lat_bend_2_min_p and lat_bend_ang <= lat_bend_2_max_p):
		lat_bend_flag = 2
	else:
		lat_bend_flag = 2
    
	ext_0_min = -19
	ext_0_max = 22
	
	ext_1_min_p = ext_0_max
	ext_1_max_p = 47
	
	ext_2_min_p = ext_1_max_p
	ext_2_max_p = 69
	
	ext_1_min_n = -37
	ext_1_max_n = ext_0_min
	
	ext_2_min_n = -53
	ext_2_max_n = ext_1_min_n
	
	if ext_ang >= ext_0_min and ext_ang <= ext_0_max:
		ext_flag = 0
	elif (ext_ang >= ext_1_min_n and ext_ang < ext_1_max_n) or (ext_ang > ext_1_min_p and ext_ang <= ext_1_max_p):
		ext_flag = 1
	elif (ext_ang >= ext_2_min_n and ext_ang < ext_2_max_n) or (ext_ang > ext_2_min_p and ext_ang <= ext_2_max_p):
		ext_flag = 2
	else:
		ext_flag = 2
        
	flag = np.maximum(lat_bend_flag, ext_flag)
	sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = np.maximum(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1], flag)
	#print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))

# 	print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))
        
	return sk_coord_mat_noz 
    

def neck_rot_ang(pos,rot_ang,sk_coord_mat_noz):
    
	rot_0_min = -41
	rot_0_max = 41
	
	rot_1_min_p = rot_0_max
	rot_1_max_p = 58
	
	rot_2_min_p = rot_1_max_p
	rot_2_max_p = 69
	
	rot_1_min_n = -58
	rot_1_max_n = rot_0_min
	
	rot_2_min_n = -69
	rot_2_max_n = rot_1_min_n
	
	if rot_ang >= rot_0_min and rot_ang <= rot_0_max:
		flag = 0
	elif (rot_ang >= rot_1_min_n and rot_ang < rot_1_max_n) or (rot_ang > rot_1_min_p and rot_ang <= rot_1_max_p):
		flag = 1
	elif (rot_ang >= rot_2_min_n and rot_ang < rot_2_max_n) or (rot_ang > rot_2_min_p and rot_ang <= rot_2_max_p):
		flag = 2
	else:
		flag = 2
        
	sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1] = np.maximum(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1], flag)
	#print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))
    
# 	print('Flaaaaaag:         '+str(sk_coord_mat_noz[sk_coord_mat_noz[:,0]==pos,-1]))
    
	return sk_coord_mat_noz
