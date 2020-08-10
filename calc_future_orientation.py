import numpy as np
import skinematics as skin

#from scipy import signal


from eul2quat_bio import eul2quat_bio
from quat2eul_bio import quat2eul_bio

def calc_future_orientation(euler_o, theta, angle_axis):
    # Initial quaternion
    # quat_o = skin.rotmat.seq2quat(euler_o, seq='nautical')
    quat_o = eul2quat_bio(euler_o)
          
    # Get quaternion matrix rotation
    q_r = np.column_stack([np.cos(theta/2), angle_axis*np.sin(theta/2)])
    
    # Obtain predicted quaternion
    quat_t = skin.quat.q_mult(quat_o, q_r)
    
    # Convert predicted quaternion to predicted Euler
    # euler_t = skin.quat.quat2seq(quat_t, seq='nautical')
    euler_t = quat2eul_bio(quat_t)
    return euler_t

