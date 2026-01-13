NUM_OF_LANES = 4
LENGTH_OF_EDGE = 1000
RL_SENSING_RADIUS = 300.0
RL_MAX_SPEED_LIMIT = 40
RL_MIN_SPEED_LIMIT = 20
RL_ACC_RANGE = 2.0
RL_DCE_RANGE = 4.0
HEADING_ANGLE = 90.0
MIN_LANE_DENSITY = 0
MAX_LANE_DENSITY = 100 #this is an assumption for the traffic

EGO_ID = "av_0"

ACC_INTERVAL = 1

#IDM & MOBIL Parameters
IDM_v0 = 35  #speedFactor
IDM_s0 = 2    #minimumGap
IDM_a = 1.5        #accelerationFactor
IDM_b = 2.0        #comfortableDeceleration
IDM_T = 1.5      #desiredTimeHeadway
IDM_delta = 4.0  
# IDM_TAU = 1.87   
# IDM_ACTIONSTEPLENGTH = 0.4 #
#IDM_STEPPING = 0.1  #the internal step length (in s) when computing follow speed

MOBIL_left_Bias = 0.3
MOBIL_politeness = 0.3
MOBIL_change_threshold = 0.1



#Risk assignment parameters
SSM_D_THRESHOLD = 0.6
SSM_A_THRESHOLD = 4.8


SSM_UNCERTAINTY = 20.8


activation_steepness = 0.5
