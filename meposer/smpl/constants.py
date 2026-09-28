JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
NUM_BODY_JOINTS = 22
PELVIS, LEFT_KNEE, RIGHT_KNEE, HEAD, LEFT_WRIST, RIGHT_WRIST = 0, 4, 5, 15, 20, 21
UPPER_INDEX = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_INDEX = [0, 1, 2, 4, 5, 7, 8, 10, 11]
BONES = [(j, PARENTS[j]) for j in range(1, NUM_BODY_JOINTS)]
LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT = 7, 8, 10, 11
FOOT_JOINTS = [[LEFT_ANKLE, LEFT_FOOT], [RIGHT_ANKLE, RIGHT_FOOT]]
