# functions

def getHRIR(az, el):
    positions = sofa.SourcePosition
    angles = positions[:, :2]
    distance = np.linalg.norm(angles - np.array([az, el]), axis = 1)
    idx = np.argmin(distance)
    hrir_left = sofa.Data_IR[idx, 0, :]
    hrir_right = sofa.Data_IR[idx, 1, :]
    return(hrir_left, hrir_right)
