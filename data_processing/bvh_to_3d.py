import numpy as np
import pyquaternion as pyq

def rot_vec_to_abs_pos_vec(frames, nodes):
    """
    Transform vectors of the human motion from the joint angles to the absolute positions
    Args:
        frames: human motion in the join angles space
        nodes:  set of markers used in motion caption

    Returns:
        output_vectors : 3d coordinates of this human motion
    """
    output_lines = []

    for frame in frames:
        node_idx = 0
        for i in range(51):  # changed from 51
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            if nodes[node_idx]['name'] == 'End Site':
                node_idx = node_idx + 1
            nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
            current_node = nodes[node_idx]

            node_idx = node_idx + 1

        for start_node in nodes:
            abs_pos = np.array([0, 60, 0])
            current_node = start_node
            # = if not start_node['name'] = 'end site'
            if start_node['children'] is not None:
                for child_idx in start_node['children']:
                    child_node = nodes[child_idx]

                    child_offset = np.array(child_node['offset'])
                    qz = pyq.Quaternion(
                        axis=[0, 0, 1], degrees=start_node['rel_degs'][0])
                    qx = pyq.Quaternion(
                        axis=[1, 0, 0], degrees=start_node['rel_degs'][1])
                    qy = pyq.Quaternion(
                        axis=[0, 1, 0], degrees=start_node['rel_degs'][2])
                    qrot = qz * qx * qy
                    offset_rotated = qrot.rotate(child_offset)
                    child_node['rel_pos'] = start_node['abs_qt'].rotate(
                        offset_rotated)

                    child_node['abs_qt'] = start_node['abs_qt'] * qrot

            while current_node['parent'] is not None:

                abs_pos = abs_pos + current_node['rel_pos']
                current_node = nodes[current_node['parent']]
            start_node['abs_pos'] = abs_pos

        line = []
        for node in nodes:
            line.append(node['abs_pos'])
        output_lines.append(line)

    output_vels = []
    for idx, line in enumerate(output_lines):
        vel_line = []
        for jn, joint_pos in enumerate(line):
            if idx == 0:
                vels = np.array([0.0, 0.0, 0.0])
            else:
                vels = np.array([joint_pos[0] - output_lines[idx-1][jn][0], joint_pos[1] -
                                 output_lines[idx-1][jn][1], joint_pos[2] - output_lines[idx-1][jn][2]])
            vel_line.append(vels)
        output_vels.append(vel_line)

    out = []
    for idx, line in enumerate(output_vels):
        ln = []
        for jn, joint_vel in enumerate(line):
            ln.append(output_lines[idx][jn])
            ln.append(joint_vel)
        out.append(ln)

    output_array = np.asarray(out)
    output_vectors = np.empty([len(output_array), 384])
    for idx, line in enumerate(output_array):
        output_vectors[idx] = line.flatten()
    return output_vectors

def create_hierarchy_nodes(hierarchy):
    """
    Create hierarchy nodes: an array of markers used in the motion capture
    Args:
        hierarchy: bvh file read in a structure

    Returns:
        nodes: array of markers to be used in motion processing

    """
    joint_offsets = []
    joint_names = []

    for idx, line in enumerate(hierarchy):
        hierarchy[idx] = hierarchy[idx].split()
        if not len(hierarchy[idx]) == 0:
            line_type = hierarchy[idx][0]
            if line_type == 'OFFSET':
                offset = np.array([float(hierarchy[idx][1]), float(
                    hierarchy[idx][2]), float(hierarchy[idx][3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(hierarchy[idx][1])
            elif line_type == 'End':
                joint_names.append('End Site')

    nodes = []
    for idx, name in enumerate(joint_names):
        if idx == 0:
            parent = None
        elif idx in [6, 30]:  # spine1->shoulders
            parent = 2
        elif idx in [14, 18, 22, 26]:  # lefthand->leftfingers
            parent = 9
        elif idx in [38, 42, 46, 50]:  # righthand->rightfingers
            parent = 33
        elif idx in [54, 59]:  # hip->legs
            parent = 0
        else:
            parent = idx - 1

        if name == 'End Site':
            children = None
        elif idx == 0:  # hips
            children = [1, 54, 59]
        elif idx == 2:  # spine1
            children = [3, 6, 30]
        elif idx == 9:  # lefthand
            children = [10, 14, 18, 22, 26]
        elif idx == 33:  # righthand
            children = [34, 38, 42, 46, 50]
        else:
            children = [idx + 1]

        node = dict([('name', name), ('parent', parent), ('children', children), ('offset', joint_offsets[idx]),
                     ('rel_degs', None), ('abs_qt', None), ('rel_pos', None), ('abs_pos', None)])
        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)

    return nodes

def remove_velocity(data, dim=3):
    """Remove velocity values from raw prediction data

      Args:
          data:         array containing both position and velocity values
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   array containing only position values
    """

    starts = np.arange(0, data.shape[1], dim * 2)
    stops = np.arange(dim, data.shape[1], dim * 2)
    return np.hstack([data[:, i:j] for i, j in zip(starts, stops)])

def select_upper_body(arr):
    """2d array"""
    assert len(
        arr.shape) == 2, f"Provide shape of (t, x) 2d array, instead got {arr.shape}"
    return np.concatenate([arr[:, 3:18], arr[:, 21:30], arr[:, 93:102]], axis=1)


def vectorize_bvh(gesture_filename):

    f = open('hierarchy.txt', 'r')
    hierarchy = f.readlines()
    f.close()
    nodes = create_hierarchy_nodes(hierarchy)

    f = open(gesture_filename, 'r')
    org = f.readlines()
    frametime = org[310].split()

    del org[0:311] 

    bvh_len = len(org)

    for idx, line in enumerate(org):
        org[idx] = [float(x) for x in line.split()]

    for i in range(0, bvh_len):
        for j in range(0, int(306 / 3)):
            st = j * 3
            del org[i][st:st + 3]

    # if data is 100fps, cut it to 20 fps (every fifth line)
    # if data is approx 24fps, cut it to 20 fps (del every sixth line)
    if float(frametime[2]) == 0.0416667:
        del org[::6]
    elif float(frametime[2]) == 0.010000:
        org = org[::5]
    else:
        print("smth wrong with fps of " + gesture_filename)

    output_vectors = rot_vec_to_abs_pos_vec(org, nodes)

    f.close()

    return select_upper_body(remove_velocity(output_vectors))