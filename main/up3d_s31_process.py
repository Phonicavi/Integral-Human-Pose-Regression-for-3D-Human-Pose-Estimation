from collections import OrderedDict
import numpy as np


def enum(*sequential):
    """Reversible, ordered enum."""
    kv_tuples_complete = zip(sequential, range(len(sequential)))
    enums = OrderedDict(kv_tuples_complete)
    reverse = OrderedDict((value, key) for key, value in enums.items())
    enums['keys'] = enums.keys()  # Ignore the reverse_mapping
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)


joints_lsp = enum('rankle', 'rknee', 'rhip',
                  'lhip', 'lknee', 'lankle',
                  'rwrist', 'relbow', 'rshoulder',
                  'lshoulder', 'lelbow', 'lwrist',
                  'neck', 'head')

connections_lsp = [(joints_lsp.head, joints_lsp.neck, (0, 255, 0), False),
                   (joints_lsp.neck, joints_lsp.rshoulder, (0, 255, 0), False),
                   (joints_lsp.neck, joints_lsp.lshoulder, (0, 255, 0), True),
                   (joints_lsp.rshoulder, joints_lsp.relbow, (0, 0, 255), False),
                   (joints_lsp.relbow, joints_lsp.rwrist, (0, 0, 255), False),
                   (joints_lsp.lshoulder, joints_lsp.lelbow, (0, 0, 255), True),
                   (joints_lsp.lelbow, joints_lsp.lwrist, (0, 0, 255), True),
                   (joints_lsp.rshoulder, joints_lsp.rhip, (255, 0, 0), False),
                   (joints_lsp.rhip, joints_lsp.rknee, (255, 255, 0), False),
                   (joints_lsp.rknee, joints_lsp.rankle, (255, 255, 0), False),
                   (joints_lsp.lshoulder, joints_lsp.lhip, (255, 0, 0), True),
                   (joints_lsp.lhip, joints_lsp.lknee, (255, 255, 0), True),
                   (joints_lsp.lknee, joints_lsp.lankle, (255, 255, 0), True)]


def robust_person_size(pose,  # pylint: disable=too-many-branches
                       joints=None,  # pylint: disable=redefined-outer-name
                       connections=None,  # pylint: disable=redefined-outer-name
                       return_additional_info=False):
    """Get an estimate for the person size from its pose in LSP format."""
    assert pose.ndim == 2
    pose = pose.transpose()
    if joints is None and connections is None:
        joints = joints_lsp
        connections = connections_lsp
    longest_conn = None
    longest_conn_length = -1
    for connection in connections:
        if ((pose[2, connection[0]] < 1 or pose[2, connection[1]] < 1) or
                (connection[0] in [joints.neck, joints.lshoulder, joints.rshoulder] and
                 connection[1] in [joints.neck, joints.lshoulder, joints.rshoulder])):
            # Unreliable joint.
            # Neck to shoulder is too unstable (error factor ~10).
            continue
        conn_length = np.linalg.norm(pose[:2, connection[0]] -
                                     pose[:2, connection[1]])
        if conn_length > longest_conn_length:
            longest_conn_length = conn_length
            longest_conn = connection
            person_size_estimate = None
    if longest_conn is None:
        raise Exception("No connection found with necessary specs!")
    if (longest_conn[0] in [joints.head, joints.neck] and
            longest_conn[1] in [joints.head, joints.neck]):
        person_size_estimate = longest_conn_length  * 6.
    elif ((longest_conn[0] in [joints.neck, joints.lshoulder] and
           longest_conn[1] in [joints.neck, joints.lshoulder]) or
          (longest_conn[0] in [joints.neck, joints.rshoulder] and
           longest_conn[1] in [joints.neck, joints.rshoulder])):
        person_size_estimate = longest_conn_length * 10.3
    elif ((longest_conn[0] in [joints.lshoulder, joints.lhip] and
           longest_conn[1] in [joints.lshoulder, joints.lhip]) or
          (longest_conn[0] in [joints.rshoulder, joints.rhip] and
           longest_conn[1] in [joints.rshoulder, joints.rhip])):
        person_size_estimate = longest_conn_length * 3.4
    elif ((longest_conn[0] in [joints.lknee, joints.lhip] and
           longest_conn[1] in [joints.lknee, joints.lhip]) or
          (longest_conn[0] in [joints.rknee, joints.rhip] and
           longest_conn[1] in [joints.rknee, joints.rhip])):
        person_size_estimate = longest_conn_length * 4.
    elif ((longest_conn[0] in [joints.lknee, joints.lankle] and
           longest_conn[1] in [joints.lknee, joints.lankle]) or
          (longest_conn[0] in [joints.rknee, joints.rankle] and
           longest_conn[1] in [joints.rknee, joints.rankle])):
        person_size_estimate = longest_conn_length * 4.4
    elif ((longest_conn[0] in [joints.lshoulder, joints.lelbow] and
           longest_conn[1] in [joints.lshoulder, joints.lelbow]) or
          (longest_conn[0] in [joints.rshoulder, joints.relbow] and
           longest_conn[1] in [joints.rshoulder, joints.relbow])):
        person_size_estimate = longest_conn_length * 5.4
    elif ((longest_conn[0] in [joints.lelbow, joints.lwrist] and
           longest_conn[1] in [joints.lelbow, joints.lwrist]) or
          (longest_conn[0] in [joints.relbow, joints.rwrist] and
           longest_conn[1] in [joints.relbow, joints.rwrist])):
        person_size_estimate = longest_conn_length * 5.8
    else:
        raise Exception("Unknown connection! Unable to estimate person size!")
    if return_additional_info:
        return person_size_estimate, longest_conn, longest_conn_length
    else:
        return person_size_estimate


def get_crop(image, person_center, crop):
    """Crop the image to the given maximum size. Use the person center."""
    if image.shape[0] > crop:
        crop_y = np.array([int(np.floor(person_center[1]) - np.floor(crop / 2.)),
                           int(np.floor(person_center[1]) + np.ceil(crop / 2.))], dtype='int')
        remaining_region_size = [crop_y[0], image.shape[0] - crop_y[1]]
        if remaining_region_size[0] < remaining_region_size[1]:
            if remaining_region_size[0] < 0 < remaining_region_size[1]:
                crop_y += min(remaining_region_size[1], -remaining_region_size[0])
        else:
            if remaining_region_size[1] < 0 < remaining_region_size[0]:
                crop_y -= min(remaining_region_size[0], -remaining_region_size[1])
        assert crop_y[1] - crop_y[0] == crop
    else:
        crop_y = [0, image.shape[0]]
    if image.shape[1] > crop:
        crop_x = np.array([int(np.floor(person_center[0]) - np.floor(crop / 2.)),
                           int(np.floor(person_center[0]) + np.ceil(crop / 2.))], dtype='int')
        remaining_region_size = [crop_x[0], image.shape[1] - crop_x[1]]
        if remaining_region_size[0] < remaining_region_size[1]:
            if remaining_region_size[0] < 0 < remaining_region_size[1]:
                crop_x += min(remaining_region_size[1], -remaining_region_size[0])
        else:
            if remaining_region_size[1] < 0 < remaining_region_size[0]:
                crop_x -= min(remaining_region_size[0], -remaining_region_size[1])
        assert crop_x[1] - crop_x[0] == crop
    else:
        crop_x = [0, image.shape[1]]
    return crop_y, crop_x
