import os, struct, math
import h5py
import numpy as np
from scipy import misc
from PIL import Image
import torch
import torchvision.transforms as transforms

import util
import gc

def load_hdf5_data(filename, num_classes):
    print filename
    assert os.path.isfile(filename)
    gc.collect()

    with h5py.File(filename, 'r') as f:
        volumes = f['data'][:].astype(np.float32)
        labels = f['label'][:]
        frames = f['frames'][:]
        world_to_grids = f['world_to_grid'][:]
    labels[np.greater(labels, num_classes - 1)] = num_classes - 1
    volumes = torch.from_numpy(volumes)
    labels = torch.from_numpy(labels.astype(np.int64))
    frames = torch.from_numpy(frames.astype(np.int32))
    world_to_grids = torch.from_numpy(world_to_grids)
    return volumes, labels, frames, world_to_grids


def load_pose(filename):
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image


def load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize):
    color_image = misc.imread(color_file)
    depth_image = misc.imread(depth_file)
    pose = load_pose(pose_file)
    # preprocess
    depth_image = resize_crop_image(depth_image, depth_image_dims)
    color_image = resize_crop_image(color_image, color_image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0
    color_image =  np.transpose(color_image, [2, 0, 1])  # move feature to front
    color_image = normalize(torch.Tensor(color_image.astype(np.float32) / 255.0))
    return depth_image, color_image, pose


def load_scene(filename, num_classes, load_gt):
    assert os.path.isfile(filename)
    fin = open(filename, 'rb')
    # read header
    width = struct.unpack('<I', fin.read(4))[0]
    height = struct.unpack('<I', fin.read(4))[0]
    depth = struct.unpack('<I', fin.read(4))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]

    numElems = width * height * depth
    sdfs = struct.unpack('f'*numElems, fin.read(numElems*4))  #grid3<float>
    labels = None
    if load_gt:
        labels = struct.unpack('B'*numElems, fin.read(numElems))  #grid3<uchar>
    fin.close()
    sdfs = np.asarray(sdfs, dtype=np.float32).reshape([depth, height, width])
    if load_gt:
        labels = np.asarray(labels, dtype=np.uint8).reshape([depth, height, width])
    occ = np.ndarray((2, depth, height, width), np.dtype('B')) #occupancy grid for occupied/empty space, known/unknown space
    occ[0] = np.less_equal(np.abs(sdfs), 1)
    occ[1] = np.greater_equal(sdfs, -1)
    if load_gt:
        # ensure occupied space has non-zero labels
        labels[np.logical_and(np.equal(occ[0], 1), np.equal(labels, 0))] = num_classes - 1
        # ensure non-occupied space has zero labels
        labels[np.equal(occ[0], 0)] = 0
        labels[np.greater_equal(labels, num_classes)] = num_classes - 1
    return occ, labels


def load_label_frame(label_file, image_dims, num_classes):
    assert os.path.isfile(label_file)
    label_image = misc.imread(label_file).astype(np.uint8)
    label_image = resize_crop_image(label_image, image_dims)
    label_image[np.greater(label_image, num_classes - 1)] = num_classes - 1
    return torch.from_numpy(label_image)


def load_label_frames(data_path, frame_indices, label_images, num_classes):
    # construct files
    num_images = frame_indices.shape[1] - 2
    scan_names = ['scene' + str(scene_id).zfill(4) + '_' + str(scan_id).zfill(2) for scene_id, scan_id in frame_indices[:,:2].numpy()]
    scan_names = np.repeat(scan_names, num_images)
    frame_ids = frame_indices[:, 2:].contiguous().view(-1).numpy()
    label_files = [os.path.join(data_path, scan_name, 'label', str(frame_id) + '.png') for scan_name, frame_id in zip(scan_names, frame_ids)]
    batch_size = frame_indices.size(0) * num_images
    # load data
    image_dims = [label_images.shape[2], label_images.shape[1]]
    for k in range(batch_size):
        label_images[k] = load_label_frame(label_files[k], image_dims, num_classes)


def load_frames_multi(data_path, frame_indices, depth_images, color_images, poses, color_mean, color_std):
    # construct files
    num_images = frame_indices.shape[1] - 2
    scan_names = ['scene' + str(scene_id).zfill(4) + '_' + str(scan_id).zfill(2) for scene_id, scan_id in frame_indices[:,:2].numpy()]
    scan_names = np.repeat(scan_names, num_images)
    frame_ids = frame_indices[:, 2:].contiguous().view(-1).numpy()
    depth_files = [os.path.join(data_path, scan_name, 'depth', str(frame_id) + '.png') for scan_name, frame_id in zip(scan_names, frame_ids)]
    color_files = [os.path.join(data_path, scan_name, 'color', str(frame_id) + '.jpg') for scan_name, frame_id in zip(scan_names, frame_ids)]
    pose_files = [os.path.join(data_path, scan_name, 'pose', str(frame_id) + '.txt') for scan_name, frame_id in zip(scan_names, frame_ids)]

    batch_size = frame_indices.size(0) * num_images
    depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
    color_image_dims = [color_images.shape[3], color_images.shape[2]]
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # load data
    for k in range(batch_size):
        depth_image, color_image, pose = load_depth_label_pose(depth_files[k], color_files[k], pose_files[k], depth_image_dims, color_image_dims, normalize)
        color_images[k] = color_image
        depth_images[k] = torch.from_numpy(depth_image)
        poses[k] = pose


def load_scene_image_info_multi(filename, scene_name, image_path, depth_image_dims, color_image_dims, num_classes, color_mean, color_std):
    assert os.path.isfile(filename)
    fin = open(filename, 'rb')
    # read header
    width = struct.unpack('<Q', fin.read(8))[0]
    height = struct.unpack('<Q', fin.read(8))[0]
    max_num_images = struct.unpack('<Q', fin.read(8))[0]
    numElems = width * height * max_num_images
    frame_ids = struct.unpack('i'*numElems, fin.read(numElems*4))  #grid3<int>
    _width = struct.unpack('<Q', fin.read(8))[0]
    _height = struct.unpack('<Q', fin.read(8))[0]
    assert width == _width and height == _height
    numElems = width * height * 4 * 4
    world_to_grids = struct.unpack('f'*numElems, fin.read(numElems*4))  #grid2<mat4f>
    fin.close()
    frame_ids = np.asarray(frame_ids, dtype=np.int32).reshape([max_num_images, height, width])
    world_to_grids = np.asarray(world_to_grids, dtype=np.float32).reshape([height, width, 4, 4])
    # load data
    unique_frame_ids = np.unique(frame_ids)
    depth_images = {}
    color_images = {}
    poses = {}
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    for f in unique_frame_ids:
        if f == -1:
            continue
        depth_file = os.path.join(image_path, scene_name, 'depth', str(f) + '.png')
        color_file = os.path.join(image_path, scene_name, 'color', str(f) + '.jpg')
        pose_file = os.path.join(image_path, scene_name, 'pose', str(f) + '.txt')
        depth_image, color_image, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize)
        depth_images[f] = torch.from_numpy(depth_image.astype(np.float32))
        color_images[f] = color_image
        poses[f] = pose
    return depth_images, color_images, poses, frame_ids, world_to_grids


