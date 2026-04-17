import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from typing import Dict, List, Tuple
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from helpers import sparse_quantize

RangeImages = Dict['dataset_pb2.LaserName.Name', List[dataset_pb2.MatrixFloat]]
CameraProjections = Dict['dataset_pb2.LaserName.Name',
                         List[dataset_pb2.MatrixInt32]]
SegmentationLabels = Dict['dataset_pb2.LaserName.Name',
                          List[dataset_pb2.MatrixInt32]]
ParsedFrame = Tuple[RangeImages, CameraProjections, SegmentationLabels,
                    dataset_pb2.MatrixFloat]


def parse_range_image_and_camera_projection(frame):

  range_images = {}
  camera_projections = {}
  seg_labels = {}
  range_image_top_pose: dataset_pb2.MatrixFloat = dataset_pb2.MatrixFloat()
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(range_image_str_tensor.numpy())
      range_images[laser.name] = [ri]

      if laser.name == dataset_pb2.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = dataset_pb2.MatrixFloat()
        range_image_top_pose.ParseFromString(range_image_top_pose_str_tensor.numpy())

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(camera_projection_str_tensor.numpy())
      camera_projections[laser.name] = [cp]

      if len(laser.ri_return1.segmentation_label_compressed) > 0:
        seg_label_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.segmentation_label_compressed, 'ZLIB')
        seg_label = dataset_pb2.MatrixInt32()
        seg_label.ParseFromString(seg_label_str_tensor.numpy())
        seg_labels[laser.name] = [seg_label]
    if len(laser.ri_return2.range_image_compressed) > 0:
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(range_image_str_tensor.numpy())
      range_images[laser.name].append(ri)

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(camera_projection_str_tensor.numpy())
      camera_projections[laser.name].append(cp)

      if len(laser.ri_return2.segmentation_label_compressed) > 0:
        seg_label_str_tensor = tf.io.decode_compressed(
            laser.ri_return2.segmentation_label_compressed, 'ZLIB')
        seg_label = dataset_pb2.MatrixInt32()
        seg_label.ParseFromString(seg_label_str_tensor.numpy())
        seg_labels[laser.name].append(seg_label)
  return range_images, camera_projections, seg_labels, range_image_top_pose



DATA_DIR = "waymo"
VOXEL_DIR = "waymo_voxels"

os.makedirs(VOXEL_DIR, exist_ok=True)

tfrecords = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".tfrecord")
])

frame_id = 0

for tfrecord in tfrecords:
    dataset = tf.data.TFRecordDataset(tfrecord, compression_type="")

    for i, data in enumerate(tqdm(dataset, desc=f"Processing {os.path.basename(tfrecord)}")):
        if i >= 100:
            break

        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        range_images, camera_projections, _, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
        )
        points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1,
        )

        xyz = np.concatenate(points + points_ri2, axis=0)[:, :3]

        # Apply bounds
        min_bound = np.array([-75.2, -75.2, -2])
        max_bound = np.array([75.2, 75.2, 4])
        mask = np.all((xyz >= min_bound) & (xyz <= max_bound), axis=1)
        xyz = xyz[mask]
        voxel_size = (0.1, 0.1, 0.15)
        voxel_coords = sparse_quantize(xyz, voxel_size)
        np.save(os.path.join(VOXEL_DIR, f"waymo_{frame_id:06d}_voxels.npy"), voxel_coords)

        frame_id += 1