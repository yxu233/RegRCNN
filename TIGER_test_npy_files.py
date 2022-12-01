#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:20:55 2022

@author: user
"""

import numpy as np

### a is float 16, b is uint8 segmentation

# a = np.load('/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/cyl1ps_dev/train/20.npy')
# b = np.load('/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/cyl1ps_dev/train/20_seg.npy')


# import napari
# viewer = napari.view_image(a)
# viewer.add_image(b)

# import _pickle as cPickle

# with open(r"/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/cyl1ps_dev/train/info_df.pickle", "rb") as input_file:
#      e = cPickle.load(input_file)



# zzz


# a = np.load('/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/cyl1ps_dev_npz/train/6.npz')
# b = np.load('/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/cyl1ps_dev_npz/train/6_seg.npz')

# a = a['6']
# b = b['6_seg']

# # import napari
# # viewer = napari.view_image(a)
# # viewer.add_image(b)





# import _pickle as cPickle

# with open(r"/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/cyl1ps_dev_npz/train/info_df.pickle", "rb") as input_file:
#      e = cPickle.load(input_file)
     
     
     
     
# zzz
     
# a = np.load('/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/MULTI/cyl1ps_dev/train/20.npy')
# b = np.load('/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/MULTI/cyl1ps_dev/train/20_seg.npy')


# import napari
# viewer = napari.view_image(a)
# viewer.add_image(b)

# import _pickle as cPickle

# with open(r"/media/user/FantomHD/Lightsheet data/RegRCNN_maskrcnn_testing/toy/MULTI/cyl1ps_dev/train/info_df.pickle", "rb") as input_file:
#      e = cPickle.load(input_file)
     
     
     
     
     
     
# zzz
     
# a = np.load('/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/OL_data/Tiger/test/906.npy')
# b = np.load('/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/OL_data/Tiger/test/906_seg.npy')



a = np.load('/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/OL_data/Tiger/train/420.npy')
b = np.load('/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/OL_data/Tiger/train/420_seg.npy')


import napari
viewer = napari.view_image(a)
viewer.add_image(b)

import _pickle as cPickle

with open(r"/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/OL_data/Tiger/train/info_df.pickle", "rb") as input_file:
     e = cPickle.load(input_file)
     




zzz

### for Caspr

a = np.load('/media/user/FantomHD/710_invivo_imaging/Caspr_tdT_homozygous/Caspr_training/Caspr_training_RegRCNN/Caspr_data/train/20.npy')
b = np.load('/media/user/FantomHD/710_invivo_imaging/Caspr_tdT_homozygous/Caspr_training/Caspr_training_RegRCNN/Caspr_data/train/20_seg.npy')



c = np.load('/media/user/FantomHD/710_invivo_imaging/Caspr_tdT_homozygous/Caspr_training/Caspr_training_RegRCNN/Caspr_data/train/20.npy').astype('float16')[np.newaxis]

# import napari
# viewer = napari.view_image(a)
# viewer.add_image(b)

import _pickle as cPickle

with open(r"/media/user/FantomHD/710_invivo_imaging/Caspr_tdT_homozygous/Caspr_training/Caspr_training_RegRCNN/Caspr_data/train/info_df.pickle", "rb") as input_file:
     e = cPickle.load(input_file)
     



