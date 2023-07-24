#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:43:24 2023

@author: user
"""

"""for presentations etc"""

import plotting as plg

import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
import utils.model_utils as mutils
from predictor import Predictor
from evaluator import Evaluator

import matplotlib.pyplot as plt
import tifffile as tiff



import pandas as pd
pd.set_option('mode.chained_assignment', None)

import shapely
from shapely.geometry import Polygon, LineString

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 


import geopandas as gpd       
from skimage.draw import polygon, polygon_perimeter
import random




### get all boxes on current slice
 
def plot_all_boxes_slice(slice_im, box_vert, slice_num):
     slice_boxes = np.where((box_vert[:, -1] >= slice_num) & (box_vert[:, -2] <= slice_num))[0]
     sb = box_vert[slice_boxes]
 
     list_poly = []
     for b in sb:
         list_poly.append(np.asarray([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]))
 
     #list_poly = np.vstack(list_poly)
 
 

     #poly_coordinates = np.array(list(gdf.iloc[0].values[0].exterior.coords))
     
     perim_c = []
     slice_perims = np.zeros(np.shape(slice_im))
     for p in list_poly:
         rr, cc = polygon_perimeter(
                    p[:, 0],
                    p[:, 1],
                    shape=slice_im.shape, clip=True)
     
     
         #rr, cc = polygon_perimeter(poly_coordinates[:,0], poly_coordinates[:,1], slice_im.shape)
 
         coords = np.transpose([cc, rr])  ### x,y,z coordinates
         
         perim_c.append(coords)
         
         slice_perims[rr, cc] = random.randint(1, 10)
         #print('speed')
         
     return slice_perims






""" Remove overlap between bounding boxes """
### since 3d polygons are complex, we will do 2d slice-by-slice to cut up the boxes
def split_boxes_by_slice(box_coords, vol_shape):
    box_2d = []
    box_ids = []
    box_depth = []
    for box_id, box3d in enumerate(box_coords):
       
       for depth in range(box3d[4], box3d[5] + 1):
           box_2d.append(box3d[0:4])   ### only get x1, y1, x2, y2
           box_depth.append(depth)
           box_ids.append(box_id)
           
    
   
    df = pd.DataFrame(data={'ids': box_ids, 'depth': box_depth, 'bbox':box_2d})
   
    df_cleaned = pd.DataFrame()
   
    ### Go slice by slice to correct the masks
    for i_d in range(vol_shape[0] + 1):
       df_slice = df.iloc[np.where(df['depth'] == i_d)[0]]
       
       
       if len(df_slice) == 0:
           #print(vol_shape[0])
           continue
       
       boxes_slice = df_slice['bbox']  
       gdf = gpd.GeoDataFrame({'geometry': [box2polygon(b) for b in boxes_slice]})
        
       #inter = gdf.loc[gdf.intersects(gdf.iloc[1].geometry)]
        
       res_df = slice_all(gdf, margin=0.5, line_mult=1.2)
       

    
    
       # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5), dpi=120)
       # gdf.plot(ax=ax1, alpha=0.5, color='gray')
       # #gdf.plot(ax=ax2, alpha=0.1, facecolor='gray')
       # res_df.plot(ax=ax2, alpha=0.5, color='olive')
       # ax1.axis('equal')
       # ax2.axis('equal')
       # ax1.set_title('Original bounding boxes')
       # ax2.set_title('Splited bounding boxes')
       # fig.tight_layout()
       
       #print('Slice_num: ' + str(i_d))
       
       # Create the basic mask
       # For each polygon, draw the polygon inside the mask
       bbox_coords = []
       bbox_perim = []
       for id_r, polygon_geom in res_df.iterrows():
           
           #a_mask = np.zeros(shape=[quad_size, quad_size], dtype="bool") # original
            
           poly_coordinates = np.array(list(polygon_geom.values[0].exterior.coords))
            
                     
           #poly_coordinates = np.array(list(gdf.iloc[0].values[0].exterior.coords))
           rr, cc = polygon(poly_coordinates[:,0], poly_coordinates[:,1], vol_shape[1:])
           
           #a_mask[cc,rr] = 1            
           coords = np.transpose([cc, rr, np.ones(len(cc)) * i_d])  ### x,y,z coordinates
           bbox_coords.append(coords)
           
           
           
           # rr, cc = polygon_perimeter(poly_coordinates[:,0], poly_coordinates[:,1], vol_shape[1:])
           
           # #a_mask[cc,rr] = 1            
           # coords = np.transpose([cc, rr, np.ones(len(cc)) * i_d])  ### x,y,z coordinates
           # bbox_perim.append(coords)
                      
           
       df_slice['bbox_coords'] = bbox_coords
       #df_slice['bbox_perims'] = bbox_perim
       #print('SLOW1: ' + str(id_r))
           
       df_cleaned = df_cleaned.append(df_slice, ignore_index=True)
       
       #print('SLOW: ' + str(id_r))
       
       
    return df_cleaned
  

def box2polygon(x):
     return Polygon([(x[0], x[1]), (x[2], x[1]), (x[2], x[3]), (x[0], x[3])])

                         
def slice_box(box_A:Polygon, box_B:Polygon, margin=10, line_mult=10):
     "Returns box_A sliced according to the distance to box_B."
     vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])
     vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])
     vec_AB_norm = np.linalg.norm(vec_AB)
     split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)*margin
     line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])
     split_box = shapely.ops.split(box_A, line)
     if len(split_box) == 1: return split_box, None, line
     is_center = [s.contains(box_A.centroid) for s in split_box]
     if sum(is_center) == 0: 
         warnings.warn('Polygon do not contain the center of original box, keeping the first slice.')
         return split_box[0], None, line
     where_is_center = np.argwhere(is_center).reshape(-1)[0]
     where_not_center = np.argwhere(~np.array(is_center)).reshape(-1)[0]
     split_box_center = split_box[int(where_is_center)]
     split_box_out = split_box[int(where_not_center)]
     return split_box_center, split_box_out, line
                                                   
def intersection_list(polylist):
     r = polylist[0]
     for p in polylist:
         r = r.intersection(p)
     return r
     
def slice_one(gdf, index, margin, line_mult):
     inter = gdf.loc[gdf.intersects(gdf.iloc[index].geometry)]
     if len(inter) == 1: return inter.geometry.values[0]
     box_A = inter.loc[index].values[0]
     inter = inter.drop(index, axis=0)
     polys = []
     for i in range(len(inter)):
         box_B = inter.iloc[i].values[0]
         
         ### weird error... for some reason have a couple of boxes that are IDENTICAL (or share centroid)
         if box_A == box_B or ((box_A.centroid.x == box_B.centroid.x) and (box_A.centroid.y == box_B.centroid.y)):
             print('same box')
             continue
         
         polyA, *_ = slice_box(box_A, box_B, margin, line_mult)
         polys.append(polyA)
         
     ### weird error...
     if len(polys) == 0: 
         return inter.geometry.values[0]
         
     return intersection_list(polys)
 
def slice_all(gdf, margin, line_mult):
     polys = []
     for i in range(len(gdf)):
         polys.append(slice_one(gdf, i, margin, line_mult))
     return gpd.GeoDataFrame({'geometry': polys})        
 
 
 

""" Tiger function """
def plot_max(im, ax=0, plot=1):
     max_im = np.amax(im, axis=ax)
     if plot:
         plt.figure(); plt.imshow(max_im)
     
     return max_im



""" Tiger function """
def boxes_to_mask(cf, results_dict, thresh):

    label_arr = np.copy(results_dict['seg_preds'],)
    new_labels = np.zeros(np.shape(results_dict['seg_preds']))
    for box_id, box_row in enumerate(results_dict['boxes'][0]):
        
 
        
        #if cf.dim == 2 and box_row['box_score'] < cf.merge_3D_iou:
        #    continue
        #else:
            
        if box_row['box_score'] >= thresh:
                    
            box_arr = np.zeros(np.shape(results_dict['seg_preds']))
            bc = box_row['box_coords']
            bc = np.asarray(bc, dtype=int)
            
            bc[np.where(bc < 0)[0]] = 0  ### cannot be negative
            
            ### also cannot be larger than image size
            if cf.dim == 2:
                bc[np.where(bc >= label_arr.shape[-1])[0]] = label_arr.shape[-1]

                box_arr[bc[4]:bc[5], 0, bc[0]:bc[2], bc[1]:bc[3]] = box_id + 1    ### +1 because starts from 0
                box_arr[label_arr == 0] = 0
                
                new_labels[box_arr > 0] = box_id + 1

            else:
                bc[0:4][np.where(bc[0:4] >= label_arr.shape[-2])[0]] = label_arr.shape[-2]
                
                bc[4:6][np.where(bc[4:6] >= label_arr.shape[-1])[0]] = label_arr.shape[-1]
                
                box_arr[0, 0, bc[0]:bc[2], bc[1]:bc[3], bc[4]:bc[5],] = box_id + 1    ### +1 because starts from 0
                box_arr[label_arr == 0] = 0
                
                new_labels[box_arr > 0] = box_id + 1           

        else:
            print('box_score: ' + str(box_row['box_score']))
            

                        
    label_arr = new_labels
    
    
    return label_arr





def find_pid_in_splits(pid, exp_dir=None):
    if exp_dir is None:
        exp_dir = cf.exp_dir
    check_file = os.path.join(exp_dir, 'fold_ids.pickle')
    with open(check_file, 'rb') as handle:
        splits = pickle.load(handle)

    finds = []
    for i, split in enumerate(splits):
        if pid in split:
            finds.append(i)
            print("Pid {} found in split {}".format(pid, i))
    if not len(finds)==1:
        raise Exception("pid {} found in more than one split: {}".format(pid, finds))
    return finds[0]







def plot_train_forward(slices=None):
    with torch.no_grad():
        batch = next(val_gen)
        
        results_dict = net.train_forward(batch, is_validation=True) #seg preds are int preds already
        print(results_dict['seg_preds'].shape)
        print(batch['data'].shape)
        
        
        out_file = os.path.join(anal_dir, "straight_val_inference_fold_{}".format(str(cf.fold)))
        #plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True,
        #                          out_file=out_file)#, slices=slices)

        ### TIGER - SAVE AS TIFF
        truth_im = np.expand_dims(batch['seg'], axis=0)
        
        
        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)    
        import tifffile as tiff
        tiff.imwrite(out_file + '_TRUTH.tif', np.asarray(truth_im, dtype=np.uint8),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        
                
        seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)    
        import tifffile as tiff
        tiff.imwrite(out_file + '_seg.tif', np.asarray(seg_im, dtype=np.uint8),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        

        input_im = np.expand_dims(batch['data'], axis=0)
        
        
        ### if 3D
        #input_im = np.moveaxis(batch['data'], -1, 1)
        tiff.imwrite(out_file + '_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        

        

        ### TIGER ADDED
        import utils.exp_utils as utils
        print('Plotting output')
        utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
                                show_gt_labels=True, get_time="val-example plot",
                                out_file=os.path.join(cf.plot_dir, 'batch_example_val_{}.png'.format(cf.fold)))


def plot_forward(pid, slices=None):
    with torch.no_grad():
        batch = batch_gen['test'].generate_train_batch(pid=pid)
        results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.

        if 'seg_preds' in results_dict.keys():
            results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]

        out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))
        



        print(results_dict['seg_preds'].shape)
        print(batch['data'].shape)

        ### TIGER - SAVE AS TIFF
        truth_im = np.expand_dims(batch['seg'], axis=0)
        
        
        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)    
        import tifffile as tiff
        tiff.imwrite(out_file + '_TRUTH.tif', np.asarray(truth_im, dtype=np.uint8),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        
        

        roi_mask = np.expand_dims(batch['roi_masks'][0], axis=0)
        
        
        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)    
        import tifffile as tiff
        tiff.imwrite(out_file + '_roi_mask.tif', np.asarray(roi_mask, dtype=np.uint8),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                
      

    
        seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
        
        
        ### if 3D
        #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)    
        import tifffile as tiff
        tiff.imwrite(out_file + '_seg.tif', np.asarray(seg_im, dtype=np.uint8),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        

        input_im = np.expand_dims(batch['data'], axis=0)
        
        
        ### if 3D
        #input_im = np.moveaxis(batch['data'], -1, 1)
        tiff.imwrite(out_file + '_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                      imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
        


        
        ### This below hangs
        
        # plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True, show_gt_labels=True,
        #                           out_file=out_file, sample_picks=slices, has_colorchannels=False)
        
        print('Plotting output')
        utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
                                show_gt_labels=True, get_time="val-example plot",
                                out_file=os.path.join(cf.plot_dir, 'batch_SINGLE_PID_{}.png'.format(pid)))
        
        
        
        


def plot_merged_boxes(results_list, pid, plot_mods=False, show_seg_ids="all", show_info=True, show_gt_boxes=True,
                      s_picks=None, vol_slice_picks=None, score_thres=None):
    """

    :param results_list: holds (results_dict, pid)
    :param pid:
    :return:
    """
    results_dict = [res_dict for (res_dict, pid_) in results_list if pid_==pid][0]
    #seg preds are discarded in predictor pipeline.
    #del results_dict['seg_preds']

    batch = batch_gen['test'].generate_train_batch(pid=pid)
    out_file = os.path.join(anal_dir, "merged_boxes_fold_{}_pid_{}_thres_{}.png".format(str(cf.fold), pid, str(score_thres).replace(".","_")))

    utils.save_obj({'res_dict':results_dict, 'batch':batch}, os.path.join(anal_dir, "bytes_merged_boxes_fold_{}_pid_{}".format(str(cf.fold), pid)))

    plg.view_batch(cf, batch, res_dict=results_dict, show_info=show_info, legend=False, sample_picks=s_picks,
                   show_seg_pred=True, show_seg_ids=show_seg_ids, show_gt_boxes=show_gt_boxes,
                   box_score_thres=score_thres, vol_slice_picks=vol_slice_picks, show_gt_labels=True,
                   plot_mods=plot_mods, out_file=out_file, has_colorchannels=cf.has_colorchannels, dpi=600)

    return


