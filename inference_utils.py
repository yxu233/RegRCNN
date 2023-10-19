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
from skimage.draw import polygon, polygon_perimeter, line, line_aa
import skimage
import random

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import shapely.geometry
import shapely.ops
from skimage import measure
from skimage import morphology


 
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
def split_boxes_by_Voronoi3D(box_coords, vol_shape):
    debug = 0
    
    box_ids = []
    bbox = []
    for box_id, box3d in enumerate(box_coords):
       
       #for depth in range(box3d[4], box3d[5] + 1):
          bbox.append(box3d)   ### only get x1, y1, x2, y2
       #    box_depth.append(depth)
          box_ids.append(box_id + 1)  ### cant start from zero!!!
           

    y1 = box_coords[:,0]
    x1 = box_coords[:,1]
    y2 = box_coords[:,2]
    x2 = box_coords[:,3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    #if dim == 3:
    z1 = box_coords[:, 4]
    z2 = box_coords[:, 5]
    areas *= (z2 - z1 + 1)

    
   

    centroids = [np.round((box_coords[:, 5] - box_coords[:, 4])/2 + box_coords[:, 4]), np.round((box_coords[:, 2] - box_coords[:, 0])/2 + box_coords[:, 0]), np.round((box_coords[:, 3] - box_coords[:, 1])/2 + box_coords[:, 1])]
    centroids = np.transpose(centroids)
    centroids = np.asarray(centroids, dtype=int)
    
    df = pd.DataFrame(data={'ids': box_ids, 'bbox':bbox, 'bbox_coords':[ [] for _ in range(len(box_ids)) ]})
    

    ### removes infinite lines from voronoi if you add the corners of the image!
    hack = np.asarray([[-100,-100,-100],
                        [-100, vol_shape[1]*10, -100],
                        [-100, -100, vol_shape[2]*10],
                        [-100, vol_shape[1]*10, vol_shape[2]*10],
                        
                        [vol_shape[0]*10, -100, -100],
                        [vol_shape[0]*10, vol_shape[1]*10, -100],
                        [vol_shape[0]*10, -100, vol_shape[2]*10],
                        [vol_shape[0]*10, vol_shape[1]*10, vol_shape[2]*10]
                        ])
                        
  
    centroids_vor = np.concatenate((centroids, hack))
    
    
    no_overlap = np.zeros(vol_shape)
    tmp_boxes = np.zeros(vol_shape)
    list_coords = []
    for b_id, box in enumerate(box_coords):
        
        a,b,c = np.meshgrid(np.arange(box[4], box[5]), np.arange(box[0], box[2]), np.arange(box[1], box[3]))
        
        
        ### TIGER UPDATE - shrink bounding boxes because they are expanded!!!
        # z_r = np.arange(box[4] + 1, box[5] - 1)
        # if len(z_r) == 0: z_r = box[4] + 1
        
        # x_r = np.arange(box[0] + 1, box[2] - 1)
        # if len(x_r) == 0: x_r = box[0] + 1        
        
        # y_r = np.arange(box[1] + 1, box[3] - 1)
        # if len(y_r) == 0: y_r = box[3] + 1        
        
        # a,b,c = np.meshgrid(z_r, x_r, y_r)
        
        
        coords = np.vstack([a.ravel(), b.ravel(), c.ravel()]).T  # unravel and transpose
                       
        list_coords.append(coords)
        
        
        no_overlap[coords[:, 0], coords[:, 1], coords[:, 2]] = b_id + 1  ### can't start from zero!!!
        tmp_boxes[coords[:, 0], coords[:, 1], coords[:, 2]]  = tmp_boxes[coords[:, 0], coords[:, 1], coords[:, 2]]  + 1
        
 

    intersect_ids = []
    for b_id, coords in enumerate(list_coords):
           val = np.max(tmp_boxes[coords[:, 0], coords[:, 1], coords[:, 2]])
           if val > 1:
               intersect_ids.append(b_id)
           # else:

           #     df.at[b_id, 'bbox_coords'] = coords  ### Unnecessary, because ALL boxes are added at the end with NO exceptions
                  
    
    
    from scipy.spatial import cKDTree
    voronoi_kdtree = cKDTree(centroids)  ### only split centroids of cells with overlap
    
    split_im = np.zeros(vol_shape)
    for b_id in intersect_ids:  
        
        coords = list_coords[b_id]
        test_point_dist, test_point_regions = voronoi_kdtree.query(coords, k=1)
        
        split_im[coords[:, 0], coords[:, 1], coords[:, 2]] = test_point_regions + 1  ### can't start from zero!!!
        
        #plot_max(split_im); print(b_id)
        
        #zzz
        
        #for idp, p in enumerate(coords):
        #    split_im[p[0], p[1], p[2]] = test_point_regions[idp]

    #plot_max(split_im)
    
    
    ### Now set the overlap regions to be of value in split_im
    overlap_assigned = np.copy(tmp_boxes)                           ### start with the overlap array
    overlap_assigned[tmp_boxes <= 1] = 0                            ### remove everything that is NOT overlap
    overlap_assigned[tmp_boxes > 1] = split_im[tmp_boxes > 1]       ### set all overlap regions to have value from split_im array

    overlap_assigned[tmp_boxes <= 1] = no_overlap[tmp_boxes <= 1]   ### Now add in all the rest of the boxes!!! INCLUDING PROPER NUMBER INDEXING
    
    overlap_assigned = np.asarray(overlap_assigned, dtype=int)
    
    if debug:
        plot_max(no_overlap)
        plot_max(overlap_assigned)
        
        plt.figure(); plt.imshow(no_overlap[20])
        plt.figure(); plt.imshow(overlap_assigned[20])
    
    
    cc = measure.regionprops(overlap_assigned, intensity_image=overlap_assigned)
    for b_id, cell in enumerate(cc):
        coords = cell['coords']
        
        box_id = cell['max_intensity'] - 1  ### Convert from array value back to index of array which starts from zero!!!
        
        df.at[b_id, 'bbox_coords'] = coords

    return df
  

""" Remove overlap between bounding boxes """
### since 3d polygons are complex, we will do 2d slice-by-slice to cut up the boxes
def split_boxes_by_Voronoi(box_coords, vol_shape):
    debug = 0
    
    box_2d = []
    box_ids = []
    box_depth = []
    for box_id, box3d in enumerate(box_coords):
       
       for depth in range(box3d[4], box3d[5] + 1):
           box_2d.append(box3d[0:4])   ### only get x1, y1, x2, y2
           box_depth.append(depth)
           box_ids.append(box_id)
           

    df = pd.DataFrame(data={'ids': box_ids, 'depth': box_depth, 'bbox':box_2d, 'bbox_coords':[ [] for _ in range(len(box_ids)) ]})
   
    df_cleaned = pd.DataFrame()

    

    ### Go slice by slice to correct the masks
    for i_d in range(vol_shape[0] + 1):
        
        
       df_slice = df.iloc[np.where(df['depth'] == i_d)[0]].reset_index()

      
       if len(df_slice) == 0:
           continue
       
       boxes_slice = df_slice['bbox']  
       boxes_slice = np.vstack(boxes_slice)
        
       ### also get boxes
       tmp_boxes = np.zeros(vol_shape[1:])
       no_overlap = np.zeros(vol_shape[1:])
       for b_id, box in enumerate(boxes_slice):
           tmp_boxes[box[0]:box[2], box[1]:box[3]] = tmp_boxes[box[0]:box[2], box[1]:box[3]] + 1
           no_overlap[box[0]:box[2], box[1]:box[3]] = b_id

    
       no_overlap[tmp_boxes > 1] = 0

       intersect_ids = []
       for b_id, box in enumerate(boxes_slice):
           val = np.max(tmp_boxes[box[0]:box[2], box[1]:box[3]])
           if val > 1:
               intersect_ids.append(b_id)
           else:
               a,b = np.meshgrid(np.arange(box[0], box[2]), np.arange(box[1], box[3]))
               coords = np.vstack(np.transpose([a, b]))
               
               ### SWAP axes to match older code
               coords[:, [0, 1]] = coords[:, [1, 0]]
               
               
               coords = np.append(coords, np.ones([len(coords), 1]) * i_d, axis=1)
               coords = np.asarray(coords, dtype=int)
               
               df_slice.at[b_id, 'bbox_coords'] = coords
               
               
       ### Skip if no intersecting boxes
       if len(intersect_ids) == 0:
           ### append to mother list
           df_cleaned = df_cleaned.append(df_slice, ignore_index=True)
           continue
             
       b_inter = boxes_slice[intersect_ids]
       boxes_slice = b_inter
        
       centroids = [np.round((boxes_slice[:, 2] - boxes_slice[:, 0])/2 + boxes_slice[:, 0]), np.round((boxes_slice[:, 3] - boxes_slice[:, 1])/2 + boxes_slice[:, 1])]
       centroids = np.transpose(centroids)
       centroids = np.asarray(centroids, dtype=int)
       
       ### removes infinite lines from voronoi if you add the corners of the image!
       hack = np.asarray([[-100,-100],
                          [0, vol_shape[2] *10],
                          [vol_shape[1] * 10,0],
                          [vol_shape[1] *10 , vol_shape[2]]])
       
       centroids_vor = np.concatenate((centroids, hack))
       
        
       vor = Voronoi(centroids_vor)
       lines = [shapely.geometry.LineString(vor.vertices[line]) for line in 
           vor.ridge_vertices if -1 not in line]
       
       
       line_coords = []
       line_im = np.zeros(vol_shape[1:])
       for id_p, line in enumerate(lines):
           #print(poly)
           
       
           lc = np.array(list(line.coords))
           lc = np.asarray(lc, dtype=int)
           

                    
            
           rr, cc = skimage.draw.line(lc[0,0],lc[0,1],lc[1,0],lc[1,1])
                  
           coords = np.transpose([rr, cc])
           
           
           ### if coordinates go beyond image, just delete them!!! Dont try to bound them
           # coords[np.where(coords[:, 0] >= vol_shape[1])[0], 0] = vol_shape[1] - 1
           # coords[np.where(coords[:, 1] >= vol_shape[2])[0], 1] = vol_shape[2] - 1
           
           # coords[np.where(coords[:, 0] < 0)[0], 0] = 0
           # coords[np.where(coords[:, 1] < 0)[0], 1] = 0
                   
           
           coords = coords[np.where((coords[:, 0] < vol_shape[1]) & (coords[:, 0] >= 0))[0]]
           coords = coords[np.where((coords[:, 1] < vol_shape[2]) & (coords[:, 1] >= 0))[0]]
           coords[np.where(coords[:, 1] >= vol_shape[2])[0], 1] = vol_shape[2] - 1
          
           
           
           line_coords.append(coords)
           
           line_im[coords[:, 0], coords[:, 1]] = 1
           
           
       ### remove edges of line_im with focal_cube to prevent weird cut offs

       ### also get boxes
       clean_boxes = np.zeros(vol_shape[1:])
       for b_id, box in enumerate(boxes_slice):
           clean_boxes[box[0]:box[2], box[1]:box[3]] = 1
           
       # for DEBUG
       if debug:
           plt.figure(); plt.imshow(clean_boxes + line_im)
       clean_boxes[line_im > 0] = 0
       


       ####################### Now figure out which boxes are cut, and which ID they correspond to
       
       lab_boxes = measure.label(clean_boxes, connectivity=1)
       
       
       ### remove holes from non-specific voronoi line cutting
       disk = morphology.disk(1)
            
       dil_box = morphology.dilation(lab_boxes, selem=disk)
       lab_boxes = morphology.erosion(dil_box, selem=disk)
           
       
       
       cc = measure.regionprops(lab_boxes)
       
    
       for obj in cc:
           
           coords = obj['coords']
           
           
           ### SWAP axes to match older code
           coords[:, [0, 1]] = coords[:, [1, 0]]
           
           #coords = np.transpose([cc, rr, np.ones(len(cc)) * i_d])
           
           ### append i_d to the coordinates
           coords = np.append(coords, np.ones([len(coords), 1]) * i_d, axis=1)
           coords = np.asarray(coords, dtype=int)

           # get most frequent value, not just max intensity
           intensity = no_overlap[coords[:, 1], coords[:, 0]]
           intensity = np.asarray(intensity, dtype=int)
           count = np.bincount(intensity[intensity != 0])
           
           val_id = -1
           if len(count) > 0:
               val_id = np.argmax(count) 

           else:  ### if no matching values found, will have to index by intersection id
               unq, count = np.unique(np.concatenate((coords[:, 0:2], centroids)), axis=0, return_counts=True)
               
               if np.max(count) == 1:
                   #print('discard due to lack of intersections and was cut up')
                   #print(len(coords[:, 0:2]))
                   continue
               else:
                   val_id = np.argwhere(np.isin(centroids, unq[count > 1][0]).all(axis=1))[0][0]
                   
                 
                   
           ### if a previous set of coordinates already exists at the spot, then concatenate with new coordinates
           if val_id != -1:
               if len(df_slice.at[val_id, 'bbox_coords']) == 0:
                   df_slice.at[val_id, 'bbox_coords'] = coords
                   
               else:
                   df_slice.at[val_id, 'bbox_coords'] = np.concatenate((df_slice.at[val_id, 'bbox_coords'], coords))

           #print(val_id)
           
    
       # for DEBUG
       if debug:
           clean_boxes = np.zeros(vol_shape[1:])
           for b_id, box in enumerate(df_slice['bbox_coords'].values):
                   if len(box) == 0:
                       print('empty box')
                       continue
                   clean_boxes[box[:, 1], box[:, 0]] = b_id + 1
           plt.figure(); plt.imshow(clean_boxes) 
           plt.figure(); plt.imshow(tmp_boxes)
           


       df_cleaned = df_cleaned.append(df_slice, ignore_index=True)
       
 
    return df_cleaned
  

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

        
       boxes_slice = np.vstack(boxes_slice)
        
   
#        multiBox = shapely.geometry.MultiPolygon([box2polygon(b) for b in boxes_slice])
     
#        # collection of individual linestrings for splitting in a list and add the polygon lines to it.
#        for line in multiBox.boundary:
#           lines.append(line)
         

#        merged_lines = shapely.ops.linemerge(lines)     
#        border_lines = shapely.ops.unary_union(merged_lines)

#        decomposition, cuts, dangles, invalids = shapely.ops.polygonize_full(border_lines)
#        #decomposition.geoms[2]
                    
               


#        match_df = {'bbox_coords':[], 'shape_order':[], 'vols':[], 'poly_id':[]}
#        #match_df = pd.DataFrame(match_df)


#        bbox_coords = []
#        shape_order = []
#        for id_p, poly in enumerate(decomposition):
#            #print(poly)
           
       
#            poly_coordinates = np.array(list(poly.exterior.coords))
            
#            rr, cc = polygon(poly_coordinates[:,1], poly_coordinates[:,0], vol_shape[1:])
                  
#            coords = np.transpose([cc, rr])
       
        
#            match = np.array([x for x in set(tuple(x) for x in coords) & set(tuple(x) for x in centroids)])
#            print(len(match))
#            if len(match) > 0:
#                id_m = np.argwhere(np.isin(centroids, match).all(axis=1))[0][0]  ### should only ever been 1 centroid so will be unique
               
#                # if id_m == 8:
#                #     print(id_p)
#                #     empty = np.zeros(vol_shape[1:])
#                #     empty[cc, rr] = 1
#                #     plt.figure(); plt.imshow(empty)
#                #     print
                   
#                    #zzz
               
#                coords = np.transpose([cc, rr, np.ones(len(cc)) * i_d])
               
#                coords = np.asarray(coords, dtype=int)
               
               
#                match_df['shape_order'].append(id_m)
#                match_df['bbox_coords'].append(coords)
#                match_df['vols'].append(len(coords))
#                match_df['poly_id'].append(id_p)
               
               
#                #shape_order.append(id_m)
#                #bbox_coords.append(coords)
           
#                #zzz
       
#        match_df = pd.DataFrame(match_df)
        
#        ### Because some polygons are fully enclosed, must pick smallest of the matched identical IDs
       
#        vals, idx_start, count = np.unique(match_df['shape_order'], return_counts=True, return_index=True)
#        dup_ids = match_df['shape_order'][np.where(count > 1)[0]]
       
#        to_del = []
#        for dup in dup_ids:
           
#            match_ids = np.where(match_df['shape_order'] == dup)[0]
           
#            volume = match_df.iloc[match_ids]['vols']
           
#            to_del.append(match_ids[np.argmax(volume)])
           
          
#        match_df = match_df.drop(index=to_del)
           
           
#        bbox_coords = match_df['bbox_coords'].values
       
       
#        empty = np.zeros(vol_shape[1:])
#        for id_check, check in enumerate(bbox_coords):
       
#                #if id_m == 8:
#                    print(id_p)
                   
#                    empty[check[:, 0], check[:, 1]] = id_check
#        plt.figure(); plt.imshow(empty)
       
        


       
        
     
           
#        df_slice['bbox_coords'] = bbox_coords
#        #df_slice['bbox_perims'] = bbox_perim
#        #print('SLOW1: ' + str(id_r))
           
#        df_cleaned = df_cleaned.append(df_slice, ignore_index=True)
       
       
       
       
       
       
       
#        multiVor = shapely.multipolygons(shapely.get_parts(shapely.polygonize(multiLines)))
       
       
#        multiLines= shapely.geometry.MultiLineString(lines)
      
       
       
       
       
#     #def voronoi_volumes(points):
# #         v = Voronoi(centroids)
# #         vol = np.zeros(v.npoints)
# #         for i, reg_num in enumerate(v.point_region):
# #             indices = v.regions[reg_num]
# #             if -1 in indices: # some regions can be opened
# #                 vol[i] = np.inf
# #             else:
# #                 vol[i] = ConvexHull(v.vertices[indices]).volume
                
# #                 hull = ConvexHull(v.vertices[indices])
# #                 bools = in_hull(p, hull)
# #                 zzz
                
# #         return vol


# # def in_hull(p, hull):
# #     """
# #     Test if points in `p` are in `hull`

# #     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
# #     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
# #     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
# #     will be computed
# #     """
# #     if not isinstance(hull,Delaunay):
# #         hull = Delaunay(hull)

# #     return hull.find_simplex(p)>=0


# # p = np.transpose(np.where(patch_im > -1))



# # v.points[reg_num]



# # from scipy.spatial import Voronoi,Delaunay
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def tetravol(a,b,c,d):
# #  '''Calculates the volume of a tetrahedron, given vertices a,b,c and d (triplets)'''
# #  tetravol=abs(np.dot((a-d),np.cross((b-d),(c-d))))/6
# #  return tetravol

# # def vol(vor,p):
# #  '''Calculate volume of 3d Voronoi cell based on point p. Voronoi diagram is passed in v.'''
# #  dpoints=[]
# #  vol=0
# #  for v in vor.regions[vor.point_region[p]]:
# #      dpoints.append(list(vor.vertices[v]))
# #  tri=Delaunay(np.array(dpoints))
# #  for simplex in tri.simplices:
# #      vol+=tetravol(np.array(dpoints[simplex[0]]),np.array(dpoints[simplex[1]]),np.array(dpoints[simplex[2]]),np.array(dpoints[simplex[3]]))
# #  return vol

# # # x= [np.random.random() for i in xrange(50)]
# # # y= [np.random.random() for i in xrange(50)]
# # # z= [np.random.random() for i in xrange(50)]
# # # dpoints=[]
# # # points=zip(x,y,z)
# # centroids = [(box_vert[:, 2] - box_vert[:, 0])/2 + box_vert[:, 0], (box_vert[:, 3] - box_vert[:, 1])/2 + box_vert[:, 1], (box_vert[:, 5] - box_vert[:, 4])/2 + box_vert[:, 4]]

# # centroids = np.transpose(centroids)
# # vor=Voronoi(centroids)
# # vtot=0




# # for i,p in enumerate(vor.points):
# #  out=False
# #  for v in vor.regions[vor.point_region[i]]:
# #      if v<=-1: #a point index of -1 is returned if the vertex is outside the Vornoi diagram, in this application these should be ignorable edge-cases
# #        out=True
# #      else:
# #          if not out:
# #              pvol=vol(vor,i)
# #              vtot+=pvol
# #              print ("point "+str(i)+" with coordinates "+str(p)+" has volume "+str(pvol))

# # print "total volume= "+str(vtot)

       
#        zzz
       
     #   polys = shapely.ops.polygonize(lines)
     #   voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
        
     #   result = gpd.overlay(df1=voronois, df2=gdf, how="union")
       
     #   fig, ax = plt.subplots(figsize=(15, 15))
     #   #polydf.boundary.plot(ax=ax, edgecolor="blue", linewidth=6)
     #   result.plot(ax=ax, color="red", alpha=0.3, edgecolor="black")
     #   plt.xlim([0, 128]); plt.ylim([0, 128])
        
        
        
     # #line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])
     
     
     
     # multiBox.boundary
     # line_split_collection=lines
     # # collection of individual linestrings for splitting in a list and add the polygon lines to it.
     # for line in multiBox.boundary:
     #     line_split_collection.append(line)
         
     # #line_split_collection.append(multiBox.boundary).append(multiBox.boundary) 
     
     
     
     # line_split_collection=lines.append(multiBox[0].boundary)
     # merged_lines = shapely.ops.linemerge(line_split_collection)     
     # #merged_line = shapely.ops.linemerge(multiLines)
     
     
     
     
     # #split_box = shapely.ops.split(multiBox, merged_line)        
        
     # border_lines = shapely.ops.unary_union(merged_lines)
     # decomposition = shapely.ops.polygonize(border_lines)     
            
     # for line in lines:
     #     s = shapely.ops.split(multiBox, line)
     #     print(s)
     #    #points.plot(ax=ax, color="maroon")       

     #   zzz


     #   fig = voronoi_plot_2d(vor)
     #   plt.show()












        
        
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



import time
from multiprocessing import Pool
#from multiprocessing import freeze_support


# def run_multiprocessing(func, i, n_processors):
#     with Pool(processes=n_processors) as pool:
#         return pool.map(func, i)
    
 
def slice_all(gdf, margin, line_mult):
    
    
     polys = []
     for i in range(len(gdf)):
         polys.append(slice_one(gdf, i, margin, line_mult))
         
         
         
         #zzz
         #freeze_support()
     # p = Pool(2)
     # kwargs  = [{"gdf":gdf, "index":i, "margin":margin, "line_mult":line_mult} for i in range(len(gdf))]
     # polys = p.map(slice_one, kwargs)
     #zzz
         
        
         #gdf, index, margin, line_mult):
    
         
         
     return gpd.GeoDataFrame({'geometry': polys})        


# def slice_one(kwargs):
    
#       #print('hello')
#      gdf = kwargs["gdf"]
#      index = kwargs["index"]
#      margin = kwargs["margin"]
#      line_mult = kwargs["line_mult"]
     
     #print(line_mult)
    
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
     









# margin=0.5; line_mult=1.2
 
# def slice_all(gdf, margin, line_mult):
#      polys = []
#      for i in range(len(gdf)):
#          #polys.append(slice_one(gdf, i, margin, line_mult))
#      #return gpd.GeoDataFrame({'geometry': polys})    
#          index = i


#          inter = gdf.loc[gdf.intersects(gdf.iloc[index].geometry)]
#          if len(inter) == 1: 
             
#              a = inter.geometry.values[0]
#              polys.append(a)
             
#          box_A = inter.loc[index].values[0]
#          inter = inter.drop(index, axis=0)
#          polys = []
#          for c in range(len(inter)):
#              box_B = inter.iloc[c].values[0]
#              zzz
#              ### weird error... for some reason have a couple of boxes that are IDENTICAL (or share centroid)
#              if box_A == box_B or ((box_A.centroid.x == box_B.centroid.x) and (box_A.centroid.y == box_B.centroid.y)):
#                  print('same box')
#                  continue
             
#              polyA, *_ = slice_box(box_A, box_B, margin, line_mult)
#              polys.append(polyA)
             
#          ### weird error...
#          if len(polys) == 0: 
#              return inter.geometry.values[0]
             
#      return intersection_list(polys)
                         
# def slice_box(box_A:Polygon, box_B:Polygon, margin=10, line_mult=10):
#      "Returns box_A sliced according to the distance to box_B."
#      vec_AB = np.array([box_B.centroid.x - box_A.centroid.x, box_B.centroid.y - box_A.centroid.y])
#      vec_ABp = np.array([-(box_B.centroid.y - box_A.centroid.y), box_B.centroid.x - box_A.centroid.x])
#      vec_AB_norm = np.linalg.norm(vec_AB)
#      split_point = box_A.centroid + vec_AB/2 - (vec_AB/vec_AB_norm)*margin
#      line = LineString([split_point-line_mult*vec_ABp, split_point+line_mult*vec_ABp])
#      split_box = shapely.ops.split(box_A, line)
#      if len(split_box) == 1: return split_box, None, line
#      is_center = [s.contains(box_A.centroid) for s in split_box]
#      if sum(is_center) == 0: 
#          warnings.warn('Polygon do not contain the center of original box, keeping the first slice.')
#          return split_box[0], None, line
#      where_is_center = np.argwhere(is_center).reshape(-1)[0]
#      where_not_center = np.argwhere(~np.array(is_center)).reshape(-1)[0]
#      split_box_center = split_box[int(where_is_center)]
#      split_box_out = split_box[int(where_not_center)]
#      return split_box_center, split_box_out, line


# def intersection_list(polylist):
#      r = polylist[0]
#      for p in polylist:
#          r = r.intersection(p)
#      return r








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


