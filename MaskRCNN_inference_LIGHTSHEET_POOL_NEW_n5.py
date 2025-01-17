#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:54:13 2023

@author: user
"""

                  
import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
#import utils.model_utils as mutils

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')   

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
from os import listdir
from os.path import isfile, join
import glob, os

from tifffile import *
import tkinter
from tkinter import filedialog
    
from skimage.measure import label, regionprops, regionprops_table

from inference_utils import *

#import h5py

from inference_analysis_OL_TIFFs_small_patch_POOL import post_process_async, expand_add_stragglers

#import time
from multiprocessing.pool import ThreadPool
from scipy.stats import norm


from functional.matlab_crop_function import *
from functional.tree_functions import *  

from predictor import apply_wbc_to_patient
import time

from tqdm import tqdm

import concurrent.futures

import z5py



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



if __name__=="__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"
            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/94) newest_CLEANED_shrunk_det_thresh_02_min_conf_01/'
            #self.exp_dir = '/media/user/fa2f9451-069e-4e2f-a29b-3f1f8fb64947/Training_checkpoints_RegRCNN/96) new_FOV_data_det_thresh_09_check_300'
            self.exp_dir = '/media/user/ce86e0dd-459a-4cf1-8194-d1c89b7ef7f6/Training_checkpoints_RegRCNN/96) new_FOV_data_det_thresh_09_check_300'
            
            self.server_env = False


    args = Args()
    data_loader = utils.import_module('dl', os.path.join(args.dataset_name, "data_loader.py"))

    config_file = utils.import_module('cf', os.path.join(args.exp_dir, "configs.py"))
    cf = config_file.Configs()
    cf.exp_dir = args.exp_dir
    cf.test_dir = cf.exp_dir

    cf.fold = 0
    if cf.dim == 2:
        cf.merge_2D_to_3D_preds = True
        if cf.merge_2D_to_3D_preds:
            cf.dim==3
    else:
        cf.merge_2D_to_3D_preds = False
        
    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))
    anal_dir = os.path.join(cf.exp_dir, "inference_analysis")
    
    

    logger = utils.get_logger(cf.exp_dir)
    
    ###^^^this causing a lot of output statements later
    
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    
    
    #zzz
    torch.backends.cudnn.benchmark = cf.dim == 3
    

    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    
    model_selector = utils.ModelSelector(cf, logger)

    starting_epoch = 1

    last_check = 1

    onlyfiles_check.sort(key = natsort_key1)
    weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
    net = model.net(cf, logger).cuda(device)


    # load already trained model weights
    with torch.no_grad():
        pass
    
    
        
        if last_check:
            optimizer = torch.optim.AdamW(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                   exclude_from_wd=cf.exclude_from_wd,
                                                                   ), 
                                      lr=cf.learning_rate[0])        
            checkpoint_path = os.path.join(cf.fold_dir, "last_state.pth")
            starting_epoch, net, optimizer, model_selector = \
                utils.load_checkpoint(checkpoint_path, net, optimizer, model_selector)
                

            net.eval()
            net.cuda(device)




        else:
    
            net.load_state_dict(torch.load(weight_path))
            net.eval()
            net = net.cuda(device)
    
    

    """ Select multiple folders for analysis AND creates new subfolder for results output """
    root = tkinter.Tk()
    # get input folders
    another_folder = 'y';
    list_folder = []
    input_path = "./"
    
    # initial_dir = '/media/user/storage/Data/'
    # while(another_folder == 'y'):
    #     input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
    #                                         title='Please select input directory')
    #     input_path = input_path + '/'
        
    #     print('Do you want to select another folder? (y/n)')
    #     another_folder = input();   # currently hangs forever
    #     #another_folder = 'y';
    
    #     list_folder.append(input_path)
    #     initial_dir = input_path    
        

    
    ### M125 - is dimmer
    ### M120 - is super bright somehow
    ### M123 - also dimmer
    
    list_folder = [#'/media/user/TigerDrive2/20221006_M125_MoE_BDV_fused/'
                  #'/media/user/TigerDrive2/20221010_M120_MoE_AAVs_lowest_LOW_RI_RIMS_14648_40perc_UREA_pH_12_zoom05_NOFUSE/BDV/',
                  #'/media/user/TigerDrive2/20221011_M123_MoE_AAVs_HIGHEST_RI_RIMS_15127_40perc_UREA_fresh_pH_12_zoom06_NOFUSE/BDV/BDV_BrainReg/to_reg/',
                  #'/media/user/FantomHD/20220813_M103_MoE_Tie2Cre_Cuprizone_6_weeks_RI_14614_3dayhot_7daysRIMSRT/BDV/BDV_BrainReg/to_reg/'

                  ### For new .n5 files!
                  #'/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231012_M230_MoE_PVCre_SHIELD_delip_RIMS_RI_1500_3days_5x/'
                  #'/media/user/ce86e0dd-459a-4cf1-8194-d1c89b7ef7f6/20231031_M229_MoE_PVCre_P56_SHIELD_delp_RIMS_50perc_then_100perc_expanded_slightly_more_5x/',
                  
                  #'/media/user/ce86e0dd-459a-4cf1-8194-d1c89b7ef7f6/20231115_M124_MoE_CasprtdT_Cuprizone_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/',
                
                  #'/media/user/ce86e0dd-459a-4cf1-8194-d1c89b7ef7f6/20231114_M147_MoE_FVB_7days_delip_RIMS_RI_1487_5days_5x_60perc_laser_SUNFLOWER/',
                                
                  #'/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231115_M139_MoE_CasprtdT_Cuprizone_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/'
                   
                  #'/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231117_M222_96rik_Evans_Blue_SHIELD_RIMS_RI_1499_5x_80perc_laser_SUNFLOWER/',
                    
                  #'/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231117_M115_MoE_P56_5days_delip_RIMS_RI_1487_7days_5x_80perc_laser_REDO_WITH_SUNFLOWER/'
                  
                  '/media/user/ce86e0dd-459a-4cf1-8194-d1c89b7ef7f6/20231116_M138_MoE_CasprtdT_Cup_CONTROL_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/'
                  
                  
                  ]
    

    ### Initiate poolThread
    poolThread_load_data = ThreadPool(processes=1)
    #poolThread_post_process = ThreadPool(processes=2)

    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:

        """ For testing ILASTIK images """
        images = glob.glob(os.path.join(input_path,'*.n5'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('.n5','.xml'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]


        input_name = images[0]  
        filename = input_name.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)      
        
        sav_dir = input_path + '/' + filename + '_MaskRCNN_patches'
    
         
        try:
            # Create target Directory
            os.mkdir(sav_dir)
            print("\nSave directory " , sav_dir ,  " Created ") 
        except FileExistsError:
            print("\nSave directory " , sav_dir ,  " already exists")
            
        sav_dir = sav_dir + '/'
        
        # Required to initialize all
        for file_num in range(len(examples)):
             
         
           """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
           with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[file_num]['input']   
                # input_im = tiff.imread(input_name)
            
            
                #import z5py
                f = z5py.File(examples[file_num]['input'], "r")
                
                dset = f['setup0/timepoint0/s0']
                
                #lowest_res = f['t00000']['s00']['7']['cells']
                #highest_res = f['t00000']['s00']['0']['cells']
                
                #dset = highest_res
                
                ### channel 2
                #highest_res = f['t00000']['s01']['0']['cells']
                coords_df = pd.DataFrame(columns = ['offset', 'block_num', 'Z', 'X', 'Y', 'Z_scaled', 'X_scaled', 'Y_scaled', 'equiv_diam', 'vol'])

                """ Or save to memmapped TIFF first... """
                print('creating memmap save file on disk')
                #memmap_save = tiff.memmap(sav_dir + 'temp_SegCNN.tif', shape=dset.shape, dtype='uint8')
                #memmap_save = tiff.memmap('/media/user/storage/Temp_lightsheet/temp_SegCNN_' + str(file_num) + '_.tif', shape=dset.shape, dtype='uint8')
    
    
                """ Figure out how many blocks need to be generated first, then loop through all the blocks for analysis """
                print('Extracting block sizes')
                
                """ Or use chunkflow instead??? https://pychunkflow.readthedocs.io/en/latest/tutorial.html"""
                im_size = np.shape(dset);
                depth_imL = im_size[0];  heightL = im_size[1];  widthL = im_size[2]; 
                
                total_blocks = 0;
                all_xyz = []                                               
             
            
            
                """ These should be whole values relative to the chunk size so that it can be uploaded back later! """
                # quad_size = 128 * 10
                # quad_depth = 64 * 4
                
                Lpatch_size = 128 * 10
                Lpatch_depth = 64 * 4
                
                # quad_size = round(input_size * 1/XY_scale * 3)
                # quad_depth = round(depth * 1/XY_scale * 3)
                
                #overlap_percent = 0
                ### Add padding to input_im

                # print('Total num of patches: ' + str(factorx * factory * factorz))
                    
                all_xyzL = [] 
                
                thread_post = 0
                called = 0
                
                for z in range(0, depth_imL, round(Lpatch_depth)):
                    #if z + Lpatch_depth > depth_imL:  continue

                    for x in range(0, widthL, round(Lpatch_size)):
                          #if x + Lpatch_size > widthL:  continue

                          for y in range(0, heightL, round(Lpatch_size)):
                               #if y + Lpatch_size > heightL: continue

                               #print([x, y, z]
                               all_xyzL.append([x, y, z])
                               

                ### how many total blocks to analyze:
                #print(len(all_xyzL))
                    
                def get_im(dset, s_c, Lpatch_depth, Lpatch_size):
                    
                        #tic = time.perf_counter()
                        
                        ### If nearing borders of image, prevent going out of bounds!
                        z_top = s_c[2] + Lpatch_depth
                        if z_top >= dset.shape[0]: z_top = dset.shape[0]
                        
                        y_top = s_c[1] + Lpatch_size
                        if y_top >= dset.shape[1]: y_top = dset.shape[1]
                        
                        x_top = s_c[0] + Lpatch_size
                        if x_top >= dset.shape[2]: x_top = dset.shape[2]
                        
                        input_im = dset[s_c[2]:z_top, s_c[1]:y_top, s_c[0]:x_top]
                        og_shape = input_im.shape
                        
                        #toc = time.perf_counter()
                        print('loaded asynchronously')
                        
                        #print(f"Opened subblock in {toc - tic:0.4f} seconds")
                        
                        return input_im, og_shape            
            
            
                """ Then loop through """
                #for id_c, s_c in enumerate(all_xyzL):
                    
                
                ### for continuing the run
                for id_c in range(0, len(all_xyzL)):
                    
                    #id_c = 100
                    
                    s_c = all_xyzL[id_c]

                    ### for debug:
                    #s_c = all_xyz[10]
                    tic = time.perf_counter()
                     
                    
                    
         
                    ### Load first tile normally, and then the rest as asynchronous processes
                    if id_c == 0:
                        input_im, og_shape = get_im(dset, s_c, Lpatch_depth, Lpatch_size)
                        print('loaded normally')
                        
                    else:   ### get tile from asynchronous processing instead!
                        input_im, og_shap = async_result.get()  # get the return value from your function.    
                        
                        #poolThread_load_data.close()
                        #poolThread_load_data.join()
                    

                    ### get NEXT tile asynchronously!!!
                    if id_c + 1 < len(all_xyzL):   ### but stop once it's reached the end
                        async_result = poolThread_load_data.apply_async(get_im, (dset, all_xyzL[id_c + 1], Lpatch_depth, Lpatch_size)) 
                    
                    print('one loop')
                    
                    #zzz



                    toc = time.perf_counter()
                    print(f"\nOpened subblock in {toc - tic:0.4f} seconds")                    
                    
                    """ Detect if blank in uint16 """
                    num_voxels = len(np.where(input_im > 300)[0])
                    if num_voxels < 10000:
                         print('skipping: ' + str(s_c))
                         print('num voxels with signal: ' + str(num_voxels))
                         
                         #time.sleep(10)
                         continue                

                    """ ### Run in a way that only re-does the missed ones """
                    if os.path.isfile(sav_dir + filename + '_' + str(int(id_c)) + '_df.pkl'):
                        continue
  
        
  
                    print('Analyzing: ' + str(s_c))
                    print('Which is: ' + str(id_c) + ' of total: ' + str(len(all_xyzL)))
                    
                      
  
                    """ Start inference on volume """
                    tic = time.perf_counter()
                    
                    
                    import warnings
                    overlap_percent = 0.1
                    
                    
                    #%% MaskRCNN analysis!!!
                    """ Analyze each block with offset in all directions """ 
                    #print('Starting inference on volume: ' + str(file_num) + ' of total: ' + str(len(examples)))
                    
                    ### Define patch sizes
                    patch_size=128; patch_depth=16
                    
                    ### Define overlap and focal cube to remove cells that fall within this edge
                    overlap_pxy = 14; overlap_pz = 3
                    step_xy = patch_size - overlap_pxy * 2
                    step_z = patch_depth - overlap_pz * 2
                    
                    focal_cube = np.ones([patch_depth, patch_size, patch_size])
                    focal_cube[overlap_pz:-overlap_pz, overlap_pxy:-overlap_pxy, overlap_pxy:-overlap_pxy] = 0                
                    focal_cube = np.moveaxis(focal_cube, 0, -1)
    
                    thresh = 0.9
                    
                    #thresh = 0.9
                    cf.merge_3D_iou = thresh
                    
                    im_size = np.shape(input_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
    
                    
                    ### Add padding to input_im
                    factorx = 0
                    while factorx < width/step_xy:   factorx+=1
                    end_padx = (factorx * step_xy) - width
                    
                    factory = 0
                    while factory < height/step_xy:   factory+=1
                    end_pady = (factory * step_xy) - height
                    
                    factorz = 0
                    while factorz < depth_im/step_z:   factorz+=1
                    end_padz = (factorz * step_z) - depth_im     
                    
                    print('Total num of patches: ' + str(factorx * factory * factorz))
                    new_dim_im = np.zeros([overlap_pz*2 + depth_im + end_padz, overlap_pxy*2 + width + end_padx, overlap_pxy*2 + height + end_pady])
                    new_dim_im[overlap_pz: overlap_pz + depth_im, overlap_pxy: overlap_pxy + width, overlap_pxy: overlap_pxy + height] = input_im
                    input_im = new_dim_im
                    
                    im_size = np.shape(input_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];                
                    
                    
                    ### Define empty items
                    box_coords_all = []; total_blocks = 0;
                    segmentation = np.zeros([depth_im, width, height])
                    #colored_im = np.zeros([depth_im, width, height])
                    
                    #split_seg = np.zeros([depth_im, width, height])
                    all_xyz = [] 
                    all_blknum = []
                    
                    
                    all_patches = []
                    all_output = []
                    
                    batch_size = 1
                    batch_im = []
                    batch_xyz = []
                    
                    
                    debug = 0
                    
                    
                    ### SET THE STEP SIZE TO BE HALF OF THE IMAGE SIZE
                    #step_z = patch_depth/2
                    #step_xy = patch_size/2
                    
                    ### add progress bar
                    # estimate number of blocks
                    pbar = tqdm(total=factorx * factory * factorz,
                                desc="Loading",
                                ncols = 75)
                    
                    
                    
                    
                    #%% START MASK-RCNN analysis
                    for z in range(0, depth_im + patch_depth, round(step_z)):
                        if z + patch_depth > depth_im:  continue
    
                        for x in range(0, width + patch_size, round(step_xy)):
                              if x + patch_size > width:  continue
    
                              for y in range(0, height + patch_size, round(step_xy)):
                                   if y + patch_size > height: continue
    
 
                                   quad_intensity = input_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size];  
                                   quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                                   quad_intensity = np.asarray(np.expand_dims(np.expand_dims(quad_intensity, axis=0), axis=0), dtype=np.float16)
                                   
                                   


                                   """ Detect if blank in uint16 """
                                   num_voxels = len(np.where(quad_intensity > 300)[0])
                                   if num_voxels < 300:
                                         #print('skipping: ' + str(s_c))
                                         #print('num voxels with signal: ' + str(num_voxels))
                                         pbar.update(1)  ### update pbar even if skipped!
                                         continue     
 
                                   if len(batch_im) > 0:
                                       batch_im = np.concatenate((batch_im, quad_intensity))
                                   else:
                                       batch_im = quad_intensity
                                   
                                   
                                   batch_xyz.append([x,y,z])
                                   
                                   #print(total_blocks)
                                   total_blocks += 1   
                                   
                                   if total_blocks % batch_size == 0:
          
    
                                       batch = {'data':batch_im, 'seg': np.zeros([batch_size, 1, patch_size, patch_size, patch_depth]), 
                                                 'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                                 'roi_masks': np.zeros([batch_size, 1, 1, patch_size, patch_size, patch_depth]),
                                                 'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                                 'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                       
    
                                       output = net.test_forward(batch, main_brain=True) #seg preds are only seg_logits! need to take argmax.
                                                                        
                                       all_output.append(output)

                                           
                                       ### Reset batch
                                       batch_im = []
                                       batch_xyz = []
                                       
                                       #patch = np.moveaxis(quad_intensity, -1, 1)
                                       #save = np.concatenate((patch, seg_im), axis=2)
                                       
                                       all_xyz.append([x, y, z])
                                       all_blknum.append(total_blocks)
                                       
                                       pbar.update(1)

                    pbar.close()
                    #zzz
                    

                    toc = time.perf_counter()
                    
                    print(f"MaskRCNN analysis in {toc - tic:0.4f} seconds")     


                    #%% Process in parallel outputs to extract result_dict
                    def parse_outputs_to_coords(kwargs):
                        #print(kwargs)
                        
                        output = kwargs[0]
                        xyz_patch = kwargs[1]
                        thresh = kwargs[2]
                        blk_num = kwargs[3]
                        
                        #print(output.shape)
                        #print(blk_num)
                        
                        
                        results_dict = net.get_results(img_shape=output[0], detections=output[1], 
                                                       detection_masks=output[2], return_masks=output[3]) 
                        im_shape = output[0]
     
                        ### Add box patch factor
                        new_box_list = []
                        tmp_check = np.zeros(im_shape[2:])
                        
                        for bid, box in enumerate(results_dict['boxes'][0]):
                              """ 
                              
                                  Some bounding boxes have no associated segmentations!!! Skip these
    
                              """
                              if len(box['mask_coords']) == 0:
                                  continue
                                                                 
                              ### Only add if above threshold
                              if box['box_score'] > thresh:
                                  c = box['box_coords']
     
                                  box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                                  box_centers.append((c[4] + c[5]) / 2)
     
                                  # factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in \
                                  #                   zip(box_centers, np.array(quad_intensity[0][0].shape) / 2)]
                                  # slightly faster call
                                  pc =  np.array(im_shape[2:]) / 2
                                  factor = np.mean([norm.pdf(box_centers, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 ])
                                     
                                     
                                  box['box_patch_center_factor'] = factor
                                  new_box_list.append(box)     
                                
       
                        
                        results_dict['boxes'] = [new_box_list]
                        #results_dict = results_dict
                    
     
    
     
                        if 'seg_preds' in results_dict.keys():
                              results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                            # results_dict['colored_boxes'] = np.expand_dims(results_dict['colored_boxes'][:, 1, :, :, :], axis=0)
                             
                        ### Add to segmentation
                        seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                        
                        
                        ### get coords so can plot into segmentation later
                        coords = np.transpose(np.where(seg_im[0, :, 0, :, :] > 0))
                        
                        coords[:, 0] = coords[:, 0] + xyz_patch[2]
                        coords[:, 1] = coords[:, 1] + xyz_patch[0]
                        coords[:, 2] = coords[:, 2] + xyz_patch[1]
                        
                        #color_im = np.moveaxis(results_dict['colored_boxes'], -1, 1) 
                        #segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] + seg_im[0, :, 0, :, :]
                        #colored_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = color_im[0, :, 0, :, :]
                        
                        
                        ### add option for larger batching?
                        #for bs in range(batch_size):
 
                        # box_df = results_dict['boxes'][bs]
                        # box_vert = []
                        # box_score = []
                        # mask_coords = []
                        # for box in box_df:
                           
                        #         box_vert.append(box['box_coords'])
                        #         box_score.append(box['box_score'])
                        #         mask_coords.append(box['mask_coords'])
                                
                        # if len(box_vert) == 0:
                        #     continue
 
 
                        #patch_im = batch_im[bs][0]
                        #patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first
                     
                        # save memory by deleting seg_preds
                        results_dict['seg_preds'] = []
     
                        # all_patches.append({ 'results_dict': results_dict, 'total_blocks': bs % total_blocks + (total_blocks - batch_size), 
                        #                       #'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                        #                       'xyz':xyz_patch})
   
                        #all_patches.append({ 'results_dict': results_dict, 'total_blocks': total_blocks, 
                        #                      #'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                        #                      'xyz':xyz_patch})
                        
                        
                        return results_dict, blk_num, xyz_patch, coords
    
                        
                    tic = time.perf_counter()
                    ### add x,y,z and save segmentation
                    kwargs = zip(all_output, all_xyz, [thresh] * len(all_xyz), all_blknum)
                    
                    
                    #kwargs = {'output':all_output, 'xyz_patch':all_xyz, 'thresh':[thresh]*len(all_xyz), 'total_blocks':[total_blocks]*len(all_xyz)}
                        
                    exec_results = concurrent.futures.ThreadPoolExecutor(10)
                    results = list(tqdm(exec_results.map(parse_outputs_to_coords, kwargs), total=len(all_output)))
                    exec_results.shutdown(wait=True)
                    
                    toc = time.perf_counter()
                    print(f"Parse output coords in {toc - tic:0.4f} seconds")     
                    
                    
                    seg_coords = []
                    for result in results:
                        all_patches.append({ 'results_dict': result[0], 'total_blocks': result[1], 
                                              #'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                                              'xyz':result[2]})
                        seg_coords.append(result[3])
                    
                    ### plot to make segmentation array with overlapping regions showing
                    for coords in seg_coords:
                        ### create segmentation image
                        segmentation[coords[:, 0], coords[:, 1], coords[:, 2]] = segmentation[coords[:, 0], coords[:, 1], coords[:, 2]] + 1
                    
                    # pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                    # result = pool.map(parse_outputs_to_coords, cc)
                    # pool.shutdown(wait=True)


                    # def parse_outputs_to_coords(output, xyz_patch, thresh, total_blocks):
                        
                        
                    #zzz

    
    

                    #%% Post-process boxes       
                    
                    ### First save the files and remove them from RAM so we don't max out
                    print('saving files')                                 
                    im_size = np.shape(input_im);
                    
                    filename = input_name.split('/')[-1].split('.')[0:-1]
                    filename = '.'.join(filename)
                
                    input_im = np.expand_dims(input_im, axis=0)
                    input_im = np.expand_dims(input_im, axis=2)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                          imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                          metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                                                 
                
                    segmentation = np.asarray(segmentation, np.uint8)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_segmentation_overlap3.tif', segmentation)                    
                    #post_process_async(cf, input_im, segmentation, input_name, sav_dir, all_patches, patch_size, patch_depth, id_c, focal_cube, s_c, debug)
                    
                    #zzz
                    ### Then call asynchronous post-processing to sort out boxes
                    if called:
                        executor.submit(post_process_async, cf, all_patches, im_size, filename, sav_dir, overlap_pxy, overlap_pz,
                                                                 patch_size, patch_depth, id_c, focal_cube, s_c=s_c, debug=debug)
                    
                    else:
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                        
                        executor.submit(post_process_async, cf, all_patches, im_size, filename, sav_dir, overlap_pxy, overlap_pz,
                                                                 patch_size, patch_depth, id_c, focal_cube, s_c=s_c, debug=debug)
                        
                        called = 1
                    
            
                    ### clean-up
                    segmentation = []; input_im = []; all_patches = []
                    
                    

                    #%% Output to memmap array
                    
                    """ Output to memmmapped arr """
                    
                    # tic = time.perf_counter()
                    # memmap_save[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size] = segmentation
                    # memmap_save.flush()
                    
                    # toc = time.perf_counter()
                    
                    # print(f"Save in {toc - tic:0.4f} seconds")
                    
                    #tic = time.perf_counter()
                

    print('\n\nSegmented outputs saved in folder: ' + sav_dir)
    
    
    ### REMEMBER TO CLOSE POOL THREADS
    poolThread_load_data.close()
    #poolThread_post_process.close()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    