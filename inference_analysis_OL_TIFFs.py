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

import inference_utils


if __name__=="__main__":
    class Args():
        def __init__(self):
            #self.dataset_name = "datasets/prostate"
            #self.dataset_name = "datasets/lidc"
            self.dataset_name = "datasets/toy"
            
            self.dataset_name = "datasets/OL_data"
            
            #self.dataset_name = "datasets/Caspr_data"

            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/66) new_training_data_NO_dil_good_adjusted_LR_at300_epoch/'
                        
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/67) det_nms_iou_0_1_BETTER/'

            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/69) dilated_data_det_nms_iou_0_1/'  
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/71) dil_batch_norm/'
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/72) dil_group_norm_det_nms_thresh_0_1/'
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/73) dil_group_norm_det_thresh_0_2/'
                      
            self.server_env = False

    mask = 0
            

    args = Args()


    data_loader = utils.import_module('dl', os.path.join(args.dataset_name, "data_loader.py"))

    config_file = utils.import_module('cf', os.path.join(args.exp_dir, "configs.py"))
    cf = config_file.Configs()
    cf.exp_dir = args.exp_dir
    cf.test_dir = cf.exp_dir

    #pid = '0811a'
    #cf.fold = find_pid_in_splits(pid)   ### TIGER -- super buggy for some reason...
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
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    torch.backends.cudnn.benchmark = cf.dim == 3
    
    test_predictor = Predictor(cf, None, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    
    
    
    cf.plot_dir = anal_dir ### TIGER ADDED FOR VAL_GEN
    val_gen = data_loader.get_train_generators(cf, logger, data_statistics=False)['val_sampling']
    batch_gen = data_loader.get_test_generator(cf, logger)
    #weight_paths = [os.path.join(cf.fold_dir, '{}_best_params.pth'.format(rank)) for rank in
    #                test_predictor.epoch_ranking]
    #weight_path = weight_paths[rank]
    
    ### TIGER - missing currently ability to find best model
    

    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    from natsort import natsort_keygen, ns
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    from os import listdir
    from os.path import isfile, join
    import glob, os
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    onlyfiles_check.sort(key = natsort_key1)
    
    
    """ Find last checkpoint """       
    weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
    
    net = model.net(cf, logger).cuda(device)
    
    #weight_path = os.path.join(cf.fold_dir, '68_best_params.pth') 
    
    
    
    try:
        pids = batch_gen["test"].dataset_pids
    except:
        pids = batch_gen["test"].generator.dataset_pids
    print("pids in test set: ",  pids)
    #pid = pids[0]
    #assert pid in pids

    # load already trained model weights
    rank = 0
    
    with torch.no_grad():
        pass
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        
        net = net.cuda(device)
        
    # generate a batch from test set and show results
    if not os.path.isdir(anal_dir):
        os.mkdir(anal_dir)

    
    #plot_train_forward(val_gen)
    #plot_forward('906')
    
    num_to_plot = 50
    plot_boxes = 0
    
    thresh_2D_to_3D_boxes = 0.5
    
    
    from natsort import natsort_keygen, ns
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    import glob, os
    
    #from csbdeep.internals import predict
    from tifffile import *
    import tkinter
    from tkinter import filedialog
        
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
        

    list_folder = ['/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Test_RegRCNN/']



    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_output_PYTORCH_73)'
    
        """ For testing ILASTIK images """
        images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), mask=i.replace('.tif','_MASK.tif')) for i in images]
    

         
        try:
            # Create target Directory
            os.mkdir(sav_dir)
            print("Directory " , sav_dir ,  " Created ") 
        except FileExistsError:
            print("Directory " , sav_dir ,  " already exists")
            
        sav_dir = sav_dir + '/'
        
        # Required to initialize all
        batch_size = 1;
        
        batch_x = []; batch_y = [];
        weights = [];
        
        plot_jaccard = [];
        
        output_stack = [];
        output_stack_masked = [];
        all_PPV = [];
        input_im_stack = [];
        for i in range(len(examples)):
             
        
            
             """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
             with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[i]['input']            
                input_im = tiff.imread(input_name)
                
                
               
                if mask:
                    mask_im = tiff.imread(examples[i]['mask'])
                    input_im[mask_im > 0] = 0
                    
       
                """ Analyze each block with offset in all directions """
                
                # Display the image
                #max_im = plot_max(input_im, ax=0)
                
                print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
                #plot_max(input_im)
                     
            
                overlap_percent = 0.5
                input_size = 128
                depth = 16
                num_truth_class = 2
                
                
                quad_size=input_size
                quad_depth=depth
                skip_top=1
                
                
                thresh = 0.99
                cf.merge_3D_iou = thresh
                
                
                im_size = np.shape(input_im);
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
                    
                segmentation = np.zeros([depth_im, width, height])
                total_blocks = 0;
                all_xyz = []                                               
                 
                
                box_coords_all = []
                box_scores_all = []
                    
                for x in range(0, width + quad_size, round(quad_size - quad_size * overlap_percent)):
                      if x + quad_size > width:
                           difference = (x + quad_size) - width
                           x = x - difference
                                
                      for y in range(0, height + quad_size, round(quad_size - quad_size * overlap_percent)):
                           
                           if y + quad_size > height:
                                difference = (y + quad_size) - height
                                y = y - difference
                           
                           for z in range(0, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                               #batch_x = []; batch_y = [];
                     
                               if z + quad_depth > depth_im:
                                    difference = (z + quad_depth) - depth_im
                                    z = z - difference
                               
                                   
                               """ Check if repeated """
                               skip = 0
                               for coord in all_xyz:
                                    if coord == [x,y,z]:
                                         skip = 1
                                         break                      
                               if skip:  continue
                                    
                               all_xyz.append([x, y, z])
                               
                               quad_intensity = input_im[z:z + quad_depth,  x:x + quad_size, y:y + quad_size];  
                               
                          
                               
                               quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                               quad_intensity = np.expand_dims(quad_intensity, axis=0)
                               quad_intensity = np.expand_dims(quad_intensity, axis=0)
                               quad_intensity = np.asarray(quad_intensity, dtype=np.float16)
                               
                               
                               
                               if cf.dim == 3:
                               
                                   batch = {'data':quad_intensity, 'seg': np.zeros([1, 1, input_size, input_size, depth]), 
                                            'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                            'roi_masks': np.zeros([1, 1, 1, input_size, input_size, depth]),
                                            'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                            'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                   
                                   
                               else:
                                   
                                   quad_intensity = quad_intensity[0]
                                   quad_intensity = np.moveaxis(quad_intensity, -1, 0)
                                   
                                   batch = {'data':quad_intensity, 'seg': np.zeros([depth, 1, input_size, input_size]), 
                                            'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                            'roi_masks': np.zeros([depth, input_size, input_size]),
                                            'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                            'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                                                      
   
                               results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                               
                               
                               
                               if 'seg_preds' in results_dict.keys():
                                    results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                        
                               #out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))
                                
                                
    
                    
                               import matplotlib
                               matplotlib.use('Qt5Agg')    
                                ### TIGER - SAVE AS TIFF
                               if cf.dim == 2:
                                    #input_im = np.expand_dims(batch['data'], axis=0)
                                    truth_im = np.expand_dims(batch['seg'], axis=0)
                                    #seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
                                    
                                    
                                    seg_im = results_dict['seg_preds'][:, 0,...]
                                    

                                    """ roi_mask means we dont need bounding boxes!!! 
                                    
                                            - if there are NO objects in this image, then will have a weird shape, so need to parse it by len()
                                            - otherwise, it is a list of lists (1 list for each slice, which contains arrays for every object on that slice)
                                    """
                                    if len(np.unique(seg_im)) == 1:
                                        continue   ### no objects found (only 0 - background)
                                         
                                        
                                    elif cf.merge_2D_to_3D_preds:
                                        """ NEED MORE WORK IF WANT TO CONVERT 2D to 3D"""
                                        
                                        print('merge 2D to 3D with iou thresh of: ' + str(cf.merge_3D_iou))
                                        
                                        import predictor as pred
                                        results_2to3D = {}
                                        results_2to3D['2D_boxes'] = results_dict['boxes']
                                        merge_dims_inputs = [results_dict['boxes'], 'dummy_pid', cf.class_dict, cf.merge_3D_iou]
                                        results_2to3D['boxes'] = pred.apply_2d_3d_merging_to_patient(merge_dims_inputs)[0]
                                        results_dict['boxes'] = results_2to3D['boxes'] 
                                        
                    
                                        label_arr = boxes_to_mask(cf, results_dict=results_dict, thresh=cf.merge_3D_iou)
                            
                                        
                            
                            
                                    else:                  
                                                        
                                        """ THIS CODE IS WRONG --- roi_masks is the INPUT, not the output of MaskRCNN, have to use bounding boxes """
                                        # label_arr = np.zeros(np.shape(batch['data']))
                                        # for slice_id, slice_masks in enumerate(batch['roi_masks']):
                                        #     for obj_id, obj_mask in enumerate(slice_masks):
                                        #         label_arr[slice_id][obj_mask > 0] = obj_id + 1   ### +1 because starts at index 0 (background)
                                         
                                    
                                    label_arr = np.asarray(label_arr, dtype=np.uint16)
                                        
                                    label_arr = np.expand_dims(label_arr, axis=0)
                            
                                    # tiff.imwrite(out_file + '_roi_masks_LABEL.tif', np.asarray(label_arr, dtype=np.uint16),
                                    #               imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})        
                                    
                                    # ### This below hangs
                                    
                                    # # plg.view_batch(cf, batch, res_dict=results_dict, show_info=False, legend=True, show_gt_labels=True,
                                    # #                           out_file=out_file, sample_picks=slices, has_colorchannels=False)
                                    # if plot_boxes:
                                    #     print('Plotting boxes png')
                                    #     utils.split_off_process(plg.view_batch, cf, batch, results_dict, has_colorchannels=cf.has_colorchannels,
                                    #                             show_gt_labels=True, get_time="val-example plot",
                                    #                             out_file=os.path.join(cf.plot_dir, 'batch_SINGLE_PID_{}.png'.format(pid)))
                                        
                                

                               elif cf.dim == 3:
                                    
                                    
                                    #input_im = np.moveaxis(batch['data'], -1, 1) 
                                    truth_im = np.moveaxis(batch['seg'], -1, 1) 
                                    seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                    


                                    box_df = results_dict['boxes'][0]
                                   
                                   
                                    box_coords = []
                                    box_score = []
                                    for box in box_df:
                                       
                                       if box['box_score'] > thresh:
                                           box_coords.append(box['box_coords'])
                                           box_score.append(box['box_score'])
                                   
                                   

                                
                                    
                                    if len(np.unique(seg_im)) == 1 or len(box_score) == 0:
                                        continue   ### no objects found (only 0 - background) 
                                    else:
                                        
                                        label_arr = np.copy(results_dict['seg_preds'],)
                                        
                                        ### SKIP EVERYTHING BELOW FOR NOW - do it at the very very end
                                        
                                        
                                      
                                        # box_coords = np.vstack(box_coords)
                                        # box_score = np.vstack(box_score)        

                                                                                                              
                                        # """ Remove overlap between bounding boxes """
                                        # vol_shape = (quad_depth, quad_size, quad_size)
                                        # df_cleaned = split_boxes_by_slice(box_coords, vol_shape)
                                            
                                        
                                        
                                        
                                        # ### merge coords back into coherent boxes!
                                        
                                        # merged_coords = []
                                        # for box_id in np.unique(df_cleaned['ids']):
                                            
                                        #     coords = df_cleaned.iloc[np.where(df_cleaned['ids'] == box_id)[0]]['bbox_coords']
                            
                                        #     coords = np.vstack(coords)
                                            
                                        #     merged_coords.append(coords)
        

                                    
                                        # label_arr = np.copy(results_dict['seg_preds'],)
                                        # new_labels = np.zeros(np.shape(results_dict['seg_preds']))
                                        # for box_id, box_coords in enumerate(merged_coords):
       
                                        #     box_arr = np.zeros(np.shape(results_dict['seg_preds']))
                                        #     bc = box_coords
                                        #     bc = np.asarray(bc, dtype=np.int)
                                            
                                            
                                        #     bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
                                            
                                        #     #bc[np.where(bc < 0)[0]] = 0  ### cannot be negative
                                            
                                        #     ### also cannot be larger than image size

                                        #     # bc[:, 0][np.where(bc[:, 0] >= label_arr.shape[-2])[0]] = label_arr.shape[-3]
                                        #     # bc[:, 1][np.where(bc[:, 1] >= label_arr.shape[-2])[0]] = label_arr.shape[-2]
                                            
                                        #     # bc[:, 2][np.where(bc[:, 2] >= label_arr.shape[-2])[0]] = label_arr.shape[-1]
                                            
                                        #     #box_arr[0, 0, bc[0]:bc[2], bc[1]:bc[3], bc[4]:bc[5],] = box_id + 1    ### +1 because starts from 0
                                            
                                        #     box_arr[0, 0, bc[:, 1], bc[:, 0], bc[:, 2]] = 1
                                        #     box_arr[label_arr == 0] = 0
                                            
                                        #     new_labels[box_arr > 0] = box_id + 1           
                                
                                        #     #else:
                                        #     #    print('box_score: ' + str(box_row['box_score']))
                                                
                                    
                                                            
                                        # label_arr = new_labels
    



                                    
                                        """ roi_mask means we dont need bounding boxes!!! 
                                        
                                                - if there are NO objects in this image, then will have a weird shape, so need to parse it by len()
                                                - otherwise, it is a list of lists (1 list for each slice, which contains arrays for every object on that slice)
                                        """
                                        

                                        # label_arr = boxes_to_mask(cf, results_dict=results_dict, thresh=cf.merge_3D_iou)
                                        
                                        if len(np.unique(label_arr)) == 1:
                                            continue   ### no objects found (only 0 - background)      
 
                                    
                                    
                                    ### if want colorful mask split up by boxes
                                    label_arr = np.asarray(label_arr, dtype=np.uint16)   
                                    label_arr = np.moveaxis(label_arr, -1, 1)                                      
                                    

                                

                               """ Keep track of boxes for final nms """

                               box_df = results_dict['boxes'][0]
                               
                               
                               box_coords = []
                               box_score = []
                               for box in box_df:
                                   
                                   if box['box_score'] > thresh:
                                       box_coords.append(box['box_coords'])
                                       box_score.append(box['box_score'])
                               
                               
                               box_coords = np.vstack(box_coords)
                               box_score = np.vstack(box_score)
                               
                               
                               
                               ### scale to real life dimensions
                               box_coords[:, 0] = box_coords[:, 0] + x
                               box_coords[:, 2] = box_coords[:, 2] + x
                               
                               box_coords[:, 1] = box_coords[:, 1] + y
                               box_coords[:, 3] = box_coords[:, 3] + y
                               
                               box_coords[:, 4] = box_coords[:, 4] + z
                               box_coords[:, 5] = box_coords[:, 5] + z
                               

                               
                               box_coords_all.append(box_coords)
                               box_scores_all.append(box_score)






                               cleaned_seg = np.asarray(label_arr, dtype=np.uint8)
                              
                               cleaned_seg = cleaned_seg[0, :, 0, :, :]
                            
                               
                               """ Plot output to check """
                               # inp = batch['data'][0]
                               # #truth_im = np.expand_dims(batch['seg'], axis=0)
                               # #seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
                               
                               # seg_im = np.copy(cleaned_seg)
                               # seg_im = np.moveaxis(seg_im, 0, -1)
                               # seg_im = np.expand_dims(seg_im, 0)
                                
                               # ### plot concatenated TIFF
                               # #truth_im[truth_im > 0] = 65535
                               # #seg_im[seg_im > 0] = 65535
                               # concat  = np.concatenate((inp, np.asarray(seg_im, dtype=np.uint16)))
                
                               # concat = np.moveaxis(concat, -1, 0)       
                               # #concat = np.moveaxis(concat, 0, 1)             
                               # concat = np.expand_dims(concat, 0)
                               
                                
                               # tiff.imwrite(sav_dir + 'Block_' + str(int(total_blocks))  + '_COMPOSITE.tif', concat,
                               #                imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                
                                                               
                                                               
                               """ ADD IN THE NEW SEG??? or just let it overlap??? """                     
                               
                               ### this simply uses the new seg
                               #segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg       
                               
                               
                               ### this is adding, it is bad
                               segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                     
                               print(total_blocks)
                               total_blocks += 1
                               
                               # if total_blocks == 40:
                               #     zzz
                               
                               
                               
                               
                """ Final NMS """

                import utils.model_utils as model_utils
                # ### takes as input: box_coords, scores, thresh
                thresh_nms = 0.1  ### det_nms_thresh essentially right?
                thresh_nms = cf.detection_nms_threshold
                
                thresh_nms = 0.01
                
                
                
                box_vert = np.vstack(box_coords_all)
                box_scores = np.vstack(box_scores_all)
                
                keep = model_utils.nms_numpy(box_vert, box_scores, thresh_nms=thresh_nms)
                keep = np.vstack(keep)
                keep = keep[:, 0]
                box_clean = box_vert[keep]
                box_score = box_scores[keep]




                ### Test out single slice
                slice_num = 10
                slice_im = input_im[10]
                
                plt.figure(); plt.imshow(slice_im)
                
 

                slice_perims =  inference_utils.plot_all_boxes_slice(slice_im, box_vert, slice_num)
                plt.figure(); plt.imshow(slice_perims)


                slice_perims =  inference_utils.plot_all_boxes_slice(slice_im, box_clean, slice_num)
                plt.figure(); plt.imshow(slice_perims)
                
                
                plt.figure(); plt.imshow(segmentation[slice_num])
                
                
                """ Now try splitting boxes again! """
                ### about 35 minutes - slowest part by far!!!
                df_cleaned = inference_utils.split_boxes_by_slice(box_clean, vol_shape=input_im.shape)
                
                
                perims = df_cleaned.iloc[np.where(df_cleaned['depth'] == slice_num)[0]]['bbox_coords']
                
                slice_perims_split = np.zeros(np.shape(slice_perims))
                import random
                for p in perims:
                    p = np.asarray(p, dtype=int)
                    slice_perims_split[p[:, 1], p[:, 0]] = random.randint(0, 10)
                    
                plt.figure(); plt.imshow(slice_perims_split)                
                
                
                
                """ and then merge boxes??? """

                ### merge coords back into coherent boxes!
                
                merged_coords = []
                for box_id in np.unique(df_cleaned['ids']):
                    
                    coords = df_cleaned.iloc[np.where(df_cleaned['ids'] == box_id)[0]]['bbox_coords']
    
                    coords = np.vstack(coords)
                    
                    merged_coords.append(coords)


                ### Then APPLY these boxes to mask out the objects in the main segmentation!!!
                label_arr = np.copy(segmentation)
                new_labels = np.zeros(np.shape(segmentation))
                
                
                for box_id, box_coords in enumerate(merged_coords):
                    
                    bc = box_coords
                    """ HACK: --- DOUBLE CHECK ON THIS SUBTRACTION HERE """
                    
                    bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
                    
                    bc = np.asarray(bc, dtype=np.int)                    
                    new_labels[bc[:, 2], bc[:, 1], bc[:, 0]] = box_id + 1     
                    
                    
                    
                
                new_labels[label_arr == 0] = 0
                    
                    
   
                    # box_arr = np.zeros(np.shape(label_arr))
                    # bc = box_coords
                    # bc = np.asarray(bc, dtype=np.int)
                    
                    # #zzz

                    
                    # #bc[np.where(bc < 0)[0]] = 0  ### cannot be negative
                    
                    # ### also cannot be larger than image size

                    # # bc[:, 0][np.where(bc[:, 0] >= label_arr.shape[-2])[0]] = label_arr.shape[-3]
                    # # bc[:, 1][np.where(bc[:, 1] >= label_arr.shape[-2])[0]] = label_arr.shape[-2]
                    
                    # # bc[:, 2][np.where(bc[:, 2] >= label_arr.shape[-2])[0]] = label_arr.shape[-1]
                    
                    # #box_arr[0, 0, bc[0]:bc[2], bc[1]:bc[3], bc[4]:bc[5],] = box_id + 1    ### +1 because starts from 0
                    
                    # box_arr[bc[:, 2], bc[:, 1], bc[:, 0]] = 1
                    # box_arr[label_arr == 0] = 0
                    
                    # new_labels[box_arr > 0] = box_id + 1           
                    
                    # print('adding box: ' + str(box_id) + ' of total: ' + str(len(merged_coords)))
        
                    # #else:
                    # #    print('box_score: ' + str(box_row['box_score']))
                                        

                
                zzz
                """ Do EVERYTHING in post-processing:
                            - final NMS above
                            - try WBC instead of NMS???
                            - then run the bbox split function to cut up any overlaps (may have to do on a regional basis?)
                            
                            - ***might have to then do ANOTHER NMS on the full image...??? or at least along the edes?
                    
                    
                    
                            STILL HAVE TO CUT UP IN Z-dimension as well???
                            
                            
                    
                    """
                
                               

                #segmentation[segmentation > 0] = 255
                filename = input_name.split('/')[-1].split('.')[0:-1]
                filename = '.'.join(filename)
                
                ### if operating system is Windows, must also remove \\ slash
                #if os_windows:
                #     filename = filename.split('\\')[-1]



                new_labels = np.asarray(new_labels, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_SPLIT_LABELS.tif', new_labels)



                     
                segmentation = np.asarray(segmentation, np.uint16)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation.tif', segmentation)
                segmentation[segmentation > 0] = 1
                
                #input_im = np.asarray(input_im, np.uint8)
                input_im = np.expand_dims(input_im, axis=0)
                input_im = np.expand_dims(input_im, axis=2)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', input_im,
                                     imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                     metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                        

        
        
        
        
                               
                ### if want unique labels:
                from skimage import measure
                labels = measure.label(segmentation)
                labels = np.asarray(labels, dtype=np.uint16)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation_LABELLED.tif', labels)
               

        
        
