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

from inference_utils import *


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



                
                ### number of pixels to exclude if centroid of object falls within that range
                overlap_px = 14
                overlap_pz = 2
                
                
                step_x = quad_size - overlap_px * 2
                step_z = quad_depth - overlap_pz * 2
                
                                    
                focal_cube = np.ones([quad_depth, quad_size, quad_size])
                focal_cube[overlap_pz:-overlap_pz, overlap_px:-overlap_px, overlap_px:-overlap_px] = 0                

    

                skip_top=1
                
                
                thresh = 0.99
                cf.merge_3D_iou = thresh
                
                
                im_size = np.shape(input_im);
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
                    

                
                total_blocks = 0;
                                                             
                 
                
                box_coords_all = []
                box_scores_all = []
                    
                
                ### Add padding to input_im
                factorx = 0
                while factorx < height/step_x:   factorx+=1
                end_padx = (factorx * step_x) - height
                
                
                factorz = 0
                while factorz < depth_im/step_z:   factorz+=1
                end_padz = (factorz * step_z) - depth_im     
                
                ### ADD ANOTHER ONE FOR DIMENSION Y!!!
                
                print('Total num of patches: ' + str(factorx * factorx * factorz))
                
                
                new_dim_im = np.zeros([overlap_pz*2 + depth_im + end_padz, overlap_px*2 + width + end_padx, overlap_px*2 + height + end_padx])
                
                new_dim_im[overlap_pz: overlap_pz + depth_im, overlap_px: overlap_px + width, overlap_px: overlap_px + height] = input_im
                
                input_im = new_dim_im
                
                im_size = np.shape(input_im);
                width = im_size[1];  height = im_size[2]; depth_im = im_size[0];                
                


                segmentation = np.zeros([depth_im, width, height])
                split_seg = np.zeros([depth_im, width, height])
                all_xyz = [] 
                for z in range(0, depth_im + quad_depth, round(step_z)):
                    #batch_x = []; batch_y = [];
          
                    if z + quad_depth > depth_im:
                         print('reached end of dim')
                         continue
                         # difference = (z + quad_depth) - depth_im
                         # z = z - difference

                    for x in range(0, width + quad_size, round(step_x)):
                          if x + quad_size > width:
                               print('reached end of dim')
                               continue
                               # difference = (x + quad_size) - width
                               # x = x - difference
                                    
                          for y in range(0, height + quad_size, round(step_x)):
                               
                               if y + quad_size > height:
                                    print('reached end of dim')
                                    continue
                                    # difference = (y + quad_size) - height
                                    # y = y - difference
                               
                                
                               print([x, y, z])

                               all_xyz.append([x, y, z])
                               
                               quad_intensity = input_im[z:z + quad_depth,  x:x + quad_size, y:y + quad_size];  
                               
                               quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                               quad_intensity = np.expand_dims(quad_intensity, axis=0)
                               quad_intensity = np.expand_dims(quad_intensity, axis=0)
                               quad_intensity = np.asarray(quad_intensity, dtype=np.float16)
                               
                               
                              
                               batch = {'data':quad_intensity, 'seg': np.zeros([1, 1, input_size, input_size, depth]), 
                                         'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                         'roi_masks': np.zeros([1, 1, 1, input_size, input_size, depth]),
                                         'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                         'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                

                               results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                               
                               
                               
                               if 'seg_preds' in results_dict.keys():
                                    results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                        
                               #out_file = os.path.join(anal_dir, "straight_inference_fold_{}_pid_{}".format(str(cf.fold), pid))
                                
                                
    
                    
                               import matplotlib
                               matplotlib.use('Qt5Agg')    
                                ### TIGER - SAVE AS TIFF

                                #input_im = np.moveaxis(batch['data'], -1, 1) 
                               truth_im = np.moveaxis(batch['seg'], -1, 1) 
                               seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                


                               box_df = results_dict['boxes'][0]
                              
                               
                               box_vert = []
                               box_score = []
                               for box in box_df:
                                   seg_im
                                   if box['box_score'] > thresh:
                                       box_vert.append(box['box_coords'])
                                       box_score.append(box['box_score'])
                               
                               

                               
                                
                               if len(np.unique(seg_im)) == 1 or len(box_score) == 0:
                                    continue   ### no objects found (only 0 - background) 
                               else:
                                    
                                    label_arr = np.copy(results_dict['seg_preds'],)
                                    
                                    ### SKIP EVERYTHING BELOW FOR NOW - do it at the very very end
                                    
                                    patch_im = np.copy(quad_intensity[0][0])
                                    patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first
                                    
                                    ### remove all boxes with centroids on edge of image - but first need to process ALL boxes, including with box split? To be sure...
                                    
                                    """ Now try splitting boxes again! """
                                    ### about 35 minutes - slowest part by far!!!
                                    df_cleaned = split_boxes_by_slice(box_vert, vol_shape=patch_im.shape)
                                    
                                    
                                    
                                    """ and then merge boxes??? """
                                    
                                    ### merge coords back into coherent boxes!
                                    
                                    merged_coords = []
                                    for box_id in np.unique(df_cleaned['ids']):
                                    
                                        coords = df_cleaned.iloc[np.where(df_cleaned['ids'] == box_id)[0]]['bbox_coords']
                                        
                                        coords = np.vstack(coords).astype(int)
                                        
                                        merged_coords.append(coords)                                    
                                    
                                    
                                    
                                    
                                    """ 
                                            ***ALSO SPLIT IN Z-dimension???
                                                
                                    """
                                    box_vert = np.vstack(box_vert)
                                    box_vert_z = np.copy(box_vert)
                                    box_vert_z[:, 0] = box_vert[:, 4]
                                    box_vert_z[:, 2] = box_vert[:, 5]
                                    box_vert_z[:, 4] = box_vert[:, 0]
                                    box_vert_z[:, 5] = box_vert[:, 2]

                                    df_cleaned_z = split_boxes_by_slice(box_vert_z, vol_shape=np.moveaxis(patch_im, 0, 1).shape)
                                                                              
                                    ### merge coords back into coherent boxes!
                                    merged_coords_z = []
                                    for box_id in np.unique(df_cleaned_z['ids']):
                                    
                                        coords = df_cleaned_z.iloc[np.where(df_cleaned_z['ids'] == box_id)[0]]['bbox_coords']
                                        
                                        coords = np.vstack(coords).astype(int)
                                        
                                        ### swap axis
                                        coords[:, [2, 1]] = coords[:, [1, 2]]
                                        
                                        merged_coords_z.append(coords)      
                                        
                                        
                                        
                                    if len(merged_coords) != len(merged_coords_z):
                                        print('NOT MATCHED LENGTH')
                                    
                                    ### COMBINE all the splits in XY and Z
                                    match_xyz = []
                                    for id_m in range(len(merged_coords)):
                                        
                                        a = merged_coords[id_m]
                                        b = merged_coords_z[id_m]
                                        
                                        unq, count = np.unique(np.concatenate((a, b)), axis=0, return_counts=True)
                                        
                                        matched = unq[count > 1]
                                        match_xyz.append(matched)
                                        
                                    merged_coords = match_xyz
                                                                              

                                        
                                    
                                    ### Then APPLY these boxes to mask out the objects in the main segmentation!!!
                                    new_labels = np.zeros(np.shape(label_arr))
                                    
                                    #all_labels = np.copy(label_arr)
                                    for box_id, box_coords in enumerate(merged_coords):
                                    
                                        bc = box_coords
                                        """ HACK: --- DOUBLE CHECK ON THIS SUBTRACTION HERE """
                                        
                                        bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
                                        
                                        bc = np.asarray(bc, dtype=np.int)                    
                                        new_labels[0, 0, bc[:, 1], bc[:, 0], bc[:, 2]] = box_id + 1   
                                        

                                    new_labels[label_arr == 0] = 0   ### This is way simpler and faster than old method of looping through each detection



                                    """ Then find which ones are on the edge and remove, then save coords of cells for the very end """
                                    from skimage.measure import label, regionprops, regionprops_table
                                    new_labels = np.asarray(new_labels, dtype=int)
                                    cc = regionprops_table(new_labels[0][0], properties=('centroid', 'coords'))
                                    
                                    df = pd.DataFrame(cc)
                                    
                                    ### Then filter using focal cube
                                    
                                    x_px = np.asarray(df['centroid-0'])
                                

                                    edge_ids = np.where(focal_cube[np.asarray(df['centroid-2']), np.asarray(df['centroid-0']), np.asarray(df['centroid-1'])])[0]
                                    
                                    
                                    
                                    print('num deleted: ' + str(len(df) - len(edge_ids)))
                                    
                                    
                                    


                                    
                                    df = df.drop(edge_ids)
                                    
                                    
                                    if len(df) > 0:
                                        ### save the coordinates so we can plot the cells later
                                        #empty = np.zeros(np.shape(patch_im))
                                        for row_id, row in df.iterrows():
                                            coords = row['coords']
                                            
                                            coords[:, 0] = coords[:, 0] + x
                                            coords[:, 1] = coords[:, 1] + y
                                            coords[:, 2] = coords[:, 2] + z
                                               
                                            box_coords_all.append(coords)
                                            #empty[coords[:, 2], coords[:, 0], coords[:, 1]] = row_id
                                            


                                    # plot_max(patch_im, ax=0)
                                    # plot_max(new_labels[0][0], ax=-1)
                                    # plot_max(empty)
                                    
                                    
                                    # if total_blocks > 4:
                                    #     zzz
                                    


                               ### if want to save NON-additive, and no splitting of boxes
                               new_labels = np.asarray(new_labels, dtype=np.uint16)   
                               split_labs = np.moveaxis(new_labels, -1, 1)                                      
                               #split_labs = np.asarray(new_labels, dtype=np.uint8)        
                               split_labs = split_labs[0, :, 0, :, :]
                               
                               split_seg[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = split_labs 




                               ### if want just pure segmentations without any subtraction
                               label_arr = np.asarray(label_arr, dtype=np.uint16)   
                               label_arr = np.moveaxis(label_arr, -1, 1)                                      
                                    
                               whole_seg = np.asarray(label_arr, dtype=np.uint8)        
                               whole_seg = whole_seg[0, :, 0, :, :]
                            
                               ### this is adding,
                               segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = whole_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                     
                               print(total_blocks)
                               total_blocks += 1
                               
                               # if total_blocks == 40:
                               #     zzz



                               
                               """ Plot output to check """
                               # inp = batch['data'][0]
                               #  #truth_im = np.expand_dims(batch['seg'], axis=0)
                               #  #seg_im = np.expand_dims(results_dict['seg_preds'], axis=0)
                               
                               # seg_im = np.copy(whole_seg)
                               # seg_im = np.moveaxis(seg_im, 0, -1)
                               # seg_im = np.expand_dims(seg_im, 0)
                                
                               #  ### plot concatenated TIFF
                               #  #truth_im[truth_im > 0] = 65535
                               #  #seg_im[seg_im > 0] = 65535
                               # concat  = np.concatenate((inp, np.asarray(seg_im, dtype=np.uint16)))
                
                               # concat = np.moveaxis(concat, -1, 0)       
                               #  #concat = np.moveaxis(concat, 0, 1)             
                               # concat = np.expand_dims(concat, 0)
                               
                                
                               # tiff.imwrite(sav_dir + 'Block_' + str(int(total_blocks))  + '_COMPOSITE.tif', concat,
                               #                  imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
                
                                                               

                zzz
                """ Post-processing 
                
                        - find doublets (identified twice) and resolve them
                        
                        - then clean super small objects
                        - and clean 
                
                """

                doublets = np.zeros(np.shape(segmentation))
                
                sizes = []
                size_thresh = 80    ### 150 is too much!!!
                #shuffled = np.copy(box_coords_all)
                #random.shuffle(shuffled)    ### for plotting so it's more visible
                for id_sw, coords in enumerate(box_coords_all):
                    
                    
                    sizes.append(len(coords))
                    if len(coords) < size_thresh:
                        continue
                    
                    doublets[coords[:, 2], coords[:, 0], coords[:, 1]] = doublets[coords[:, 2], coords[:, 0], coords[:, 1]] + 1
                    
                    
                    
                doublets[doublets <= 1] = 0
                doublets[doublets > 0] = 1
                
                lab = label(doublets)
                cc = regionprops(lab)

                print('num doublets: ' + str(len(cc)))
                
                cleaned = np.zeros(np.shape(doublets))

                arr_doublets = [ [] for _ in range(len(cc) + 1) ]
                for id_sw, coords in enumerate(box_coords_all):
                    
                    region = lab[coords[:, 2], coords[:, 0], coords[:, 1]]
                    
                    if len(np.where(region)[0]) > 0:
                        
                        ids = np.unique(region)
                        ids = ids[ids != 0]   ### remove zero
                        if len(ids) > 1:
                            #zzz
                            print(ids)
                            
                        arr_doublets[ids[0]].append(coords)
                        
                    
                    
                ### loop through all doublets and determine iou
                for case_id, case in enumerate(arr_doublets):
                    
                    if len(case) == 0:
                        continue
                    
                    
                    
                    #unq, count = np.unique(a, axis=0, return_counts=True) ### what do we do if more than one???
                
                    ### Case #1: if iou is very close to 1, then just choose larger seg
                    ### Case #2: if iou is very small...
                    
                    ### HACK: just choose largest for now...
                    all_vols = []
                    for vol in case:
                        all_vols.append(len(vol))
                    
                    unique_coords = case[np.argmax(all_vols)]
                    
                    
                    
                    ### or just ALL of the coords to be the same!!! consensus
                    
                    ### OR JUST GET AVERAGE SHAPE, and then delete all of the other coords in the main image!!!
                    
                    unique_coords = np.vstack(case)
                    cleaned[unique_coords[:, 2], unique_coords[:, 0], unique_coords[:, 1]] = case_id
                
                cleaned = np.asarray(cleaned, np.int32)
                
                
                stepwise_clean = np.copy(stepwise_im)
                stepwise_clean[cleaned > 0] = 0
                #cleaned[cleaned > 0] = cleaned[cleaned > 0] + np.max(stepwise_clean)  ### add to make indices unique
                
                stepwise_clean[cleaned > 0] = cleaned[cleaned > 0]
                stepwise_clean = np.asarray(stepwise_clean, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_cleaned.tif', stepwise_clean)







###########################################################################################

                filename = input_name.split('/')[-1].split('.')[0:-1]
                filename = '.'.join(filename)

                
                ### Go through all coords and assemble into whole image with splitting and step-wise
                
                stepwise_im = np.zeros(np.shape(segmentation))
                
                shuffled = np.copy(box_coords_all)
                random.shuffle(shuffled)    ### for plotting so it's more visible
                for id_sw, coords in enumerate(shuffled):


                    #sizes.append(len(coords))
                    if len(coords) < size_thresh:
                        continue                    
                    stepwise_im[coords[:, 2], coords[:, 0], coords[:, 1]] = id_sw
                    
                    print(id_sw)
                    
                    
                stepwise_im = np.asarray(stepwise_im, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_stepwise_im.tif', stepwise_im)
                    
                    
                ### if operating system is Windows, must also remove \\ slash
                #if os_windows:
                #     filename = filename.split('\\')[-1]



                split_seg = np.asarray(split_seg, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_split_seg.tif', split_seg)



                     
                segmentation = np.asarray(segmentation, np.uint16)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_whole_seg.tif', segmentation)
                segmentation[segmentation > 0] = 1
                
                #input_im = np.asarray(input_im, np.uint8)
                input_im = np.expand_dims(input_im, axis=0)
                input_im = np.expand_dims(input_im, axis=2)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                     imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                     metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                        

        
        
        
        

        
        
