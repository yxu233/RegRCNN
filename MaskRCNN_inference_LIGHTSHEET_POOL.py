import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
import utils.model_utils as mutils

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

import h5py

from inference_analysis_OL_TIFFs_small_patch_POOL import post_process_boxes


if __name__=="__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/73) dil_group_norm_det_thresh_0_2/'      
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/76) dil_group_norm_NEW_DATA_edges_wbc/'                
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
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    torch.backends.cudnn.benchmark = cf.dim == 3
    

    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    onlyfiles_check.sort(key = natsort_key1)
    weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
    net = model.net(cf, logger).cuda(device)

    # load already trained model weights
    with torch.no_grad():
        pass
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        net = net.cuda(device)
        
    # generate a batch from test set and show results
    if not os.path.isdir(anal_dir):
        os.mkdir(anal_dir)



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
    
    list_folder = ['/media/user/TigerDrive2/20221006_M125_MoE_BDV_fused/'
                  #'/media/user/TigerDrive2/20221010_M120_MoE_AAVs_lowest_LOW_RI_RIMS_14648_40perc_UREA_pH_12_zoom05_NOFUSE/BDV/',
                  #'/media/user/TigerDrive2/20221011_M123_MoE_AAVs_HIGHEST_RI_RIMS_15127_40perc_UREA_fresh_pH_12_zoom06_NOFUSE/BDV/BDV_BrainReg/to_reg/',
                  #'/media/user/FantomHD/20220813_M103_MoE_Tie2Cre_Cuprizone_6_weeks_RI_14614_3dayhot_7daysRIMSRT/BDV/BDV_BrainReg/to_reg/'
                  ]
    

    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_MaskRCNN'
    

        """ For testing ILASTIK images """
        images = glob.glob(os.path.join(input_path,'*.h5'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('.h5','.xml'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]
         
        try:
            # Create target Directory
            os.mkdir(sav_dir)
            print("\nSave directory " , sav_dir ,  " Created ") 
        except FileExistsError:
            print("\nSave directory " , sav_dir ,  " already exists")
            
        sav_dir = sav_dir + '/'
        
        # Required to initialize all
        for i in range(len(examples)):
             
         
           """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
           with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[i]['input']   
                # input_im = tiff.imread(input_name)
            
            
                import h5py
                f = h5py.File(examples[i]['input'], "r")
                
                print(f.keys())
                print(f['s00'].keys())
                print(f['t00000'].keys())
                print(f['t00000']['s00'].keys())
                
                #lowest_res = f['t00000']['s00']['7']['cells']
                highest_res = f['t00000']['s00']['0']['cells']
                
                dset = highest_res
                
                ### channel 2
                #highest_res = f['t00000']['s01']['0']['cells']
                
                coords_df = pd.DataFrame(columns = ['offset', 'block_num', 'Z', 'X', 'Y', 'Z_scaled', 'X_scaled', 'Y_scaled', 'equiv_diam', 'vol'])
                
            

                """ Or save to memmapped TIFF first... """
                print('creating memmap save file on disk')
                #memmap_save = tiff.memmap(sav_dir + 'temp_SegCNN.tif', shape=dset.shape, dtype='uint8')
                
                memmap_save = tiff.memmap('/media/user/storage/Temp_lightsheet/temp_SegCNN_' + str(i) + '_.tif', shape=dset.shape, dtype='uint8')
    
    
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

                
                    
                all_xyz = [] 
                
                for z in range(0, depth_imL + Lpatch_depth, round(Lpatch_depth)):
                    if z + Lpatch_depth > depth_imL:  continue

                    for x in range(0, widthL + Lpatch_size, round(Lpatch_size)):
                          if x + Lpatch_size > widthL:  continue

                          for y in range(0, heightL + Lpatch_size, round(Lpatch_size)):
                               if y + Lpatch_size > heightL: continue

                               print([x, y, z])

                               all_xyz.append([x, y, z])
                               
                                
            
                ### how many total blocks to analyze:
                print(len(all_xyz))
                    
            
            
            
                """ Then loop through """
                for id_c, s_c in enumerate(all_xyz):
                    
                
                ### for continuing the run
                #for id_c in range(93, len(all_xyz)):
                    s_c = all_xyz[id_c]
                    
                    ### for debug:
                    #s_c = all_xyz[10]
                    
                    
                    import time
                    tic = time.perf_counter()
                    
         
                    
                    input_im = dset[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size]
                    og_shape = input_im.shape
    
                
                    """ Detect if blank in uint16 """
                    num_voxels = len(np.where(input_im > 300)[0])
                    if num_voxels < 10000:
                         print('skipping: ' + str(s_c))
                         print('num voxels with signal: ' + str(num_voxels))
                         continue                
                

                                        
                    print('Analyzing: ' + str(s_c))
                    print('Which is: ' + str(id_c) + ' of total: ' + str(len(all_xyz)))
                            
                    
                    toc = time.perf_counter()
                    
                    print(f"Opened subblock in {toc - tic:0.4f} seconds")
    
                    """ Start inference on volume """
                    tic = time.perf_counter()
                    
                    
                    import warnings
                    overlap_percent = 0.1
                    
                    
                    #%% MaskRCNN analysis!!!
                    """ Analyze each block with offset in all directions """ 
                    print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
                    
                    ### Define patch sizes
                    patch_size=128; patch_depth=16
                    
                    ### Define overlap and focal cube to remove cells that fall within this edge
                    overlap_pxy = 14; overlap_pz = 2
                    step_xy = patch_size - overlap_pxy * 2
                    step_z = patch_depth - overlap_pz * 2
                    
                    focal_cube = np.ones([patch_depth, patch_size, patch_size])
                    focal_cube[overlap_pz:-overlap_pz, overlap_pxy:-overlap_pxy, overlap_pxy:-overlap_pxy] = 0                
            
    
                    thresh = 0.99
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
                    split_seg = np.zeros([depth_im, width, height])
                    #all_xyz = [] 
                    
                    
                    all_patches = []
                    
                    for z in range(0, depth_im + patch_depth, round(step_z)):
                        if z + patch_depth > depth_im:  continue
    
                        for x in range(0, width + patch_size, round(step_xy)):
                              if x + patch_size > width:  continue
    
                              for y in range(0, height + patch_size, round(step_xy)):
                                   if y + patch_size > height: continue
    
                                   print([x, y, z])
    
                                   all_xyz.append([x, y, z])
    
                                   print(total_blocks)
                                   total_blocks += 1
                                   
                                   quad_intensity = input_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size];  
                                   quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                                   quad_intensity = np.asarray(np.expand_dims(np.expand_dims(quad_intensity, axis=0), axis=0), dtype=np.float16)
          
    
                                   batch = {'data':quad_intensity, 'seg': np.zeros([1, 1, patch_size, patch_size, patch_depth]), 
                                             'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                             'roi_masks': np.zeros([1, 1, 1, patch_size, patch_size, patch_depth]),
                                             'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                             'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                    
                                   ### run MaskRCNN
                                   results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                                   
    
                                   if 'seg_preds' in results_dict.keys():
                                        results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
    
                                   seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                    
    
                                   box_df = results_dict['boxes'][0]
                                   box_vert = []
                                   box_score = []
                                   for box in box_df:
                                       if box['box_score'] > thresh:
                                           box_vert.append(box['box_coords'])
                                           box_score.append(box['box_score'])
                                           
                                           
                                   if len(box_vert) == 0:
                                       continue
    
    
                                   patch_im = np.copy(quad_intensity[0][0])
                                   patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first
                                    
                
                                   all_patches.append({'box_vert':box_vert, 'results_dict': results_dict, 'patch_im': patch_im, 'total_blocks':total_blocks, 
                                                       'focal_cube':focal_cube, 'xyz':[x,y,z]})
                                   
                                   # if total_blocks == 80:
                                   #      zzz
                                        
    
                                   # ### if want just pure segmentations without any subtraction
                                   #label_arr = np.asarray(label_arr, dtype=np.uint16)   
                                   #whole_seg = np.asarray(seg_im, dtype=np.uint8)        
                                   #whole_seg = whole_seg[0, :, 0, :, :]
                                
                                   ### this is adding,
                                   #segmentation[z:z + patch_depth, x:x + patch_size, y:y + patch_size] = whole_seg + segmentation[z:z + patch_depth, x:x + patch_size, y:y + patch_size]
                              
                    p = Pool(8)
                    #kwargs  = [{"gdf":gdf, "index":i, "margin":margin, "line_mult":line_mult} for i in range(len(gdf))]
                    box_coords_all = p.map(post_process_boxes, all_patches)
                    
                    p.close()  ### to prevent memory leak
    
    
                    filename = input_name.split('/')[-1].split('.')[0:-1]
                    filename = '.'.join(filename)
    
                    size_thresh = 80    ### 150 is too much!!!
                    ### Go through all coords and assemble into whole image with splitting and step-wise
                    
                    stepwise_im = np.zeros(np.shape(segmentation))
                    
                    cell_num = 0
                    for patch_box_coords in box_coords_all:
                        #print(patch_box_coords)
                        if patch_box_coords:  ### FOR SOME REASON SOME IS NONETYPE???
                            
                            shuffled = np.copy(patch_box_coords)
                            random.shuffle(shuffled)    ### for plotting so it's more visible
                            for coords in shuffled:
        
                                #sizes.append(len(coords))
                                if len(coords) < size_thresh:
                                    continue                    
                                stepwise_im[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_num
                                
                                cell_num += 1
                            
                                print(cell_num)
                        
                        
                    stepwise_im = np.asarray(stepwise_im, np.int32)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_stepwise_im.tif', stepwise_im)
                        
    
                    #input_im = np.asarray(input_im, np.uint8)
                    input_im = np.expand_dims(input_im, axis=0)
                    input_im = np.expand_dims(input_im, axis=2)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                          imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                          metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                            
    
            
    
    ###########################################################################################        
            
    
                    """ Post-processing 
                    
                            - find doublets (identified twice) and resolve them
                            
                            - then clean super small objects
                            - and clean 
                    
                    """
    
                    doublets = np.zeros(np.shape(segmentation))
                    
                    sizes = []
                    
                    #shuffled = np.copy(box_coords_all)
                    #random.shuffle(shuffled)    ### for plotting so it's more visible
                    for patch_box_coords in box_coords_all:
                        #print(patch_box_coords)
                        if patch_box_coords:  ### FOR SOME REASON SOME IS NONETYPE???
                            
                            for id_sw, coords in enumerate(patch_box_coords):
                                
                                
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
    
                    arr_doublets = [ [] for _ in range(len(cc) + 1)]
                    
                    for patch_box_coords in box_coords_all:
                        #print(patch_box_coords)
                        if patch_box_coords:  ### FOR SOME REASON SOME IS NONETYPE???
                            
                            for id_sw, coords in enumerate(patch_box_coords):
                                
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
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_cleaned.tif', stepwise_clean)
    
                        
                        
                    
                    
                    
                    #########################################################################################################################
                    
                    
                    
                    
                    
                    segmentation = np.copy(stepwise_clean)
    
                    segmentation = segmentation[overlap_pz: overlap_pz + Lpatch_depth, overlap_pxy: overlap_pxy + Lpatch_size, overlap_pxy: overlap_pxy + Lpatch_size]

                    #%% Output to memmap array
                    
                    toc = time.perf_counter()
                    
                    print(f"Inference in {toc - tic:0.4f} seconds")
                    

                    """ Output to memmmapped arr """
                    
                    tic = time.perf_counter()
                    memmap_save[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size] = segmentation
                    memmap_save.flush()
                    
                    toc = time.perf_counter()
                    
                    print(f"Save in {toc - tic:0.4f} seconds")
                    
                    
                    """ Also save list of coords of where the cells are located so that can easily access later (or plot without making a whole image!) """
                    print('saving coords')
                    tic = time.perf_counter()
                    from skimage import measure
                    labels = measure.label(segmentation)
                    #blobs_labels = measure.label(blobs, background=0)
                    cc = measure.regionprops(labels)
                    
                    
                    ########################## TOO SLOW TO USE APPEND TO ADD EACH ROW!!!
                    ######################### MUCH FASTER TO JUST MAKE A DICTIONARY FIRST, AND THEN CONVERT TO DATAFRAME AND CONCAT
                    d = {}
                    for i_list, cell in enumerate(cc):
                        center = cell['centroid']
                        center = [round(center[0]), round(center[1]), round(center[2])]
                        
                        d[i_list] = {'offset': s_c, 'block_num': id_c, 
                               'Z': center[0], 'X': center[1], 'Y': center[2],
                               'Z_scaled': center[0] + s_c[0], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[2],
                               'equiv_diam': cell['equivalent_diameter'], 'vol': cell['area']}
                    
                    
                    
                    df = pd.DataFrame.from_dict(d, "index")
            
                    coords_df = pd.concat([coords_df, df])           
                                            
                    
                    toc = time.perf_counter()
                    
                    
                    np.save(sav_dir + filename + '_numpy_arr', coords_df)
                    print(f"Save coords in {toc - tic:0.4f} seconds")

        
    print('\n\nSegmented outputs saved in folder: ' + sav_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    