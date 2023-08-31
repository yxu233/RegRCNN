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

from inference_analysis_OL_TIFFs_small_patch_POOL import post_process_boxes, expand_add_stragglers

import time
from multiprocessing.pool import ThreadPool
from scipy.stats import norm


from functional.matlab_crop_function import *
from functional.tree_functions import *  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


if __name__=="__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/73) dil_group_norm_det_thresh_0_2/'      
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/76) dil_group_norm_NEW_DATA_edges_wbc/' 

            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/90) new_dense_cleaned_edges_det_02_min_conf_01_later_check437/'              
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
    
    
    
    
    
    
    
    
    """ HAD BEEN USING BEST CHECKPOINT, NOT LAST CHECKPOINT"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
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
        sav_dir = input_path + '/' + foldername + '_MaskRCNN_NEW'
    

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
                
                #memmap_save = tiff.memmap('/media/user/storage/Temp_lightsheet/temp_SegCNN_' + str(i) + '_.tif', shape=dset.shape, dtype='uint8')
    
    
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
                
                for z in range(0, depth_imL + Lpatch_depth, round(Lpatch_depth)):
                    if z + Lpatch_depth > depth_imL:  continue

                    for x in range(0, widthL + Lpatch_size, round(Lpatch_size)):
                          if x + Lpatch_size > widthL:  continue

                          for y in range(0, heightL + Lpatch_size, round(Lpatch_size)):
                               if y + Lpatch_size > heightL: continue

                               print([x, y, z])

                               all_xyzL.append([x, y, z])
                               
                                
            
                ### how many total blocks to analyze:
                #print(len(all_xyzL))
                    
                def get_im(dset, s_c, Lpatch_depth, Lpatch_size):
                    
                        #tic = time.perf_counter()
                        dset_im = dset[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size]
                        og_shape = dset_im.shape
                        
                        #toc = time.perf_counter()
                        print('loaded asynchronously')
                        
                        #print(f"Opened subblock in {toc - tic:0.4f} seconds")
                        
                        return dset_im, og_shape            
            
            
                """ Then loop through """
                for id_c, s_c in enumerate(all_xyzL):
                    
                
                ### for continuing the run
                #for id_c in range(93, len(all_xyz)):
                    s_c = all_xyzL[id_c]
                    
                    ### for debug:
                    #s_c = all_xyz[10]
                    
                    

                    tic = time.perf_counter()
                    
         
                    ### Load first tile normally, and then the rest as asynchronous processes
                    if id_c == 0:
                        dset_im, og_shape = get_im(dset, s_c, Lpatch_depth, Lpatch_size)
                        print('loaded normally')
                        
                    else:   ### get tile from asynchronous processing instead!
                        dset_im, og_shap = async_result.get()  # get the return value from your function.    
                        
                        poolThread.close()
                        poolThread.join()
                    
                    
                    
                    
                    ### Initiate poolThread
                    poolThread = ThreadPool(processes=1)
                    
                    ### get NEXT tile asynchronously!!!

                    async_result = poolThread.apply_async(get_im, (dset, all_xyzL[id_c + 1], Lpatch_depth, Lpatch_size)) 



                    toc = time.perf_counter()
                    print(f"\nOpened subblock in {toc - tic:0.4f} seconds")                    
                    
                    """ Detect if blank in uint16 """
                    num_voxels = len(np.where(dset_im > 300)[0])
                    if num_voxels < 10000:
                         print('skipping: ' + str(s_c))
                         print('num voxels with signal: ' + str(num_voxels))
                         
                         #time.sleep(10)
                         continue                
                     
                    
                                        
                    print('Analyzing: ' + str(s_c))
                    print('Which is: ' + str(id_c) + ' of total: ' + str(len(all_xyzL)))
                    
                      
  
                    """ Start inference on volume """
                    tic = time.perf_counter()
                    
                    
                    import warnings
                    overlap_percent = 0.1
                    
                    
                    #%% MaskRCNN analysis!!!
                    """ Analyze each block with offset in all directions """ 
                    #print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
                    
                    ### Define patch sizes
                    patch_size=128; patch_depth=16
                    
                    ### Define overlap and focal cube to remove cells that fall within this edge
                    overlap_pxy = 14; overlap_pz = 3
                    step_xy = patch_size - overlap_pxy * 2
                    step_z = patch_depth - overlap_pz * 2
                    
                    focal_cube = np.ones([patch_depth, patch_size, patch_size])
                    focal_cube[overlap_pz:-overlap_pz, overlap_pxy:-overlap_pxy, overlap_pxy:-overlap_pxy] = 0                
            
    
                    thresh = 0.99
                    cf.merge_3D_iou = thresh
                    
                    im_size = np.shape(dset_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
    
                    
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
                    
                    #print('Total num of patches: ' + str(factorx * factory * factorz))
                    new_dim_im = np.zeros([overlap_pz*2 + depth_im + end_padz, overlap_pxy*2 + width + end_padx, overlap_pxy*2 + height + end_pady])
                    new_dim_im[overlap_pz: overlap_pz + depth_im, overlap_pxy: overlap_pxy + width, overlap_pxy: overlap_pxy + height] = dset_im
                    input_im = new_dim_im
                    
                    im_size = np.shape(input_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];                
                    
                    
                    ### Define empty items
                    box_coords_all = []; total_blocks = 0;
                    #all_xyz = [] 
  
                    all_patches = []
                    batch_size = 1
                    batch_im = []
                    batch_xyz = []
                    
                    segmentation = np.zeros([depth_im, width, height])
                    for z in range(0, depth_im + patch_depth, round(step_z)):
                        if z + patch_depth > depth_im:  continue
    
                        for x in range(0, width + patch_size, round(step_xy)):
                              if x + patch_size > width:  continue
    
                              for y in range(0, height + patch_size, round(step_xy)):
                                   if y + patch_size > height: continue
    
                                   
    
                                   #all_xyz.append([x, y, z])
    
    
                                   
                                   quad_intensity = input_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size];  
                                   quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                                   quad_intensity = np.asarray(np.expand_dims(np.expand_dims(quad_intensity, axis=0), axis=0), dtype=np.float16)
                                   
                                   


                                   """ Detect if blank in uint16 """
                                   num_voxels = len(np.where(quad_intensity > 300)[0])
                                   if num_voxels < 300:
                                         #print('skipping: ' + str(s_c))
                                         #print('num voxels with signal: ' + str(num_voxels))
                                         continue     
                                   #else:
                                   #      print([x, y, z])
                                   #     plot_max(quad_intensity[0][0], ax=-1)
                                   #     plt.title('num_voxels')
                                   
                                   
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
                                       
                                       
                                       ### run MaskRCNN
                                       output = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.

                                       ### Add box patch factor
                                       for bid, box in enumerate(output['boxes'][0]):
                                           
                                          #box_centers = box_coords
                                           
                                           c = box['box_coords']
                                           box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                                          #if self.cf.dim == 3:
                                           box_centers.append((c[4] + c[5]) / 2)
                            
                            
                                           factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in \
                                                             zip(box_centers, np.array(quad_intensity[0][0].shape) / 2)])
          
                                           
                                           box['box_patch_center_factor' ] = factor
                                           
                                           
                                           
                                           output['boxes'][0][bid] = box
                                           
                                       results_dict = output
        

        
                                       if 'seg_preds' in results_dict.keys():
                                            results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
        
                                       seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                       segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] + seg_im[0, :, 0, :, :]
                                        
                                   
                                       for bs in range(batch_size):
        
                                           box_df = results_dict['boxes'][bs]
                                           box_vert = []
                                           box_score = []
                                           for box in box_df:
                                               if box['box_score'] > thresh:
                                                   box_vert.append(box['box_coords'])
                                                   box_score.append(box['box_score'])
                                                   
                                                   
                                           if len(box_vert) == 0:
                                               continue
            
            
                                           patch_im = np.copy(batch_im[bs][0])
                                           patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first
                                            
                        
                                           all_patches.append({'box_vert':box_vert, 'results_dict': results_dict, 'patch_im': patch_im, 'total_blocks': bs % total_blocks + (total_blocks - batch_size), 
                                                                'focal_cube':focal_cube, 'xyz':batch_xyz[bs]})
                                           
                                       ### Reset batch
                                       batch_im = []
                                       batch_xyz = []


                    toc = time.perf_counter()
                    
                    print(f"MaskRCNN and splitting in {toc - tic:0.4f} seconds")



                    filename = input_name.split('/')[-1].split('.')[0:-1]
                    filename = '.'.join(filename)
                
                                           
                    pool_for_wbc = []
                    for patch in all_patches:
                        
                        xyz = patch['xyz']
                        results = patch['results_dict']
                        boxes = results['boxes'][0]
                        
                        #scaled_boxes = np.copy(results)
                        for idb, box in enumerate(boxes):
                            
                            
                            c = box['box_coords']
                            c[0] = c[0] + xyz[0]
                            c[1] = c[1] + xyz[1]
                            c[2] = c[2] + xyz[0]
                            c[3] = c[3] + xyz[1]
                            c[4] = c[4] + xyz[2]
                            c[5] = c[5] + xyz[2]
                            
                            
                            #if c[5] > 200:
                            #    zzz
                            
                            box['box_coords'] = c
                            
                            box['box_n_overlaps'] = 1
                            
                            box['patch_id'] = '0_0'
                            
                            #results['boxes'][0][idb] = box
                            
                            pool_for_wbc.append(box)
                            
                    #zzz        
                    ### SKIP IF NOTHING AFTER POOLING
                    if len(pool_for_wbc) == 0:
                       print('empty after pooling')
                       continue
                    
                    
                    
    
                    regress_flag = False
                    n_ens = 1
                    wbc_input = [regress_flag, [pool_for_wbc], 'dummy_pid', cf.class_dict, cf.clustering_iou, n_ens]
                    from predictor import *
                    out = apply_wbc_to_patient(wbc_input)[0]   
                    
                    #results_dict = output
                    
    
                    #patch_depth = patch_im.shape[0]
                    #patch_size = patch_im.shape[1]
                
                
                    box_vert = []
                    for box in out[0]:
                        box_vert.append(box['box_coords'])
                
                    box_vert = np.asarray(np.round(np.vstack(box_vert)), dtype=int)
                    
                    
                    seg_overall = np.copy(segmentation)
                    seg_overall[seg_overall > 0] = 1
                    
                    #label_arr = np.copy(results_dict['seg_preds'],)
                    
                
                
                    df_cleaned = split_boxes_by_Voronoi3D(box_vert, vol_shape = seg_overall.shape)
                    merged_coords = df_cleaned['bbox_coords'].values
                
                    
                
                    
                    ### Then APPLY these boxes to mask out the objects in the main segmentation!!!
                    new_labels = np.zeros(np.shape(seg_overall))
                    overlap = np.zeros(np.shape(seg_overall))
                    all_lens = []
                    for box_id, box_coords in enumerate(merged_coords):
                    
                        if len(box_coords) == 0:
                            continue
                        all_lens.append(len(box_coords))
                        
                        
                        
                        ### swap axis
                        #box_coords[:, [2, 0]] = box_coords[:, [0, 2]]  
                        bc = box_coords
                        """ HACK: --- DOUBLE CHECK ON THIS SUBTRACTION HERE """
                        
                        #bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
                        bc = np.asarray(bc, dtype=int)                    
                        new_labels[bc[:, 0], bc[:, 1], bc[:, 2]] = box_id + 1   
                        overlap[bc[:, 0], bc[:, 1], bc[:, 2]] = overlap[bc[:, 0], bc[:, 1], bc[:, 2]] + 1
                        
    
    
                    new_labels = np.asarray(new_labels, np.int32)
                    #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_labels_BOXES_02.tif', new_labels)
                    
                
                    new_labels[seg_overall == 0] = 0   ### This is way simpler and faster than old method of looping through each detection
                    #plot_max(new_labels)
                    
                    new_labels = np.asarray(new_labels, np.int32)
                    #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_labels_overlap02.tif', new_labels)
                    
                    
                    
                    """ CLEANUP 
                    
                            (1) Find objects that are not connected at all (spurious assignments) and only keep the larger object
                            
                            (2) The rest of the spurious assignments can be assigned to nearest object by dilating the spurious assignments
                            
                            (3) Same goes for leftover hanging bits of segmentation that were not assigned due to bounding box issues
                    
                    
                    
                            optional: do this earlier --> but delete any bounding boxes with a centroid that does NOT contain a segmented object?
                    
                    """
                    from functional.matlab_crop_function import *
                    from functional.tree_functions import *               
                
                    to_assign = np.zeros(np.shape(new_labels))
                    cc = measure.regionprops(new_labels)
                    
                    obj_num = 1
                    for id_o, obj in enumerate(cc):
                        
                        tmp = np.zeros(np.shape(new_labels))
                        coords = obj['coords']
                        center = np.asarray(np.round(obj['centroid']), dtype=int)
                        
                        tmp[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                        
                        
                        
                        ### If center is blank, also skip?
                        # if tmp[center[0], center[1], center[2]] == 0:
                            
                        #     #clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
                        #     to_assign[coords[:, 0], coords[:, 1], coords[:, 2]] = obj_num
                        #     print('empty center')
                        #     obj_num += 1
                        #     continue
                        
                        ### SPEED THIS UP BY CROPPING IT OUT
                        
                        #bw_lab = measure.label(tmp, connectivity=1)
                        
    
                        
                        crop_size = 50
                        z_size = 20
                        crop_input, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(np.moveaxis(tmp, 0, -1), y=center[2],  \
                                                                                                        x=center[1], z=center[0], crop_size=crop_size, z_size=z_size,  \
                                                                                                        height=height, width=width, depth=depth_im)                
                        bw_lab = measure.label(np.moveaxis(crop_input, -1, 0), connectivity=1)
                            
                            
                        
                        
                        if np.max(bw_lab) > 1:
                            check_cc = measure.regionprops(bw_lab)
                            
                            #print(str(id_o))
                            all_lens = []
                            all_coords = []
                            for check in check_cc:
                                all_lens.append(len(check['coords']))
                                all_coords.append(check['coords'])
                            
                            all_coords = np.asarray(all_coords)
                            
                            min_thresh = 30
                            if np.max(all_lens) > min_thresh: ### keep the main object if it's large enough else delete EVERYTHING
                            
                                amax = np.argmax(all_lens)
                                
                                ### delete all objects that are NOT the largest conncected component
                                ind = np.delete(np.arange(len(all_lens)), amax)
                                to_del = all_coords[ind]
                                
                                to_del = np.vstack(to_del)
                                
                            else:
                                to_del = np.vstack(all_coords)
                                
                                
                            ### Have to loop through coord by coord to make sure they remain separate
                            for coord_ass in to_del:
                                
                                ### scale coordinates back up
                                coord_ass = scale_coords_of_crop_to_full(np.roll([coord_ass], -1), box_xyz, box_over)
                                coord_ass = np.roll(coord_ass, 1)
                                
                                to_assign[coord_ass[:, 0], coord_ass[:, 1], coord_ass[:, 2]] = obj_num
                                
                                obj_num += 1
    
                            # else:  ### for debug
                            #     print('Too small')
                            #     plot_max(bw_lab)
                            
                        #print('hey')
                    
                    to_assign = np.asarray(to_assign, np.int32)
                    #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_to_assign.tif', to_assign)
                                    
                    
                    
                    ### Expand each to_assign to become a neighborhood!  ### OR JUST DILATE THE WHOLE IMAGE?
                    clean_labels = np.copy(new_labels)
                    clean_labels[to_assign > 0] = 0
                    
           
    
                    clean_labels = expand_add_stragglers(to_assign, clean_labels)
                    #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_ass_step1.tif', clean_labels)                
                    
                    
                    ### Expand each leftover segmentation piece to be a part of the neighborhood!
                    bw_seg = np.copy(segmentation)
                    bw_seg[bw_seg > 0] = 1
                    bw_seg[clean_labels > 0] = 0
                    
                    
                    stragglers = measure.label(bw_seg)
                    clean_labels = expand_add_stragglers(stragglers, clean_labels)
                    #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_ass_step2.tif', clean_labels)                      
                    
                    
                    ### Also clean up small objects and add them to nearest object that is large
                    min_size = 50
                    
                    all_obj = measure.regionprops(clean_labels)
                    small = np.zeros(np.shape(clean_labels))
                    counter = 1
                    for o_id, obj in enumerate(all_obj):
                        c = obj['coords']
                        
                        if len(c) < min_size:
                            small[c[:, 0], c[:, 1], c[:, 2]] = counter
                            counter += 1
                        
                        
                        
                    small = np.asarray(small, np.int32)
    
                    clean_labels = expand_add_stragglers(small, clean_labels)
    
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_ass_step3.tif', clean_labels)                      
                    
    
                    #input_im = np.asarray(input_im, np.uint8)
                    input_im = np.expand_dims(input_im, axis=0)
                    input_im = np.expand_dims(input_im, axis=2)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                          imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                          metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                    
                    
                    segmentation = clean_labels
            

###########################################################################################        
    

                    
                    
                    # stepwise_clean = np.asarray(stepwise_clean, np.int32)
                    # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_cleaned_NEW.tif', stepwise_clean)
                            
                        
                    
                    
                    #########################################################################################################################
                    
                    

                    # segmentation = stepwise_clean[overlap_pz: overlap_pz + Lpatch_depth, overlap_pxy: overlap_pxy + Lpatch_size, overlap_pxy: overlap_pxy + Lpatch_size]

                    #%% Output to memmap array
                    


                    """ Output to memmmapped arr """
                    
                    # tic = time.perf_counter()
                    # memmap_save[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size] = segmentation
                    # memmap_save.flush()
                    
                    # toc = time.perf_counter()
                    
                    # print(f"Save in {toc - tic:0.4f} seconds")
                    
                    
                    """ Also save list of coords of where the cells are located so that can easily access later (or plot without making a whole image!) """
                    #print('saving coords')
                    #tic = time.perf_counter()
                    from skimage import measure
                    #labels = measure.label(segmentation)
                    #blobs_labels = measure.label(blobs, background=0)
                    cc = measure.regionprops(segmentation)
                    
                    
                    ########################## TOO SLOW TO USE APPEND TO ADD EACH ROW!!!
                    ######################### MUCH FASTER TO JUST MAKE A DICTIONARY FIRST, AND THEN CONVERT TO DATAFRAME AND CONCAT
                    d = {}
                    for i_list, cell in enumerate(cc):
                        center = cell['centroid']
                        center = [round(center[0]), round(center[1]), round(center[2])]
                        
                        
                        coords = cell['coords']
                        
                        coords[:, 0] = coords[:, 0] + s_c[2]
                        coords[:, 1] = coords[:, 1] + s_c[1]
                        coords[:, 2] = coords[:, 2] + s_c[0]
                        
                        d[i_list] = {'offset': s_c, 'block_num': id_c, 
                               'Z': center[0], 'X': center[1], 'Y': center[2],
                               'Z_scaled': center[0] + s_c[2], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[0],
                               'equiv_diam': cell['equivalent_diameter'], 'vol': cell['area'], 'coords':coords}
                        
                        
                        """
                        
                        
                            TO DO 
                            
                                1) Remember to scale s_c so that the coordinates are related to ORIGINAL location NOT with overlap_xy, overlap_xz
                            
                                2) ***ALSO, find some way to remove overlap from LARGE patch edges? - i.e. edges of s_c
                        
                        
    
                           MAKE PRE-LOADING DATA possible, so it doesn't have to wait so long each time
    
                        
                        
                        """
                        
                    
                    
                    
                    # df = pd.DataFrame.from_dict(d, "index")
            
                    # coords_df = pd.concat([coords_df, df])           
                                            
                    
                    toc = time.perf_counter()
                    
                    
                    # np.save(sav_dir + filename + '_numpy_arr', coords_df)
                    # #print(f"Save coords in {toc - tic:0.4f} seconds \n\n")
                    
                    # print('Saved asynchronously')
                    
                    # return coords_df
                    
                    




                    #tic = time.perf_counter()
                
                    
                    #%% Post-processing poolthread
                    # if thread_post == 1:   ### previous poolthread was running and now needs to be joined
                        
                    #     coords_df = async_post.get()  # get the return value from your function.    
                        
                    #     poolThread_postprocess.close()
                    #     poolThread_postprocess.join()   
                        
                        


                    # ### Initiate poolThread 
                    
                    # poolThread_postprocess = ThreadPool(processes=1)


                    # ### start post-processing asynchronously
                    # async_post = poolThread_postprocess.apply_async(box_to_arr_async, (box_coords_all, input_name, dset_im, coords_df, input_im, id_c)) # tuple of args for foo
                    
                    # thread_post = 1  ### to allow async on next iteration


                    # toc = time.perf_counter()
                    
                    print(f"Post-proccesing in {toc - tic:0.4f} seconds")
                                        
                


        
    print('\n\nSegmented outputs saved in folder: ' + sav_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    