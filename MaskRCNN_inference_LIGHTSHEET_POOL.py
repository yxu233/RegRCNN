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

import time
from multiprocessing.pool import ThreadPool


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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
                                       results_dict = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                                       
        
                                       if 'seg_preds' in results_dict.keys():
                                            results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
        
                                       #seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                        
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
                                           
                    p = Pool(8)
                    #kwargs  = [{"gdf":gdf, "index":i, "margin":margin, "line_mult":line_mult} for i in range(len(gdf))]
                    box_coords_all = p.map(post_process_boxes, all_patches)
                    
                    p.close()  ### to prevent memory leak
    
    
                    toc = time.perf_counter()
                    
                    print(f"MaskRCNN and splitting in {toc - tic:0.4f} seconds")
                    

    
                    """ 
                    
                    
                    
                    
                                Can run all of the rest of this asynchronously!!! 
                    
                    
                    
                    

                    """
    
                    def get_boxes_from_doublets(all_box_coords, im_size, size_thresh):
                        doublets = np.zeros(im_size)
                    
                        doublets = np.moveaxis(doublets, 0, -1)                    
                        sizes = []
                        for coords in all_box_coords:
                                    sizes.append(len(coords))
                                    if len(coords) < size_thresh:
                                        continue
                                    doublets[coords[:, 0], coords[:, 1], coords[:, 2]] = doublets[coords[:, 0], coords[:, 1], coords[:, 2]] + 1
                                    
                        doublets[doublets <= 1] = 0
                        doublets[doublets > 0] = 1                    
                        
                        
                        lab = label(doublets)
                        cc = regionprops(lab)
        
                        print('num doublets: ' + str(len(cc)))
            
                        cleaned = np.zeros(np.shape(doublets))
                        arr_doublets = [ [] for _ in range(len(cc) + 1)]  ### start at 1 because no zero value objects in array!!!
                        box_ids = [ [] for _ in range(len(cc) + 1)]
                        for b_id, coords in enumerate(all_box_coords):  
                                    region = lab[coords[:, 0], coords[:, 1], coords[:, 2]]
                                    if len(np.where(region)[0]) > 0:
                                        ids = np.unique(region)
                                        ids = ids[ids != 0]   ### remove zero
                                        for id_o in ids:
                                            arr_doublets[id_o].append(coords)
                                            box_ids[id_o].append(b_id)
                                    
                        return doublets, arr_doublets, box_ids, cc
                    
                    
                    
                    def box_to_arr_async(box_coords_all, input_name, dset_im, coords_df, input_im, id_c):
    
                        size_thresh = 80    ### 150 is too much!!!
        
                        ### Clean up boxes so it's just one large list
                        all_box_coords = []
                        for patch_box_coords in box_coords_all:
                            #print(patch_box_coords)
                            if patch_box_coords:  ### FOR SOME REASON SOME IS NONETYPE???
                                for coords in patch_box_coords:
                                    all_box_coords.append(coords)
                            
                                             
                        
                        filename = input_name.split('/')[-1].split('.')[0:-1]
                        filename = '.'.join(filename)
        
                        
                        ### Go through all coords and assemble into whole image with splitting and step-wise
                        
                        stepwise_im = np.zeros(np.shape(input_im))
                        cell_num = 0
                        
                        cell_ids = np.arange(len(all_box_coords))
                        random.shuffle(cell_ids)
                        for coords in all_box_coords:
                            if len(coords) < size_thresh:
                                continue                    
                            stepwise_im[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_ids[cell_num]
                            
                            cell_num += 1
        
                            
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
                          
                        doublets, arr_doublets, box_ids, cc = get_boxes_from_doublets(all_box_coords, im_size, size_thresh)
                                    
                        sav_doubs = np.moveaxis(doublets, -1, 0)
                        sav_doubs = np.asarray(sav_doubs, dtype=np.int32)
                        #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_doublets.tif', sav_doubs)
        
                        
        
                            
                        """First pass through just identifies all overlaps with 2 boxes only 
                                ***and ADDS their segmentations together so less weird subtraction artifacts
                        
                        """
                        iou_thresh = 0.2
                        clean_box_coords = np.copy(all_box_coords)
                        num_doubs = 0
                        for case_id, case in enumerate(arr_doublets):
                            
                            if len(case) == 0:
                                continue
                            box_nums = np.asarray(box_ids[case_id])
        
                            ### Find identical rows that match to overlap region
                            overlap = cc[case_id - 1]['coords']    ### case_id minus 1 when indexing cc because no region of value zero from above
                            
                            ### Next calculate iou for each individual case
                            iou_per_case = []
                            for reg in case:
                                intersect = (reg[:, None] == overlap).all(-1).any(-1)
                                intersect = reg[intersect]
                                intersect = len(intersect)
                                
                                union = len(np.unique(np.vstack([overlap, reg]), axis=0))
                                
                                iou_per_case.append(intersect/union)
                            iou_per_case = np.asarray(iou_per_case)
        
                            
                            box_nums = np.asarray(box_ids[case_id])                
                            ### The majority of cases only have 2 things overlapping AND have high overlap
                            if len(box_nums) == 2 and len(np.where(iou_per_case > iou_thresh)[0]) == len(box_nums):  
                                ### In the case of doublets with HIGH overlap, just pick the one with lowest iou_thresh, and discard the other
                                exclude_box = np.argmin(iou_per_case)
                                
                                ### Case 2: anything ABOVE iou thresh is fully deleted
                                to_del = np.where(iou_per_case > iou_thresh)[0]    
                                to_del = to_del[to_del != exclude_box]  ### make sure not to delete current box
                                
                                del_boxes = box_nums[to_del]
                                
                                clean_box_coords[del_boxes] = [[]]
                                
                                
                                ### also ADD coordinates from BOTH boxes to the clean_box_coords at the exclude_box ID
                                clean_box_coords[box_nums[exclude_box]] = np.unique(np.vstack([case[0], case[1]]), axis=0)
                                
                                num_doubs += 1
                        
                                
                                
                        clean_box_coords = [x for x in clean_box_coords if x != []]
                                
                        first_pass_doublets, arr_doublets, box_ids, cc = get_boxes_from_doublets(clean_box_coords, im_size, size_thresh)                             
                        
                        
                        # sav_doubs = np.moveaxis(first_pass_doublets, -1, 0)
                        # sav_doubs = np.asarray(sav_doubs, dtype=np.int32)
                        # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_doublets_FP.tif', sav_doubs)
        
                                        
                        
                        
                        ### loop through all doublets and determine iou
                        for case_id, case in enumerate(arr_doublets):
                            
                            if len(case) == 0:
                                continue
                            
                            box_nums = np.asarray(box_ids[case_id])
        
                            ### Find identical rows that match to overlap region
                            overlap = cc[case_id - 1]['coords']    ### case_id minus 1 when indexing cc because no region of value zero from above
                            
                            # matched = case[0]  ### start with first case
                            # for reg in case:
                            #     matched = (reg[:, None] == matched).all(-1).any(-1)
                            #     matched = reg[matched]
                                
                            
                            ### HACK -- sometimes doesn't match across ALL, so need to just do ANYTHING that matches across any of them for now..
                            # vals, counts = np.unique(np.vstack(case), axis=0, return_counts=True)
                            # matched = vals[counts >= 2]
        
                            ### Next calculate iou for each individual case
                            iou_per_case = []
                            for reg in case:
                                intersect = (reg[:, None] == overlap).all(-1).any(-1)
                                intersect = reg[intersect]
                                intersect = len(intersect)
                                
                                union = len(np.unique(np.vstack([overlap, reg]), axis=0))
                                
                                iou_per_case.append(intersect/union)
                            iou_per_case = np.asarray(iou_per_case)
        
              
                            """ 3 possible conclusions: 
                                        1) Highest iou gets to keep overlap area (consensus)
                                        2) All other iou above iou threshold (0.7?) is deleted fully
                                        3) All other iou BELOW threshold is kept, but the overlapping coordinates are deleted from the cell
        
        
                                """     
                            ### Case 1: LOWEST iou is kept as overlap area #do nothing
                            exclude_box = np.argmax(iou_per_case)
                            
                            ### Case 2: anything ABOVE iou thresh is fully deleted
                            # to_del = np.where(iou_per_case > iou_thresh)[0]
                            # to_del = to_del[to_del != exclude_box]  ### make sure not to delete current box
                            
                            # del_boxes = box_nums[to_del]
                            # clean_box_coords[del_boxes] = [[]]
                            
                            # other_boxes = np.delete(box_nums, np.concatenate((to_del, [exclude_box])))
                            
                            ### Case 3: anything else that is not fully overlapped (low iou) only has these coords deleted
                            other_boxes = np.delete(box_nums, exclude_box)
                            for obox_num in other_boxes:
                                obox = clean_box_coords[obox_num]
                                
                                ### HACK if box is empty already
                                if len(obox) == 0: 
                                    continue
                                #all_c = np.concatenate((obox, matched))
                                
                                not_matched = obox[~(obox[:, None] == overlap).all(-1).any(-1)] ### the only coords that do NOT overlap
                                
                                #vals, counts = np.unique(all_c, axis=0, return_counts=True)
                                
                                #unique = vals[counts < 2]   ### the only coords that do NOT overlap
                                
                                ### Can we also add a way to find the connectivity of the objects at the end here???
                                ### Like, if there's almost no discrete objects left, just delete the whole darn thing?
                                clean_box_coords[obox_num] = not_matched
                                                    
                            ### To prevent holes, add the area of the overlap to the box that was excluded from subtraction
                            clean_box_coords[box_nums[exclude_box]] = np.unique(np.vstack([overlap, clean_box_coords[box_nums[exclude_box]]]), axis=0)
                            
                            
                        clean_box_coords = [x for x in clean_box_coords if x != []]
                                
                        
                        stepwise_clean = np.zeros(np.shape(stepwise_im))
                        cell_num = 0
                        for coords in clean_box_coords:
                            if len(coords) < size_thresh:
                                continue                    
                            stepwise_clean[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_ids[cell_num]
                            
                            cell_num += 1                
                        
                        
                        stepwise_clean = np.asarray(stepwise_clean, np.int32)
                        tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_cleaned_NEW.tif', stepwise_clean)
                                
                            
                        
                        
                        #########################################################################################################################
                        
                        
    
                        segmentation = stepwise_clean[overlap_pz: overlap_pz + Lpatch_depth, overlap_pxy: overlap_pxy + Lpatch_size, overlap_pxy: overlap_pxy + Lpatch_size]
    
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
                            
                            d[i_list] = {'offset': s_c, 'block_num': id_c, 
                                   'Z': center[0], 'X': center[1], 'Y': center[2],
                                   'Z_scaled': center[0] + s_c[0], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[2],
                                   'equiv_diam': cell['equivalent_diameter'], 'vol': cell['area'], 'coords':cell['coords']}
                            
                            
                            """
                            
                            
                                TO DO 
                                
                                    1) Remember to scale s_c so that the coordinates are related to ORIGINAL location NOT with overlap_xy, overlap_xz
                                
                                    2) ***ALSO, find some way to remove overlap from LARGE patch edges? - i.e. edges of s_c
                            
                            
        
                               MAKE PRE-LOADING DATA possible, so it doesn't have to wait so long each time
        
                            
                            
                            """
                            
                        
                        
                        
                        df = pd.DataFrame.from_dict(d, "index")
                
                        coords_df = pd.concat([coords_df, df])           
                                                
                        
                        #toc = time.perf_counter()
                        
                        
                        np.save(sav_dir + filename + '_numpy_arr', coords_df)
                        #print(f"Save coords in {toc - tic:0.4f} seconds \n\n")
                        
                        print('Saved asynchronously')
                        
                        return coords_df
                        
                        




                    tic = time.perf_counter()
                
                    
                    #%% Post-processing poolthread
                    if thread_post == 1:   ### previous poolthread was running and now needs to be joined
                        
                        coords_df = async_post.get()  # get the return value from your function.    
                        
                        poolThread_postprocess.close()
                        poolThread_postprocess.join()   
                        
                        


                    ### Initiate poolThread 
                    
                    poolThread_postprocess = ThreadPool(processes=1)


                    ### start post-processing asynchronously
                    async_post = poolThread_postprocess.apply_async(box_to_arr_async, (box_coords_all, input_name, dset_im, coords_df, input_im, id_c)) # tuple of args for foo
                    
                    thread_post = 1  ### to allow async on next iteration


                    toc = time.perf_counter()
                    
                    print(f"Post-proccesing in {toc - tic:0.4f} seconds")
                                        
                


        
    print('\n\nSegmented outputs saved in folder: ' + sav_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    