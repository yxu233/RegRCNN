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

from scipy.stats import norm

from functional.matlab_crop_function import *
from functional.tree_functions import *  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def expand_add_stragglers(to_assign, clean_labels):
    cc_ass = measure.regionprops(to_assign)
    for ass in cc_ass:
        
        coords = ass['coords']


        match = to_assign[coords[:, 0], coords[:, 1], coords[:, 2]]
        vals = np.unique(match)
        

        exp = expand_coord_to_neighborhood(coords, lower=1, upper=1 + 1)   ### always need +1 because of python indexing
        exp = np.vstack(exp)

        values = clean_labels[exp[:, 0], exp[:, 1], exp[:, 2]]
        
        vals, counts = np.unique(values, return_counts=True)
        
        if np.max(vals) > 0:  ### if there is something other than background matched
            
            ### Assign to the nearest object with the MOST matches
            ass_val = np.argmax(counts[1:])   ### skip 0
            ass_val = vals[1:][ass_val]
            
            
            clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = ass_val   ### Give these pixels the value of the associated object
            
        
            clean_labels = np.asarray(clean_labels, np.int32)
    return clean_labels
        


def post_process_boxes(kwargs):                   

    ### NEED TO ADAPT IF BATCH_SIZE > 1    

    box_vert = kwargs['box_vert']
    results_dict = kwargs['results_dict']
    patch_im = kwargs['patch_im']    
    total_blocks = kwargs['total_blocks']
    xyz = kwargs['xyz']
    focal_cube = kwargs['focal_cube']
    
    box_coords_all = []
    box_factor_all = []
    box_center_all = []
    patch_depth = patch_im.shape[0]
    patch_size = patch_im.shape[1]

       
    box_vert = np.asarray(np.round(np.vstack(box_vert)), dtype=int)
    
    label_arr = np.copy(results_dict['seg_preds'],)
    

    if len(box_vert) == 0:
        return []


    df_cleaned = split_boxes_by_Voronoi3D(box_vert, vol_shape = patch_im.shape)
    merged_coords = df_cleaned['bbox_coords'].values


    ### Then APPLY these boxes to mask out the objects in the main segmentation!!!
    new_labels = np.zeros(np.shape(label_arr))
    for box_id, box_coords in enumerate(merged_coords):
    
        ### SOME BOXES GET LOST?
        if len(box_coords) == 0:
            print('BOX WAS FULLY SUBTRACTED')
            continue
            
        ### swap axis
        box_coords[:, [2, 0]] = box_coords[:, [0, 2]]  
        bc = box_coords
        """ HACK: --- DOUBLE CHECK ON THIS SUBTRACTION HERE """
        
        #bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
        bc = np.asarray(bc, dtype=int)                    
        new_labels[0, 0, bc[:, 1], bc[:, 0], bc[:, 2]] = box_id + 1   
        

    new_labels[label_arr == 0] = 0   ### This is way simpler and faster than old method of looping through each detection
    
    
    """ Then find which ones are on the edge and remove, then save coords of cells for the very end """
    
    if len(np.unique(new_labels)) == 1:
        print('WEIRD')
        return []
    
    #print(total_blocks)
    
    new_labels = np.asarray(new_labels, dtype=int)
    # cc = regionprops_table(new_labels[0][0], properties=('centroid', 'coords'))
    
    # df = pd.DataFrame(cc)
    # # ### Then filter using focal cube
    # edge_ids = np.where(focal_cube[np.asarray(df['centroid-2']), np.asarray(df['centroid-0']), np.asarray(df['centroid-1'])])[0]
    
    
    
    
    cc = regionprops(new_labels[0][0])
    
    df = pd.DataFrame()
    for obj in cc:
        cent = np.round(obj['centroid'])

        
      
        
       ### Find way to add center factor
         # boxes from the edges of a patch have a lower prediction quality, than the ones at patch-centers.
         # hence they will be down-weighted for consolidation, using the 'box_patch_center_factor', which is
         # obtained by a gaussian distribution over positions in the patch and average over spatial dimensions.
         # Also the info 'box_n_overlaps' is stored for consolidation, which represents the amount of
         # overlapping patches at the box's position.
        
        #zzz
         
        # box_df = results_dict['boxes'][bs]
         
         
        # all_box = []
        # all_factors = []
        # tmp = np.zeros(seg_im[0, :, 0].shape)
        # for box in box_df:
 
        #     c = box['box_coords']
        #      #box_centers = np.array([(c[ii] + c[ii+2])/2 for ii in range(len(c)//2)])
        #     box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
        #     box_centers.append((c[4] + c[5]) / 2)
            
            
        box_centers = cent
            
            
            # The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
            #    mean == pc == center of the volume
            #    scale == std
        factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in zip(box_centers, np.array(np.moveaxis(patch_im, 0, -1).shape) / 2)])
            
        #tmp[int(box_centers[2]), int(box_centers[0]), int(box_centers[1])] = factor
        
        coords = obj['coords']
        coords[:, 0] = coords[:, 0] + xyz[0]
        coords[:, 1] = coords[:, 1] + xyz[1]
        coords[:, 2] = coords[:, 2] + xyz[2]        
        
        df_dict = {'c0':int(cent[0])+xyz[0], 'c1':int(cent[1])+xyz[1], 'c2':int(cent[2])+xyz[2], 'coords':coords, 'patch_factor':factor}
        
        df = df.append(df_dict, ignore_index=True)
        
        
        #all_box.append(box_centers)
        #all_factors.append(factor)
   
        
        
        
        
        
        
   # edge_ids = np.where(focal_cube[np.asarray(df['c2'] - xyz[2], dtype=int), np.asarray(df['c0'] - xyz[0], dtype=int), np.asarray(df['c1'] - xyz[1], dtype=int)])[0]
           
        
        
    
    
    
    
    
    
    
    #print('num deleted: ' + str(len(edge_ids)))
    #df = df.drop(edge_ids)


    
    # if len(df) > 0:
    #     ### save the coordinates so we can plot the cells later
    #     #empty = np.zeros(np.shape(patch_im))
    #     for row_id, row in df.iterrows():
    #         coords = row['coords']
            
    #         coords[:, 0] = coords[:, 0] + xyz[0]
    #         coords[:, 1] = coords[:, 1] + xyz[1]
    #         coords[:, 2] = coords[:, 2] + xyz[2]
               
    #         box_coords_all.append(coords)
    #         box_factor_all.append(row['patch_factor'])
            
    #         box_center_all.append([row['c0']+xyz[0], row['c1']+xyz[1], row['c2']+xyz[2]])



    return df





if __name__=="__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/73) dil_group_norm_det_thresh_0_2/'      

            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/76) dil_group_norm_NEW_DATA_edges_wbc/'              
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/78) same_76_new_data_corrected/'               
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/79) same_78_but_det_nms_thresh_01/'  
           
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/81) same_78_det_thresh_02_pooled_validation/'
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/82) same_78_det_thresh_02_pool_val_MIN_CONF_09/'
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/85) only_new_data_dense_det_thresh_02/'
            
            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/86) new_dense_data_only_det_thresh_04/'
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/87) new_dense_CLEANED_det_thresh_02_min_conf_09/'
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/90) new_dense_cleaned_edges_det_02_min_conf_01/'
            
                
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/90) new_dense_cleaned_edges_det_02_min_conf_01_later_check437/'                    
    
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/91) new_CLEANED_training_det_thresh_02_min_conf_09/'
    
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/92) new_CLEANED_det_thresh_02_min_conf_09_groupnorm/'
            
            
            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN_device0/93) new_CLEANED_det_thresh02_min_thresh_01_groupnorm/'
            
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/94) newest_CLEANED_shrunk_det_thresh_02_min_conf_01/'
            
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/96) new_FOV_data_det_thresh_09_check_300/'
            
            
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
    
    model_selector = utils.ModelSelector(cf, logger)

    starting_epoch = 1
    #if cf.resume:
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
        

    list_folder = ['/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Test_RegRCNN/',
                   '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Test_RegRCNN_fortraining/']


    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_output_PYTORCH_96_last300_skimage_COLORED_step3_thresh09_NOfoc_fixed_neighborhoods'
    
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
        

        input_im_stack = [];
        for i in range(len(examples)):
             
             with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[i]['input']            
                input_im = tiff.imread(input_name)
                
       
                """ Analyze each block with offset in all directions """ 
                print('Starting inference on volume: ' + str(i) + ' of total: ' + str(len(examples)))
                
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
                colored_im = np.zeros([depth_im, width, height])
                
                split_seg = np.zeros([depth_im, width, height])
                all_xyz = [] 
                
                
                all_patches = []
                
                batch_size = 1
                batch_im = []
                batch_xyz = []
                
                
                
                ### SET THE STEP SIZE TO BE HALF OF THE IMAGE SIZE
                #step_z = patch_depth/2
                #step_xy = patch_size/2
                
                
                
                
                for z in range(0, depth_im + patch_depth, round(step_z)):
                    if z + patch_depth > depth_im:  continue

                    for x in range(0, width + patch_size, round(step_xy)):
                          if x + patch_size > width:  continue

                          for y in range(0, height + patch_size, round(step_xy)):
                               if y + patch_size > height: continue

                               print([x, y, z])

                               all_xyz.append([x, y, z])


                               
                               quad_intensity = input_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size];  
                               quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                               quad_intensity = np.asarray(np.expand_dims(np.expand_dims(quad_intensity, axis=0), axis=0), dtype=np.float16)
                               
                               
                               
                               # a = np.copy(input_im)
                               # a = a[0:128, 0:256, 0:256]
                               # a = np.moveaxis(a, 0, -1)
                               # a = np.asarray(np.expand_dims(np.expand_dims(a, axis=0), axis=0), dtype=np.float16)
                               
                               if len(batch_im) > 0:
                                   batch_im = np.concatenate((batch_im, quad_intensity))
                               else:
                                   batch_im = quad_intensity
                               
                               
                               batch_xyz.append([x,y,z])
                               
                               print(total_blocks)
                               total_blocks += 1  
                               
                               
                               if total_blocks % batch_size == 0:
      

                                   batch = {'data':batch_im, 'seg': np.zeros([batch_size, 1, patch_size, patch_size, patch_depth]), 
                                             'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                             'roi_masks': np.zeros([batch_size, 1, 1, patch_size, patch_size, patch_depth]),
                                             'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                             'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                   
                                   
                                   """ """
                                   ### run MaskRCNN   ### likely with NMS pooling
                                   #zzz
                                   output = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                                   
                                   
                                   """ """
                                   ### Run using Predictor instead
                                   #test_predictor = Predictor(cf, net, logger, mode='test')
                                   #output = test_predictor.predict_patient(batch)
                                   
                                   #new = np.copy(output)
                                   #new['boxes'] = []
                                   
                                   ### Add box patch factor
                                   new_box_list = []
                                   tmp_check = np.zeros(np.shape(quad_intensity[0][0]))
                                   for bid, box in enumerate(output['boxes'][0]):
                                       
                                       #box_centers = box_coords
                                       
                                       c = box['box_coords']
                                       
                                       #tmp_check[c[0]:c[2], c[1]:c[3], c[4]:c[5]] = 2 
                                       """ 
                                       
                                       
                                           Some bounding boxes have no associated segmentations!!! Skip these
                                       
                                       
                                       
                                       """
                                       if len(box['mask_coords']) == 0:
                                           #print(box)
                                           #zzz
                                           continue
                                       
                                       #else:
                                       #    tmp_check[c[0]:c[2], c[1]:c[3], c[4]:c[5]] = 1 
                                       
                                        
                                       
                                       
                                       box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                                      #if self.cf.dim == 3:
                                       box_centers.append((c[4] + c[5]) / 2)
                        
                        
                                       factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in \
                                                         zip(box_centers, np.array(quad_intensity[0][0].shape) / 2)])
      
                                       
                                       box['box_patch_center_factor'] = factor
                                       
                                       
                                       ### Only add if above threshold
                                       if box['box_score'] > thresh:
                                           new_box_list.append(box)     
                                           
                                        

                                           
                                           
                                           
                                   
                                   output['boxes'] = [new_box_list]
                                   #zzz
                                   ### DEBUG: make fake norm pdf
                                   # all_pdf = np.transpose(np.where(quad_intensity[0][0] > -1))
                                   
                                   # im_pdf = np.zeros(np.shape(quad_intensity[0][0]))
                                   # all_f = []
                                   
                                   # im_center = np.array(quad_intensity[0][0].shape) / 2
                                   
                                   # z_scale = 5
                                   # im_center[-1] = im_center[-1] * z_scale
                                   
                                   # for h in all_pdf:
                                       
                                   #     h_new = np.copy(h)
                                   #     h_new[-1] = h_new[-1] * z_scale
                                       

                                   #     all_factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in \
                                   #                           zip(h_new, im_center)])   
                                       
                                   #     #all_f.append(all_factor)
                                   #     #print(h)
                                       
                                           
                                           
                                   #     im_pdf[h[0], h[1], h[2]] = all_factor
                                       
                                           
                                   # #focal_cube = np.moveaxis(focal_cube, 0, -1)
                                   
                                   # vals = im_pdf[focal_cube > 0]
                                   # sub = np.max(vals)
                                   # copy = np.copy(im_pdf)
                                   # im_pdf[im_pdf < sub] = 0
                                   
                                   
                                   


                                   results_dict = output
                                   

    
                                   if 'seg_preds' in results_dict.keys():
                                        results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                                        results_dict['colored_boxes'] = np.expand_dims(results_dict['colored_boxes'][:, 1, :, :, :], axis=0)
                                        
                                   ### Add to segmentation
                                   seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                   
                                   color_im = np.moveaxis(results_dict['colored_boxes'], -1, 1) 
                                   segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] + seg_im[0, :, 0, :, :]
                                   colored_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = color_im[0, :, 0, :, :]
                                   
                                   for bs in range(batch_size):
    
                                       box_df = results_dict['boxes'][bs]
                                       box_vert = []
                                       box_score = []
                                       mask_coords = []
                                       for box in box_df:
                                          
                                               box_vert.append(box['box_coords'])
                                               box_score.append(box['box_score'])
                                               mask_coords.append(box['mask_coords'])
                                               #zzz
                                               if box['box_score'] < thresh:
                                                   #print('error thresh')
                                                   zzz
                                               
                                               
                                               
                                       if len(box_vert) == 0:
                                           continue
        
        
                                       patch_im = np.copy(batch_im[bs][0])
                                       patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first
                                        


                    
                                       all_patches.append({'box_vert':box_vert, 'results_dict': results_dict, 'patch_im': patch_im, 'total_blocks': bs % total_blocks + (total_blocks - batch_size), 
                                                            'focal_cube':focal_cube, 'xyz':batch_xyz[bs], 'mask_coords':mask_coords})
                                       
                                   ### Plot color image to see extent of overlap
                                   # tmp_im = np.zeros(np.shape(patch_im))
                                       
                                   # for mask_c in mask_coords:
                                   #     tmp_im[mask_c[:, 2], mask_c[:, 0], mask_c[:, 1]] = tmp_im[mask_c[:, 2], mask_c[:, 0], mask_c[:, 1]] + 1
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                   ### Reset batch
                                   batch_im = []
                                   batch_xyz = []
                                   
                                   patch = np.moveaxis(quad_intensity, -1, 1)
                                   save = np.concatenate((patch, seg_im), axis=2)
                                   #tiff.imwrite(sav_dir + '_' + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) + '_segmentation.tif', np.asarray(save, dtype=np.uint16),
                                   #             imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                   #             metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})                                     
                                   #zzz





                                   # if z > 30:
                                   #     zzz
 
                #zzz
                filename = input_name.split('/')[-1].split('.')[0:-1]
                filename = '.'.join(filename)
                
                
                segmentation = np.asarray(segmentation, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_segmentation_overlap3.tif', segmentation)


                colored_im = np.asarray(colored_im, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_colored_im.tif', colored_im)
                
                
                plot_max(segmentation)
           
                

 
                #%% For bounding box approach
                # p = Pool(8)

                # all_df = p.map(post_process_boxes, all_patches)
                 
                # p.close()  ### to prevent memory leak
                
                #zzz
                
                #focal_cube = np.moveaxis(focal_cube, 0, -1)

                pool_for_wbc = []
                exclude_edge = []
                for patch in all_patches:
                    
                    xyz = patch['xyz']
                    results = patch['results_dict']
                    boxes = np.copy(results['boxes'][0])
                    
                    #scaled_boxes = np.copy(results)
                    for idb, box in enumerate(boxes):
                        
                        
                        c = box['box_coords']
                        
                        #box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                        
                        
                        box_centers = []                        
                        
                        x_val = round((c[0] + c[2]) / 2)
                        if x_val >= patch_depth:
                            x_val = patch_depth - 1
                        box_centers.append(x_val)   
                        
                        
                        y_val = round((c[1] + c[3]) / 2)
                        if y_val >= patch_depth:
                            y_val = patch_depth - 1
                        box_centers.append(y_val)   

                        z_val = round((c[4] + c[5]) / 2)
                        if z_val >= patch_depth:
                            z_val = patch_depth - 1
                        box_centers.append(z_val)  





                        ### Add exclusion by focal_cube
                        box_centers = np.asarray(np.round(box_centers), dtype=int)
                        val = focal_cube[box_centers[0], box_centers[1], box_centers[2]]

          
                        c[0] = c[0] + xyz[0]
                        c[1] = c[1] + xyz[1]
                        c[2] = c[2] + xyz[0]
                        c[3] = c[3] + xyz[1]
                        c[4] = c[4] + xyz[2]
                        c[5] = c[5] + xyz[2]
                        
                        
 
                        coords = np.copy(box['mask_coords']) 
                        coords[:, 0] = coords[:, 0] + xyz[0]
                        coords[:, 1] = coords[:, 1] + xyz[1]                             
                        coords[:, 2] = coords[:, 2] + xyz[2]
 
    
                        #if c[5] > 200:
                        #    zzz
                        
                        box['box_coords'] = c
                        
                        box['box_n_overlaps'] = 1
                        
                        box['patch_id'] = '0_0'
                        
                        box['mask_coords'] = coords
                        
                        
                        
                        #results['boxes'][0][idb] = box
                        pool_for_wbc.append(box)
                        # if val == 0:
                        
                        #     pool_for_wbc.append(box)
                        
                        # else:
                        #     exclude_edge.append(box)   ### not needed if have overlap == 50%
                        
                        
                        
                        
                input_im = np.expand_dims(input_im, axis=0)
                input_im = np.expand_dims(input_im, axis=2)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                      imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                      metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                                        
                        
                        
                        
                #zzz        

                regress_flag = False
                n_ens = 1
                wbc_input = [regress_flag, [pool_for_wbc], 'dummy_pid', cf.class_dict, cf.clustering_iou, n_ens]
                from predictor import *
                out = apply_wbc_to_patient(wbc_input)[0]   
                
                #results_dict = output
                
                
                
                #%% Get masks directly without any other postprocessing from maskrcnn outputs -- will have overlaps!!!
                box_masks = []
                for box in out[0]:
                    box_masks.append(box['mask_coords'])
            
                box_masks = np.asarray(box_masks[0])
                
                tmp = np.zeros(np.shape(segmentation))
                tmp_overlap = np.zeros(np.shape(segmentation))
                for m_id, mask in enumerate(box_masks):
                    tmp[mask[:, 2], mask[:, 0], mask[:, 1]] = m_id + 1
                    tmp_overlap[mask[:, 2], mask[:, 0], mask[:, 1]]  = tmp_overlap[mask[:, 2], mask[:, 0], mask[:, 1]] + 1
                
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'tmp.tif', np.asarray(tmp, dtype=np.int32))        
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'tmp_overlap.tif', np.asarray(tmp_overlap, dtype=np.int32))        
                
                #zzz
                
                
                #%% For KNN based assignment of voxels to split boxes

                patch_depth = patch_im.shape[0]
                patch_size = patch_im.shape[1]
            
                #zzz
                box_vert = []
                for box in out[0]:
                    box_vert.append(box['box_coords'])
            
                box_vert = np.asarray(np.round(np.vstack(box_vert)), dtype=int)
                
                
                seg_overall = np.copy(segmentation)
                
                
                #%% Since there's so much overlap, go by consensus, only keep seg > 2
                # seg_overall[seg_overall <= 2] = 0
                
                # seg_overall[seg_overall > 0] = 1
                
                #label_arr = np.copy(results_dict['seg_preds'],)
                
                #zz
            
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
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_labels_BOXES_02_SHRUNK.tif', new_labels)
                
            
                new_labels[seg_overall == 0] = 0   ### This is way simpler and faster than old method of looping through each detection
                plot_max(new_labels)
                
                new_labels = np.asarray(new_labels, np.int32)
                #tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_labels_overlap02.tif', new_labels)
                
                
                
                """ CLEANUP 
                
                        (1) Find objects that are not connected at all (spurious assignments) and only keep the larger object
                        
                        (2) The rest of the spurious assignments can be assigned to nearest object by dilating the spurious assignments
                        
                        (3) Same goes for leftover hanging bits of segmentation that were not assigned due to bounding box issues
                
                
                
                        optional: do this earlier --> but delete any bounding boxes with a centroid that does NOT contain a segmented object?
                
                """
             
            
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
                        
                        print(str(id_o))
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
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_to_assign_FOCAL.tif', to_assign)
                                
                
                
                ### Expand each to_assign to become a neighborhood!  ### OR JUST DILATE THE WHOLE IMAGE?
                clean_labels = np.copy(new_labels)
                clean_labels[to_assign > 0] = 0

                clean_labels = expand_add_stragglers(to_assign, clean_labels)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_ass_step1.tif', clean_labels)                
                
                
                #%% ## Expand each leftover segmentation piece to be a part of the neighborhood!
                bw_seg = np.copy(segmentation)
                bw_seg[bw_seg > 0] = 1
                bw_seg[clean_labels > 0] = 0
                
                
                stragglers = measure.label(bw_seg)
                clean_labels = expand_add_stragglers(stragglers, clean_labels)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_ass_step2.tif', clean_labels)                      
                
                
                #%% ## Also clean up small objects and add them to nearest object that is large
                
                #zzz
                
                min_size = 80
                
                all_obj = measure.regionprops(clean_labels)
                small = np.zeros(np.shape(clean_labels))
                counter = 1
                for o_id, obj in enumerate(all_obj):
                    c = obj['coords']
                    
                    if len(c) < min_size:
                        small[c[:, 0], c[:, 1], c[:, 2]] = counter
                        counter += 1
                    
                    
                    
                small = np.asarray(small, np.int32)
                
                clean_labels[small > 0] = 0   ### must remember to mask out all the small areas otherwise will get reassociated back with the small area!

                clean_labels = expand_add_stragglers(small, clean_labels)


                ### Add back in all the small objects that were NOT near enough to touch anything else
                small[clean_labels > 0] = 0
                
                clean_labels[small > 0] = small[small > 0]




                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_ass_step3_FOCAL.tif', clean_labels)                      
                
                
                
                
                

                #%% ## Also go through Z-slices and remove any super thin sections in XY? Like < 10 pixels
                count = 0
                for zid, zslice in enumerate(clean_labels):
                    
                    cc = measure.regionprops(zslice)
                    
                    for obj in cc:
                        coords = obj['coords']
                        if len(coords) < 10:
                            clean_labels[zid, coords[:, 0], coords[:, 1]] = 0
                            count += 1
                            
                            
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_ass_step4_FOCAL.tif', clean_labels)           
                
                
                ### Also remove super large objects?
                
                
                
                
                #%% Pad entire image and shift by 1 to the right and 1 down?
                shift_im = np.zeros([depth_im + 1, width + 1, height + 1])
                
                shift_im[1:, 1:, 1:] = clean_labels
                
                shifted = shift_im[:-1, :-1, :-1]
                
                shifted = np.asarray(shifted, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_shfited.tif', shifted)    
                
                
                
                
                    
                #zzz
                                      # all_patches.append({'box_vert':box_vert, 'results_dict': results_dict, 'patch_im': patch_im, 'total_blocks': bs % total_blocks + (total_blocks - batch_size), 
                                      #                      'focal_cube':focal_cube, 'xyz':batch_xyz[bs]})

                #zzz 
                
                """ Clean up all df to extract factors per box, and then use k-nearest neighbor to get best matches 
                
                    linearize all boxes first so not by patch
                
                """
                # df_boxes = pd.DataFrame()
                # for patch_df in all_df:
                    
                #     df_boxes = df_boxes.append(patch_df, ignore_index=True)
                    
                    
                    
                    
                    
                    
                # centroids = [np.asarray(df_boxes['c0']), np.asarray(df_boxes['c1']), np.asarray(df_boxes['c2'])]
                # centroids = np.transpose(centroids)
                



                # from sklearn.neighbors import NearestNeighbors         
                # nbrs = NearestNeighbors(n_neighbors=8, metric='euclidean').fit(centroids)
                
                # distances, indices = nbrs.kneighbors(centroids)
                
                # dist_thresh = 10
                # all_del = np.asarray([])
                # keep_ids = []
                # for id_r, test_ids in enumerate(indices):
                    
                #     ### skip if deleted previously
                #     if id_r in all_del:
                #         continue
                    
                #     ### First find which of these are within the radius of distance
                #     test_dist = distances[id_r]
                    
                #     test_ids = test_ids[np.where(test_dist < dist_thresh)[0]]
                    
                    
                    
                #     ### Also delete any indices that have already been deleted previously???
                    
                    
                    
                #     ### Then figure out which of these have coordinates that match
                #     test_rows = df_boxes.iloc[test_ids]
                #     query_box = test_rows.iloc[0]['coords']
                    
                #     int_id = [0]  ### include current obj
                #     for id_t in range(1, len(test_rows)):
                #         intersect = (query_box[:, None] == test_rows.iloc[id_t]['coords']).all(-1).any(-1)
                        
                        
                #         ### IGNORE IF INTERSECT IS NOT ENOUGH
                #         print(len(np.where(intersect)[0]))
                #         if len(np.where(intersect)[0]) < 10:
                #                continue
                        
                        
                        
                        
                #         if len(np.where(intersect)[0]) > 0:
                #             int_id.append(id_t)

                #     int_id = np.asarray(int_id)
                    
                #     test_ids = test_ids[int_id]
                    
                #     ### Then find highest match with highest patch factor --- delete all the rest
                #     patch_factors = test_rows.iloc[int_id]['patch_factor']
                    
                #     max_id = test_ids[np.argmax(patch_factors)]
                    
                #     keep_ids.append(max_id)  ### save id
                    
                #     ### also make a list to keep track of what to delete
                #     to_del = test_ids[np.where(test_ids != max_id)[0]]
                    
                    
                #     all_del = np.concatenate((all_del, to_del))
                    
                #     #zzz
                #     #print(id_r)
                 
                # ### Only keep unique ids
                # keep_ids = np.unique(keep_ids)
                    
                
                # keep_df = df_boxes.iloc[keep_ids]
                
                
                # all_box_coords = keep_df['coords'].values
                
                
                
                
                













                ### Clean up boxes so it's just one large list
                # all_box_coords = []
                # for patch_box_coords in df_boxes['coords']:
                #     #print(patch_box_coords)
                #     if len(patch_box_coords) > 0:  ### FOR SOME REASON SOME IS NONETYPE???
                #         #for coords in patch_box_coords:
                #         all_box_coords.append(patch_box_coords)
                    
                                     
                # #zzz
                

                # size_thresh = 0    ### 150 is too much!!!
                # ### Go through all coords and assemble into whole image with splitting and step-wise
                
                # stepwise_im = np.zeros(np.shape(segmentation))
                # cell_num = 0
                
                # cell_ids = np.arange(len(all_box_coords))
                # random.shuffle(cell_ids)
                # for coords in all_box_coords:
                #     if len(coords) < size_thresh:
                #         continue                    
                #     stepwise_im[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_ids[cell_num]
                    
                #     cell_num += 1

                    
                # stepwise_im = np.asarray(stepwise_im, np.int32)
                # tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_labels_overlap5.tif', stepwise_im)
                
                
                
                # #zzz

                #input_im = np.asarray(input_im, np.uint8)

                #zzz

###########################################################################################        
        

                """ Post-processing 
                
                        - find doublets (identified twice) and resolve them
                        
                        - then clean super small objects
                        - and clean 
                
                """
            
                
                # def get_boxes_from_doublets(all_box_coords):
                #     doublets = np.zeros(np.shape(segmentation))
                
                #     doublets = np.moveaxis(doublets, 0, -1)                    
                #     sizes = []
                #     for coords in all_box_coords:
                #                 sizes.append(len(coords))
                #                 if len(coords) < size_thresh:
                #                     continue
                #                 doublets[coords[:, 0], coords[:, 1], coords[:, 2]] = doublets[coords[:, 0], coords[:, 1], coords[:, 2]] + 1
                                
                #     doublets[doublets <= 1] = 0
                #     doublets[doublets > 0] = 1                    
                    
                    
                #     lab = label(doublets)
                #     cc = regionprops(lab)
    
                #     print('num doublets: ' + str(len(cc)))
        
                #     cleaned = np.zeros(np.shape(doublets))
                #     arr_doublets = [ [] for _ in range(len(cc) + 1)]  ### start at 1 because no zero value objects in array!!!
                #     box_ids = [ [] for _ in range(len(cc) + 1)]
                #     for b_id, coords in enumerate(all_box_coords):  
                #                 region = lab[coords[:, 0], coords[:, 1], coords[:, 2]]
                #                 if len(np.where(region)[0]) > 0:
                #                     ids = np.unique(region)
                #                     ids = ids[ids != 0]   ### remove zero
                #                     for id_o in ids:
                #                         arr_doublets[id_o].append(coords)
                #                         box_ids[id_o].append(b_id)
                                
                #     return doublets, arr_doublets, box_ids, cc
                                
                                
                            
                # doublets, arr_doublets, box_ids, cc = get_boxes_from_doublets(all_box_coords)
                            
                # sav_doubs = np.moveaxis(doublets, -1, 0)
                # sav_doubs = np.asarray(sav_doubs, dtype=np.int32)
                # tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_doublets.tif', sav_doubs)

                

                    
                # """First pass through just identifies all overlaps with 2 boxes only 
                #         ***and ADDS their segmentations together so less weird subtraction artifacts
                
                # """
                # iou_thresh = 0.2
                # clean_box_coords = np.copy(all_box_coords)
                # num_doubs = 0
                # for case_id, case in enumerate(arr_doublets):
                    
                #     if len(case) == 0:
                #         continue
                #     box_nums = np.asarray(box_ids[case_id])

                #     ### Find identical rows that match to overlap region
                #     overlap = cc[case_id - 1]['coords']    ### case_id minus 1 when indexing cc because no region of value zero from above
                    
                #     ### Next calculate iou for each individual case
                #     iou_per_case = []
                #     for reg in case:
                #         intersect = (reg[:, None] == overlap).all(-1).any(-1)
                #         intersect = reg[intersect]
                #         intersect = len(intersect)
                        
                #         union = len(np.unique(np.vstack([overlap, reg]), axis=0))
                        
                #         iou_per_case.append(intersect/union)
                #     iou_per_case = np.asarray(iou_per_case)

                    
                #     box_nums = np.asarray(box_ids[case_id])                
                #     ### The majority of cases only have 2 things overlapping AND have high overlap
                #     if len(box_nums) == 2 and len(np.where(iou_per_case > iou_thresh)[0]) == len(box_nums):  
                #         ### In the case of doublets with HIGH overlap, just pick the one with lowest iou_thresh, and discard the other
                #         exclude_box = np.argmin(iou_per_case)
                        
                #         ### Case 2: anything ABOVE iou thresh is fully deleted
                #         to_del = np.where(iou_per_case > iou_thresh)[0]    
                #         to_del = to_del[to_del != exclude_box]  ### make sure not to delete current box
                        
                #         del_boxes = box_nums[to_del]
                        
                #         clean_box_coords[del_boxes] = [[]]
                        
                        
                #         ### also ADD coordinates from BOTH boxes to the clean_box_coords at the exclude_box ID
                #         clean_box_coords[box_nums[exclude_box]] = np.unique(np.vstack([case[0], case[1]]), axis=0)
                        
                #         num_doubs += 1
                
                        
                        
                # clean_box_coords = [x for x in clean_box_coords if x != []]
                        
                # first_pass_doublets, arr_doublets, box_ids, cc = get_boxes_from_doublets(clean_box_coords)                             
                
                
                # sav_doubs = np.moveaxis(first_pass_doublets, -1, 0)
                # sav_doubs = np.asarray(sav_doubs, dtype=np.int32)
                # tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_doublets_FP.tif', sav_doubs)

                                
                
                
                # ### loop through all doublets and determine iou
                # for case_id, case in enumerate(arr_doublets):
                    
                #     if len(case) == 0:
                #         continue
                    
                #     box_nums = np.asarray(box_ids[case_id])

                #     ### Find identical rows that match to overlap region
                #     overlap = cc[case_id - 1]['coords']    ### case_id minus 1 when indexing cc because no region of value zero from above
                    
                #     # matched = case[0]  ### start with first case
                #     # for reg in case:
                #     #     matched = (reg[:, None] == matched).all(-1).any(-1)
                #     #     matched = reg[matched]
                        
                    
                #     ### HACK -- sometimes doesn't match across ALL, so need to just do ANYTHING that matches across any of them for now..
                #     # vals, counts = np.unique(np.vstack(case), axis=0, return_counts=True)
                #     # matched = vals[counts >= 2]

                #     ### Next calculate iou for each individual case
                #     iou_per_case = []
                #     for reg in case:
                #         intersect = (reg[:, None] == overlap).all(-1).any(-1)
                #         intersect = reg[intersect]
                #         intersect = len(intersect)
                        
                #         union = len(np.unique(np.vstack([overlap, reg]), axis=0))
                        
                #         iou_per_case.append(intersect/union)
                #     iou_per_case = np.asarray(iou_per_case)

      
                #     """ 3 possible conclusions: 
                #                 1) Highest iou gets to keep overlap area (consensus)
                #                 2) All other iou above iou threshold (0.7?) is deleted fully
                #                 3) All other iou BELOW threshold is kept, but the overlapping coordinates are deleted from the cell


                #         """     
                #     ### Case 1: LOWEST iou is kept as overlap area #do nothing
                #     exclude_box = np.argmax(iou_per_case)
                    
                #     ### Case 2: anything ABOVE iou thresh is fully deleted
                #     # to_del = np.where(iou_per_case > iou_thresh)[0]
                #     # to_del = to_del[to_del != exclude_box]  ### make sure not to delete current box
                    
                #     # del_boxes = box_nums[to_del]
                #     # clean_box_coords[del_boxes] = [[]]
                    
                #     # other_boxes = np.delete(box_nums, np.concatenate((to_del, [exclude_box])))
                    
                #     ### Case 3: anything else that is not fully overlapped (low iou) only has these coords deleted
                #     other_boxes = np.delete(box_nums, exclude_box)
                #     for obox_num in other_boxes:
                #         obox = clean_box_coords[obox_num]
                        
                #         ### HACK if box is empty already
                #         if len(obox) == 0: 
                #             continue
                #         #all_c = np.concatenate((obox, matched))
                        
                #         not_matched = obox[~(obox[:, None] == overlap).all(-1).any(-1)] ### the only coords that do NOT overlap
                        
                #         #vals, counts = np.unique(all_c, axis=0, return_counts=True)
                        
                #         #unique = vals[counts < 2]   ### the only coords that do NOT overlap
                        
                #         ### Can we also add a way to find the connectivity of the objects at the end here???
                #         ### Like, if there's almost no discrete objects left, just delete the whole darn thing?
                #         clean_box_coords[obox_num] = not_matched
                                            
                #     ### To prevent holes, add the area of the overlap to the box that was excluded from subtraction
                #     clean_box_coords[box_nums[exclude_box]] = np.unique(np.vstack([overlap, clean_box_coords[box_nums[exclude_box]]]), axis=0)
                    
                    
                # clean_box_coords = [x for x in clean_box_coords if x != []]
                        
                
                # stepwise_clean = np.zeros(np.shape(stepwise_im))
                # cell_num = 0
                # for coords in clean_box_coords:
                #     if len(coords) < size_thresh:
                #         continue                    
                #     stepwise_clean[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_ids[cell_num]
                    
                #     cell_num += 1                
                
                
                # stepwise_clean = np.asarray(stepwise_clean, np.int32)
                # tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_cleaned_NEW.tif', stepwise_clean)

        

        
        
