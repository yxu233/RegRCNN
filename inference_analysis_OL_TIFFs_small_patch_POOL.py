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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def post_process_boxes(kwargs):                   

    box_vert = kwargs['box_vert']
    results_dict = kwargs['results_dict']
    patch_im = kwargs['patch_im']    
    total_blocks = kwargs['total_blocks']
    xyz = kwargs['xyz']
    focal_cube = kwargs['focal_cube']
    
    box_coords_all = []
    patch_depth = patch_im.shape[0]
    patch_size = patch_im.shape[1]

    #print(total_blocks)
       
    box_vert = np.vstack(box_vert)
    
    label_arr = np.copy(results_dict['seg_preds'],)
    

    ### remove all boxes with centroids on edge of image - but first need to process ALL boxes, including with box split? To be sure...
    
    """ Now try splitting boxes again! """
    ### about 35 minutes - slowest part by far!!!


    """ Filter boxes by EDGE of focal cube BEFORE running split box analysis """
    
    ### find centroids of boxes
    centroids = [(box_vert[:, 2] - box_vert[:, 0])/2 + box_vert[:, 0], (box_vert[:, 3] - box_vert[:, 1])/2 + box_vert[:, 1], (box_vert[:, 5] - box_vert[:, 4])/2 + box_vert[:, 4]]
    
    centroids = np.transpose(centroids)
    centroids = np.round(centroids).astype(int)
    #centroids = centroids - 1 ### cuz indices dont start from 0 from polygon function?
    
    
    centroids[np.where(centroids[:, 2] >= patch_depth)[0], 2] = patch_depth - 1
    centroids[np.where(centroids[:, 1] >= patch_size)[0], 1] = patch_size - 1
    centroids[np.where(centroids[:, 0] >= patch_size)[0], 0] = patch_size - 1
    
  
    ### Then filter using focal cube
    keep_ids = np.where(focal_cube[centroids[:, 2], centroids[:, 0], centroids[:, 1]] == 0)[0]
    print('num deleted: ' + str(len(centroids) - len(keep_ids)))

    box_vert = box_vert[keep_ids]



    
    if len(box_vert) == 0:
        return []


    df_cleaned = split_boxes_by_Voronoi(box_vert, vol_shape=patch_im.shape)

     
     ### REMOVE ROWS THAT HAD NO MATCHING BOX AT THE END (usually due to too much overlap)
     #df_cleaned = df_cleaned[df_cleaned['bbox_coords'].map(len) > 0]
     
     
    ### merge coords back into coherent boxes!
    merged_coords = [ [] for _ in range(len(box_vert)) ]
    for box_id in np.unique(df_cleaned['ids']):
         coords = df_cleaned.iloc[np.where(df_cleaned['ids'] == box_id)[0]]['bbox_coords']
         
         ### REMOVE ROWS THAT HAD NO MATCHING BOX AT THE END (usually due to too much overlap)
         coords = coords[coords.map(len) > 0]
         if len(coords) > 0:                                        
             coords = np.vstack(coords).astype(int)
             
         merged_coords[box_id] = coords                                            

    """ 
            ***ALSO SPLIT IN Z-dimension???     
    """
    box_vert = np.vstack(box_vert)
    box_vert_z = np.copy(box_vert)
    box_vert_z[:, 0] = box_vert[:, 4]
    box_vert_z[:, 2] = box_vert[:, 5]
    box_vert_z[:, 4] = box_vert[:, 0]
    box_vert_z[:, 5] = box_vert[:, 2]


    df_cleaned_z = split_boxes_by_Voronoi(box_vert_z, vol_shape=np.moveaxis(patch_im, 0, 1).shape)
                
    ### REMOVE ROWS THAT HAD NO MATCHING BOX AT THE END (usually due to too much overlap)
    #df_cleaned_z = df_cleaned_z[df_cleaned_z['bbox_coords'].map(len) > 0]
    
                                        
    
    ### merge coords back into coherent boxes!
    merged_coords_z = [ [] for _ in range(len(box_vert_z)) ]
    for box_id in np.unique(df_cleaned_z['ids']):
    
        coords = df_cleaned_z.iloc[np.where(df_cleaned_z['ids'] == box_id)[0]]['bbox_coords']   
        
        ### REMOVE ROWS THAT HAD NO MATCHING BOX AT THE END (usually due to too much overlap)
        coords = coords[coords.map(len) > 0]
        if len(coords) > 0:
            coords = np.vstack(coords).astype(int)
            
            #print()
            
            ### swap axis
            coords[:, [2, 1]] = coords[:, [1, 2]]   ### used to be this for split_boxes no voronoi
            #coords[:, [2, 0]] = coords[:, [0, 2]]
            
        merged_coords_z[box_id] = coords      
        

    ### COMBINE all the splits in XY and Z
    if len(merged_coords) != len(merged_coords_z):
        print('NOT MATCHED LENGTH')
        #zzz
        
        
    match_xyz = []
    for id_m in range(len(merged_coords)):
        
        a = merged_coords[id_m]
        b = merged_coords_z[id_m]
        
         ### REMOVE ROWS THAT HAD NO MATCHING BOX AT THE END (usually due to too much overlap)
        if len(a) == 0 and len(b) == 0:
            continue
        elif len(a) == 0:
            match_xyz.append(b)
        elif len(b) == 0:
            match_xyz.append(a)
        else:
            unq, count = np.unique(np.concatenate((a, b)), axis=0, return_counts=True)                                        
            matched = unq[count > 1]
            
            match_xyz.append(matched)
                                        

    merged_coords = match_xyz
    

                                              
    
    ### Then APPLY these boxes to mask out the objects in the main segmentation!!!
    new_labels = np.zeros(np.shape(label_arr))
    for box_id, box_coords in enumerate(merged_coords):
    
        bc = box_coords
        """ HACK: --- DOUBLE CHECK ON THIS SUBTRACTION HERE """
        
        #bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
        bc[np.where(bc[:, 2] >= patch_depth)[0], 2] = patch_depth - 1
        bc[np.where(bc[:, 1] >= patch_size)[0], 1] = patch_size - 1
        bc[np.where(bc[:, 0] >= patch_size)[0], 0] = patch_size - 1
        
        
        bc = np.asarray(bc, dtype=int)                    
        new_labels[0, 0, bc[:, 1], bc[:, 0], bc[:, 2]] = box_id + 1   
        

    new_labels[label_arr == 0] = 0   ### This is way simpler and faster than old method of looping through each detection


    """ Then find which ones are on the edge and remove, then save coords of cells for the very end """
    
    if len(np.unique(new_labels)) == 1:
        print('WEIRD')
        return []
    
    print(total_blocks)
    
    new_labels = np.asarray(new_labels, dtype=int)
    cc = regionprops_table(new_labels[0][0], properties=('centroid', 'coords'))
    
    df = pd.DataFrame(cc)
    # ### Then filter using focal cube
    # edge_ids = np.where(focal_cube[np.asarray(df['centroid-2']), np.asarray(df['centroid-0']), np.asarray(df['centroid-1'])])[0]
    # print('num deleted: ' + str(len(edge_ids)))
    # df = df.drop(edge_ids)
    




    #toc = time.perf_counter()
    
    #print(f"Inference in {toc - tic:0.4f} seconds")
    

    
    if len(df) > 0:
        ### save the coordinates so we can plot the cells later
        #empty = np.zeros(np.shape(patch_im))
        for row_id, row in df.iterrows():
            coords = row['coords']
            
            coords[:, 0] = coords[:, 0] + xyz[0]
            coords[:, 1] = coords[:, 1] + xyz[1]
            coords[:, 2] = coords[:, 2] + xyz[2]
               
            box_coords_all.append(coords)



    return box_coords_all





if __name__=="__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/73) dil_group_norm_det_thresh_0_2/'      

            #self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/76) dil_group_norm_NEW_DATA_edges_wbc/'                  
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
        

    list_folder = ['/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Test_RegRCNN/']



    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_output_PYTORCH_76)'
    
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
                all_xyz = [] 
                
                
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

                ### Clean up boxes so it's just one large list
                all_box_coords = []
                for patch_box_coords in box_coords_all:
                    #print(patch_box_coords)
                    if patch_box_coords:  ### FOR SOME REASON SOME IS NONETYPE???
                        for coords in patch_box_coords:
                            all_box_coords.append(coords)
                    
                                     
                
                filename = input_name.split('/')[-1].split('.')[0:-1]
                filename = '.'.join(filename)

                size_thresh = 80    ### 150 is too much!!!
                ### Go through all coords and assemble into whole image with splitting and step-wise
                
                stepwise_im = np.zeros(np.shape(segmentation))
                cell_num = 0
                
                cell_ids = np.arange(len(all_box_coords))
                random.shuffle(cell_ids)
                for coords in all_box_coords:
                    if len(coords) < size_thresh:
                        continue                    
                    stepwise_im[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_ids[cell_num]
                    
                    cell_num += 1

                    
                stepwise_im = np.asarray(stepwise_im, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_stepwise_im.tif', stepwise_im)
                    

                #input_im = np.asarray(input_im, np.uint8)
                input_im = np.expand_dims(input_im, axis=0)
                input_im = np.expand_dims(input_im, axis=2)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                      imagej=True, #resolution=(1/XY_res, 1/XY_res),
                                      metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                
                
                
                
                        
                from skimage import morphology
                ball = morphology.ball(1)
                dil_box = morphology.dilation(stepwise_im, selem=ball)
                
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_stepwise_im_DIL.tif', dil_box)
                                
                
                
                
                lab_boxes = morphology.erosion(dil_box, selem=ball)
        
                
###########################################################################################        
        

                """ Post-processing 
                
                        - find doublets (identified twice) and resolve them
                        
                        - then clean super small objects
                        - and clean 
                
                """
                
                zzz

                doublets = np.zeros(np.shape(segmentation))
                
                sizes = []
                
                #shuffled = np.copy(box_coords_all)
                #random.shuffle(shuffled)    ### for plotting so it's more visible
                for coords in all_box_coords:
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
                box_ids = [ [] for _ in range(len(cc) + 1)]
                
                for b_id, coords in enumerate(all_box_coords):  
                            region = lab[coords[:, 2], coords[:, 0], coords[:, 1]]
                            
                            if len(np.where(region)[0]) > 0:
                                
                                ids = np.unique(region)
                                ids = ids[ids != 0]   ### remove zero
                                #if len(ids) > 1:
                                    #zzz
                                    #print(ids)
                                    
                                for id_o in ids:
                                    arr_doublets[id_o].append(coords)
                                    box_ids[id_o].append(b_id)
                                
                            
                    
                ### loop through all doublets and determine iou
                iou_thresh = 0.7
                
                clean_box_coords = np.copy(all_box_coords)
                for case_id, case in enumerate(arr_doublets):
                    
                    if len(case) == 0:
                        continue
                    
                    box_nums = np.asarray(box_ids[case_id])
                    
                    ### Find identical rows across ALL overlapped regions
                    # matched = case[0]  ### start with first case
                    # for reg in case:
                    #     matched = (reg[:, None] == matched).all(-1).any(-1)
                    #     matched = reg[matched]
                        
                    
                    ### HACK -- sometimes doesn't match across ALL, so need to just do ANYTHING that matches across any of them for now..
                    vals, counts = np.unique(np.vstack(case), axis=0, return_counts=True)
                    matched = vals[counts >= 2]
                    
                    
                        
                        
                    ### Next calculate iou for each individual case
                    iou_per_case = []
                    for reg in case:
                        intersect = (reg[:, None] == matched).all(-1).any(-1)
                        intersect = reg[intersect]
                        intersect = len(intersect)
                        
                        union = len(np.unique(np.vstack([matched, reg]), axis=0))
                        
                        iou_per_case.append(intersect/union)
                    iou_per_case = np.asarray(iou_per_case)
                    print(iou_per_case)
                    
                    """ 3 possible conclusions: 
                                1) Highest iou gets to keep overlap area (consensus)
                                2) All other iou above iou threshold (0.7?) is deleted fully
                                3) All other iou BELOW threshold is kept, but the overlapping coordinates are deleted from the cell

                        """
                        
                    ### Case 1: highest iou is kept as overlap area #do nothing
                    exclude_box = np.argmax(iou_per_case)
                    
                    ### Case 2: anything ABOVE iou thresh is fully deleted
                    to_del = np.where(iou_per_case > iou_thresh)[0]
                        
                    del_boxes = box_nums[to_del]
                    clean_box_coords[del_boxes] = [[]]
                    
                    
                    ### Case 3: anything else that is not fully overlapped (low iou) only has these coords deleted
                    # other_boxes = np.delete(box_nums, np.concatenate((to_del, [exclude_box])))
                    # for obox_num in other_boxes:
                    #     obox = clean_box_coords[obox_num]
                        
                    #     ### HACK if box is empty already
                    #     if len(obox) == 0: 
                    #         continue
                        
                    #     #all_c = np.concatenate((obox, matched))
                        
                    #     not_matched = obox[~(obox[:, None] == matched).all(-1).any(-1)] ### the only coords that do NOT overlap
                        
                    #     #vals, counts = np.unique(all_c, axis=0, return_counts=True)
                        
                    #     #unique = vals[counts < 2]   ### the only coords that do NOT overlap
                        
                    #     ### Can we also add a way to find the connectivity of the objects at the end here???
                    #     ### Like, if there's almost no discrete objects left, just delete the whole darn thing?
                    #     clean_box_coords[obox_num] = not_matched
                                            
                    
                    ### WHAT IF THERE ARE REGIONS OVERLAPPED BY only a couple of the multiple cells??? 
                    
                    
                    
                    
                    
                    
                    """
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
                    cleaned[unique_coords[:, 2], unique_coords[:, 0], unique_coords[:, 1]] = cell_ids[cell_num]
                    cell_num += 1
                    """
                
                # cleaned = np.asarray(cleaned, np.int32)
 
                # stepwise_clean = np.copy(stepwise_im)
                # stepwise_clean[cleaned > 0] = 0
                # #cleaned[cleaned > 0] = cleaned[cleaned > 0] + np.max(stepwise_clean)  ### add to make indices unique
                
                # stepwise_clean[cleaned > 0] = cleaned[cleaned > 0]
                
                stepwise_clean = np.zeros(np.shape(stepwise_im))
                cell_num = 0
                for coords in clean_box_coords:
                    if len(coords) < size_thresh:
                        continue                    
                    stepwise_clean[coords[:, 2], coords[:, 0], coords[:, 1]] = cell_ids[cell_num]
                    
                    cell_num += 1                
                
                
                #stepwise_clean = np.asarray(stepwise_clean, np.int32)
                tiff.imwrite(sav_dir + filename + '_' + str(int(i)) +'_cleaned_NEW.tif', stepwise_clean)

        

        
        
