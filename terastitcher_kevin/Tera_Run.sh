#!/bin/bash

#######i/o folders###########################################################################################################
im_location='C:/Users/Bergles Lab/Documents/Kevin Yu/7d_PI_5x_TILE_um_3d' #wherever your two-tiered parent folder is
im_out='C:/Users/Bergles Lab/Documents/Kevin Yu/test_stitched' #wherever you would like your final image saved
#############################################################################################################################


######change if you compiled terastitcher from src instead of binaries (uncomment + change terastitcher_install)#############
#terastitcher_install='C:/Users/Bergles Lab/Source/Repos/TeraStitcher/src/out/install/x64-Debug/bin/' #where your TeraStitcher is installed
#cd "${terastitcher_install}" #change to your install path for terastitcher
#alternatively you can add terastitcher to your PATH and it should be callable from everywhere 
#(you can then skip this step and delete the ./ from the start of each terastitcher step
#############################################################################################################################


#voxel sizes (can be determined from zeiss metadata, rounded to the nearest 100th of a micron)
voxel_x=$(cat "${im_location}/voxeldim_x.txt")
voxel_y=$(cat "${im_location}/voxeldim_y.txt")
voxel_z=$(cat "${im_location}/voxeldim_z.txt")

#terastitcher pipeline
terastitcher --import --volin="${im_location}" --ref1=x --ref2=y --ref3=z --vxl1=${voxel_x} --vxl2=${voxel_y} --vxl3=${voxel_z} --volin_plugin="TiledXY|3Dseries" --imout_depth=16
terastitcher --displcompute --projin="${im_location}/xml_import.xml"
terastitcher --displproj --projin="${im_location}/xml_displcomp.xml"
terastitcher --displthres --projin="${im_location}/xml_displproj.xml" --threshold=0.7
terastitcher --placetiles --projin="${im_location}/xml_displthres.xml"
terastitcher --merge --projin="${im_location}/xml_merging.xml" --volout="${im_out}" --resolutions=0 --volout_plugin="TiledXY|3Dseries" --imout_format="tif" --imout_depth=16