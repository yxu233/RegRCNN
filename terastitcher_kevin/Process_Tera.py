from ij import IJ, io
from java.io import File
from loci.formats import ImageReader,MetadataTools
import os
import math

def makeSix(num):
	temp = str(num)
	while(len(temp) < 6):
		temp = "0"+temp
	while(len(temp) > 6):
		temp = temp[1:]
	sixDigitString=temp
	return sixDigitString

## get file
fname = None
while fname is None:
	dc = io.DirectoryChooser("Select source folder")
	fold = dc.getDirectory()
	od = io.OpenDialog("Choose your .czi", None)
	fname = od.getFileName()

dc = io.DirectoryChooser("Choose Destination folder")
dest = dc.getDirectory()
## setup for metadata
f = File(fold + fname)
imageReader = ImageReader()
meta = MetadataTools.createOMEXMLMetadata()
imageReader.setMetadataStore(meta)
imageReader.setId(f.getAbsolutePath())
globalMetaData = imageReader.getGlobalMetadata()

## get the # of series in the tile
nSeries = int(globalMetaData.get("Information|Image|SizeM #1"))

## get physical pixel sizes (should by consistent over entire tile?)
pixelX = float(meta.getPixelsPhysicalSizeX(0).toString().split('[')[1].split(']')[0])
pixelY = float(meta.getPixelsPhysicalSizeY(0).toString().split('[')[1].split(']')[0])
pixelZ = float(meta.getPixelsPhysicalSizeZ(0).toString().split('[')[1].split(']')[0])	

## get # of pixels in each dimension
pX = float(meta.getPixelsSizeX(0).toString())
pY = float(meta.getPixelsSizeY(0).toString())
pZ = float(meta.getPixelsSizeZ(0).toString())

## get overlap for each tile
comment = globalMetaData.get("Information|Document|Comment #1")
overlapIndex = comment.find("Tiles overlap: ") + 15
overlap=float(comment[overlapIndex:(overlapIndex+4)])/100

## calculate frame size based on overlap
fsizeX = (1-overlap) * pX
fsizeY = (1-overlap) * pY

## bidirectionality
bidir = globalMetaData.get("Experiment|AcquisitionBlock|AcquisitionModeSetup|BiDirectional #1")

## illumination
illum = int(meta.getPixelsSizeC(0).toString())

## find the # of unique rows/columns
x=[]
y=[]
for i in xrange(nSeries):
	num = i+1
	if i != 0:
		x.append(float(meta.getPlanePositionX(i,0).toString().split('[')[1].split(']')[0]))
		y.append(float(meta.getPlanePositionY(i,0).toString().split('[')[1].split(']')[0]))	
uq_x = list(set(x))
uq_y = list(set(y))
init_x = min(uq_x)
init_y = min(uq_y)
off_x = init_x * math.copysign(1, init_x)
off_y = init_y * math.copysign(1, init_y)

xDim = len(uq_x)
yDim = len(uq_y)

## zero your start position
mX=0
mY=0
mZ=0

for i in xrange(nSeries):
	num = i+1
	IJ.run("Bio-Formats Importer", "open=["+fold+fname+"] color_mode=Default split_channels use_virtual_stack series_"+str(num))
	if illum > 1:
		IJ.run("Close")
	if i==0:
		Y=(init_y+off_y)*pixelY*10
		X=(init_x+off_x)*pixelX*10
		nY=makeSix(int(Y))
		nX=makeSix(int(X))
		os.makedirs(dest+"/"+nX)
		os.makedirs(dest+"/"+nX+"/"+nX+"_"+nY)
		sDir = dest+"/"+nX+"/"+nX+"_"+nY+"/"
		temp=IJ.getImage()
		IJ.saveAs(temp, "Tiff", sDir+str(0))
		IJ.run("Close")
	else:
		newX = (float(meta.getPlanePositionX(i,0).toString().split('[')[1].split(']')[0]) + off_x)*pixelX*10
		newY = (float(meta.getPlanePositionY(i,0).toString().split('[')[1].split(']')[0]) + off_y)*pixelY*10
		nY=makeSix(int(newY))
		nX=makeSix(int(newX))				
		if(not os.path.exists(dest+"/"+nX)):
			os.makedirs(dest+"/"+nX)
		os.makedirs(dest+"/"+nX+"/"+nX+"_"+nY)
		sDir = dest+"/"+nX+"/"+nX+"_"+nY+"/"
		temp=IJ.getImage()
		IJ.saveAs(temp, "Tiff", sDir+str(0))
		IJ.run("Close")

with open(str(dest)+'/voxeldim_x.txt', 'w') as f:
	f.write(str(round(pixelX,2)))
with open(str(dest)+'/voxeldim_y.txt', 'w') as f:
	f.write(str(round(pixelY,2)))
with open(str(dest)+'/voxeldim_z.txt', 'w') as f:
	f.write(str(pixelZ))


