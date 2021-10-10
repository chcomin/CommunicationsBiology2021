"""Calculate length and number of branching points of curvilinear structures."""

import numpy as np
import os
from igraph import *
import readimg
import cv2
import natsort
	
def get_measurements(files, inFolderNetwork, inFolderOriginal, inFolderBinary):

	vessel_length = []
	num_branch_point = []
	z_interp = np.zeros(len(files), dtype=int)

	for q in range(len(files)):	
	
		subFolder = ''.join(files[q][0:-1])
		originalFolder = inFolderOriginal+subFolder
		ROIFolder = inFolderOriginal+subFolder+'ROI/'
		binaryFolder = inFolderBinary+subFolder
		networkFolder = inFolderNetwork+subFolder+'core/numpy/'
		
	
		ind = files[q][-1].rfind('.')
		fileName = files[q][-1][0:ind]
		print("\nAnalyzing image \""+str(fileName)+"\" ("+str(q+1)+" of "+str(len(files))+")")
		
		img_np_or,scale_or = readimg.ReadImg(originalFolder+files[q][-1], 0)
		scale = [np.min(scale_or)]*3	
			
		file = binaryFolder+'numpy/'+fileName+'.npy'
		imgBin = np.load(file)
				
		size_z,size_x,size_y = imgBin.shape
		
		if img_np_or.shape[1:]==imgBin.shape[1:]:
			z_interp[q] = 1
		else:
			z_interp[q] = 0
		
		file = networkFolder+fileName+'.npy'
		g = np.load(file).item()		
		
		# Read ROI
		arq_name = ROIFolder+fileName+'.png'
		if (os.path.isfile(arq_name)):
			img_ROIs = cv2.imread(arq_name,0).astype(np.int)
		else:
			img_ROIs = 255*np.ones([size_x,size_y],dtype=int)
		
		vals = np.unique(img_ROIs)
		
		vals = vals[vals>0]
		
		vessel_length_ROI = []
		num_branch_point_ROI = []
		for indROI in range(len(vals)-1,-1,-1):
				
			img_ROI = (img_ROIs==vals[indROI])*1
				
			volume = np.sum(img_ROI)*size_z*scale[0]*scale[1]*scale[2]*1e-9
			tam = 0
			for i in range(g.ecount()):				
				ed = np.array(g.es['pos_ed'][i])
				
				is_inside_ROI = img_ROI[ed[:,1],ed[:,2]]
				is_inside_ROI = np.tile(is_inside_ROI[1:],(3,1)).T
				
							
				ded = np.diff(ed,axis=0)
				ded *= is_inside_ROI
				
				
				tam_aux = np.zeros(ded.shape[0])
				for j in range(ded.shape[0]):
					tam_aux[j] = np.sqrt(np.sum((ded[j]*scale)**2))

				tam += np.sum(tam_aux)
				
			tam *= 1e-3	
			vessel_length_ROI.append(tam/float(volume))
					
			grau = np.array(g.degree())
			ind = np.nonzero(grau>=3)[0]
			
			pos_no = np.array(g.vs['pos_no'])[ind]
			is_inside_ROI = img_ROI[pos_no[:,1],pos_no[:,2]]
			
			ind2 = np.nonzero(is_inside_ROI==1)[0]
			
			num_branch_point_ROI.append(ind2.size/volume)
			
		vessel_length.append(vessel_length_ROI)
		num_branch_point.append(num_branch_point_ROI)
		
	return 	vessel_length,num_branch_point, z_interp
	
batchName = 'Vascular Project_Picture CD31-PDGFR-vGLUT2-IBA1'

inFolderNetwork = '../data/Network/'+batchName+'/'
inFolderOriginal = '/media/Data/cesar/Baptiste data/Project neuroinflammation/'+batchName+'/'
inFolderBinary = '../data/Binary/'+batchName+'/'
outFolder = '../results/'

dataTree = os.walk(inFolderOriginal)
files = []
for Pfolder, subFolders, subFolderFiles in dataTree:
	if len(subFolderFiles)>0:
		ind = Pfolder.find(batchName)+len(batchName)+1
		if os.name=='nt':
			folders = Pfolder[ind:].split('\\') 
		else:
			folders = Pfolder[ind:].split('/') 
		
		if ((folders[-1]!='ROI') and (folders[-1]!='maximum_projection') and (folders[-1]!='equalized')):
			if (len(folders[0])>0):
				folders = [folder+'/' for folder in folders]
					
			for subFolderFile in subFolderFiles:
				if '.tif' not in subFolderFile:
					files.append(folders+[subFolderFile])
files = natsort.natsorted(files)	
				
vessel_length, num_branch_point, z_interp = get_measurements(files, inFolderNetwork, inFolderOriginal, inFolderBinary)

strOutData = ''
for i in range(len(files)):
	folders,file = files[i][0:-1],files[i][-1]
	ind = file.rfind('.')
	fileName = file[0:ind]	
	
	if len(vessel_length[i])==1:	# If only one ROI
		strOutData += ','.join(folders[0:-1]).replace('/','')+','+fileName+','
		strOutData += '%.2f,%.2f\n'%(vessel_length[i][0],num_branch_point[i][0])
	else:
		for j in range(len(vessel_length[i])):
			strOutData += ','.join(folders[0:-1]).replace('/','')+','+fileName+' region #%d,'%(j+1)
			strOutData += '%.2f,%.2f\n'%(vessel_length[i][j],num_branch_point[i][j])
		
fd = open(outFolder+'results_'+batchName+'.txt','w')		
fd.write(strOutData)
fd.close()		

	
	

	
	
	