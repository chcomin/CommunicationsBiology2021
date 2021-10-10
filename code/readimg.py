"""Read 3D images on different formats"""

import oiffile as oif
import tifffile_old as tifffile
import numpy as np
import Czifile

def ReadImg(file, channel=None):

	fileType = file.split('.')[-1]

	if ((fileType == 'tif') | (fileType == 'tiff')):
	
		data = tifffile.TIFFfile(file)
		img = data.asarray().astype(np.float)
		img = 255*img/np.max(img)		
		
		scale_z,scale_x,scale_y = Find_scale_tiff(data)
		
	elif (fileType == 'oib'):	

		data = oif.OifFile(file)
		img = data.asarray()[0].astype(np.float)
		img = 255*img/np.max(img)

		imgInfo = data.mainfile['Reference Image Parameter']
		scale_z = 1.
		scale_x = imgInfo['HeightConvertValue']
		scale_y = imgInfo['WidthConvertValue']	
		
	elif (fileType == 'lsm'):	
		data = tifffile.TIFFfile(file)
		img = data.asarray(series=0)[0,0].astype(np.float)
		img = 255*img/np.max(img)
		
		aux = data.pages[0].cz_lsm_scan_info['line_spacing']
		scale_x = scale_y = float(aux)
		scale_z = data.pages[0].cz_lsm_scan_info['plane_spacing']
	
	elif (fileType == 'czi'):
		data = Czifile.CziFile(file)
		img = data.asarray()[0,0,...,0].astype(np.float)
		if img.shape[0]>1:
			if channel==None:
				pcaChannels = np.zeros(3)
				for i in range(3):
					pcaChannels[i] = 0.50336421*np.mean(img[i]) + 0.86407434*np.std(img[i])
				ind = np.argmin(pcaChannels)
			else:
				ind = channel
			img = img[ind]
			print("Multichannel image, using channel %d"%ind)
		else:
			img = img[0]
		img = 255.*img/float(np.max(img))
		
		escala = [-1,-1,-1]
		metadata = data.metadata
		scaling = metadata.find('.//Scaling')
		for i,axis in enumerate(["Z", "X", "Y"]):
			axisTag = scaling.find('.//Distance[@Id="%s"]'%axis)
			try:
				scaleValue = float(axisTag.find('Value').text)
			except ValueError:
				raise ValueError('Pixel size not recognized')
				
			if axisTag.find('DefaultUnitFormat').text != u'\xb5m':
				print('Warning, pixel size unit is not microns')
			else:
				escala[i] = scaleValue
		
		scale_z, scale_x, scale_y = [1e6*item for item in escala]
	
	escala = np.array([scale_z,scale_x,scale_y])	
		
	return img, escala
	
def Find_scale_tiff(data):

	scale_z = 1.
	numChar = 21
	imagej_tags = data.pages[0].imagej_tags
	if 'spacing' in imagej_tags:
		scale_z = imagej_tags['spacing']
	if ('info' in imagej_tags):
		imgInfo=data.pages[0].imagej_tags['info']
		k1 = imgInfo.find('HeightConvertValue')
		if k1!=-1:
			aux = imgInfo[k1+numChar:k1+numChar+10]
			k2 = aux.find('\n')
			scale_x = float(aux[:k2])
			k1 = imgInfo.find('WidthConvertValue')
			aux = imgInfo[k1+numChar-1:k1+numChar+10-1]
			k2 = aux.find('\n')
			scale_y = float(aux[:k2])	
		else:
			scale_x, scale_y = -1, -1
		
	else:
		p = data.pages[0]	
		v = p.tags['x_resolution'].value
		scale_x = v[1]/float(v[0])
		v = p.tags['y_resolution'].value
		scale_y = v[1]/float(v[0])	

	return scale_z,scale_x,scale_y