"""Utility routines for calculating blood vessel tortuosity"""

import numpy as np
import scipy

def fit3D(edgeSegment):

	ded = np.diff(edgeSegment,axis=0)
	ds = np.sqrt(np.sum(ded**2, axis=1))
	s = np.array([0] + (np.cumsum(ds)/sum(ds)).tolist())

	az, bz = scipy.polyfit(s, edgeSegment[:,0], 1)
	ax, bx = scipy.polyfit(s, edgeSegment[:,1], 1)
	ay, by = scipy.polyfit(s, edgeSegment[:,2], 1)

	z = az*s + bz
	x = ax*s + bx
	y = ay*s + by	
	
	return z, x, y

# Distance from point p to line (v2-v1)	
def pointLineDist(p, v1, v2):

	if p.ndim==1:
		p = np.expand_dims(p, 0)

	# Line versor 
	v12 = v2 - v1
	n12 = v12/(np.sqrt(np.sum(v12**2)))
		
	# Vector from v1 to p
	v1p = v1-p
	
	# Scalar product between v1p and n12 (i.e., size of v1p projected into n12)
	v1pProjn12 = np.sum(v1p*n12, axis=1)
	
	v1pProjn12 = v1pProjn12.reshape(p.shape[0],1)
	
	# Vector from point p to the closest point in the line
	vpx = v1p-v1pProjn12*n12	
	
	# Norm of vector vpx
	d = np.sqrt(np.sum(vpx**2, axis=1))
		
	return d
	
def findEdgeSegment(ed, pixelIndex, L0):

	pixel = ed[pixelIndex]

	dist = np.sqrt(np.sum((ed-pixel)**2, axis=1))
	distNorm = dist - L0
	indLeft = np.nonzero(distNorm[:pixelIndex+1]>=0)[0]
	indRight = np.nonzero(distNorm[pixelIndex:]>=0)[0]
	
	if len(indLeft)==0:
		indLeft = 0
		isLeftBorder = True
	else:
		indLeft = indLeft[-1]
		isLeftBorder = False
		
	if len(indRight)==0:
		indRight = distNorm.size
		isRightBorder = True
	else:
		indRight = pixelIndex + indRight[0]
		isRightBorder = False
		
	edgeSegment = ed[indLeft:indRight]	
	
	return edgeSegment, isLeftBorder, isRightBorder
	
def getEdgeTortuosity(ed, L0, debug=False):

	edgeTortuosity = np.zeros(ed.shape[0])
	leftBorderIndex = ed.shape[0]
	rightBorderIndex = 0
	foundLeftBorder = False
	foundRightBorder = False
	
	if debug:
		_debug_vars = []
	
	for pixelIndex, pixel in enumerate(ed):

		edgeSegment, isLeftBorder, isRightBorder = findEdgeSegment(ed, pixelIndex, L0)
		if (isLeftBorder==False) and (foundLeftBorder==False):
			leftBorderIndex = pixelIndex
			foundLeftBorder = True
		if (isRightBorder==True) and (foundRightBorder==False):
			rightBorderIndex = pixelIndex			
			foundRightBorder = True
			
		if edgeSegment.shape[0]<2:
			raise ValueError("edgeSegment must have at least two points for fitting")
			
		z, x, y = fit3D(edgeSegment)

		v1 = np.array((z[0], x[0], y[0]))
		v2 = np.array((z[-1], x[-1], y[-1]))

		d = pointLineDist(edgeSegment, v1, v2)

		tortuosityPixel = np.mean(d)
		edgeTortuosity[pixelIndex] = tortuosityPixel
		
		if (debug==True) and (pixelIndex%200==0):
			_debug_vars.append([edgeSegment, (z, x, y), ed[pixelIndex]])

	if debug==False:
		return edgeTortuosity, leftBorderIndex, rightBorderIndex
	else:
		return edgeTortuosity, leftBorderIndex, rightBorderIndex, _debug_vars		
		
		
		
		