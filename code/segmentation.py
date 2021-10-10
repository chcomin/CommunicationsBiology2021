"""Segmentation of blood vessels and pericytes. A much improved 
version of this code is available at https://github.com/chcomin/pyvesto"""

import numpy as np
import scipy.ndimage.measurements as me
import scipy.ndimage as nd
import scipy.ndimage.filters as fil
import scipy
import matplotlib.pyplot as plt
import os
from igraph import *
import readimg
from skimage.filters import threshold_adaptive
import natsort
import ctypes as ct

skel_module = ct.CDLL('./libskeleton.so') 

def readThresholds(fileName):
	fd = open(fileName, 'rb')
	data = fd.readlines()
	fd.close()
	
	tDict = {}
	for lineIndex, line in enumerate(data):
		if line[0]=='#':
			expName = line[1:].strip()
			if expName in tDict:
				print('Error reading thresholds, experiment already in dictionary')
			else:
				tDict[expName] = {}
		else:
			splittedLine = line.strip().split('\t')
			index, stackName, bestC = splittedLine
			tDict[expName][stackName] = [int(index), float(bestC)]
			
	return tDict

def Binariza(img_np,scale, C):

	size_z,size_x,size_y = img_np.shape

	if img_np.dtype!='float64':
		img_np = img_np.astype(np.float64)

	sigma = np.array([0.1,1.,1.])
	img_np_diffused = fil.gaussian_filter(img_np,sigma=sigma/scale) 


	radius = 40
	comp_size = 500 
	img_corr = np.zeros([size_z,size_x,size_y])
	img_T = np.zeros([size_z,size_x,size_y],dtype=np.bool)
	for i in range(size_z):
		img_blurred = fil.gaussian_filter(img_np_diffused[i],sigma=radius/2.)
		img_corr[i] = img_np_diffused[i] - img_blurred
		img_T[i] = img_corr[i] > C
			
	
	img_tmp = Elimina_v2(img_T, comp_size)

	img_lab,num_comp = me.label(1-img_tmp)
	tam_comp = me.sum(1-img_tmp, labels=img_lab, index=range(1,num_comp+1))
	ind = np.argmax(tam_comp)+1
	img_tmp2 = img_lab!=ind
	
	return img_tmp2
	
def Esqueleto(img):

	img = np.ascontiguousarray(img,dtype=np.uint16)

	size_z, size_x, size_y = img.shape	
	size_z, size_x, size_y = int(size_z), int(size_x), int(size_y)

	img_res = np.zeros([size_z,size_x, size_y],dtype=np.uint16)	

	skel_module.Esqueleto(img.ctypes.data_as(ct.POINTER(ct.c_ushort)), 
					  img_res.ctypes.data_as(ct.POINTER(ct.c_ushort)), 
					  size_z, size_x, size_y)	
	return img_res

def Cria_rede(img):


	size_z,size_x,size_y = img.shape

	size_z += 2
	size_x += 2
	size_y += 2

	img_pad = np.zeros([size_z,size_x,size_y])		
	img_pad[1:-1,1:-1,1:-1] = img
	img = img_pad

	s = np.ones([3,3,3])


	img_lab,num_comp = me.label(img,s)
	tam_comp = me.sum(img, labels=img_lab, index=range(1,num_comp+1))
	img = Elimina(img,img_lab,tam_comp,2)								

	res = fil.correlate(img, s, output=None, mode='constant', cval=0., origin=0)
	res = res*img - 1
	res_aux = res.copy()

	img_label,num_comp = me.label((res>=3) | (res==1),s)
	img_label -= 1


	ind_at = 0		

	g = Graph()		
	node_att = []

	for grand_z in range(size_z):
		print(grand_z)
		for grand_x in range(size_x):
			for grand_y in range(size_y):

				if (res[grand_z,grand_x,grand_y]==1):
					
					res[grand_z,grand_x,grand_y] = -1
					viz = vizinhos(res,[grand_z,grand_x,grand_y])[0]
					if (res[viz[0],viz[1],viz[2]]==2):								
						g.add_vertex()
						node_att.append([[[grand_z,grand_x,grand_y]],[viz]])
					elif (res[viz[0],viz[1],viz[2]]>=3):	
						node = Flood_fill(res,viz[0],viz[1],viz[2])
						g.add_vertex()
						node_att.append(node)
			
				if (res[grand_z,grand_x,grand_y]>=3):
				
					node = Flood_fill(res,grand_z,grand_x,grand_y)

					g.add_vertex()
					node_att.append(node)

		
	ed_att = []
	print(g.vcount())
	for node in range(g.vcount()):
		print(node)
		
		tips = node_att[node][1]
		for j in range(len(tips)):
			prox_point = tips[j]
			
			if (res[prox_point[0],prox_point[1],prox_point[2]]!=-1):		# Verify if edge is already in the network
			
				ed_points = []
				prox_point = [prox_point]
				while (len(prox_point)>0):
				
					prox_point = prox_point[0]
					ed_points.append(prox_point)
					res[prox_point[0],prox_point[1],prox_point[2]] = -1
					prox_point = vizinhos(res,prox_point)
				
				viz = vizinhos(img_label,ed_points[-1])
				
				if (len(viz)==1):
					viz_ind = img_label[viz[0][0],viz[0][1],viz[0][2]]
					g.add_edge(int(node),int(viz_ind))
					ed_att.append(ed_points)
				else:									# Treat the case where a segment has size equal to one pixel
					viz_ind = img_label[viz[0][0],viz[0][1],viz[0][2]]
					if (viz_ind!=node):	
						g.add_edge(int(node),int(viz_ind))
						ed_att.append(ed_points)
					else:
						viz_ind = img_label[viz[1][0],viz[1][1],viz[1][2]]
						if (viz_ind!=node):
							g.add_edge(int(node),int(viz_ind))
							ed_att.append(ed_points)

						

	node_pos = []								# Correct positions for unpadded image
	for node in node_att:
		points = np.array(node[0])-1
		node_pos.append(points.tolist())

	ed_pos = []
	for edge in ed_att:
		points = np.array(edge)-1
		ed_pos.append(points.tolist())

	g.vs.set_attribute_values('pos_no',node_pos)
	g.es.set_attribute_values('pos_ed',ed_pos)

	degree = np.array(g.degree())
	ind = np.nonzero(degree==0)[0]
	g.delete_vertices(ind.tolist())				# Disconsider isolated bifurcations
	
	return g
	
def Simplify(g):

	grau = np.array(g.degree())
	vet_pos_no = g.vs.get_attribute_values('pos_no')
	vet_pos_ed = g.es.get_attribute_values('pos_ed')

	for no in range(g.vcount()):

		print(no)

		pos_no = vet_pos_no[no]
		
		if (len(pos_no)>1):
			m_pos_no = np.round(np.mean(pos_no,axis=0)).astype(np.int)
			
			viz = g.neighbors(no)

			for k in range(grau[no]):
				eid = g.get_eid(no,viz[k])
				
				n_rep = g.count_multiple(eid)
				
				if (n_rep>1):
					list_ed = Get_eid(g,eid)
				else:
					list_ed = [eid]
			
				for eid in list_ed:
					e1np = np.array(vet_pos_ed[eid][:])

					if (g.is_loop(eid)==False):
						
					
						# Make edges run in the same direction (if we travel along both edges, we go from [0] to [-1] in edge 1
						# and then from [0] to [-1] in edge 2)
						dist = [np.sqrt(np.sum((m_pos_no-e1np[0])**2.)),np.sqrt(np.sum((m_pos_no-e1np[-1])**2.))]
						if (dist[0]>dist[1]):
							e1np = e1np[::-1]
						
						vet = e1np[0]-m_pos_no
						t = np.linspace(0,1,1000)
						seg = m_pos_no + vet*np.array([t]).T
						seg = np.round(seg).astype(np.int)
						
						new_seg = [seg[0]]
						for j in range(1,len(seg)):
							if ((seg[j][0] != seg[j-1][0]) | (seg[j][1] != seg[j-1][1]) | (seg[j][2] != seg[j-1][2])):
								new_seg.append(seg[j])
						
						new_seg.pop(0)
						if (len(new_seg)>0):
							new_seg.pop(-1)
							
						if (len(new_seg)>0):
							new_edge = np.concatenate((np.array(new_seg),e1np))
						else:
							new_edge = e1np.copy()
						
						vet_pos_ed[eid] = new_edge.tolist()
						
					else:

						vet = e1np[0]-m_pos_no
						t = np.linspace(0,1,1000)
						seg = m_pos_no + vet*np.array([t]).T
						seg = np.round(seg).astype(np.int)
						
						new_seg1 = [seg[0]]
						for j in range(1,len(seg)):
							if ((seg[j][0] != seg[j-1][0]) | (seg[j][1] != seg[j-1][1]) | (seg[j][2] != seg[j-1][2])):
								new_seg1.append(seg[j])
						
						new_seg1.pop(0)
						if (len(new_seg1)>0):
							new_seg1.pop(-1)
						
						vet = e1np[-1]-m_pos_no
						t = np.linspace(0,1,1000)
						seg = m_pos_no + vet*np.array([t]).T
						seg = np.round(seg).astype(np.int)
						
						new_seg2 = [seg[0]]
						for j in range(1,len(seg)):
							if ((seg[j][0] != seg[j-1][0]) | (seg[j][1] != seg[j-1][1]) | (seg[j][2] != seg[j-1][2])):
								new_seg2.append(seg[j])
						
						new_seg2.pop(0)
						if (len(new_seg2)>0):
							new_seg2.pop(-1)
						
						if (len(new_seg1)>0):
							new_edge = np.concatenate((np.array(new_seg1),e1np))
						else:
							new_edge = e1np.copy()
							
						if (len(new_seg2)>0):
							new_edge = np.concatenate((new_edge,np.array(new_seg2[::-1])))
						
						vet_pos_ed[eid] = new_edge.tolist()				
						

			vet_pos_no[no] = [m_pos_no.tolist()]
				
	g_f = g.copy()

	for i in range(g.vcount()):
		vet_pos_no[i] = vet_pos_no[i][0]
		
	g_f.vs.set_attribute_values('pos_no',vet_pos_no)
	g_f.es.set_attribute_values('pos_ed',vet_pos_ed)
	
	return g_f
	
def Apaga_bif(g,T,escala):

	escala = np.array(escala)

	g_f = g.copy()
	
	# Get edges size
	for i in range(g_f.ecount()):
		
		ed = np.array(g_f.es['pos_ed'][i])
		
		ded = np.diff(ed,axis=0)
		tam_aux = np.zeros(ded.shape[0])
		for j in range(ded.shape[0]):
			tam_aux[j] = np.sqrt(np.sum((ded[j]*escala)**2))

		tam = np.sum(tam_aux)
	
		no1,no2 = g_f.es[i].tuple
		if ((g_f.degree(no1)==1) | (g_f.degree(no2)==1)):
			g_f.es[i]['is_branch'] = 1
		else:
			g_f.es[i]['is_branch'] = 0
		
		g_f.es[i]['tam'] = tam
		
	g_f2 = Remove_small_mul_all(g_f,2*T,escala)
	g_f2 = Connect_all(g_f2,None,T,escala)
		
	g_f2 = Del_branch(g_f2,T,escala)

	return g_f2
	
def Elimina(img,img_lab,tam_comp,tam=20):

	ind = np.nonzero(tam_comp<=tam)[0]+1
	
	if (ind.size < tam_comp.size/2):

		img_f = img.copy()
	
		print(ind.size)
		for i in range(0,ind.size):
		
			comp = ind[i]
			ind2 = np.nonzero(img_lab==comp)
			img_f[ind2] = 0
		
			print(i)
		
	else:
		ind = np.nonzero(tam_comp>tam)[0]+1
		
		img_f = np.zeros_like(img)
		
		print(ind.size)
		for i in range(0,ind.size):
		
			comp = ind[i]
			ind2 = np.nonzero(img_lab==comp)
			img_f[ind2] = 1
		
			print(i)
		

	return img_f
	
def Elimina_v2(imgBin, tamThreshold=20):
	
	imgLab,numComp = nd.label(imgBin)
	tamComp = nd.sum(imgBin,imgLab,range(numComp+1))

	mask = tamComp>tamThreshold
	mask[0] = False
		
	img_f = mask[imgLab]
	
	return img_f	
	
def Flood_fill(res,grand_z,grand_x,grand_y):

	bif_nodes = []
	prox_bif = []
	tips = []
	
	bif_nodes.append([grand_z,grand_x,grand_y])
	prox_bif.append([grand_z,grand_x,grand_y])
	res[grand_z,grand_x,grand_y] = -1
		
	while (len(prox_bif)>0):
		z,x,y = prox_bif.pop()
		list_viz = vizinhos(res,[z,x,y])
		for viz in list_viz:
			z_viz,x_viz,y_viz = viz
			
			if (res[z_viz,x_viz,y_viz]==0):			# Isolated pixels in the image
				res[z_viz,x_viz,y_viz] = -1
				
			if (res[z_viz,x_viz,y_viz]==1):			# Disconsider tips connected to bifurcations
				res[z_viz,x_viz,y_viz] = -1	
			
			elif (res[z_viz,x_viz,y_viz]==2):
				tips.append([z_viz,x_viz,y_viz])
				
			elif (res[z_viz,x_viz,y_viz]>=3):
				prox_bif.append([z_viz,x_viz,y_viz])
				bif_nodes.append([z_viz,x_viz,y_viz])
				res[z_viz,x_viz,y_viz] = -1
				
	return [bif_nodes,tips]
	
def vizinhos(res,point):
	
	z,x,y = point
	
	ind = []

	for k in range(z-1,z+2):
		for i in range(x-1,x+2):
			for j in range(y-1,y+2):			
				if (res[k,i,j]>=0):
					ind.append([k,i,j])


	return ind	
	
def Get_eid(g,ed):	

	edges = np.array(g.get_edgelist())
	no1,no2 = g.es[ed].tuple
	
	ind = np.nonzero((no1==edges[:,0])&(no2==edges[:,1]))[0]
	ind = ind.tolist()
	
	return ind		
	
def Remove_small_mul_all(g,T,escala):	
	
	g_f = g.copy()
	
	isLoop = np.array(g_f.is_loop())
	ind = np.nonzero(isLoop==1)[0].tolist()
	eds2rem = []
	for i in ind:
	
		tam = g_f.es[i]['tam']
	
		if (tam<T):
		
			eds2rem.append(i)
				
	g_f.delete_edges(eds2rem)			

	
	numMult = np.array(g_f.count_multiple())
	ind = np.nonzero(numMult>=2)[0].tolist()	
	eds2rem = []
	for i in ind:
	
		eids = Get_eid(g_f,int(i))		# Get all edges between the given pair of nodes
	
		tam = np.zeros(len(eids))
		k = 0
		for eid in eids:
		
			tam[k] = g_f.es[eid]['tam']
			k+=1
	
		currEdInd = eids.index(i)
		if (tam[currEdInd]<T):
			if (np.max(tam)>=T):	# If one edge is bigger than threshold, and not i, remove i

				eds2rem.append(i)
				
			else:
			
				biggerEd = eids[np.argmax(tam)]
				
				if (biggerEd!=i):
				
					eds2rem.append(i)
					
	g_f.delete_edges(eds2rem)	

	d = np.array(g_f.degree())
	ind = np.nonzero(d==1)[0].tolist()
	for i in ind:
		eid = g_f.incident(i)[0]
		g_f.es[eid]['is_branch'] = 1
	
	return g_f
	
def Remove_small_mul(g,viz,T,escala):	
	
	i = g.get_eid(viz[0],viz[1])

	eids = Get_eid(g,i)		# Get all edges between the given pair of nodes
	
	tam = np.zeros(len(eids))
	k = 0
	for eid in eids:
	
		tam[k] = g.es[eid]['tam']
		k+=1

	smallEdgeInd = np.argmin(tam)
	if (tam[smallEdgeInd]<T):	# If one edge is bigger than threshold, and not i, remove i

		g.delete_edges(eids[smallEdgeInd])
		wasRemoved = True
		
	else:
		
		wasRemoved = False
		
		
	return wasRemoved	
		
def Connect_all(g_f,no_ini,T,escala):

	ind_no = []
	if (no_ini!=None):
		viz = g_f.neighbors(no_ini)
		if (viz[0]!=viz[1]):
			ind_no.append(no_ini)
	else:
		grau = np.array(g_f.degree())
		degree_2_nodes = np.nonzero(grau==2)[0].tolist()
		for i in degree_2_nodes:
			viz = g_f.neighbors(i)
			if (viz[0]!=viz[1]):
				ind_no.append(i)
	
	while (len(ind_no)>0):
		no = ind_no.pop()
		print(no)
		
		pos_no = g_f.vs[no]['pos_no']
		
		viz = g_f.neighbors(no)
		eid1 = g_f.get_eid(no,viz[0])
		if (viz[0]!=viz[1]):
			eid2 = g_f.get_eid(no,viz[1])
			isMultiple = False
		else:
			aux = Get_eid(g_f,eid1)
			if (aux[0]==eid1):
				eid2 = aux[1]
			else:
				eid2 = aux[0]	
			isMultiple = True
		
		e1 = g_f.es['pos_ed'][eid1][:]
		e2 = g_f.es['pos_ed'][eid2][:]

		e1np = np.array(e1)
		e2np = np.array(e2)
		
		# Make both edges run in the same direction (if we travel along both edges, we go from [0] to [-1] in edge 1
		# and then from [0] to [-1] in edge 2)
		dist = [np.sqrt(np.sum((pos_no-e1np[0])**2.)),np.sqrt(np.sum((pos_no-e1np[-1])**2.))]
		if (dist[0]<dist[1]):
			e1np = e1np[::-1]
			
		dist = [np.sqrt(np.sum((pos_no-e2np[0])**2.)),np.sqrt(np.sum((pos_no-e2np[-1])**2.))]
		if (dist[1]<dist[0]):
			e2np = e2np[::-1]	
		
		new_edge = np.concatenate((e1np,np.array([pos_no]),e2np))

		ded = np.diff(new_edge,axis=0)
		tam_aux = np.zeros(ded.shape[0])
		for j in range(ded.shape[0]):
			tam_aux[j] = np.sqrt(np.sum((ded[j]*escala)**2))

		tam = np.sum(tam_aux)		
			
		if (isMultiple==True):	
			
			if (g_f.degree(viz[0])==2):
				ind_no.remove(viz[0])
			
			g_f.delete_edges([eid1,eid2])		
			g_f.add_edge(viz[0],viz[1],pos_ed=new_edge.tolist(),tam=tam,is_branch=0)			
				
		else:

			if (g_f.are_connected(viz[0],viz[1])==False):
			
				if ((g_f.degree(viz[0])==1) | (g_f.degree(viz[1])==1)):
					isBranch = 1
				else:
					isBranch = 0		
			
				g_f.delete_edges([eid1,eid2])		
				g_f.add_edge(viz[0],viz[1],pos_ed=new_edge.tolist(),tam=tam,is_branch=isBranch)
				
			else:
			
				g_f.delete_edges([eid1,eid2])		
				g_f.add_edge(viz[0],viz[1],pos_ed=new_edge.tolist(),tam=tam,is_branch=0)

				wasRemoved = Remove_small_mul(g_f,viz,2*T,escala)
				
				if (wasRemoved==True):
	
					if (g_f.degree(viz[0])==2):
						ind_no.append(viz[0])
					if (g_f.degree(viz[1])==2):
						ind_no.append(viz[1])
						
					eid = g_f.get_eid(viz[0],viz[1])	
					if (g_f.degree(viz[0])==1):
						g_f.es[eid]['is_branch'] = True
						ind_no.remove(viz[0])
					if (g_f.degree(viz[1])==1):
						g_f.es[eid]['is_branch'] = True
						ind_no.remove(viz[1])						
						
	return g_f
	
def Connect(g,no,escala):
	
	pos_no = g.vs['pos_no'][no]
		
	viz = g.neighbors(no)

	eid1 = g.get_eid(no,viz[0])
	if (g.is_loop(eid1)==False):
		if (g.count_multiple(eid1)==1):
			eid2 = g.get_eid(no,viz[1])
		else:
			aux = Get_eid(g,eid1)
			if (aux[0]==eid1):
				eid2 = aux[1]
			else:
				eid2 = aux[0]
	
		e1 = g.es['pos_ed'][eid1][:]
		e2 = g.es['pos_ed'][eid2][:]

		e1np = np.array(e1)
		e2np = np.array(e2)
		
		# Make both edges run in the same direction (if we travel along both edges, we go from [0] to [-1] in edge 1
		# and then from [0] to [-1] in edge 2)
		dist = [np.sqrt(np.sum((pos_no-e1np[0])**2.)),np.sqrt(np.sum((pos_no-e1np[-1])**2.))]
		if (dist[0]<dist[1]):
			e1np = e1np[::-1]
			
		dist = [np.sqrt(np.sum((pos_no-e2np[0])**2.)),np.sqrt(np.sum((pos_no-e2np[-1])**2.))]
		if (dist[1]<dist[0]):
			e2np = e2np[::-1]	
		
		new_edge = np.concatenate((e1np,np.array([pos_no]),e2np))
		
		ded = np.diff(new_edge,axis=0)
		tam_aux = np.zeros(ded.shape[0])
		for j in range(ded.shape[0]):
			tam_aux[j] = np.sqrt(np.sum((ded[j]*escala)**2))

		tam = np.sum(tam_aux)		
				
		if ((g.degree(viz[0])==1) | (g.degree(viz[1])==1)):
			isBranch = 1
		else:
			isBranch = 0

		g.delete_edges([eid1,eid2])			
		g.add_edge(viz[0],viz[1],pos_ed=new_edge.tolist(),tam=tam,is_branch=isBranch)

	return viz

def Del_branch(g,T,escala):

	escala = np.array(escala)

	g_f = g.copy()

	while (1==1):

		eids = g_f.es.select(is_branch_eq=1).indices
		
		if (len(eids)==0):												# If no more edges, finish
			break
			
		vet_tam = np.array(g_f.es['tam'])[eids]
		ind = np.argmin(vet_tam)									# Get smallest branch
		
		if (vet_tam[ind]>=T):										# If larger than the threshold, finish
			break
			
		ind = int(eids[ind])
		no1,no2 = g_f.es[ind].tuple
			
		g_f.delete_edges(ind)

		if (g_f.degree(no1)==0):
			no_target = no2
		else:
			no_target = no1
			

		if (g_f.degree(no_target)==2):								# If neighbor degree was two, we need to delete it

			Connect_all(g_f,no_target,T,escala)

		elif (g_f.degree(no_target)==1):
			eid = g_f.get_eid(no_target,g_f.neighbors(no_target)[0])
			g_f.es[eid]['is_branch'] = 1
		
	grau = np.array(g_f.degree())
	ind = np.nonzero(grau==0)[0]		
	g_f.delete_vertices(ind.tolist())	
		
	return g_f
	
def remove_small_clusters(g, T=50):

	clu = g.clusters()
	gs = clu.subgraphs()
	compSize = np.zeros(len(gs))
	for i in range(len(gs)):
		compSize[i] = np.sum(gs[i].es['tam'])
	ind = np.nonzero(compSize<T)[0]
	
	nodesToRem = []
	for compInd in ind:
		nodesToRem.extend(clu[compInd])
		
	g_out = g.copy()	
	g_out.delete_vertices(nodesToRem)	

	return g_out	
	
def Make_folders(inFolder, outFolder, filesTree):
	
	isFile = np.array(filesTree.vs['isFile'], dtype=int)
	batchName = filesTree.vs[0]['name']
	nodesToVisit = [0]
	while len(nodesToVisit)>0:
		node = nodesToVisit.pop(0)
		if isFile[node]==0:
			pathName = filesTree.vs[node]['fullName']
			sons = filesTree.neighbors(node, mode=OUT)
			hasFile = max(isFile[sons])

			dirName = outFolder+'Binary/'+pathName
			if os.path.isdir(dirName)==False:	
				os.mkdir(dirName)	
			dirName = outFolder+'interpolated/'+pathName
			if os.path.isdir(dirName)==False:	
				os.mkdir(dirName)					
			dirName = outFolder+'Skeleton/'+pathName
			if os.path.isdir(dirName)==False:	
				os.mkdir(dirName)	
			dirName = outFolder+'Network/'+pathName
			if os.path.isdir(dirName)==False:	
				os.mkdir(dirName)				
				
			if hasFile==1:		
				for dirKind in ['maximum_projection', 'numpy']:
					dirName = outFolder+'Binary/'+pathName+dirKind
					if os.path.isdir(dirName)==False:	
						os.mkdir(dirName)
					dirName = outFolder+'interpolated/'+pathName+dirKind
					if os.path.isdir(dirName)==False:	
						os.mkdir(dirName)						
					dirName = outFolder+'Skeleton/'+pathName+dirKind
					if os.path.isdir(dirName)==False:	
						os.mkdir(dirName)

				for netType in ['original', 'simple', 'core']:
					dirName = outFolder+'Network/'+pathName+netType
					if os.path.isdir(dirName)==False:	
						os.mkdir(dirName)				
					for dirKind in ['maximum_projection', 'numpy']:
						dirName = outFolder+'Network/'+pathName+netType+'/'+dirKind
						if os.path.isdir(dirName)==False:	
							os.mkdir(dirName)
			
			nodesToVisit.extend(sons)
								
def getFiles(inFolder, batchName, condStr=''):

	forbiddenFolders = ['ROI', 'maximum_projection', 'equalized']

	if os.name=='nt':
		if inFolder[-1]!='\\':
			if inFolder[-1]=='/':
				properInFolder = inFolder[:-1] + '\\'
				inFolder = inFolder[:-1] + '/'
			else:
				properInFolder = inFolder + '\\'
				inFolder = inFolder + '/'
	else:
		properInFolder = inFolder

	foldersTree = Graph(directed=True)
	foldersTree.add_vertex(name=batchName, fullName=batchName+'/', isFile=0)
	nameMap = {batchName:0}
	dataTree = os.walk(properInFolder+batchName)
	k = 1
	for fullPfolder, folders, files in dataTree:
		if os.name=='nt':
			Pfolder = fullPfolder.split('\\')[-1]
		else:
			Pfolder = fullPfolder.split('/')[-1]
		if Pfolder not in forbiddenFolders:
			sortedFiles = natsort.natsorted(files)
			sortedFolders = natsort.natsorted(folders)
			for file in sortedFiles:
				if condStr in file:
					foldersTree.add_vertex(name=file, fullName=foldersTree.vs[nameMap[Pfolder]]['fullName']+file, isFile=1)
					nameMap[file] = k
					k += 1
					foldersTree.add_edge(nameMap[Pfolder], nameMap[file])
			for folder in sortedFolders:
				if folder not in forbiddenFolders:
					foldersTree.add_vertex(name=folder, fullName=foldersTree.vs[nameMap[Pfolder]]['fullName']+folder+'/', isFile=0)
					nameMap[folder] = k
					k += 1			
					foldersTree.add_edge(nameMap[Pfolder], nameMap[folder])

	files = foldersTree.vs.select(isFile_eq=1)['fullName']
	
	return foldersTree, files
									
def Graph2Img(g,shape):

	node_pos = g.vs['pos_no']
	ed_pos = g.es['pos_ed']
	img = np.zeros(shape)
	for i in range(len(node_pos)):
		node = node_pos[i]
		if (np.isscalar(node[0])==True):
			img[node[0],node[1],node[2]] = 1
		else:
			for pos in node:
				img[pos[0],pos[1],pos[2]] = 1
			
	for i in range(len(ed_pos)):		
		edge = ed_pos[i]
		for point in edge:
			img[point[0],point[1],point[2]] = 1

	return img
	
def testThresholds(img_np, scale, outFolder, batchName, fileName):

	C_list = [2,3,4,5,6,7,8,9,10,11,12]

	dirName = outFolder + batchName
	if os.path.isdir(dirName)==False:	
		os.mkdir(dirName)	
	
	sigma = np.array([0.1,1.,1.])
	img_np_diffused = fil.gaussian_filter(img_np,sigma=sigma/scale)

	size_z,size_x,size_y = img_np.shape

	radius = 40
	img_corr = np.zeros([size_z,size_x,size_y])
	for i in range(size_z):
		img_blurred = fil.gaussian_filter(img_np_diffused[i],sigma=radius/2.)
		img_corr[i] = img_np_diffused[i] - img_blurred


	for C in C_list:
		img_T = img_corr > C	
		outFolderComplete = outFolder+batchName+'/'
		plt.imsave(outFolderComplete+fileName+'_%d.png'%C,np.max(img_T,axis=0),cmap='gray')
	
	
T = 10. #25.
saveImgs = True	# Wheter to save intermediate steps and maximum projections

batchName = 'Vascular Project_Picture CD31-PDGFR-vGLUT2-IBA1'
batchNameOut = '{}_pericytes'.format(batchName)

inFolder = '/media/Data/cesar/Baptiste data/Project neuroinflammation/'
outFolder = '../data/'

shouldTestThresholds = False

tDict = readThresholds('../data/thresholds.txt')

filesTree, files = getFiles(inFolder, batchName, '.czi')
for idx, folder in enumerate(filesTree.vs['fullName']):
	filesTree.vs[idx]['fullName'] = folder.replace(batchName, batchNameOut)
Make_folders(inFolder, outFolder, filesTree)

for q in range(len(files)):	

	splitIndex = files[q].rfind('/')
	filePath, file = files[q][:splitIndex]+'/', files[q][splitIndex+1:]
	dotIndex = file.rfind('.')
	fileName = file[:dotIndex]
	
	print("\nProcessing image \""+str(fileName)+"\" ("+str(q+1)+" of "+str(len(files))+")")
	
	img_np_or,scale = readimg.ReadImg(inFolder+filePath+file, 1)

	filePath = filePath.replace(batchName, batchNameOut)
	
	max_img_np_or = np.max(img_np_or,axis=0)
		
	shape = img_np_or.shape
	print("Image size is %s, scale is (%.5f,%.5f,%.5f)"%(shape.__str__(),scale[0],scale[1],scale[2]))
	
	zoomRatio = scale/np.min(scale)
	img_np = nd.interpolation.zoom(img_np_or,zoomRatio,order=2)	
	newScale = [np.min(scale)]*3	
	
	del img_np_or
	
	img_np_shape = img_np.shape
	
	if (saveImgs == True):
		plt.imsave(outFolder+'interpolated/'+filePath+'maximum_projection/'+fileName+'.png',np.max(img_np,axis=0),cmap='gray')
	
	if shouldTestThresholds==True:
		testThresholds(img_np, scale, '../tests/threshold/', batchName, fileName)	
	else:
		print("Binarizing...")
		img_bin = Binariza(img_np, newScale, tDict[batchNameOut][fileName][1])	
		
		del img_np
		
		if (saveImgs == True):
			outFolderComplete = outFolder+'Binary/'+filePath
			np.save(outFolderComplete+'numpy/'+fileName,img_bin)
			plt.imsave(outFolderComplete+'maximum_projection/'+fileName+'.png',np.sum(img_bin,axis=0),cmap='jet')
		
		print("\nObtaining skeleton...")
		img_skel = Esqueleto(img_bin)
		
		del img_bin
		
		if (saveImgs == True):
			outFolderComplete = outFolder+'Skeleton/'+filePath
			np.save(outFolderComplete+'numpy/'+fileName,img_skel)
			plt.imsave(outFolderComplete+'maximum_projection/'+fileName+'.png',np.max(img_skel,axis=0),cmap='gray')
		
		print("\nTransforming into network...")
		g = Cria_rede(img_skel)
		
		del img_skel
		
		if (saveImgs == True):
			outFolderComplete = outFolder+'Network/'+filePath
			np.save(outFolderComplete+'original/numpy/'+fileName,g)
			img_gor = Graph2Img(g,img_np_shape)
			plt.imsave(outFolderComplete+'original/maximum_projection/'+fileName+'.png',np.max(img_gor,axis=0),cmap='gray')
		
		g_simple = Simplify(g)
		
		del g
		
		if (saveImgs == True):
			outFolderComplete = outFolder+'Network/'+filePath
			np.save(outFolderComplete+'simple/numpy/'+fileName,g_simple)
			img_gsi = Graph2Img(g_simple,img_np_shape)
			plt.imsave(outFolderComplete+'simple/maximum_projection/'+fileName+'.png',np.max(img_gsi,axis=0),cmap='gray')

		g_final = Apaga_bif(g_simple,T,newScale)
		g_final = remove_small_clusters(g_final, T=20)
			
		outFolderComplete = outFolder+'Network/'+filePath
		np.save(outFolderComplete+'core/numpy/'+fileName,g_final)
		
		if (saveImgs == True):
			img_gfi = Graph2Img(g_final,img_np_shape)
			plt.imsave(outFolderComplete+'core/maximum_projection/'+fileName+'.png',np.max(img_gfi,axis=0),cmap='gray')	