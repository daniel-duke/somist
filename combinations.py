### Author: Daniel Duke
### Last updated: 4/5/24

test words

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import pickle
import sys

### Notes
# this script searches the combinatorial space to find the optimum commensurate supercell 
  # between two materials given their lattice vectors (Lvec) in the plane of contact and the
  # strain constants (ec_normal, ec_shearP, ec_shearD) of the material to be strained.
# this method assumes the relative orientation between the two lattices is irrelevnt. 
# because the program assumes only one material is strained, the elastic constants are only
  # required for the one (strained) material; see elasic constant descriptions below.
# as mentioned above, the shear elastic response is defined by two constants: ec_shearP 
  # (for parallel to x axis) and ec_sheaerD (for diagonal between x and y axes); the code
  # assumes that the elastic response constant sinusoidally between these two values
  # (periodicity of 90 degrees); this describes the behavior of perovskites well, with
  # a lower response (roughly half) when the perovskite is strained parallel to the octahedron
  # edges (the diagonal direction in the standard root2/root2 cell).
# to ignore this sinusoidal variance of the shear response (in other words, to assume the 
  # material is shear isotropic in the plane of contact), simply set the two shear elastic
  # constants equal.
# the "optimum" cell is that which minimizes the deformation energy of the strained material,
  # as calculated from the given elastic constants (see below), while still keeping the size
  # of the supercell reasonable
# to this end, the discovered cells are ranked according to a "superscore" which balances the
  # size and strain of the supercell according to a given "size_strain_tradeoff" parameter, 
  # which describes how huch additional area (A2) you are willing to simulate for 1 mEv/atom
  # reduction in strain energy.
# all lattice vectors are defined as column vectors; for example, lattice = [[1,2],[3,4]] would
  # be interpreted as a lattice with basis vectors v1 = [1,3] and v2 = [2,4].
# although this script was written assuming the two materials are graphene and a perovskite,
  # it is extensible to any two materials; thus any reference to p (perovskite) may be thought
  # of as material 1, and any reference to g (graphene) may thought of as material 2.
# because of unaccounted symmetry, the combination searcher usually finds several supercells
  # with identical (or nearly identical if the lattice vector magnitudes are slightly different)
  # areas and energies; to make analysis easier, the program identifies these "clusters" and
  # picks the supercell within each cluster with the lattice vector angle closest to 90 degrees
  # to represent the cluster.

### Elastic Constants
# ec_normalE = [ energy increase (meV) / atom ] /
#              [ area increase from uniform biaxial expansion / unit cell area * 100 ]
# ec_normalC = [ energy increase (meV) / atom ] /
#              [ area increase from uniform biaxial compression / unit cell area * 100 ]
# ec_shearP = [ energy increase (meV) / atom ] /
# 			  [ simple shear strain parallel to x axis * 100 ]
# ec_shearD = [ energy increase (meV) / atom ] /
# 			  [ simple shear strain parallel to x=y line (the xy plane diagonal) * 100 ]

### Output Variables
# areas - supercell areas
# enrgs - [strain energy, normal strain, shear strain]
# sTidxs - supercell transformation indices, graphene then perovskite ([supercell] = [unit cell]*[T])
# sLvecs - supercell lattice vectors, graphene then perovskite
# A suffix - all discovered
# C suffix - clustered


################################################################################
### Parameters

def main():

	### choose whether to load or calculate data
	load_data = False
	saveFile = './supercells.pkl'
	loadFile = './supercells.pkl'

	### for calculating data
	if load_data == False:

		### lattice vectors - graphene (material 1)
					#Lvec1		#Lvec2
		gLvec = [[	2.462729, 	1.231395	],\
				 [	0.000000, 	2.133057	]]

		### lattice vectors - perovskite (material 2) - 
					#Lvec1		#Lvec2
		pLvec = [[	11.794684,	-0.059246	],\
				 [	0.000000,	11.794561	]]

		### elastic constants
		ec_normalE = 0.2768				#expansive normal strain elastic constant
		ec_normalC = 0.2906				#compressive normal strain elastic constant
		ec_shearP = 0.1421				#shear strain parallel to axis elastic constant
		ec_shearD = 0.0719				#shear strain diagonal to axis elastic constant

		### search parameters
		g_max = 8						#max integer multiple of graphene lattice vectors 
		p_max = 2  						#max integer multiple of perovskite lattice vectors 
		max_area = 600					#max area supercell to consider
		max_normal = 0.04  				#max normal strain to consider
		max_shear = 0.08 				#max shear strain to consider
		max_find = 1000					#max number of supercells to find (consider restricting search parameters this many are found)
		chirality = False				#whether to test mirror image of perovskite (useless if lattice vector magnitudes are equal)

		### analysis parameters
		size_strain_tradeoff = 100		#scoring parameter (see notes)
		separate_area = 10 				#difference in area (A2) that distinguishes between separate supercell clusters
		separate_energy = 0.1   		#difference in energy (mEv/atom) that distinguishes between separate supercell clusters


################################################################################
### Calculations

		### combine elastic constants into convenient array
		ec = np.array([ec_normalE,ec_normalC,ec_shearP,ec_shearD])

		### make math work
		gLvec = np.array(gLvec)
		pLvec = np.array(pLvec)

		### search with given unit vectors
		areasA, enrgsA, sTidxsA, sLvecsA = search(ec, gLvec, pLvec, g_max, p_max, max_area, max_normal, max_shear, max_find)

		### flip and search again
		if chirality == True:
			pLvecF = pLvec[:, [1,0]]
			areasF, enrgsF, sTidxsF, sLvecsF = search(ec, gLvec, pLvecF, g_max, p_max, max_area, max_normal, max_shear, max_find)
			
			areasA = np.concatenate((areasA, areasF))
			enrgsA = np.concatenate((enrgsA, enrgsF))
			sTidxsA = np.concatenate((sTidxsA, sTidxsF))
			sLvecsA = np.concatenate((sLvecsA, sLvecsF))

		### identify clusters and rank
		areasC, enrgsC, sTidxsC, sLvecsC, superscores = process_options(size_strain_tradeoff, areasA, enrgsA, sTidxsA, sLvecsA, separate_area, separate_energy)

		### calculate maximum theoretical energy (for plotting)
		max_enrg = max([ec_normalE,ec_normalC])*(max_normal*100)**2 + max([ec_shearP,ec_shearD])*(max_shear*100)**2
		max_enrg = np.round(max_enrg,1)

		### storing calculations
		with open(saveFile, 'wb') as f:
			pickle.dump([areasA, enrgsA, sTidxsA, sLvecsA, areasC, enrgsC, sTidxsC, sLvecsC, superscores, max_area, max_enrg], f)

	### for loading data
	else:
		with open(loadFile, 'rb') as f:
			areasA, enrgsA, sTidxsA, sLvecsA, areasC, enrgsC, sTidxsC, sLvecsC, superscores, max_area, max_enrg = pickle.load(f)


################################################################################
### Results

	### choose index of supercell to inspect (0 for best supercell)
	s = 0

	### print basic search results
	print()
	print(f"{len(areasA)} supercells found")
	print(f"{len(areasC)} clusters identified")

	### plot all the found supercells and clusters
	plt.figure('Results',figsize=(8,6))
	plt.scatter(areasA,enrgsA[:,0],s=50)
	plt.scatter(areasC,enrgsC[:,0],s=20)
	plt.xlim((0,max_area))
	plt.ylim((0,max_enrg))
	plt.xlabel("Area [A2]")
	plt.ylabel("Strain Energy [mEv/atom]")
	plt.title("Discovered Supercells")
	plt.grid(True)

	### print chosen supercell transition matrix and lattice vectors
	print("\nGraphene Transition Indices:")
	printMatrix(sTidxsC[s,0],"int")
	print("Perovskite Transition Indices:")
	printMatrix(sTidxsC[s,1],"int")
	print("\nGraphene Supercell:")
	printMatrix(sLvecsC[s,0],"float")
	print("Perovskite Supercell:")
	printMatrix(sLvecsC[s,1],"float")
	print("\nSupercell Area:")
	print(f"{areasC[s]:1.2f} A2")
	print("\nStrain Energy:")
	print(f"{enrgsC[s,0]:1.2f} mEv/atom")
	print("\nNormal Strain: \tShear Strain:")
	print(f"{enrgsC[s,1]*100:1.2f} % \t\t\t{enrgsC[s,2]*100:1.2f} %")
	print()

	# Plot the vectors with custom colors
	plt.figure('Lattices',figsize=(8, 6))
	plt.quiver([0,0], [0,0], sLvecsC[s,0,0,:], sLvecsC[s,0,1,:], angles='xy', scale_units='xy', scale=1, color='grey', label='Graphene')
	plt.quiver([0,0], [0,0], sLvecsC[s,1,0,:], sLvecsC[s,1,1,:], angles='xy', scale_units='xy', scale=1, color='orange', label='Perovskite')
	plt.xlim(min(np.min(sLvecsC[s,:,0,:]),0) - 1, max(np.max(sLvecsC[s,:,0,:]),0) + 1)
	plt.ylim(min(np.min(sLvecsC[s,:,1,:]),0) - 1, max(np.max(sLvecsC[s,:,1,:]),0) + 1)
	plt.title('Supercell Lattice Vectors')
	plt.grid(True)
	plt.legend()
	plt.show()

################################################################################
### Calculation Managers

### searches combinatorial space for best supercell
def search(ec, gLvec, pLvec, g_max, p_max, max_area, max_normal, max_shear, max_find):

	### adjust lattice vectors
	gLvec,gswapped,gR = fix_lattice(gLvec)
	pLvec,pswapped,pR = fix_lattice(pLvec)

	### calculate areas and cells for perovskite
	Ap = np.zeros((p_max * 2 + 1, p_max + 1, p_max * 2 + 1, p_max + 1))
	X  = np.zeros((p_max * 2 + 1, p_max + 1, p_max * 2 + 1, p_max + 1, 2, 2))
	for j2 in range(p_max + 1):
		for i2 in range(-p_max, p_max + 1):
			for j1 in range(p_max + 1):
				for i1 in range(-p_max, p_max + 1):
					X[i1 + p_max, j1, i2 + p_max, j2] = np.dot(pLvec, np.array([[i1, i2], [j1, j2]]))
					Ap[i1 + p_max, j1, i2 + p_max, j2] = np.abs(np.linalg.det(np.dot(pLvec, np.array([[i1, i2], [j1, j2]]))))
					if Ap[i1 + p_max, j1, i2 + p_max, j2] > max_area:
						Ap[i1 + p_max, j1, i2 + p_max, j2] = 0

	### calculate areas and cells for graphene
	Ag = np.zeros((g_max * 2 + 1, g_max + 1, g_max * 2 + 1, g_max + 1))
	S = np.zeros((g_max * 2 + 1, g_max + 1, g_max * 2 + 1, g_max + 1, 2, 2))
	for j2 in range(g_max + 1):
		for i2 in range(-g_max, g_max + 1):
			for j1 in range(g_max + 1):
				for i1 in range(-g_max, g_max + 1):
					S[i1 + g_max, j1, i2 + g_max, j2] = np.dot(gLvec, np.array([[i1, i2], [j1, j2]]))
					Ag[i1 + g_max, j1, i2 + g_max, j2] = np.abs(np.linalg.det(np.dot(gLvec, np.array([[i1, i2], [j1, j2]]))))
					if Ag[i1 + g_max, j1, i2 + g_max, j2] > max_area:
						Ag[i1 + g_max, j1, i2 + g_max, j2] = 0

	### initializations
	pass_n = 0
	pass_s = 0
	ref = np.array([1,0]).T
	options_A = np.zeros(max_find)
	options_E = np.zeros((max_find, 3))
	options_T = np.zeros((max_find, 2,2,2))
	options_L = np.zeros((max_find, 2,2,2))

	### loop thru perovskite cells
	for j2p in range(p_max + 1):
		for i2p in range(-p_max, p_max + 1):
			for j1p in range(p_max + 1):
				for i1p in range(-p_max, p_max + 1):

					### test if v2 is couterclockwise from v1
					### test if area is non-zero
					if i2p / max(1,np.linalg.norm([i2p, j2p])) < i1p / max(1,np.linalg.norm([i1p, j1p])) and Ap[i1p + p_max, j1p, i2p + p_max, j2p] > 1:
						
						### loop thru graphene cells
						for j2g in range(g_max + 1):
							for i2g in range(-g_max, g_max + 1):
								for j1g in range(g_max + 1):
									for i1g in range(-g_max, g_max + 1):

										### test if v2 is couterclockwise from v1
										### test if area is non-zero
										if i2g / max(1,np.linalg.norm([i2g, j2g])) < i1g / max(1,np.linalg.norm([i1g, j1g])) and Ag[i1g + g_max, j1g, i2g + g_max, j2g] > 1:

											### normal strain test
											normal = Ag[i1g + g_max, j1g, i2g + g_max, j2g] / Ap[i1p + p_max, j1p, i2p + p_max, j2p] - 1
											if np.abs(normal) < max_normal:
												pass_n += 1

												### shear strain test
												Sr = S[i1g + g_max, j1g, i2g + g_max, j2g]
												Xr = X[i1p + p_max, j1p, i2p + p_max, j2p]
												F = np.dot(Sr, np.linalg.inv(Xr))
												D_eig, _ = np.linalg.eig(np.real(sqrtm(np.dot(F.T, F) / np.linalg.det(F))))
												shear = np.abs(D_eig[0] - D_eig[1])
												if shear < max_shear:
													pass_s += 1

													### calculate normal strain energy
													if normal > 0:
														E_normal = ec[0] * (normal * 100) ** 2
													else:
														E_normal = ec[1] * (normal * 100) ** 2

													### calculate shear strain energy
													s_dir = calc_s_dir(Sr, Xr, F, D_eig, ref)
													E_shearP = ec[2] * (shear * 100) ** 2
													E_shearD = ec[3] * (shear * 100) ** 2
													E_shear = (E_shearP-E_shearD)/2 * np.cos(s_dir * 2 * np.pi / 90) + (E_shearP+E_shearD)/2
												
													### total strain energy
													E_total = E_normal + E_shear

													### calculate supercell transition indices and lattice vectors
													gsTidx = np.array([[i1g, i2g],[j1g, j2g]])
													gsLvec,gsTidx = unfix_lattice(Sr,gsTidx,gswapped,gR)
													psTidx = np.array([[i1p, i2p],[j1p, j2p]])
													psLvec,psTidx = unfix_lattice(Xr,psTidx,pswapped,pR)

													### record option
													options_A[pass_s-1] = Ag[i1g + g_max, j1g, i2g + g_max, j2g]
													options_E[pass_s-1] = [E_total, normal, shear]
													options_T[pass_s-1] = [gsTidx,psTidx]
													options_L[pass_s-1] = [gsLvec,psLvec]

													### raise flag if too many found
													if pass_s == max_find:
														print("Criteria too lenient, stopped searching.")
														sys.exit()

	### trim results
	options_A = options_A[:pass_s]
	options_E = options_E[:pass_s]
	options_T = options_T[:pass_s]
	options_L = options_L[:pass_s]

	### raise flag if none found, or return
	if pass_s == 0:
		print("Criteria too stringent, no supercells found.")
		sys.exit()
	return options_A, options_E, options_T, options_L


### sorts options and groups them into clusters
def process_options(size_strain_tradeoff, areas, enrgs, sTidxs, sLvecs, sep_area, sep_energy):
	### calculate angles between lattice vectors
	n_options = len(areas)
	angles_int = np.zeros(n_options)
	angles_xax = np.zeros(n_options)
	for i in range(n_options):
		angles_int[i] = vec2ang(sLvecs[i,0,:,0], sLvecs[i,0,:,1])
		angles_xax[i] = vec2ang(sLvecs[i,0,:,0], [1, 0]) + vec2ang(sLvecs[i,0,:,1], [1, 0])

	### initialize
	n_cluster = 1
	cluster_idxs = np.zeros(n_options, dtype=int)
	cluster_idxs[n_cluster - 1] = 0

	### find cluster indices
	for i in range(1, n_options):
		matched = False
		for j in range(0, n_cluster):
			if abs(areas[i] - areas[cluster_idxs[j]]) < sep_area:
				if abs(enrgs[i][0] - enrgs[cluster_idxs[j]][0]) < sep_energy:
					matched = True
					if abs(angles_int[i] - 90) - abs(angles_int[cluster_idxs[j]] - 90) < 0:
						cluster_idxs[j] = i
					elif abs(angles_int[i] - angles_int[cluster_idxs[j]]) < 0.01:
						if abs(angles_xax[i]) - abs(angles_xax[cluster_idxs[j]]) < 0:
							cluster_idxs[j] = i
					break
		if not matched:
			n_cluster += 1
			cluster_idxs[n_cluster - 1] = i

	### initialize clusters
	areasC = np.zeros(n_cluster)
	enrgsC = np.zeros((n_cluster, 3))
	sTidxsC = np.zeros((n_cluster, 2,2,2))
	sLvecsC = np.zeros((n_cluster, 2,2,2))
	superscores = np.zeros(n_cluster)

	### assign cluster values
	area_min = np.min(areasC)
	enrg_at_area_min = enrgsC[np.argmin(areasC)][0]
	for i in range(n_cluster):
		areasC[i] = areas[cluster_idxs[i]]
		enrgsC[i] = enrgs[cluster_idxs[i]]
		sTidxsC[i] = sTidxs[cluster_idxs[i]]
		sLvecsC[i] = sLvecs[cluster_idxs[i]]
		superscores[i] = (area_min - areasC[i]) / size_strain_tradeoff - (enrgsC[i][0] - enrg_at_area_min)

	### sort by superscore and return
	idx = np.argsort(-superscores)
	areasC = areasC[idx]
	enrgsC = enrgsC[idx]
	sTidxsC = sTidxsC[idx]
	sLvecsC = sLvecsC[idx]
	return areasC, enrgsC, sTidxsC, sLvecsC, superscores


################################################################################
### Utility Functions

### standardizes unit cell lattice vectors
def fix_lattice(Lvec):
	### if v2 is clockwise from v1, swap them
	swapped = False
	if vec2ang(Lvec[:,0],Lvec[:,1]) < 0:
		Lvec[:,[0,1]] = Lvec[:,[1,0]]
		swapped = True

	### align v1 with x axis
	R = ang2R(vec2ang(Lvec[:,0],np.array([1,0])))
	Lvec = np.round(np.dot(R,Lvec),6)

	### return result and change log
	return Lvec, swapped, R


### frames supercell lattice vectors and transition matrices in terms of original lattice
def unfix_lattice(sLvec,sTidx,swapped,R):
	### if v2 was clockwise from v1, swap them back
	if swapped == True:
		sLvec = sLvec[:,[1,0]]
		sTidx = sTidx[[0,1],:]

	### return lattice to original orientation
	sLvec = np.dot(R.T,sLvec)

	### return result
	return sLvec,sTidx


### calculates shear direction
def calc_s_dir(S, X, F, D, ref):
	Fn = F / np.sqrt(np.linalg.det(F))
	Rp = np.dot(Fn, np.linalg.inv((sqrtm(np.dot(Fn.T, Fn)))))
	theta = np.arctan(np.abs(D[0] - D[1]) / 2)
	Rt = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	_,V = np.linalg.eig(np.dot(S, np.linalg.inv(np.dot(np.dot(np.dot(Rt, Rp), X), np.sqrt(np.linalg.det(F))))))
	v1 = np.dot(np.dot(Rt, Rp), ref)
	v2 = np.real(V[:,0])
	s_dir = np.arccos(np.dot(v1,v2)) * 180 / np.pi
	return s_dir


### calculates angle between two 2D vectors (range = [-180,180])
def vec2ang(v1, v2):
	v1 = v1 / np.linalg.norm(v1)
	v2 = v2 / np.linalg.norm(v2)
	dot = np.dot(v1, v2)
	crs = np.cross([v1[0], v1[1], 0], [v2[0], v2[1], 0])
	angle = np.sign(crs[2]) * np.arccos(dot) * 180 / np.pi
	return angle


### calculates rotation matrix given angle
def ang2R(phi):
	phi = phi/180*np.pi
	return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


### print 3x3 matrix nicely
def printMatrix(M,type):
	for i in range(M.shape[0]):
		print("[",end="")
		for j in range(M.shape[1]):
			if type == "float":
				print('\t{0:.6f}'.format(M[i][j]),end="")
			elif type == "int":
				print('\t{0:.0f}'.format(M[i][j]),end="")
			else:
				print("Warning: unknown number type.")
		print("\t]")


### run program
main()

