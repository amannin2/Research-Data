import argparse
import copy
import numpy as np
import sklearn.decomposition
import types

from cnmim_func_gromacs import *
from cnmim_func_tinker import *

"""
"atoms" have the following attributes:
	atname       (T)
	atnum        (G, T)
	attype       (G, T)
	resname      (G)
	resnum       (G)
	vxyz         (G)
	xyz          (G, T)
	connectivity (T)
"""

# Read in input arguments
def parseinputlength(nmin=4,nmax=20):
	parser = argparse.ArgumentParser()
	parser.add_argument("n", help="Number of side carbons in CnMIm", type=int)
	args = parser.parse_args()

	n = args.n

	if n % 2:
		raise ValueError("Number of carbon atoms in side chain must be even.")
	if n < nmin:
		raise ValueError(f"Number of carbon atoms in side chain must be at least {nmin}.")
	if n > nmax:
		raise ValueError(f"Number of carbon atoms in side chain must be no more than {nmax}.")

	return n

### Error-checking functions ###

# Check indices of the form (i, j)
def checkidx(idx, fname):
	if len(idx) != 2:
		raise ValueError(f"idx in {fname} must be a tuple of (i,j)!")
	i,j = idx
	if not (i in range(3) and j in range(3)):
		raise ValueError(f"i and j in {fname} must be in [0,1,2]!")
	if i >= j:
		raise ValueError(f"j must be greater than i in {fname}!")

# Check that an array is of size Nx3
def checkNx3(xyz, fname):
	if xyz.shape[1] != 3:
		raise ValueError(f"xyz in {fname}() must be Nx3, but got Nx{xyz.shape[1]} instead!")

### General-use functions ###

# Determines whether a file is for GROMACS ("G") or Tinker ("T")
def filetype(fname):
	suffix = fname.split(".")[-1]
	typedict = {
		"gro": "G",
		"arc": "T",
		"txyz": "T",
		"xyz": "T",
	}
	return typedict.get(suffix, None)

# General-use file reading
def readfile(infile):
	t = filetype(infile)
	if t == "G":
		return readgro(infile)
	if t == "T":
		return readtxyz(infile)
	raise TypeError("Filetype not supported yet!")

# General-use file writing
def writefile(atoms, headers, outfile):
	t = filetype(outfile)
	if t == "G":
		return writegro(atoms, headers, outfile)
	if t == "T":
		return writetxyz(atoms, headers, outfile)
	raise TypeError("Filetype not supported yet!")

# Applies periodic boundary conditions
# ***Assumes rectangular box***
# A point at exactly box/2 will be wrapped to -box/2
def pbc(var, box):
	v = np.array(var)
	b = np.diagonal(np.array(box))
	return (v + b/2) % b - b/2

# Unwrap all atoms
def unwrap(atoms, box):
	if box is not None:
		for atom in atoms:
			atom.xyz = pbc(atom.xyz, box)
	return atoms

# Adds a delta shift to coordinates (xyz + delta*mult)
def adddeltaxyz(xyz, delta, mult):
	if len(xyz) == len(delta):
		return [xyz[i] + delta[i]*mult for i in range(len(xyz))]
	if len(delta) == 1:
		return [x + delta*mult for x in xyz]
	raise ValueError("Size mismatch in adddeltaxyz().")

# Calculate the center of mass of coordinates (assumes mass=1)
def centerofmass(xyz):
	checkNx3(xyz, "centerofmass")
	return np.sum(xyz, axis=0)/len(xyz)

# Subtract out the COM (center of mass)
def removeCOM(xyz):
	checkNx3(xyz, "removeCOM")
	return xyz - centerofmass(xyz)

# Perform PCA (principal component analysis) on coordinates
def getpca(xyz):
	checkNx3(xyz, "getpca")
	if len(xyz) == 1:
		return np.identity(3)
	return np.array(sklearn.decomposition.PCA().fit(removeCOM(xyz)).components_)

# Makes a shifted and rotated copy of coordinates
def rotcopy(xyz, shift, rot, f=1.0):
	checkNx3(xyz, "rotcopy")
	xyz = np.array(xyz)
	shift = np.array(shift)
	if shift.size != 3:
		raise ValueError(f"shift in rotcopy() must have 3 elements, but got {shift.size} instead!")
	rot = np.array(rot)
	if rot.size != 3:
		raise ValueError(f"rot in rotcopy() must have 3 elements, but got {rot.size} instead!")
	xyz -= f*shift
	R  = np.diag(rot)
	return np.append(xyz, np.dot(xyz,R), axis=0)

# Aligns planes in molecules and separates them by sep
# This is specifically for systems where refatoms form a plane and some of the
# atoms form a plane indexed with idx
def alignplanes(refatoms, atoms, sep, idx=[]):
	# WARNING: assumes idx applies to refatoms
	if not isinstance(refatoms[0],types.SimpleNamespace) or not isinstance(atoms[0],types.SimpleNamespace):
		raise TypeError("Atoms must be an array of type SimpleNamespace!")
	if idx == []:
		idx = range(len(refatoms)) # use all atoms

	rxyz = np.array([atom.xyz for atom in refatoms])
	ref = rxyz[idx,:]
	pca = getpca(ref)

	xyz = np.array([atom.xyz for atom in atoms])
	R = np.dot(getpca(xyz).T,pca) # inverse is transpose!
	xyz = np.dot(removeCOM(xyz),R) + centerofmass(ref)

	xyz1 = xyz + pca[2]*np.abs(sep) # shift in pos. orthogonal direction
	pca1 = sklearn.decomposition.PCA().fit(np.append(rxyz, xyz1, axis=0)).explained_variance_ratio_[2]
	xyz2 = xyz - pca[2]*np.abs(sep) # shift in neg. orthogonal direction
	pca2 = sklearn.decomposition.PCA().fit(np.append(rxyz, xyz2, axis=0)).explained_variance_ratio_[2]
	# We want a long configuration, meaning we want a small z-component
	return xyz1 if pca1 < pca2 else xyz2

# Aligns atoms to angle bisector of refatoms and separates them by sep
# The idx needs to be 3 indices with the angle being idx[0]-idx[1]-idx[2]
# This is designed to work with a single-atom molecule named "atoms",
# but may work with larger molecules
def alignbisector(refatoms, atoms, sep, idx=[]):
	# WARNING: assumes idx applies to refatoms
	if not isinstance(refatoms[0],types.SimpleNamespace) or not isinstance(atoms[0],types.SimpleNamespace):
		raise TypeError("Atoms must be an array of type SimpleNamespace!")
	if idx == []:
		idx = [0, 1, 2] # use first 3 atoms
	if len(idx) != 3:
		raise ValueError("Must pass 3 indices into alignbisector()")

	rxyz = np.array([atom.xyz for atom in refatoms])
	ref = rxyz[idx, :]
	
	# Create both vectors u and v from the center atom
	u = ref[0, :] - ref[1, :]
	v = ref[2, :] - ref[1, :]
	# Like a parallelogram, we can simply add the vectors
	bisector = u + v
	# We usually want to place the atom opposite the refatoms,
	# so flip the sign and normalize
	bisector = -bisector/np.linalg.norm(bisector)

	# Put atoms at distance sep away from the center atom
	xyz = np.array([atom.xyz for atom in atoms])
	xyz = removeCOM(xyz) + ref[1, :] + sep*bisector

	return xyz

### Transformation matrices
# th is a vector of 3 angles (in radians)

# Creates a matrix to shear
def shearmat(th, idx):
	checkidx(idx, "shearmat")
	S = np.eye(3)
	S[idx] = np.tan(th[idx])
	return S

# Creates a matrix to rotate
def rotmat(th, idx):
	checkidx(idx, "rotmat")
	i, j = idx
	c, s = np.cos(-th[idx]), np.sin(-th[idx])
	R = np.eye(3)
	R[i,i] = c; R[i,j] = -s
	R[j,i] = s; R[j,j] =  c
	return R

# Creates a matrix to scale
def scalemat(th, idx):
	checkidx(idx, "scalemat")
	i, j = idx
	k = 1/np.cos(th[idx])
	Q = np.eye(3)
	Q[i,i] = k
	Q[j,j] = k
	return Q

# Shears, rotates, then scales molecules
def shearrotscale(mols, th, idx):
	shear = shearmat(th, idx)
	rot = rotmat(th, idx)
	scale = scalemat(th, idx)
	M = []
	for mol in mols:
		checkNx3(mol, "shearrotscale")
		c = np.atleast_2d(centerofmass(mol)).T
		mol = mol.T
		mol = (mol-c) + np.dot(shear,c) # shear
		mol = np.dot(rot,(mol-c)) + c # rotate
		mol = (mol-c) + np.dot(scale,c) # scale
		M.append(mol.T)
	return M

### Box-related functions

def scalebox(th, idx):
	checkidx(idx, "scalebox")
	i, j = idx
	k = 1/np.cos(th[idx])
	Q = np.eye(3)
	Q[i, i] = k
	Q[j, j] = 1/k
	return Q

def replicatecell(xyz, box, dims):
	checkNx3(xyz, "replicatecell")
	checkNx3(box, "replicatecell")
	if len(dims) != 3:
		raise ValueError(f"dims in replicatecell() must have 3 values, but got {len(dims)} instead!")
	data = np.concatenate([(xyz + np.dot(box,[i,j,k])) for i in range(dims[0]) for j in range(dims[1]) for k in range(dims[2])])
	return data, box*dims

def build_unitcell(f_cntfsi, f_Li, f_out, n=18, copies=(1,1,1), chg_idx=[], th=[0,0,0], triclinic=False, units=1, fudge=1.05):
	u = units # units to convert nm to whatever (1 for nm)
	f = fudge # adds spacing to alleviate forces

	th = np.deg2rad([[0, th[0], th[1]], [0, 0, th[2]], [0, 0, 0]])

	# import data and extract coordinates
	cntfsi, headers = readfile(f_cntfsi)
	cntfsi = unwrap(cntfsi, headers.box)
	Li, headers = readfile(f_Li)
	Li = unwrap(Li, headers.box)
	for a in cntfsi+Li:
		a.xyz = [x/units for x in a.xyz] # remove native units

	xyzLi = alignbisector(cntfsi, Li, 0.3*f, idx=chg_idx)
	for i, x in enumerate(xyzLi):
		Li[i].xyz = x

	atoms = cntfsi + Li
	xyz = np.array([atom.xyz for atom in atoms])
	natoms_cntfsi = len(cntfsi)
	natoms_Li = len(Li)
	natoms_pair = natoms_cntfsi + natoms_Li
	if natoms_Li != 1: raise ValueError(f"Li should have 1 atoms, but has {natoms_Li}!")

	# perform PCA on cntfsi and align with z-axis
	# xyzcntfsi = copy.copy([atom.xyz for atom in cntfsi])
	xyzcntfsi = np.array([atom.xyz for atom in cntfsi])
	pca = getpca(xyzcntfsi)
	R = np.flip(pca.T,axis=1)
	R[:,1] *= -1
	# R = pca.T[:,[2,0,1]]
	xyz = np.dot(removeCOM(xyz),R) # apply rotation to all atoms
	
	# Define rotation and shift
	rot1 = [0.22,    0,     0.36]
	rot2 = [   0, 0.22,        0]
	zfunc = lambda i : 0.64 + 0.06*i
	# zdict = {8:1.15, 10:1.25, 12:1.35, 14:1.45, 16:1.55, 18:1.75, 20:1.85}
	rot3 = [   0,    0, zfunc(n)]

	# make rotated copies of the molecules
	xyz = rotcopy(xyz, rot1, [-1, 1,-1], f) # copy in xz
	xyz = rotcopy(xyz, rot2, [-1,-1, 1], f) # copy in xy
	xyz = rotcopy(xyz, rot3, [ 1,-1,-1], f) # copy in yz

	# set box size
	boxsize = f*np.ptp(xyz,axis=0) # max - min
	boxsize[0:2] = np.max(boxsize[0:2]) # ensure aspect ratio of unity in xy

	### Shear & Rotate & Scale ###

	# pairs of indices for transformations
	idx = [(i,j) for i in range(3) for j in range(i+1,3)]

	# apply the shear, rotation, and scaling
	boxtens = np.diag(boxsize)
	xyzm = np.reshape(xyz,(-1,natoms_pair,3))
	for i in idx:
		xyzm = shearrotscale(xyzm,th,i)
		boxtens = np.matmul(scalebox(th,i),boxtens)
		boxtens = np.matmul(shearmat(th,i),boxtens)
	xyz = np.reshape(xyzm,(-1,3))

	xyz, boxtens = replicatecell(xyz, boxtens, copies)
	xyz -= np.min(xyz, axis=0)
	xyz *= u # apply native units
	boxtens *= u

	iters = len(xyz)//natoms_pair

	### Original Ordering - commented out; need to sort atom pairs
	# create new atoms
#	outatoms = []
#	for j in range(iter):
#		for i, atom in enumerate(atoms):
#			offset = j*natoms_pair
#			idx = i + offset
#			newatom = copy.deepcopy(atom)
#			newatom.xyz = xyz[idx]
#			newatom.atnum = idx + 1
#			newatom.resnum = atom.resnum + 2*j if atom.resnum is not None else None # since we have 2 molecules
#			newatom.connectivity = [c+offset+((i>=natoms_cmi)*natoms_cntfsi) for c in atom.connectivity] if atom.connectivity is not None else []
#			outatoms += [newatom]

	### New grouped ordering: all molecules A, all molecules B
	outatoms = []
	offset_total = iters*natoms_cntfsi
	for j in range(iters):
		offset_pair = j*natoms_pair
		for i, atom in enumerate(atoms):
			idx = i + offset_pair
			pair_idx = ((atom.resnum + 2*j) - 1)//2
			newatom = copy.deepcopy(atom)
			newatom.xyz = xyz[idx]
			newatom.atnum = pair_idx*natoms_cntfsi+i+1 if i < natoms_cntfsi else offset_total + pair_idx*natoms_Li + (i-natoms_cntfsi) + 1
			newatom.resnum = pair_idx + 1 if i < natoms_cntfsi else iters + pair_idx + 1 if atom.resnum is not None else None
			newatom.connectivity = None # No implementation yet
			outatoms += [newatom]
	outatoms.sort(key=lambda x: x.atnum)

	headers.natoms = idx+1

	if not triclinic:
		boxtens = np.diag(np.diag(boxtens)) # remove off-diagonal elements
	headers.box = boxtens

	writefile(outatoms, headers, f_out)

# If you already have a pair of ions setup
# WARNING: This will align the *pair* of molecules to the z-axis!
def build_unitcell_pair(f_pair, f_out, n=18, copies=(1,1,1), chg_idx=[], th=[0,0,0], triclinic=False, units=1, fudge=1.05):
	u = units # units to convert nm to whatever (1 for nm)
	f = fudge # adds spacing to alleviate forces

	th = np.deg2rad([[0, th[0], th[1]], [0, 0, th[2]], [0, 0, 0]])

	# import data and extract coordinates
	atoms, headers = readfile(f_pair)
	atoms = unwrap(atoms, headers.box)
	for atom in atoms:
		atom.xyz = [x/units for x in atom.xyz] # remove native units

	xyz = np.array([atom.xyz for atom in atoms])
	natoms_pair = len(atoms)
	natoms_Li = 1 # HARD-CODED
	natoms_cntfsi = natoms_pair - natoms_Li

	# perform PCA on cntfsi and align with z-axis
	xyz_pca = np.array([atom.xyz for atom in atoms])
	pca = getpca(xyz_pca)
	R = np.flip(pca.T, axis=1)
	R[:,1] *= -1
	# R = pca.T[:,[2,0,1]]
	xyz = np.dot(removeCOM(xyz),R) # apply rotation to all atoms

	# Define rotation and shift
	rot1 = [0.3,    0,    -0.2]
	rot2 = [   0, 0.18,        0]
	zfunc = lambda i : 0.50 + 0.06*i
	# zdict = {8:1.15, 10:1.25, 12:1.35, 14:1.45, 16:1.55, 18:1.75, 20:1.85}
	rot3 = [   0,    0, zfunc(n)]

	# make rotated copies of the molecules
	xyz = rotcopy(xyz, rot1, [-1, 1,-1], f) # copy in xz
	xyz = rotcopy(xyz, rot2, [-1,-1, 1], f) # copy in xy
	xyz = rotcopy(xyz, rot3, [ 1,-1,-1], f) # copy in yz

	# set box size
	boxsize = f*np.ptp(xyz,axis=0) # max - min
	boxsize[0:2] = np.max(boxsize[0:2]) # ensure aspect ratio of unity in xy

	### Shear & Rotate & Scale ###

	# pairs of indices for transformations
	idx = [(i,j) for i in range(3) for j in range(i+1,3)]

	# apply the shear, rotation, and scaling
	boxtens = np.diag(boxsize)
	xyzm = np.reshape(xyz,(-1,natoms_pair,3))
	for i in idx:
		xyzm = shearrotscale(xyzm,th,i)
		boxtens = np.matmul(scalebox(th,i),boxtens)
		boxtens = np.matmul(shearmat(th,i),boxtens)
	xyz = np.reshape(xyzm,(-1,3))

	xyz, boxtens = replicatecell(xyz, boxtens, copies)
	xyz -= np.min(xyz, axis=0)
	xyz *= u # apply native units
	boxtens *= u

	iters = len(xyz)//natoms_pair

	### Original Ordering - commented out; need to sort atom pairs
	# create new atoms
#	outatoms = []
#	for j in range(iter):
#		for i, atom in enumerate(atoms):
#			offset = j*natoms_pair
#			idx = i + offset
#			newatom = copy.deepcopy(atom)
#			newatom.xyz = xyz[idx]
#			newatom.atnum = idx + 1
#			newatom.resnum = atom.resnum + 2*j if atom.resnum is not None else None # since we have 2 molecules
#			newatom.connectivity = [c+offset+((i>=natoms_cmi)*natoms_cntfsi) for c in atom.connectivity] if atom.connectivity is not None else []
#			outatoms += [newatom]

	### New grouped ordering: all molecules A, all molecules B
	outatoms = []
	offset_total = iters*natoms_cntfsi
	for j in range(iters):
		offset_pair = j*natoms_pair
		for i, atom in enumerate(atoms):
			idx = i + offset_pair
			pair_idx = ((atom.resnum + 2*j) - 1)//2
			newatom = copy.deepcopy(atom)
			newatom.xyz = xyz[idx]
			newatom.atnum = pair_idx*natoms_cntfsi+i+1 if i < natoms_cntfsi else offset_total + pair_idx*natoms_Li + (i-natoms_cntfsi) + 1
			newatom.resnum = pair_idx + 1 if i < natoms_cntfsi else iters + pair_idx + 1 if atom.resnum is not None else None
			newatom.connectivity = None # No implementation yet
			outatoms += [newatom]
	outatoms.sort(key=lambda x: x.atnum)

	headers.natoms = idx+1

	if not triclinic:
		boxtens = np.diag(np.diag(boxtens)) # remove off-diagonal elements
	headers.box = boxtens

	writefile(outatoms, headers, f_out)


################ Conversion functions ################

def txyz2gro(infile, outfile):
	atoms, headers = readtxyz(infile)
	print("Not supported, but trying anyway!")

	# idx = 0
	# prevres = ""
	# for atom in atoms:
	# 	print(f"res: {atom.resname}, attype: {atom.attype}")
	# 	atom.resname = "CMI" if len(atom.attype) == 2 else "NO3"
	# 	if atom.resname != prevres:
	# 		idx += 1
	# 	atom.resnum = idx

	writegro(atoms, headers, outfile)

def gro2txyz(infile, outfile, reffile):
	atoms, headers = readgro(infile)
	headers.box *= 10
	nnn = len(atoms)//512
	refatoms, refheaders = readtxyz(reffile)
	idxd = {i:0 for i in opls2amoeba}
	for i in range(len(atoms[::nnn])):
		shift = nnn*i
		mol = atoms[shift:shift+nnn]
		for atom in mol:
			atom.xyz = [10*x for x in atom.xyz] # nm to A
			# atom.vxyz does not exist in Tinker files
			gromacs_t = atom.attype
			idx = idxd[gromacs_t]
			idx_len = len(opls2amoeba[gromacs_t])
			tinker_t = opls2amoeba[gromacs_t][idx]
			occurrence = opls2amoeba[gromacs_t][:idx].count(tinker_t) # number of occurrence
			j = 0
			for refatom in refatoms[:nnn]:
				if refatom.attype == tinker_t:
					if j == occurrence:
						atom.atname = refatom.atname
						atom.attype = refatom.attype
						atom.atnum = refatom.atnum + shift
						atom.connectivity = [c + shift for c in refatom.connectivity] # adjust index!!!
						idxd[gromacs_t] = (idx + 1) % idx_len # add 1 to counter, wrap by length
						break # found match!
					else:
						j += 1
			if not atom.atname:
				raise LookupError(f"Could not find atom \"{atom}\" in reference file {reffile}!")
	atoms.sort(key=lambda x: x.atnum)
	writetxyz(atoms, headers, outfile)

def convert(infile, outfile, reffile=None):
	it = filetype(infile)
	ot = filetype(outfile)
	if it == ot:
		writefile(*readfile(infile), outfile)
	elif it == "G" and ot == "T":
		if reffile and filetype(reffile) != "T":
			raise TypeError("Reference file must be in Tinker format!")
		gro2txyz(infile, outfile, reffile)
	elif it == "T" and ot == "G":
		txyz2gro(infile, outfile)
	else:
		raise TypeError("Filetype conversion not supported yet!")

amoeba2opls = {
	"401": "CS",
	"402": "CS",
	"403": "CS",
	"404": "CS",
	"405": "CS",
	"406": "CS",
	"407": "CS",
	"408": "CT",
	"409": "CS",
	"410": "CS",
	"411": "CS",
	"412": "CS",
	"413": "CS",
	"414": "CA",
	"415": "CS",
	"416": "CS",
	"417": "CM",
	"418": "NA",
	"419": "NA",
	"420": "CR",
	"421": "CW",
	"422": "CW",
	"423": "HS",
	"424": "HT",
	"425": "HS",
	"426": "HS",
	"427": "HS",
	"428": "HS",
	"429": "HS",
	"430": "HS",
	"431": "HS",
	"432": "HS",
	"433": "HS",
	"434": "HS",
	"435": "HS",
	"436": "HS",
	"437": "HS",
	"438": "HA",
	"439": "HM",
	"440": "HR",
	"441": "HW",
	"442": "HW",
	"101": "NNO",
	"102": "ONO",
}

# The order matters here!
opls2amoeba = {
	"NA": ["419", "418"],
	"CW": ["422", "421"],
	"HW": ["442", "441"],
	"CR": ["420"],
	"HR": ["440"],
	"CM": ["417"],
	"HM": ["439"]*3,
	"CA": ["414"],
	"HA": ["438"]*2,
	"CS": ["416", "413", "415", "412", "411", "405", "410", "404", "409", "403", "407", "402", "406", "401"],
	"HS": [ # list comprehension to double them all up
		val for i in
		["434", "437", "435", "436", "433", "432", "427", "431", "426", "430", "425", "428", "423", "429"]
		for val in (i, i)
	],
	"CT": ["408"],
	"HT": ["424"]*3,
	"NNO": ["101"],
	"ONO": ["102"]*3,
}

for key in amoeba2opls:
	if key not in opls2amoeba[amoeba2opls[key]]:
		raise ValueError("Dictionaries aren't right...")
