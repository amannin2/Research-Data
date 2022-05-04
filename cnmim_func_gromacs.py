import numpy as np
import types

### GROMACS-specific functions ###

def boxtens2gro(box):
	return [box[0,0], box[1,1], box[2,2], box[1,0], box[2,0], box[0,1], box[2,1], box[0,2], box[1,2]]

def gro2boxtens(gbox):
	if len(gbox) == 3:
		box = np.diag(gbox)
	else:
		box = np.array([
			[gbox[0], gbox[5], gbox[7]],
			[gbox[3], gbox[1], gbox[8]],
			[gbox[4], gbox[6], gbox[2]],
		])
	return box

def test_boxconv():
	box = np.triu(np.random.rand(3, 3))
	if not np.allclose(box, gro2boxtens(boxtens2gro(box))):
		raise ValueError("Something has gone wrong in box conversion! (GROMACS)")

def str2gro(str):
	# Check for box size!
	if len(str.split()) in (3,9):
		try:
			return [float(i) for i in str.split()]
		except ValueError:
			pass

	data = types.SimpleNamespace()
	data.resnum = int(str[0:5])
	data.resname = str[5:10].strip()
	data.attype = str[10:15].strip()
	data.atnum = int(str[15:20])
	data.atname = None
	xyzvvv = str[20:].split()
	data.xyz = [float(xyzvvv[i]) for i in range(0,3)]
	if len(xyzvvv) == 6:
		data.vxyz = [float(xyzvvv[i]) for i in range(3,6)]
	data.connectivity = []
	return data

def gro2str(data):
	def pos_f(name, i):
		return f"{{{name}[{i}]:8.3f}}"
		# return "{"+name+"["+str(i)+"]:8.3f}"
	int_f  = "{:5d}"
	str_f  = "{:>5.5s}"
	info   = [int_f, str_f, str_f, int_f]
	gro_fx = "".join(  info  +[pos_f("x", i) for i in range(3)]) # no velocities
	gro_fv = "".join([gro_fx]+[pos_f("v", i) for i in range(3)]) # with velocities
	box_f  = "  {:10.6f}"

	if isinstance(data, list):
		if len(data) in (3,9):
			return "".join([box_f.format(d) for d in data])
	elif isinstance(data, types.SimpleNamespace):
		if hasattr(data, "vxyz"):
			return gro_fv.format(
				data.resnum,
				data.resname,
				data.attype,
				data.atnum,
				x=data.xyz,
				v=data.vxyz
			)
		else:
			return gro_fx.format(
				data.resnum,
				data.resname,
				data.attype,
				data.atnum,
				x=data.xyz
			)
	else:
		raise TypeError("Expected a list or types. SimpleNamespace, but got "+str(type(data))+"!")

def readgro(infile):
	headers = types.SimpleNamespace()
	atoms = []
	with open(infile, "r") as inf:
		headers.title = inf.readline().rstrip("\n")
		headers.natoms = int(inf.readline().rstrip("\n"))
		line = inf.readline()
		while line:
			data = str2gro(line.rstrip("\n"))
			if isinstance(data, list):
				headers.box = gro2boxtens(data)
			else:
				atoms.append(data)
			line = inf.readline()
	return atoms, headers

def writegro(atoms, headers, outfile):
	if not isinstance(headers, types.SimpleNamespace):
		raise TypeError("Headers must be of type SimpleNamespace!")
	if not isinstance(atoms[0], types.SimpleNamespace):
		raise TypeError("Atoms must be an array of type SimpleNamespace!")

	with open(outfile,"w") as outf:
		outf.write(f"{headers.title}\n{headers.natoms}\n")
		for atom in atoms:
			outf.write(f"{gro2str(atom)}\n")
		outf.write(f"{gro2str(boxtens2gro(headers.box))}\n")
