import numpy as np
import types

### Tinker-specific functions ###

def boxtens2tinker(box):
	return boxtens2lattice(box)

def lattice2boxtens(latticed):
	A = latticed["A"]
	B = latticed["B"]
	C = latticed["C"]
	alpha = latticed["alpha"]
	beta = latticed["beta"]
	gamma = latticed["gamma"]
	box = np.zeros((3,3))
	box[0,0] = A
	box[0,1] = B*np.cos(np.radians(gamma))
	box[0,2] = C*np.cos(np.radians(beta))
	box[1,1] = B*np.sin(np.radians(gamma))
	box[1,2] = (B*C*np.cos(np.radians(alpha)) - box[0,1]*box[0,2])/box[1,1]
	box[2,2] = np.sqrt(C**2 - box[0,2]**2 - box[1,2]**2)
	return box

def boxtens2lattice(box):
	if not np.allclose(box, np.triu(box)):
		raise ValueError(f"Box tensor must be upper triangular!")
	A = np.linalg.norm(box[:,0])
	B = np.linalg.norm(box[:,1])
	C = np.linalg.norm(box[:,2])
	alpha = np.degrees(np.arccos((box[0,1]*box[0,2] + box[1,1]*box[1,2])/(B*C)))
	beta = np.degrees(np.arccos(box[0,2]/C))
	gamma = np.degrees(np.arccos(box[0,1]/B))
	V = A*B*box[2,2]*np.sin(np.radians(gamma))
	return {"A":A, "B":B, "C":C, "alpha":alpha, "beta":beta, "gamma":gamma, "V":V}

def test_boxconv():
	box = np.triu(np.random.rand(3, 3))
	if not np.allclose(box, lattice2boxtens(boxtens2lattice(box))):
		raise ValueError("Something has gone wrong in box conversion! (Tinker)")

def str2txyz(str):
	fields = str.split()

	# Check for box size!
	if len(fields) in (3,6):
		try:
			lattice = [float(i) for i in fields]
			latticed = {
				"A": lattice[0],
				"B": lattice[1],
				"C": lattice[2],
				"alpha": lattice[3],
				"beta": lattice[4],
				"gamma": lattice[5],
			}
			return latticed
		except:
			pass

	data = types.SimpleNamespace()
	data.atnum = int(fields[0])
	data.atname = fields[1]
	data.xyz = [float(x) for x in fields[2:5]]
	data.attype = fields[5]
	data.connectivity = [int(x) for x in fields[6:]]
	data.resnum = None
	data.resname = None
	return data

def txyz2str(data):
	txyz_f = "{:6d} {:>2s}    {:10.6f}  {:10.6f}  {:10.6f}   {:>5s}{}"
	box_f = "  {:10.6f}"

	if isinstance(data, dict):
		adata = [
			data.get("A", 0),
			data.get("B", 0),
			data.get("C", 0),
			data.get("alpha", 90),
			data.get( "beta", 90),
			data.get("gamma", 90),
		]
		return "".join([box_f.format(x) for x in adata])
	elif isinstance(data, types.SimpleNamespace):
		connect_str = " ".join(["{:6d}".format(c) for c in data.connectivity])
		return txyz_f.format(
			data.atnum,
			data.atname,
			data.xyz[0],
			data.xyz[1],
			data.xyz[2],
			data.attype,
			connect_str,
		)
	else:
		raise TypeError(f"Expected a dict or types.SimpleNamespace, but got {type(data)}!")

def readtxyz(infile):
	"""
	Reads a Tinker file and returns atoms and headers data.
	"""
	headers = types.SimpleNamespace()
	headers.box = None # most Tinker files do not have box info
	atoms = []
	with open(infile, "r") as inf:
		header = inf.readline().rstrip("\n").split(maxsplit=1)
		headers.natoms = int(header[0])
		headers.title = header[-1] if len(header) > 1 else ""
		line = inf.readline()
		while line:
			data = str2txyz(line.rstrip("\n"))
			if isinstance(data, dict):
				headers.box = lattice2boxtens(data)
			else:
				atoms.append(data)
			line = inf.readline()
	return atoms, headers

def writetxyz(atoms, headers, outfile):
	"""
	Writes a Tinker file from atoms and headers data.
	"""
	if not isinstance(headers, types.SimpleNamespace):
		raise TypeError("Headers must be of type SimpleNamespace!")
	if not isinstance(atoms[0], types.SimpleNamespace):
		raise TypeError("Atoms must be an array of type SimpleNamespace!")

	with open(outfile,"w") as outf:
		outf.write(f"{headers.natoms:6d}    {headers.title}\n")
		if headers.box != []:
			outf.write(f"{txyz2str(boxtens2tinker(headers.box))}\n")
		for atom in atoms:
			outf.write(f"{txyz2str(atom)}\n")
