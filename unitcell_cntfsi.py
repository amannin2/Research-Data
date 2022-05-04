#!/usr/bin/env python3

from cntfsi_func import *

import os
folder = os.getcwd()

n = parseinputlength()

angles = [0, 0, 0]
copies = (8, 8, 1)
f = 1.035

# GROMACS
f_cntfsi = "c18tfsi_GMX.gro"
f_Li = "Li.gro"
f_pair = "c18tfsiLi_pair.gro" # both ions together
f_out = "Final_Crystal.gro"
chg_idx = [5, 8, 9] # 5, 8, 9 in .gro
triclinic = False
units = 1

# # Tinker
# f_cmi = f"{folder}/c{n:02d}mim.xyz"
# f_no3 = f"{folder}/no3.xyz"
# f_out = f"{folder}/c{n:02d}_unitcell.xyz"
# chg_idx = [1, 2, 3, 4, 5]
# triclinic = True #False
# units = 10

# Use one of the following methods, depending on how your ions are saved

# Pass ions as separate files
# build_unitcell(f_cntfsi, f_Li, f_out, n=n, copies=copies, chg_idx=chg_idx, th=angles, triclinic=triclinic, units=units, fudge=f)

# Pass pair of ions together as one file
build_unitcell_pair(f_pair, f_out, n=n, copies=copies, chg_idx=chg_idx, th=angles, triclinic=triclinic, units=units, fudge=f)




