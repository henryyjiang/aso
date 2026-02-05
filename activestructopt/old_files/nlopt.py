# https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/

import activestructopt.old_files.nlopt as nlopt
import numpy as np
from activestructopt.common.constraints import lj_repulsion_pymatgen

gn_algs = [nlopt.GN_CRS2_LM, nlopt.GN_ESCH, nlopt.GN_ISRES, nlopt.GN_DIRECT, 
  nlopt.GN_DIRECT_L, nlopt.GN_DIRECT_L_RAND, nlopt.GN_DIRECT_L_NOSCAL, 
  nlopt.GN_DIRECT_L_RAND_NOSCAL, nlopt.GN_DIRECT_NOSCAL]

gn_alg_labels = ['CRS2 LM', 'ESCH', 'ISRES', 'DIRECT', 'DIRECT L', 
  'DIRECT L RAND', 'DIRECT L NOSCAL', 
  'DIRECT L RAND NOSCAL', 'DIRECT NOSCAL']

def run_nlopt(optfunc, args, exp, structure, N, algorithm = nlopt.GN_ISRES):
  natoms = len(structure)
  structures = []

  xstart = []
  for i in range(natoms):
    xstart.append(structure.sites[i].a)
    xstart.append(structure.sites[i].b)
    xstart.append(structure.sites[i].c)

  def modify_structure(x):
    for i in range(natoms):
      structure.sites[i].a = x[3 * i]
      structure.sites[i].b = x[3 * i + 1]
      structure.sites[i].c = x[3 * i + 2]

  def f(x, grad):
    assert not (grad.size > 0)
    modify_structure(x)
    structures.append(structure.copy())
    return np.mean((exp - optfunc(structure, **(args))) ** 2) + lj_repulsion_pymatgen(structure)

  opt = nlopt.opt(algorithm, 3 * natoms)
  opt.set_min_objective(f)
  opt.set_lower_bounds(np.zeros(3 * natoms))
  opt.set_upper_bounds(2 * np.ones(3 * natoms))
  opt.set_maxeval(N)
  modify_structure(opt.optimize(xstart))

  mses = [np.mean((exp - optfunc(s, **(args))) ** 2) for s in structures]
  return mses, structures


