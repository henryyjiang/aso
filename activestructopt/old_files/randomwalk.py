import numpy as np
import copy

def step(structure, σr):
    atom_i = np.random.choice(range(len(structure)))
    structure.sites[atom_i].a = (structure.sites[atom_i].a + 
        σr * np.random.rand() / structure.lattice.a) % 1
    structure.sites[atom_i].b = (structure.sites[atom_i].b + 
        σr * np.random.rand() / structure.lattice.b) % 1
    structure.sites[atom_i].c = (structure.sites[atom_i].c + 
        σr * np.random.rand() / structure.lattice.c) % 1

def 𝛘2(exp, th, σ):
    return np.mean((exp - th) ** 2) / (σ ** 2)

def randomwalk(optfunc, args, exp, σ, structure, N, σr = 0.5):
    structures = []
    𝛘2s = []

    for _ in range(N):
        new_structure = step(structure, σr)
        structures.append(new_structure)
        𝛘2s.append(𝛘2(exp, optfunc(new_structure, **(args)), σ))

    return structures, 𝛘2s
