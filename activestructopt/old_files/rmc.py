import numpy as np
from activestructopt.common.constraints import lj_reject

def step(structure, latticeprob, σr, σl, σθ, step_type = 'one'):
    new_struct = structure.copy()
    if np.random.rand() < latticeprob:
        lattice_step(new_struct, σl, σθ)
    else:
        positions_step(new_struct, σr, step_type = step_type)
    return new_struct

def lattice_step(structure, σl, σθ):
    structure.lattice = structure.lattice.from_parameters(
        np.maximum(0.0, structure.lattice.a + σl * np.random.randn()),
        np.maximum(0.0, structure.lattice.b + σl * np.random.randn()), 
        np.maximum(0.0, structure.lattice.c + σl * np.random.randn()), 
        structure.lattice.alpha + σθ * np.random.randn(), 
        structure.lattice.beta + σθ * np.random.randn(), 
        structure.lattice.gamma + σθ * np.random.randn()
    )

def positions_step(structure, σr, step_type = 'one'):
    if step_type == 'one':
        atom_i = np.random.choice(range(len(structure)))
        structure.sites[atom_i].a = (structure.sites[atom_i].a + 
            σr * np.random.rand() / structure.lattice.a) % 1
        structure.sites[atom_i].b = (structure.sites[atom_i].b + 
            σr * np.random.rand() / structure.lattice.b) % 1
        structure.sites[atom_i].c = (structure.sites[atom_i].c + 
            σr * np.random.rand() / structure.lattice.c) % 1
    else:
        structure.perturb(σr)

def mse(exp, th):
    return np.mean((exp - th) ** 2)

def rmc(optfunc, args, exp, structure, N, σ = 0.0025, latticeprob = 0.0, σr = 0.1, σl = 0.1, σθ = 1.0, step_type = 'one'):
    structures = [structure]
    accepts = [True]
    old_structure = structure
    old_mse = mse(exp, optfunc(old_structure, **(args)))
    mses = [old_mse]
    Δmses = [-1.]
    σs = [σ]

    for _ in range(N - 1):
        rejected = True
        while rejected:
            new_structure = step(old_structure, latticeprob, σr, σl, σθ, step_type = step_type)
            rejected = lj_reject(new_structure)
        res = optfunc(new_structure, **(args))
        new_mse = mse(exp, res)
        Δmse = new_mse - old_mse
        accept = (Δmse <= 0 or np.log(np.random.rand()) < -Δmse/(2 * σ ** 2))
        structures.append(new_structure)
        σs.append(σ)
        mses.append(new_mse)
        Δmses.append(Δmse)
        accepts.append(accept)
        if accept:
            old_structure = new_structure.copy()
            old_mse = new_mse

    return structures, mses, accepts
