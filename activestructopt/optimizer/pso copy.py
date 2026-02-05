from activestructopt.common.dataloader import prepare_data_pmg, reprocess_data
from activestructopt.common.constraints import lj_rmins, lj_repulsion, lj_reject
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure, Structure
from pymatgen.core import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.geometry import cell_to_cellpar, cellpar_to_cell
import torch
import numpy as np

import torch_sim as ts
from torch_sim.models.mace import MaceModel
from torch_sim.models.mattersim import MatterSimModel


def separate_close_atoms(atoms, min_dist=0.8):
    """Push apart atoms that are too close together."""
    positions = atoms.get_positions()
    n_atoms = len(positions)
    
    for _ in range(10):  # Max iterations
        moved = False
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff)
                if dist < min_dist and dist > 1e-10:
                    # Push atoms apart
                    direction = diff / dist
                    push = (min_dist - dist) / 2 + 0.1
                    positions[i] -= direction * push
                    positions[j] += direction * push
                    moved = True
        if not moved:
            break
    
    atoms.set_positions(positions)
    return atoms


def validate_structure_distances(atoms, min_dist=0.5):
    """Check if any atoms are too close together."""
    positions = atoms.get_positions()
    n_atoms = len(positions)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist < min_dist:
                return False
    return True


def sanitize_cell(atoms, min_length=3.0, max_length=100.0, min_angle=30.0, max_angle=150.0):
    """Ensure cell parameters are within reasonable bounds."""
    try:
        cellpar = cell_to_cellpar(atoms.cell)
        # Clamp lengths
        cellpar[:3] = np.clip(cellpar[:3], min_length, max_length)
        # Clamp angles
        cellpar[3:] = np.clip(cellpar[3:], min_angle, max_angle)
        cell = cellpar_to_cell(cellpar)
        atoms.set_cell(cell, scale_atoms=True)
    except Exception:
        pass
    return atoms


@registry.register_optimizer("PSO2")
class PSO(BaseOptimizer):
    def __init__(self) -> None:
        pass

    def run(self, model: BaseModel, dataset: BaseDataset, 
        objective: BaseObjective, sampler: BaseSampler, 
        particles: int = 10, iters: int = 50, local_steps: int = 25,
        c1: float = 1.2, c2: float = 1.2, w: float = 0.5,
        optimize_atoms: bool = True, optimize_lattice: bool = True,
        constraint_scale: float = 1.0, save_obj_values: bool = False,
        use_torchsim: bool = True, torchsim_model = None,
        torchsim_optimizer: str = "frechet_cell_fire",
        **kwargs) -> IStructure:

        device = model.device
        ljrmins = torch.tensor(lj_rmins, device=device)
        
        initial_structure = dataset.structures[0].copy()
        natoms = len(initial_structure)
        target = torch.tensor(dataset.target, device=device)
        
        # Dimension calculation
        pos_dims = 3 * natoms if optimize_atoms else 0
        cell_dims = 9 if optimize_lattice else 0
        total_dims = pos_dims + cell_dims
        
        # Initialize swarm using ASE atoms for consistency
        swarm_atoms = []  # Store ASE atoms objects
        swarm_positions = np.zeros((particles, total_dims))
        
        for p in range(particles):
            if p == 0:
                structure = initial_structure.copy()
            else:
                structure = sampler.sample()
            
            # Convert to ASE atoms
            atoms = AseAtomsAdaptor.get_atoms(structure)
            swarm_atoms.append(atoms)
            
            # Flatten to parameter vector
            idx = 0
            if optimize_lattice:
                cell_flat = atoms.get_cell().flatten()
                swarm_positions[p, idx:idx+9] = cell_flat
                idx += 9
            if optimize_atoms:
                pos_flat = atoms.get_positions().flatten()
                swarm_positions[p, idx:idx+pos_dims] = pos_flat
        
        swarm_velocities = np.zeros((particles, total_dims))
        
        # Personal and global best
        pbest_positions = swarm_positions.copy()
        pbest_costs = np.full(particles, np.inf)
        pbest_atoms = [atoms.copy() for atoms in swarm_atoms]
        
        gbest_position = swarm_positions[0].copy()
        gbest_cost = np.inf
        gbest_atoms = swarm_atoms[0].copy()
        
        obj_vals = None
        if save_obj_values:
            obj_vals = np.zeros((iters, particles))
        
        # Define bounds
        if optimize_lattice:
            lower_bound = np.concatenate([
                np.full(cell_dims, 2.0),
                np.full(pos_dims, -50.0)
            ]) if optimize_atoms else np.full(cell_dims, 2.0)
            upper_bound = np.concatenate([
                np.full(cell_dims, 100.0),
                np.full(pos_dims, 50.0)
            ]) if optimize_atoms else np.full(cell_dims, 100.0)
        else:
            lower_bound = np.full(pos_dims, -50.0)
            upper_bound = np.full(pos_dims, 50.0)
        
        # Setup torchsim optimizer function
        ts_optimizer = None
        if use_torchsim and TORCHSIM_AVAILABLE and torchsim_model is not None:
            optimizer_map = {
                "frechet_cell_fire": ts.frechet_cell_fire,
                "fire": ts.fire,
                "unit_cell_fire": ts.unit_cell_fire,
            }
            ts_optimizer = optimizer_map.get(torchsim_optimizer, ts.frechet_cell_fire)
        
        # Main PSO loop
        for iteration in range(iters):
            costs = np.zeros(particles)
            current_atoms = []
            
            # === STEP 1: Convert parameters to atoms ===
            for p in range(particles):
                atoms = swarm_atoms[p].copy()
                idx = 0
                
                if optimize_lattice:
                    cell_matrix = swarm_positions[p, idx:idx+9].reshape(3, 3)
                    try:
                        atoms.set_cell(cell_matrix, scale_atoms=False)
                    except Exception:
                        pass
                    idx += 9
                
                if optimize_atoms:
                    positions = swarm_positions[p, idx:idx+pos_dims].reshape(natoms, 3)
                    atoms.set_positions(positions)
                
                # Sanitize structure
                atoms = sanitize_cell(atoms)
                if not validate_structure_distances(atoms):
                    atoms = separate_close_atoms(atoms)
                
                current_atoms.append(atoms)
            
            # === STEP 2: Local optimization with torchsim ===
            if ts_optimizer is not None and local_steps > 0:
                try:
                    # Ensure all atoms have valid forces before optimization
                    sanitized_for_ts = []
                    for atoms in current_atoms:
                        atoms_copy = atoms.copy()
                        # Add small perturbation if needed
                        if not validate_structure_distances(atoms_copy):
                            atoms_copy = separate_close_atoms(atoms_copy)
                        sanitized_for_ts.append(atoms_copy)
                    
                    optimized_state = ts.optimize(
                        system=sanitized_for_ts,
                        model=torchsim_model,
                        optimizer=ts_optimizer,
                        autobatcher=False,
                        max_steps=local_steps
                    )
                    optimized_atoms = optimized_state.to_atoms()
                    
                    # Validate and use optimized structures
                    for i, opt_atoms in enumerate(optimized_atoms):
                        if validate_structure_distances(opt_atoms):
                            current_atoms[i] = opt_atoms
                        else:
                            # Keep original but separate close atoms
                            current_atoms[i] = separate_close_atoms(current_atoms[i])
                            
                except Exception as e:
                    print(f"Warning: torchsim optimization failed at iteration {iteration}: {e}")
                    # Fall back to just using sanitized atoms
                    for i in range(len(current_atoms)):
                        current_atoms[i] = separate_close_atoms(current_atoms[i])
            
            # === STEP 3: Evaluate objective for each particle ===
            for p in range(particles):
                atoms = current_atoms[p]
                
                # Debug: print structure info on first iteration
                if iteration == 0 and p == 0:
                    print(f"DEBUG: atoms has {len(atoms)} atoms, symbols: {atoms.get_chemical_symbols()}")
                    print(f"DEBUG: atoms cell:\n{atoms.get_cell()}")
                    print(f"DEBUG: atoms positions shape: {atoms.get_positions().shape}")
                
                try:
                    # Convert back to pymatgen for objective evaluation
                    structure = AseAtomsAdaptor.get_structure(atoms)
                    
                    if iteration == 0 and p == 0:
                        print(f"DEBUG: pymatgen structure has {len(structure)} sites")
                        print(f"DEBUG: structure lattice: {structure.lattice}")
                    
                    data = prepare_data_pmg(structure, dataset.config, pos_grad=False,
                        device=device, preprocess=True, cell_grad=False)
                    
                    with torch.inference_mode():
                        predictions = model.predict([data], prepared=True,
                            mask=dataset.simfunc.mask if hasattr(dataset.simfunc, 'mask') else None)
                    
                    _, obj_total = objective.get(predictions, target, device=device, N=1)
                    
                    # Add constraint penalty
                    obj_total = obj_total + constraint_scale * lj_repulsion(data, ljrmins)
                    obj_total = torch.nan_to_num(obj_total, nan=torch.tensor(float('inf'), device=device))
                    
                    cost = obj_total.item()
                except Exception as e:
                    if iteration == 0 and p == 0:
                        print(f"DEBUG: Objective evaluation failed: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                    cost = float('inf')
                
                costs[p] = cost
                
                if save_obj_values:
                    obj_vals[iteration, p] = cost
                
                # Update personal best
                if cost < pbest_costs[p]:
                    pbest_costs[p] = cost
                    pbest_atoms[p] = atoms.copy()
                    # Update position from optimized atoms
                    idx = 0
                    if optimize_lattice:
                        pbest_positions[p, idx:idx+9] = atoms.get_cell().flatten()
                        idx += 9
                    if optimize_atoms:
                        pbest_positions[p, idx:idx+pos_dims] = atoms.get_positions().flatten()
                
                # Update global best
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_atoms = atoms.copy()
                    idx = 0
                    if optimize_lattice:
                        gbest_position[idx:idx+9] = atoms.get_cell().flatten()
                        idx += 9
                    if optimize_atoms:
                        gbest_position[idx:idx+pos_dims] = atoms.get_positions().flatten()
            
            # === STEP 4: Update swarm positions from optimized atoms ===
            for p in range(particles):
                atoms = current_atoms[p]
                idx = 0
                if optimize_lattice:
                    swarm_positions[p, idx:idx+9] = atoms.get_cell().flatten()
                    idx += 9
                if optimize_atoms:
                    swarm_positions[p, idx:idx+pos_dims] = atoms.get_positions().flatten()
                swarm_atoms[p] = atoms
            
            # === STEP 5: PSO velocity and position update ===
            r1 = np.random.rand(particles, total_dims)
            r2 = np.random.rand(particles, total_dims)
            
            cognitive = c1 * r1 * (pbest_positions - swarm_positions)
            social = c2 * r2 * (gbest_position - swarm_positions)
            
            swarm_velocities = w * swarm_velocities + cognitive + social
            swarm_positions = swarm_positions + swarm_velocities
            swarm_positions = np.clip(swarm_positions, lower_bound, upper_bound)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best cost = {gbest_cost:.6f}, "
                      f"Mean cost = {np.mean(costs[costs < np.inf]):.6f}")
        
        # Convert best atoms back to pymatgen structure
        gbest_structure = AseAtomsAdaptor.get_structure(gbest_atoms)
        
        return gbest_structure, obj_vals