from activestructopt.common.dataloader import prepare_data_pmg, reprocess_data
from activestructopt.common.constraints import lj_rmins, lj_repulsion, lj_reject
from activestructopt.model.base import BaseModel
from activestructopt.dataset.base import BaseDataset
from activestructopt.objective.base import BaseObjective
from activestructopt.optimizer.base import BaseOptimizer
from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from pymatgen.core.structure import IStructure
from pymatgen.core import Lattice
import torch
import numpy as np

@registry.register_optimizer("PSO")
class PSO(BaseOptimizer):
  def __init__(self) -> None:
    pass

  def run(self, model: BaseModel, dataset: BaseDataset, 
    objective: BaseObjective, sampler: BaseSampler, 
    particles: int = 10, iters: int = 50, local_steps: int = 25,
    c1: float = 1.2, c2: float = 1.2, w: float = 0.5,
    optimize_atoms: bool = True, optimize_lattice: bool = False,
    constraint_scale: float = 1.0, save_obj_values: bool = False,
    local_optimizer: str = "Adam", local_lr: float = 0.001,
    **kwargs) -> IStructure:

    
    device = model.device
    ljrmins = torch.tensor(lj_rmins, device=device)
    
    initial_structure = dataset.structures[0].copy()
    natoms = len(initial_structure)
    target = torch.tensor(dataset.target, device=device)
    
    pos_dims = 3 * natoms if optimize_atoms else 0
    cell_dims = 9 if optimize_lattice else 0
    total_dims = pos_dims + cell_dims
    
    # Each particle is a flattened representation: [cell (optional), positions (optional)]
    swarm_positions = np.zeros((particles, total_dims))
    
    for p in range(particles):
      if p == 0:
        structure = initial_structure.copy()
      else:
        structure = sampler.sample()
      
      idx = 0
      if optimize_lattice:
        cell_flat = structure.lattice.matrix.flatten()
        swarm_positions[p, idx:idx+9] = cell_flat
        idx += 9
      if optimize_atoms:
        pos_flat = np.array([site.coords for site in structure]).flatten()
        swarm_positions[p, idx:idx+pos_dims] = pos_flat
    
    swarm_velocities = np.zeros((particles, total_dims))
    
    pbest_positions = swarm_positions.copy()
    pbest_costs = np.full(particles, np.inf)
    
    gbest_position = swarm_positions[0].copy()
    gbest_cost = np.inf
    gbest_structure = initial_structure.copy()
    
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
    
    # Main PSO loop
    for iteration in range(iters):
      costs = np.zeros(particles)
      
      for p in range(particles):
        structure = initial_structure.copy()
        idx = 0
        
        if optimize_lattice:
          cell_matrix = swarm_positions[p, idx:idx+9].reshape(3, 3)
          structure.lattice = Lattice(cell_matrix)
          idx += 9
        
        if optimize_atoms:
          positions = swarm_positions[p, idx:idx+pos_dims].reshape(natoms, 3)
          for i in range(natoms):
            try:
              structure[i].coords = positions[i]
            except np.linalg.LinAlgError:
              pass
        
        # Prepare data and compute objective
        try:
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
          cost = float('inf')
        
        costs[p] = cost
        
        if save_obj_values:
          obj_vals[iteration, p] = cost
        
        if cost < pbest_costs[p]:
          pbest_costs[p] = cost
          pbest_positions[p] = swarm_positions[p].copy()
        
        if cost < gbest_cost:
          gbest_cost = cost
          gbest_position = swarm_positions[p].copy()
          gbest_structure = structure.copy()
      
      r1 = np.random.rand(particles, total_dims)
      r2 = np.random.rand(particles, total_dims)
      
      cognitive = c1 * r1 * (pbest_positions - swarm_positions)
      social = c2 * r2 * (gbest_position - swarm_positions)
      
      swarm_velocities = w * swarm_velocities + cognitive + social
      swarm_positions = swarm_positions + swarm_velocities
      swarm_positions = np.clip(swarm_positions, lower_bound, upper_bound)
      
      # TODO: Local optimization step
    
    return gbest_structure, obj_vals