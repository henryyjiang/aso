import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("MAE")
class MAE(BaseObjective):
  def __init__(self, **kwargs) -> None:
    pass

  def get(self, predictions: torch.Tensor, target, device = 'cpu', N = 1):
    maes = torch.zeros(N, device = device)
    mae_total = torch.tensor([0.0], device = device)
    for i in range(N):
      mae = torch.mean(torch.abs(target - predictions[0][i]))
      mae_total = mae_total + mae
      maes[i] = mae.detach()
      del mae
    return maes, mae_total
