import torch
from activestructopt.objective.base import BaseObjective
from activestructopt.common.registry import registry

@registry.register_objective("MAEUncertainty")
class MAEUncertainty(BaseObjective):
  def __init__(self, λ = 0.1, **kwargs) -> None:
    self.λ = λ

  def get(self, predictions: torch.Tensor, target, device = 'cpu', N = 1, ):
    maes = torch.zeros(N, device = device)
    mae_total = torch.tensor([0.0], device = device)
    for i in range(N):
      mae = torch.maximum(torch.mean(torch.abs(target - predictions[0][i])) - 
        self.λ * torch.mean(torch.abs(predictions[1][i])), torch.tensor(0.)) 
      mae_total = mae_total + mae
      maes[i] = mae.detach()
      del mae
    return maes, mae_total
