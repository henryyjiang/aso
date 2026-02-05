from abc import ABC, abstractmethod
import torch

class BaseObjective(ABC):
  @abstractmethod
  def __init__(self, **kwargs) -> None:
    pass

  @abstractmethod
  def get(self, prediction: torch.Tensor, target, device = 'cpu', N = 1):
    pass
