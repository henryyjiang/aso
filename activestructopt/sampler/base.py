from abc import ABC, abstractmethod
from pymatgen.core.structure import IStructure

class BaseSampler(ABC):
  @abstractmethod
  def __init__(self, initial_structure: IStructure, **kwargs) -> None:
    pass

  @abstractmethod
  def sample(self) -> IStructure:
    pass
