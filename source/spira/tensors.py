__all__ = ['SparseTensor']

from typing import Tuple, Union, List

import torch

from spira.utils.helpers import as_tuple
from spira.utils.typing import ScalarOrTuple

class SparseTensor(object):

  def __init__(
      self,
      coordinates: torch.Tensor,
      features: torch.Tensor,
      stride: Union[int, Tuple[int, ...]] = 1,
      double: bool = False
      ):

    self._coordinates = coordinates
    self._features = features
    self._double = double
    self._stride = as_tuple(stride, size=3, name="strides")

  def cuda(self):
        self._coordinates = self._coordinates.cuda()
        self._features = self._features.cuda()
        return self

  
  def to(self, dtype):
    
    self._features = self._features.to(dtype=dtype)
    return self



  def contiguous(self):
    self._coordinates = self._coordinates.contiguous()
    self._features = self._features.contiguous()
    return self

  def half(self):
    self._features = self._features.half()
    return self
  
  def __add__(self, other):
     output = SparseTensor(
         coordinates=self._coordinates,
         features=self._features + other._features,
         stride=self._stride,
     )
     return output
  
  def order(self):
    from spira.nn.functional.indexing import flat_sort

    self._coordinates, index = flat_sort(self._coordinates, self._double)

    self._features = self._features[index]

    self._coordinates = self._coordinates.contiguous()
    self._features = self._features.contiguous()

    return self
