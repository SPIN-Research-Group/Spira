__all__ = [
 'cuda_free_buffers', 'cuda_reset_error'
]

from spira.nn.functional import _C

def cuda_free_buffers():
  return _C.cuda_free_buffers()

def cuda_reset_error():
  _C.cuda_reset_error()