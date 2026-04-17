__all__ = ['ScalarOrIterable', 'ScalarOrTuple']

from typing import TypeVar, Union, Tuple, Iterable

T = TypeVar('T')
ScalarOrTuple = Union[T, Tuple[T, ...]]

ScalarOrIterable = Union[T, Iterable[T]]

