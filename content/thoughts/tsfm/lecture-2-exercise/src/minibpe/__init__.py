from .impl import Tokenizer

__all__ = ['Tokenizer', 'TokenizerFast']


def __getattr__(name: str):
  if name == 'TokenizerFast':
    from ._core import Tokenizer as TokenizerFast

    return TokenizerFast
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
