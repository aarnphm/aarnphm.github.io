from __future__ import annotations

import ast as _quartz_ast
import importlib.abc
import importlib.util
import json
import math
import posixpath
import shlex
import sys
import time
import traceback
import types

import js

_quartz_notebook_modules = {}
_quartz_native_package_errors = {
  'flax': 'Flax depends on JAX and jaxlib, which require a native XLA runtime outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'jaxlib': 'jaxlib requires native XLA runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'keras': 'Keras depends on native TensorFlow/JAX/PyTorch runtimes unavailable in this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'optax': 'Optax depends on JAX and jaxlib, which require a native XLA runtime outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'tensorflow': 'TensorFlow requires native runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'torchaudio': 'torchaudio depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'torchtext': 'torchtext depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'torchvision': 'torchvision depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
  'triton': 'Triton requires native compiler/runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.',
}


class _QuartzNotebookLoader(importlib.abc.Loader):
  def create_module(self, spec):
    return None

  def exec_module(self, module):
    source, filename = _quartz_notebook_modules[module.__name__]
    module.__file__ = filename
    exec(compile(source, filename, 'exec'), module.__dict__)


class _QuartzNotebookFinder(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname in _quartz_notebook_modules:
      return importlib.util.spec_from_loader(
        fullname, _QuartzNotebookLoader(), origin=_quartz_notebook_modules[fullname][1]
      )
    return None


class _QuartzUnsupportedPackageFinder(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    root = fullname.split('.', 1)[0]
    if root in _quartz_native_package_errors:
      raise ImportError(_quartz_native_package_errors[root])
    return None


def __quartz_register_notebook_module(name, source, filename):
  _quartz_notebook_modules[name] = (source, filename)


if not any(isinstance(finder, _QuartzNotebookFinder) for finder in sys.meta_path):
  sys.meta_path.insert(0, _QuartzNotebookFinder())
if not any(isinstance(finder, _QuartzUnsupportedPackageFinder) for finder in sys.meta_path):
  sys.meta_path.append(_QuartzUnsupportedPackageFinder())


def _shape(value):
  return [value] if isinstance(value, int) else list(value)


def _dtype_name(dtype):
  return str(dtype) if dtype is not None else None


def _descriptor(value):
  if isinstance(value, JaxArray):
    return {'kind': 'tensor', 'id': value._quartz_ml_id}
  if value is None:
    return {'kind': 'none'}
  if isinstance(value, bool):
    return {'kind': 'scalar', 'value': value}
  if isinstance(value, (int, float, str)):
    return {'kind': 'scalar', 'value': value}
  if isinstance(value, tuple):
    return {'kind': 'tuple', 'items': [_descriptor(item) for item in value]}
  if isinstance(value, list):
    return {'kind': 'list', 'items': [_descriptor(item) for item in value]}
  if isinstance(value, dict):
    return {'kind': 'dict', 'items': {str(key): _descriptor(item) for key, item in value.items()}}
  if hasattr(value, 'tolist'):
    return _descriptor(value.tolist())
  raise TypeError(f'unsupported notebook ML value: {type(value).__name__}')


def _leaf_descriptor(value):
  descriptor = _descriptor(value)
  if descriptor['kind'] in ('dict',):
    raise TypeError('notebook ML array operations require array or scalar leaves')
  return json.dumps(descriptor)


def __quartz_ml_describe_tree(value):
  return json.dumps(_descriptor(value))


def __quartz_ml_wrap_tree(serialized):
  return _wrap_tree(json.loads(str(serialized)))


def _wrap_tree(descriptor):
  kind = descriptor.get('kind')
  if kind == 'tensor':
    return JaxArray(descriptor['id'])
  if kind == 'scalar':
    return descriptor.get('value')
  if kind == 'none':
    return None
  if kind == 'list':
    return [_wrap_tree(item) for item in descriptor['items']]
  if kind == 'tuple':
    return tuple(_wrap_tree(item) for item in descriptor['items'])
  if kind == 'dict':
    return {key: _wrap_tree(item) for key, item in descriptor['items'].items()}
  raise TypeError(f'unsupported notebook ML descriptor: {kind}')


js.quartz_notebook_ml_register_callbacks(__quartz_ml_wrap_tree, __quartz_ml_describe_tree)


def _wrap_id(value_id):
  return JaxArray(str(value_id))


def _wrap_json(serialized):
  return _wrap_tree(json.loads(str(serialized)))


def _binary(name, left, right):
  return _wrap_id(js.quartz_notebook_ml_binary(name, _leaf_descriptor(left), _leaf_descriptor(right)))


def _unary(name, value, **options):
  return _wrap_id(js.quartz_notebook_ml_unary(name, _leaf_descriptor(value), json.dumps(options)))


def _reduce(name, value, axis=None, keepdims=False):
  return _wrap_id(js.quartz_notebook_ml_reduce(name, _leaf_descriptor(value), json.dumps(axis), bool(keepdims)))


def _index_descriptor(index):
  if not isinstance(index, tuple):
    index = (index,)
  parts = []
  for item in index:
    if item is None:
      parts.append({'kind': 'none'})
    elif item is Ellipsis:
      parts.append({'kind': 'ellipsis'})
    elif isinstance(item, JaxArray):
      parts.append({'kind': 'tensor', 'id': item._quartz_ml_id})
    elif isinstance(item, int):
      parts.append({'kind': 'int', 'value': item})
    elif isinstance(item, slice):
      parts.append({'kind': 'slice', 'start': item.start, 'stop': item.stop, 'step': item.step})
    else:
      parts.append({'kind': 'tensor', 'id': asarray(item)._quartz_ml_id})
  return json.dumps(parts)


class _AtSelection:
  def __init__(self, array, index):
    self.array = array
    self.index = index

  def set(self, value):
    return _wrap_id(
      js.quartz_notebook_ml_at_set(self.array._quartz_ml_id, _index_descriptor(self.index), _leaf_descriptor(value))
    )


class _AtIndexer:
  def __init__(self, array):
    self.array = array

  def __getitem__(self, index):
    return _AtSelection(self.array, index)


class JaxArray:
  __array_priority__ = 1000

  def __init__(self, value_id):
    self._quartz_ml_id = str(value_id)

  @property
  def shape(self):
    return tuple(json.loads(str(js.quartz_notebook_ml_shape(self._quartz_ml_id))))

  @property
  def dtype(self):
    return str(js.quartz_notebook_ml_dtype(self._quartz_ml_id))

  @property
  def ndim(self):
    return int(js.quartz_notebook_ml_ndim(self._quartz_ml_id))

  @property
  def size(self):
    return int(js.quartz_notebook_ml_size(self._quartz_ml_id))

  @property
  def T(self):
    return transpose(self)

  @property
  def at(self):
    return _AtIndexer(self)

  def __repr__(self):
    return f'Array({self.tolist()}, dtype={self.dtype})'

  def __len__(self):
    if not self.shape:
      raise TypeError('len() of unsized object')
    return self.shape[0]

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def __getitem__(self, index):
    return _wrap_id(js.quartz_notebook_ml_slice(self._quartz_ml_id, _index_descriptor(index)))

  def __array__(self, dtype=None):
    value = self.tolist()
    if dtype is None:
      return value
    return value

  def __float__(self):
    return float(self.item())

  def __int__(self):
    return int(self.item())

  def __bool__(self):
    return bool(self.item())

  def __neg__(self):
    return _unary('neg', self)

  def __add__(self, other):
    return _binary('add', self, other)

  def __radd__(self, other):
    return _binary('add', other, self)

  def __sub__(self, other):
    return _binary('sub', self, other)

  def __rsub__(self, other):
    return _binary('sub', other, self)

  def __mul__(self, other):
    return _binary('mul', self, other)

  def __rmul__(self, other):
    return _binary('mul', other, self)

  def __truediv__(self, other):
    return _binary('div', self, other)

  def __rtruediv__(self, other):
    return _binary('div', other, self)

  def __pow__(self, other):
    return _binary('pow', self, other)

  def __rpow__(self, other):
    return _binary('pow', other, self)

  def __matmul__(self, other):
    return _binary('matmul', self, other)

  def __rmatmul__(self, other):
    return _binary('matmul', other, self)

  def __lt__(self, other):
    return _binary('lt', self, other)

  def __le__(self, other):
    return _binary('le', self, other)

  def __gt__(self, other):
    return _binary('gt', self, other)

  def __ge__(self, other):
    return _binary('ge', self, other)

  def __eq__(self, other):
    return _binary('eq', self, other)

  def __ne__(self, other):
    return _binary('ne', self, other)

  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = tuple(shape[0])
    return _wrap_id(js.quartz_notebook_ml_reshape(self._quartz_ml_id, json.dumps(list(shape))))

  def transpose(self, axes=None):
    return transpose(self, axes)

  def swapaxes(self, axis1, axis2):
    return swapaxes(self, axis1, axis2)

  def astype(self, dtype):
    return _unary('astype', self, dtype=_dtype_name(dtype))

  def squeeze(self, axis=None):
    return _unary('squeeze', self, axis=axis)

  def sum(self, axis=None, keepdims=False):
    return _reduce('sum', self, axis, keepdims)

  def mean(self, axis=None, keepdims=False):
    return _reduce('mean', self, axis, keepdims)

  def block_until_ready(self):
    js.quartz_notebook_ml_unary('block', _leaf_descriptor(self), '{}')
    return self

  def addressable_data(self, index):
    if index != 0:
      raise IndexError('browser notebook runtime has one addressable WebGPU device')
    return self

  def tolist(self):
    return json.loads(str(js.quartz_notebook_ml_tojson.dumps(self._quartz_ml_id)))

  def item(self):
    return js.quartz_notebook_ml_item(self._quartz_ml_id)

  def to(self, *args, **kwargs):
    return self


Tensor = JaxArray


def array(values, dtype=None):
  if isinstance(values, JaxArray):
    return values.astype(dtype) if dtype is not None else values
  return _wrap_id(js.quartz_notebook_ml_array(json.dumps(values), _dtype_name(dtype)))


def asarray(values, dtype=None):
  return array(values, dtype)


def zeros(shape, dtype=None):
  return _wrap_id(js.quartz_notebook_ml_zeros(json.dumps(_shape(shape)), _dtype_name(dtype)))


def ones(shape, dtype=None):
  return _wrap_id(js.quartz_notebook_ml_ones(json.dumps(_shape(shape)), _dtype_name(dtype)))


def full(shape, fill_value, dtype=None):
  return _wrap_id(
    js.quartz_notebook_ml_full(json.dumps(_shape(shape)), _leaf_descriptor(fill_value), _dtype_name(dtype))
  )


def arange(start, stop=None, step=None, dtype=None):
  return _wrap_id(js.quartz_notebook_ml_arange(start, stop, step, _dtype_name(dtype)))


def zeros_like(value, dtype=None):
  return _wrap_id(js.quartz_notebook_ml_like('zeros', _leaf_descriptor(value), _dtype_name(dtype)))


def ones_like(value, dtype=None):
  return _wrap_id(js.quartz_notebook_ml_like('ones', _leaf_descriptor(value), _dtype_name(dtype)))


def full_like(value, fill_value, dtype=None):
  return _wrap_id(
    js.quartz_notebook_ml_full(
      json.dumps(list(asarray(value).shape)), _leaf_descriptor(fill_value), _dtype_name(dtype or asarray(value).dtype)
    )
  )


def sin(value):
  return _unary('sin', value)


def cos(value):
  return _unary('cos', value)


def tanh(value):
  return _unary('tanh', value)


def exp(value):
  return _unary('exp', value)


def log(value):
  return _unary('log', value)


def sqrt(value):
  return _unary('sqrt', value)


def floor(value):
  return _unary('floor', value)


def maximum(left, right):
  return _binary('max', left, right)


def matmul(left, right):
  return _binary('matmul', left, right)


def dot(left, right):
  return matmul(left, right)


def mean(value, axis=None, keepdims=False):
  return _reduce('mean', value, axis, keepdims)


def sum(value, axis=None, keepdims=False):
  return _reduce('sum', value, axis, keepdims)


def transpose(value, axes=None):
  return _unary('transpose', value, axes=None if axes is None else list(axes))


def swapaxes(value, axis1, axis2):
  return _wrap_id(js.quartz_notebook_ml_swapaxes(asarray(value)._quartz_ml_id, axis1, axis2))


def split(value, indices_or_sections, axis=0):
  return [
    _wrap_id(item)
    for item in json.loads(
      str(js.quartz_notebook_ml_split(asarray(value)._quartz_ml_id, json.dumps(indices_or_sections), axis))
    )
  ]


def tril(value, k=0):
  return _wrap_id(js.quartz_notebook_ml_tril(asarray(value)._quartz_ml_id, k))


def where(cond, left, right):
  return _wrap_id(js.quartz_notebook_ml_where(_leaf_descriptor(cond), _leaf_descriptor(left), _leaf_descriptor(right)))


def take(value, indices, axis=None):
  return _wrap_id(js.quartz_notebook_ml_take(asarray(value)._quartz_ml_id, _leaf_descriptor(indices), axis))


def take_along_axis(value, indices, axis):
  return take(value, indices, axis)


def squeeze(value, axis=None):
  return asarray(value).squeeze(axis)


def relu(value):
  return _unary('relu', value)


def sigmoid(value):
  return _unary('sigmoid', value)


def gelu(value, approximate=True):
  return _unary('gelu', value, approximate=approximate is not False)


def softmax(value, axis=-1):
  return _unary('softmax', value, axis=axis)


def log_softmax(value, axis=-1):
  return _unary('log_softmax', value, axis=axis)


def one_hot(value, num_classes, dtype='float32'):
  indices = asarray(value).tolist()

  def encode(item):
    if isinstance(item, list):
      return [encode(child) for child in item]
    row = [0.0] * int(num_classes)
    row[int(item)] = 1.0
    return row

  return array(encode(indices), dtype=dtype)


def PRNGKey(seed=0):
  return _wrap_id(js.quartz_notebook_ml_random_key(seed))


def random_split(key, num=2):
  return _wrap_id(js.quartz_notebook_ml_random_split(asarray(key)._quartz_ml_id, json.dumps(num)))


def random_normal(key, shape=None, dtype=None):
  return _wrap_id(
    js.quartz_notebook_ml_random_normal(
      asarray(key)._quartz_ml_id, json.dumps(_shape(shape or ())), _dtype_name(dtype)
    )
  )


def random_uniform(key, shape=None, minval=0.0, maxval=1.0, dtype=None):
  return _wrap_id(
    js.quartz_notebook_ml_random_uniform(
      asarray(key)._quartz_ml_id, json.dumps(_shape(shape or ())), minval, maxval, _dtype_name(dtype)
    )
  )


def random_randint(key, shape, minval, maxval, dtype='int32'):
  return _wrap_id(
    js.quartz_notebook_ml_random_randint(
      asarray(key)._quartz_ml_id, json.dumps(_shape(shape)), minval, maxval, _dtype_name(dtype)
    )
  )


def tree_map(fn, tree, *rest):
  if isinstance(tree, dict):
    return {key: tree_map(fn, tree[key], *(other[key] for other in rest)) for key in tree}
  if isinstance(tree, list):
    return [tree_map(fn, item, *(other[index] for other in rest)) for index, item in enumerate(tree)]
  if isinstance(tree, tuple):
    return tuple(tree_map(fn, item, *(other[index] for other in rest)) for index, item in enumerate(tree))
  return fn(tree, *rest)


def _transform_args(args):
  return json.dumps(_descriptor(tuple(args)))


def jit_function(fn=None, **options):
  def decorate(inner):
    def wrapped(*args, **kwargs):
      if kwargs:
        raise TypeError('keyword arguments are unavailable in browser notebook jit')
      return _wrap_json(js.quartz_notebook_ml_jit(inner, _transform_args(args), json.dumps(options)))

    return wrapped

  return decorate(fn) if fn is not None else decorate


def grad_function(fn, **options):
  def wrapped(*args, **kwargs):
    if kwargs:
      raise TypeError('keyword arguments are unavailable in browser notebook grad')
    return _wrap_json(js.quartz_notebook_ml_grad(fn, _transform_args(args), json.dumps(options)))

  return wrapped


def value_and_grad(fn, **options):
  def wrapped(*args, **kwargs):
    if kwargs:
      raise TypeError('keyword arguments are unavailable in browser notebook value_and_grad')
    return tuple(_wrap_json(js.quartz_notebook_ml_value_and_grad(fn, _transform_args(args), json.dumps(options))))

  return wrapped


def make_jaxpr(fn):
  def wrapped(*args, **kwargs):
    if kwargs:
      raise TypeError('keyword arguments are unavailable in browser notebook make_jaxpr')
    return js.quartz_notebook_ml_make_jaxpr(fn, _transform_args(args))

  return wrapped


def block_until_ready(value):
  return _wrap_json(js.quartz_notebook_ml_block_until_ready(json.dumps(_descriptor(value))))


def device_put(value, device=None):
  if isinstance(value, JaxArray):
    return value
  if isinstance(value, dict):
    return {key: device_put(item, device) for key, item in value.items()}
  if isinstance(value, list):
    return [device_put(item, device) for item in value]
  if isinstance(value, tuple):
    return tuple(device_put(item, device) for item in value)
  return asarray(value)


def devices():
  return json.loads(str(js.quartz_notebook_ml_devices()))


class Mesh:
  def __init__(self, devices, axis_names):
    self.devices = devices
    self.axis_names = axis_names

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, traceback):
    return False


class PartitionSpec(tuple):
  def __new__(cls, *parts):
    return tuple.__new__(cls, parts)


class NamedSharding:
  def __init__(self, mesh, spec):
    self.mesh = mesh
    self.spec = spec


class _MeshContext:
  def __init__(self, mesh):
    self.mesh = mesh

  def __enter__(self):
    return self.mesh

  def __exit__(self, exc_type, exc, traceback):
    return False


def set_mesh(mesh):
  return _MeshContext(mesh)


def stop_gradient(value):
  return value


class TorchCuda:
  def is_available(self):
    return False

  def synchronize(self):
    return None

  def empty_cache(self):
    return None


class TorchModule:
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def forward(self, *args, **kwargs):
    raise NotImplementedError('torch.nn.Module.forward is not implemented')

  def parameters(self):
    return []

  def to(self, *args, **kwargs):
    return self


def _torch_shape(args):
  if len(args) == 1 and isinstance(args[0], (tuple, list)):
    return args[0]
  return args


def torch_randn(*shape, dtype=None, device=None):
  return random_normal(PRNGKey(int(time.time() * 1000) % 2147483647), _torch_shape(shape), dtype=dtype)


def torch_compile(fn=None, **options):
  return jit_function(fn, **options) if fn is not None else lambda inner: jit_function(inner, **options)


def set_float32_matmul_precision(value):
  return None


def _install_module(name, attrs):
  module = types.ModuleType(name)
  for key, value in attrs.items():
    setattr(module, key, value)
  sys.modules[name] = module
  return module


jnp_module = _install_module(
  'jax.numpy',
  {
    'Array': JaxArray,
    'array': array,
    'asarray': asarray,
    'arange': arange,
    'zeros': zeros,
    'ones': ones,
    'full': full,
    'zeros_like': zeros_like,
    'ones_like': ones_like,
    'full_like': full_like,
    'matmul': matmul,
    'dot': dot,
    'mean': mean,
    'sum': sum,
    'sqrt': sqrt,
    'tanh': tanh,
    'exp': exp,
    'log': log,
    'sin': sin,
    'cos': cos,
    'floor': floor,
    'maximum': maximum,
    'split': split,
    'tril': tril,
    'where': where,
    'swapaxes': swapaxes,
    'transpose': transpose,
    'take': take,
    'take_along_axis': take_along_axis,
    'squeeze': squeeze,
    'float32': 'float32',
    'float64': 'float64',
    'float16': 'float16',
    'int32': 'int32',
    'uint32': 'uint32',
    'bool_': 'bool',
    'bool': 'bool',
    'pi': math.pi,
  },
)
random_module = _install_module(
  'jax.random',
  {
    'PRNGKey': PRNGKey,
    'key': PRNGKey,
    'split': random_split,
    'normal': random_normal,
    'uniform': random_uniform,
    'randint': random_randint,
  },
)
nn_module = _install_module(
  'jax.nn',
  {
    'relu': relu,
    'sigmoid': sigmoid,
    'gelu': gelu,
    'softmax': softmax,
    'log_softmax': log_softmax,
    'logSoftmax': log_softmax,
    'one_hot': one_hot,
    'oneHot': one_hot,
  },
)
tree_util_module = _install_module('jax.tree_util', {'tree_map': tree_map})
lax_module = _install_module('jax.lax', {'stop_gradient': stop_gradient})
sharding_module = _install_module(
  'jax.sharding', {'Mesh': Mesh, 'NamedSharding': NamedSharding, 'PartitionSpec': PartitionSpec}
)
jax_module = _install_module(
  'jax',
  {
    'Array': JaxArray,
    'numpy': jnp_module,
    'random': random_module,
    'nn': nn_module,
    'tree_util': tree_util_module,
    'lax': lax_module,
    'sharding': sharding_module,
    'jit': jit_function,
    'grad': grad_function,
    'value_and_grad': value_and_grad,
    'valueAndGrad': value_and_grad,
    'block_until_ready': block_until_ready,
    'device_put': device_put,
    'devices': devices,
    'make_jaxpr': make_jaxpr,
    'set_mesh': set_mesh,
  },
)
jax_module.__path__ = []
torch_cuda = TorchCuda()
torch_nn_functional = _install_module('torch.nn.functional', {'relu': relu, 'gelu': gelu})
torch_nn = _install_module('torch.nn', {'Module': TorchModule, 'functional': torch_nn_functional})
torch_nn.__path__ = []
torch_module = _install_module(
  'torch',
  {
    'Tensor': JaxArray,
    'nn': torch_nn,
    'cuda': torch_cuda,
    'float32': 'float32',
    'randn': torch_randn,
    'zeros': lambda *shape, dtype=None, device=None: zeros(_torch_shape(shape), dtype=dtype),
    'ones': lambda *shape, dtype=None, device=None: ones(_torch_shape(shape), dtype=dtype),
    'compile': torch_compile,
    'set_float32_matmul_precision': set_float32_matmul_precision,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'exp': exp,
    'sqrt': sqrt,
    'matmul': matmul,
  },
)
torch_module.__path__ = []
torch_cuda_module = _install_module(
  'torch.cuda',
  {
    'is_available': torch_cuda.is_available,
    'synchronize': torch_cuda.synchronize,
    'empty_cache': torch_cuda.empty_cache,
  },
)
torch_module.cuda = torch_cuda_module


class _QuartzDisplayObject:
  def __init__(self, mime, data):
    self.mime = mime
    self.data = data

  def _repr_mimebundle_(self, include=None, exclude=None):
    return ({self.mime: self.data, 'text/plain': repr(self)}, {})


class Javascript(_QuartzDisplayObject):
  def __init__(self, data):
    super().__init__('application/javascript', data)

  def __repr__(self):
    return '<IPython.core.display.Javascript object>'


class HTML(_QuartzDisplayObject):
  def __init__(self, data):
    super().__init__('text/html', data)

  def __repr__(self):
    return '<IPython.core.display.HTML object>'


def display(*objects):
  for obj in objects:
    if hasattr(obj, '_repr_mimebundle_'):
      data, _metadata = obj._repr_mimebundle_()
      js.quartz_notebook_display(json.dumps(data))
    else:
      js.quartz_notebook_display(json.dumps({'text/plain': str(obj)}))


display_module = types.ModuleType('IPython.display')
display_module.display = display
display_module.Javascript = Javascript
display_module.HTML = HTML
ipython_module = types.ModuleType('IPython')
ipython_module.display = display_module
sys.modules['IPython'] = ipython_module
sys.modules['IPython.display'] = display_module
sys.modules['import_ipynb'] = types.ModuleType('import_ipynb')
nbimporter_module = types.ModuleType('nbimporter')
nbimporter_module.options = {'only_defs': False}
sys.modules['nbimporter'] = nbimporter_module


def _format_timeit(seconds):
  if seconds < 1e-6:
    return f'{seconds * 1e9:.3g} ns'
  if seconds < 1e-3:
    return f'{seconds * 1e6:.3g} us'
  if seconds < 1:
    return f'{seconds * 1e3:.3g} ms'
  return f'{seconds:.3g} s'


def __quartz_timeit(statement, global_ns, local_ns, number=None, repeat=None):
  runs = 10 if number is None else int(number)
  repeats = 3 if repeat is None else int(repeat)
  code = compile(statement, '<timeit>', 'exec')
  durations = []
  for _ in range(repeats):
    started = time.perf_counter()
    for _ in range(runs):
      exec(code, global_ns, local_ns)
    durations.append(time.perf_counter() - started)
  best = min(durations) / max(runs, 1)
  print(f'{_format_timeit(best)} per loop (best of {repeats}, {runs} loops each)')


def __quartz_notebook_path(filename):
  path = posixpath.normpath(str(filename).strip())
  if path in {'', '.'} or path == '..' or path.startswith('/') or path.startswith('../') or '/../' in path:
    raise ValueError(f'notebook file path is unavailable: {filename}')
  return path


def __quartz_writefile(filename, content, append=False):
  path = __quartz_notebook_path(filename)
  mode = 'a' if append else 'w'
  with open(path, mode, encoding='utf-8') as file:
    file.write(str(content))
  action = 'Appending to' if append else 'Writing'
  print(f'{action} {path}')


def __quartz_shell(command):
  words = shlex.split(str(command), comments=True)
  if not words:
    return
  if words[0] == 'cat':
    if len(words) == 1:
      raise ValueError('cat requires a file')
    for raw_path in words[1:]:
      if raw_path.startswith('-'):
        raise ValueError('cat options are unavailable in the browser runtime')
      path = __quartz_notebook_path(raw_path)
      with open(path, 'r', encoding='utf-8') as file:
        print(file.read(), end='')
    return
  raise ValueError(f'shell command {words[0]} is unavailable in the browser runtime')


def __quartz_run_cell(
  source,
  _parse=_quartz_ast.parse,
  _expression=_quartz_ast.Expression,
  _expr=_quartz_ast.Expr,
  _fix_missing_locations=_quartz_ast.fix_missing_locations,
  _compile=compile,
  _eval=eval,
  _globals=globals,
  _display=display,
  _json_dumps=json.dumps,
  _format_exc=traceback.format_exc,
  _python_error=js.quartz_notebook_python_error,
):
  try:
    tree = _parse(source, mode='exec')
    if len(tree.body) > 0 and isinstance(tree.body[-1], _expr):
      expr = _expression(tree.body.pop().value)
      _fix_missing_locations(tree)
      _fix_missing_locations(expr)
      exec(_compile(tree, '<notebook-cell>', 'exec'), _globals())
      value = _eval(_compile(expr, '<notebook-cell>', 'eval'), _globals())
      if value is not None:
        _display(value)
    else:
      exec(_compile(tree, '<notebook-cell>', 'exec'), _globals())
  except BaseException as exc:
    _python_error(_json_dumps({'ename': exc.__class__.__name__, 'evalue': str(exc), 'traceback': _format_exc()}))
    raise
