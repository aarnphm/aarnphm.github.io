import ast
import importlib.abc
import importlib.util
import json
import sys
import types

import js

_quartz_notebook_modules = {}
_quartz_native_package_errors = {
    "flax": "Flax depends on JAX and jaxlib, which require a native XLA runtime outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "jax": "JAX requires jaxlib and XLA native runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "jaxlib": "jaxlib requires native XLA runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "keras": "Keras depends on native TensorFlow/JAX/PyTorch runtimes unavailable in this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "optax": "Optax depends on JAX and jaxlib, which require a native XLA runtime outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "tensorflow": "TensorFlow requires native runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "torch": "PyTorch requires native CPython/CUDA or CPU wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "torchaudio": "torchaudio depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "torchtext": "torchtext depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "torchvision": "torchvision depends on PyTorch native wheels outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
    "triton": "Triton requires native compiler/runtime support outside this browser Pyodide runtime. Use QUARTZ_NOTEBOOK_MODE=execute or a Colab/server runtime for this cell.",
}


class _QuartzNotebookLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source, filename = _quartz_notebook_modules[module.__name__]
        module.__file__ = filename
        exec(compile(source, filename, "exec"), module.__dict__)


class _QuartzNotebookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _quartz_notebook_modules:
            return importlib.util.spec_from_loader(
                fullname,
                _QuartzNotebookLoader(),
                origin=_quartz_notebook_modules[fullname][1],
            )
        return None


class _QuartzUnsupportedPackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in _quartz_native_package_errors:
            raise ImportError(_quartz_native_package_errors[root])
        return None


def __quartz_register_notebook_module(name, source, filename):
    _quartz_notebook_modules[name] = (source, filename)


if not any(isinstance(finder, _QuartzNotebookFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _QuartzNotebookFinder())
if not any(isinstance(finder, _QuartzUnsupportedPackageFinder) for finder in sys.meta_path):
    sys.meta_path.append(_QuartzUnsupportedPackageFinder())


class _QuartzDisplayObject:
    def __init__(self, mime, data):
        self.mime = mime
        self.data = data

    def _repr_mimebundle_(self, include=None, exclude=None):
        return ({self.mime: self.data, "text/plain": repr(self)}, {})


class Javascript(_QuartzDisplayObject):
    def __init__(self, data):
        super().__init__("application/javascript", data)

    def __repr__(self):
        return "<IPython.core.display.Javascript object>"


class HTML(_QuartzDisplayObject):
    def __init__(self, data):
        super().__init__("text/html", data)

    def __repr__(self):
        return "<IPython.core.display.HTML object>"


def display(*objects):
    for obj in objects:
        if hasattr(obj, "_repr_mimebundle_"):
            data, _metadata = obj._repr_mimebundle_()
            js.quartz_notebook_display(json.dumps(data))
        else:
            js.quartz_notebook_display(json.dumps({"text/plain": str(obj)}))


display_module = types.ModuleType("IPython.display")
display_module.display = display
display_module.Javascript = Javascript
display_module.HTML = HTML
ipython_module = types.ModuleType("IPython")
ipython_module.display = display_module
sys.modules["IPython"] = ipython_module
sys.modules["IPython.display"] = display_module
sys.modules["import_ipynb"] = types.ModuleType("import_ipynb")
nbimporter_module = types.ModuleType("nbimporter")
nbimporter_module.options = {"only_defs": False}
sys.modules["nbimporter"] = nbimporter_module


def __quartz_run_cell(source):
    tree = ast.parse(source, mode="exec")
    if len(tree.body) > 0 and isinstance(tree.body[-1], ast.Expr):
        expr = ast.Expression(tree.body.pop().value)
        ast.fix_missing_locations(tree)
        ast.fix_missing_locations(expr)
        exec(compile(tree, "<notebook-cell>", "exec"), globals())
        value = eval(compile(expr, "<notebook-cell>", "eval"), globals())
        if value is not None:
            display(value)
    else:
        exec(compile(tree, "<notebook-cell>", "exec"), globals())
