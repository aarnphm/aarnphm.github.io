"""The Modal images of modal_interp.py. Container code imports local modules by these paths."""

from __future__ import annotations

from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent

torch_image = (
  modal.Image
  .debian_slim(python_version='3.12')
  .uv_pip_install(
    'torch==2.13.0', 'transformers==5.10.1', 'safetensors', 'numpy', 'pytest'
  )
  .env({'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'})
  .add_local_file(HERE / 'gemma_torch.py', '/root/gemma_torch.py')
  .add_local_file(HERE / 'clt.py', '/root/clt.py')
  .add_local_file(HERE / 'clt_model.py', '/root/clt_model.py')
  .add_local_file(HERE / 'clt_fit.py', '/root/clt_fit.py')
  .add_local_file(HERE / 'clt_io.py', '/root/clt_io.py')
  .add_local_dir(
    HERE / 'garden_vllm', '/root/garden_vllm', ignore=['__pycache__']
  )
  # modal_interp.py imports these at module level, so every container needs them
  .add_local_file(
    HERE / 'modal_interp_images.py', '/root/modal_interp_images.py'
  )
  .add_local_file(
    HERE / 'modal_interp_engine.py', '/root/modal_interp_engine.py'
  )
)
vllm_image = (
  modal.Image
  .debian_slim(python_version='3.12')
  .uv_pip_install('vllm==0.33.0', 'fastapi[standard]>=0.115', 'pytest')
  .env({
    'TOKENIZERS_PARALLELISM': 'false',
    'HF_HUB_OFFLINE': '1',
    'VLLM_USE_V2_MODEL_RUNNER': '0',  # runner.py wraps the classic GPUModelRunner
    'VLLM_ENABLE_V1_MULTIPROCESSING': '0',  # engine core in this process: the sink is a queue
    'VLLM_CACHE_ROOT': '/vol/vllm-cache',  # torch.compile artefacts survive cold starts
  })
  .add_local_dir(
    HERE / 'garden_vllm',
    '/root/garden_vllm',
    copy=True,
    ignore=['__pycache__'],
  )
  .run_commands(
    'pip install --no-deps /root/garden_vllm'
  )  # the vllm.general_plugins entry point
  .add_local_file(HERE / 'gemma_torch.py', '/root/gemma_torch.py')
  .add_local_file(
    HERE / 'modal_interp_images.py', '/root/modal_interp_images.py'
  )
  .add_local_file(
    HERE / 'modal_interp_engine.py', '/root/modal_interp_engine.py'
  )
)
