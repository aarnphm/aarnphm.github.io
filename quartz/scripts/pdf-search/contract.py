"""Pinned OCR models and the verified server adapter contract."""

OCR_MODELS = {
  'bf16': (
    'mlx-community/GLM-OCR-bf16',
    '24f15402e83baa0a80eeeaecf5480e172abc6f2e',
  ),
  '8bit': (
    'mlx-community/GLM-OCR-8bit',
    '350a346bea43468bcd202abb833f7d32e9a17237',
  ),
}
PREVIOUS_MODEL_REVISION = OCR_MODELS['8bit'][1]
LAYOUT_MODEL = 'PaddlePaddle/PP-DocLayoutV3_safetensors'
LAYOUT_REVISION = 'e4f489dc536556fc1be02b973ba06756e0d4a2ac'
SERVER_REVISION = 'fec3f50379bb2cb0760ef3c8a68ebce9585e4d97'
MODEL_ADAPTER = 'glm-ocr-request-positions-checkpoint-v1'
PREVIOUS_RECIPE = (
  '56a3c5da71f7599ceeccbc9dc5715267ad0cc9bfad86c8e738aaaaab7fdeba8f'
)
