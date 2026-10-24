"""Run the SDK's bounded page/layout/recognition pipeline against MLX-VLM."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import closing
import json

from glmocr.config import GlmOcrConfig
from glmocr.layout.layout_detector import PPDocLayoutDetector
from glmocr.ocr_client import OCRClient
from PIL import Image

from cache import ExtractedPage, normalize_page
from streaming import PdfPage, PdfPipeline


class CheckedOCRClient(OCRClient):
  def process(self, request_data: dict) -> tuple[dict, int]:
    if self._session is None:
      raise RuntimeError('Start the OCR pipeline before submitting regions')
    # The SDK's process() discards finish_reason and usage. Validate the native
    # response before its workers reduce a completion to region text.
    with self._session.post(
      self.api_url,
      json={**request_data, 'model': self.model, 'enable_thinking': False},
      timeout=self.request_timeout,
    ) as reply:
      reply.raise_for_status()
      response = reply.json()
    choices = response.get('choices', [])
    tokens = response.get('usage', {}).get('completion_tokens')
    if (
      len(choices) != 1
      or choices[0].get('finish_reason') != 'stop'
      or type(tokens) is not int
      or not 0 < tokens < request_data['max_tokens']
    ):
      raise ValueError('OCR did not finish; refusing a truncated region')
    return response, 200


class CheckedLayout(PPDocLayoutDetector):
  failure: Exception | None = None

  def process(
    self,
    images: list[Image.Image],
    save_visualization: bool = False,
    global_start_idx: int = 0,
    use_polygon: bool = False,
  ) -> tuple[list[list[dict]], dict]:
    try:
      return super().process(
        images, save_visualization, global_start_idx, use_polygon
      )
    except Exception as error:
      # The SDK converts a failed layout batch into empty pages. Preserve the
      # failure so a scanned page cannot be committed as a successful blank.
      self.failure = error
      raise


class OcrClient:
  def __init__(self, config: GlmOcrConfig, concurrency: int = 8) -> None:
    if not 1 <= concurrency <= 16:
      raise ValueError('OCR concurrency must be between 1 and 16')
    config.pipeline.max_workers = concurrency
    self.layout = CheckedLayout(config.pipeline.layout)
    self.pipeline = PdfPipeline(config.pipeline, layout_detector=self.layout)
    self.pipeline.ocr_client = CheckedOCRClient(config.pipeline.ocr_api)
    self.pipeline.ocr_client.api_key = None
    self.started = False

  def close(self) -> None:
    if self.started:
      self.pipeline.stop()

  def pages(
    self, pages: Sequence[PdfPage], limit: int
  ) -> Iterator[tuple[PdfPage, ExtractedPage, list[dict], bool]]:
    if not self.started:
      self.pipeline.start()
      self.started = True
    self.layout.failure = None
    results = self.pipeline.pages(pages, limit)
    with closing(results):
      for loaded, result in results:
        if self.layout.failure is not None:
          raise self.layout.failure
        raw = result.raw_json_result
        output = result.json_result
        if isinstance(output, str):
          output = json.loads(output)
        if not isinstance(raw, list) or len(raw) != 1 or len(output) != 1:
          raise ValueError('SDK must return exactly one page per input image')
        number, label = loaded.source.number, loaded.label
        try:
          original = normalize_page(number, label, raw[0])
          normalized = normalize_page(number, label, output[0])
        except ValueError as error:
          print(
            json.dumps({
              'event': 'page-rejected',
              'source': str(loaded.source.source),
              'page': number,
              'reason': str(error),
              'regions': raw[0],
            }),
            flush=True,
          )
          raise ValueError(
            f'{loaded.source.source}: page {number}: {error}'
          ) from error
        if len(normalized.regions) != len(original.regions):
          raise ValueError('Formatting discarded a recognized region')
        yield loaded.source, normalized, raw[0], loaded.has_native_text
