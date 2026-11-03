"""Run the SDK's bounded page/layout/recognition pipeline against MLX-VLM."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import closing
import json

from glmocr.config import GlmOcrConfig
from glmocr.layout.layout_detector import PPDocLayoutDetector
from glmocr.ocr_client import OCRClient
from glmocr.utils.image_utils import crop_image_region
from PIL import Image

from cache import ExtractedPage, normalize_page
from streaming import PdfPage, PdfPipeline
from recovery import (
  native_binary_text,
  native_region_text,
  reviewed_region_text,
  reviewed_scan_artifact,
  vector_aside,
)


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
      results, visualizations = super().process(
        images, save_visualization, global_start_idx, use_polygon
      )
      for image, regions in zip(images, results, strict=True):
        source = image.info.get('pdf_page')
        if not isinstance(source, PdfPage):
          continue
        for region in regions:
          if (
            region.get('label') not in self.label_task_mapping['text']
            and region.get('label') != 'table'
          ):
            continue
          reviewed = reviewed_region_text(
            source.source, source.number, region['bbox_2d'], None
          )
          if reviewed is not None and reviewed.get('review_recognition') == (
            'approved-transcription-before-ocr'
          ):
            region['task_type'] = 'skip'
      return results, visualizations
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
    if not pages:
      return
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
        recovered = False
        for region in raw[0]:
          if (
            region.get('label') in self.layout.label_task_mapping['text']
            or region.get('label') == 'table'
          ):
            reviewed = reviewed_region_text(
              loaded.source.source,
              number,
              region['bbox_2d'],
              region.get('content'),
            )
            if reviewed is not None:
              region['original_content'] = region.get('content')
              region.update(reviewed)
              recovered = True
              print(
                json.dumps({
                  'event': 'page-region-recovery',
                  'source': str(loaded.source.source),
                  'page': number,
                  'region': region['index'],
                  'method': reviewed['recovery'],
                }),
                flush=True,
              )
              continue
          if region.get('content') in (None, ''):
            native = None
            method = 'native-pdf-text-v1'
            if region.get('label') in self.layout.label_task_mapping['text']:
              native = native_region_text(
                loaded.source.source, number, region['bbox_2d']
              )
            elif region.get('label') in ('table', 'display_formula'):
              native = native_binary_text(
                loaded.source.source, number, region['bbox_2d']
              )
              method = 'native-binary-text-v1'
            if native is not None:
              region['content'] = native
              region['recovery'] = method
              recovered = True
              print(
                json.dumps({
                  'event': 'page-region-recovery',
                  'source': str(loaded.source.source),
                  'page': number,
                  'region': region['index'],
                  'method': method,
                }),
                flush=True,
              )
              continue
          if (
            region.get('label') in self.layout.label_task_mapping['text']
            and region.get('content') == ''
            and (
              vector_aside(loaded.source.source, number, region['bbox_2d'])
              or reviewed_scan_artifact(
                loaded.source.source, number, region['bbox_2d']
              )
            )
          ):
            region['original_label'] = region['label']
            region['label'] = 'image'
            region['recovery'] = (
              'reviewed-scan-artifact-v1'
              if reviewed_scan_artifact(
                loaded.source.source, number, region['bbox_2d']
              )
              else 'native-vector-aside-v1'
            )
            recovered = True
            print(
              json.dumps({
                'event': 'page-region-recovery',
                'source': str(loaded.source.source),
                'page': number,
                'region': region['index'],
                'method': region['recovery'],
              }),
              flush=True,
            )
            continue
          failed_text = (
            region.get('label') in self.layout.label_task_mapping['text']
            and region.get('label') not in ('algorithm', 'content')
            and region.get('content') is None
          )
          if (
            failed_text
            or region.get('label')
            in {'content', 'table', 'display_formula', 'algorithm'}
          ) and region.get('content') in (None, ''):
            # Structured regions can loop under the layout's recognition prompt.
            # Retry once with the other supported representation of the crop.
            task = 'table' if region['label'] == 'content' else 'text'
            bounds = region['bbox_2d']
            if failed_text or region['label'] in (
              'display_formula',
              'algorithm',
            ):
              # Structured boxes can clip edge glyphs. Keep a small page-bounded
              # margin for the retry and retain the detector's original bounds.
              left, top, right, bottom = bounds
              margin = 32 if failed_text else 8
              bounds = [
                max(0, left - margin),
                max(0, top - margin),
                min(1000, right + margin),
                min(1000, bottom + margin),
              ]
              region['recovery_bbox_2d'] = bounds
            print(
              json.dumps({
                'event': 'page-region-retry',
                'source': str(loaded.source.source),
                'page': number,
                'region': region['index'],
                'label': region['label'],
                'bounds': bounds,
              }),
              flush=True,
            )
            with crop_image_region(loaded.image, bounds) as crop:
              request = self.pipeline.page_loader.build_request_from_image(
                crop, task_type=task
              )
              response, _ = self.pipeline.ocr_client.process(request)
            region['content'] = response['choices'][0]['message']['content']
            region['recovery'] = {
              'content': 'contents-table-ocr-v1',
              'table': 'table-text-ocr-v1',
              'display_formula': 'formula-margin-text-ocr-v1',
              'algorithm': 'algorithm-margin-text-ocr-v1',
            }.get(region['label'], 'text-margin-ocr-v1')
            recovered = True
            print(
              json.dumps({
                'event': 'page-region-recovery',
                'source': str(loaded.source.source),
                'page': number,
                'region': region['index'],
                'method': region['recovery'],
              }),
              flush=True,
            )
            continue
          if (
            region.get('content') != ''
            or region.get('label')
            not in self.layout.label_task_mapping['text']
          ):
            continue
          with crop_image_region(loaded.image, region['bbox_2d']) as crop:
            # Tiny single-character crops can finish successfully without text.
            # Give the short edge enough pixels while bounding elongated crops.
            if max(crop.size) >= 112:
              continue
            scale = min(224 / min(crop.size), 896 / max(crop.size))
            with crop.resize(
              (round(crop.width * scale), round(crop.height * scale)),
              Image.Resampling.LANCZOS,
            ) as enlarged:
              request = self.pipeline.page_loader.build_request_from_image(
                enlarged, task_type='text'
              )
              response, _ = self.pipeline.ocr_client.process(request)
              content = response['choices'][0]['message']['content']
              method = 'small-region-min-edge-ocr-v2'
              if content is None or all(
                line.strip() in ('', '```', '```markdown')
                for line in content.splitlines()
              ):
                request = self.pipeline.page_loader.build_request_from_image(
                  enlarged, task_type='formula'
                )
                response, _ = self.pipeline.ocr_client.process(request)
                formula = response['choices'][0]['message']['content']
                symbol = (
                  (formula or '')
                  .strip()
                  .removeprefix('$$')
                  .removesuffix('$$')
                  .strip()
                )
                if len(symbol) != 1 or not symbol.isalnum():
                  raise ValueError(
                    'Tiny text recovery did not return one letter or digit'
                  )
                content = symbol
                method = 'small-region-symbol-ocr-v1'
                region['recovery_response'] = formula
          region['content'] = content
          region['recovery'] = method
          recovered = True
          print(
            json.dumps({
              'event': 'page-region-recovery',
              'source': str(loaded.source.source),
              'page': number,
              'region': region['index'],
              'method': region['recovery'],
            }),
            flush=True,
          )
        if recovered:
          formatted, _, _ = self.pipeline.result_formatter.process(raw)
          output = json.loads(formatted)
        empty_recognition = bool(raw[0]) and all(
          region.get('content') == ''
          and region.get('label') in self.layout.label_task_mapping['text']
          for region in raw[0]
        )
        if (not raw[0] or empty_recognition) and loaded.has_native_text:
          # Layout can miss captions and classify a photograph as empty text.
          # Recover the whole page only when no recognized text is replaced.
          request = self.pipeline.page_loader.build_request_from_image(
            loaded.image, task_type='text'
          )
          response, _ = self.pipeline.ocr_client.process(request)
          original_regions = raw[0]
          raw = [
            [
              {
                'index': 0,
                'label': 'text',
                'bbox_2d': [0, 0, 1000, 1000],
                'content': response['choices'][0]['message']['content'],
                'recovery': (
                  'empty-recognition-full-page-ocr-v1'
                  if empty_recognition
                  else 'empty-layout-full-page-ocr-v1'
                ),
                'original_regions': original_regions,
              }
            ]
          ]
          output = raw
          print(
            json.dumps({
              'event': 'page-layout-recovery',
              'source': str(loaded.source.source),
              'page': number,
              'method': 'full-page-ocr',
            }),
            flush=True,
          )
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
