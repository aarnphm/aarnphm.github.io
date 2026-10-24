"""Feed selected PDF pages through the SDK with bounded, rolling admission."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from threading import BoundedSemaphore, Thread
import time

from glmocr.parser_result import PipelineResult
from glmocr.pipeline.pipeline import Pipeline
from glmocr.pipeline._state import PipelineState
from glmocr.pipeline._unit_tracker import UnitTracker
from glmocr.pipeline._workers import layout_worker, recognition_worker
from PIL import Image
import pymupdf


@dataclass(frozen=True)
class PdfPage:
  source: Path
  number: int


@dataclass
class LoadedPage:
  source: PdfPage
  image: Image.Image
  label: str
  has_native_text: bool


class PdfPipeline(Pipeline):
  def pages(
    self, pages: Sequence[PdfPage], limit: int
  ) -> Iterator[tuple[LoadedPage, PipelineResult]]:
    if not pages:
      return
    state = PipelineState(page_maxsize=2, region_maxsize=32)
    tracker = UnitTracker(len(pages))
    state.set_tracker(tracker)
    self._current_state = state
    slots = BoundedSemaphore(limit)
    loaded: dict[int, LoadedPage] = {}
    peak_pages = 0

    def load_pages() -> None:
      nonlocal peak_pages
      try:
        for index, source in enumerate(pages):
          while not state.is_shutdown:
            if slots.acquire(timeout=0.1):
              break
          if state.is_shutdown:
            return
          started = time.monotonic()
          with pymupdf.open(source.source) as pdf:
            page = pdf[source.number - 1]
            pixels = page.get_pixmap(dpi=self.page_loader.pdf_dpi, alpha=False)
            image = Image.frombytes(
              'RGB', (pixels.width, pixels.height), pixels.samples
            )
            loaded[index] = LoadedPage(
              source, image, page.get_label(), bool(page.get_text().strip())
            )
          peak_pages = max(peak_pages, len(loaded))
          state.register_page(index, index)
          state.images_dict[index] = image
          print(
            json.dumps({
              'event': 'page-admitted',
              'source': str(source.source),
              'page': source.number,
              'activePages': len(loaded),
              'renderSeconds': time.monotonic() - started,
            }),
            flush=True,
          )
          if not state.safe_put(
            state.page_queue,
            {
              'identifier': 'image',
              'page_idx': index,
              'unit_idx': index,
              'image': image,
            },
          ):
            return
          # Each selected page is a complete SDK unit. Signal its end before
          # waiting for a free slot, otherwise the final admitted page deadlocks.
          if not state.safe_put(
            state.page_queue, {'identifier': 'unit_done', 'unit_idx': index}
          ):
            return
        state.safe_put(state.page_queue, {'identifier': 'done'})
      except Exception as error:
        state.record_exception('PdfPageLoader', error)

    threads = [
      Thread(target=load_pages, daemon=True),
      Thread(
        target=layout_worker,
        args=(
          state,
          self.layout_detector,
          False,
          self.config.layout.use_polygon,
        ),
        daemon=True,
      ),
      Thread(
        target=recognition_worker,
        args=(state, self.page_loader, self.ocr_client, self.max_workers),
        daemon=True,
      ),
      Thread(target=self._health_watchdog, args=(state,), daemon=True),
    ]
    for thread in threads:
      thread.start()
    try:
      identities = [f'{page.source}#page={page.number}' for page in pages]
      indices = {identity: index for index, identity in enumerate(identities)}
      for result in self._emit_results(
        state, tracker, identities, preserve_order=False
      ):
        if len(result.original_images) != 1:
          raise ValueError('SDK result lost its physical page identity')
        index = indices[result.original_images[0]]
        page = loaded[index]
        try:
          yield page, result
        finally:
          loaded.pop(index).image.close()
          slots.release()
      state.raise_if_exceptions()
    finally:
      state.request_shutdown()
      for thread in threads:
        thread.join(timeout=10)
      for page in loaded.values():
        page.image.close()
      self._current_state = None
      print(
        json.dumps({
          'event': 'stream-closed',
          'peakActivePages': peak_pages,
          'pageLimit': limit,
        }),
        flush=True,
      )
