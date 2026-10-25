"""Exercise durable queue clients against real PDFs and process locks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import portalocker
import pymupdf

SCRIPTS = Path(__file__).resolve().parent


class QueueClientTests(unittest.TestCase):
  def test_queue_deduplication_source_changes_and_public_boundaries(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      papers = root / 'content/thoughts/papers'
      index = root / 'public/static/searchIndex.json'
      cache = root / 'quartz/.quartz-cache/pdf-search'
      for directory in (papers, index.parent, cache):
        directory.mkdir(parents=True)
      (root / 'quartz.config.ts').write_text(
        'export default {ignorePatterns: []}'
      )
      source = papers / 'new.pdf'
      with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), 'New public paper')
        pdf.save(source)
      slug = 'thoughts/papers/new.pdf'
      index.write_text(
        json.dumps({
          slug: {'filePath': str(source), 'title': 'New'},
          'thoughts/papers/private.pdf': {
            'filePath': str(source),
            'protected': True,
          },
        })
      )

      def enqueue(*slugs: str) -> subprocess.CompletedProcess[str]:
        command = [
          sys.executable,
          str(SCRIPTS / 'queue-client.py'),
          'enqueue',
          '--root',
          str(root),
        ]
        for value in slugs:
          command.extend(['--slug', value])
        return subprocess.run(
          command, text=True, capture_output=True, timeout=20
        )

      # A real process lock keeps inference out of this queue-boundary replay.
      with portalocker.Lock(cache / '.queue-worker.lock', timeout=0):
        first = enqueue(slug)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)['queued'], 1)
        second = enqueue(slug)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)['queued'], 0)
        self.assertEqual(json.loads(second.stdout)['pending'], 1)
        for forbidden in (
          '../private.pdf',
          'thoughts/papers/private.pdf',
          'missing.pdf',
        ):
          denied = enqueue(forbidden)
          self.assertNotEqual(denied.returncode, 0)
        saved = json.loads((cache / 'queue.json').read_text())
        self.assertEqual(len(saved['jobs']), 1)
        before = saved['jobs'][0]['sourceSha256']
        with source.open('ab') as stream:
          stream.write(b'\n% source change\n')
        changed = enqueue(slug)
        self.assertEqual(changed.returncode, 0, changed.stderr)
        self.assertEqual(json.loads(changed.stdout)['queued'], 1)
        saved = json.loads((cache / 'queue.json').read_text())
        self.assertNotEqual(saved['jobs'][0]['sourceSha256'], before)
        # A publication of the previous source cannot make changed bytes complete.
        output = root / 'content/embeddings/pdf'
        output.mkdir(parents=True)
        manifest = output / 'manifest.json'
        manifest.write_text(
          json.dumps({
            'documents': {slug: {'sourceSha256': before, 'pages': [1]}}
          })
        )
        self.assertEqual(json.loads(enqueue(slug).stdout)['pending'], 1)
        with source.open('rb') as stream:
          digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest.write_text(
          json.dumps({
            'documents': {slug: {'sourceSha256': digest, 'pages': [1]}}
          })
        )
        complete = enqueue(slug)
        self.assertEqual(complete.returncode, 0, complete.stderr)
        self.assertEqual(json.loads(complete.stdout)['pending'], 0)
        self.assertEqual(json.loads(complete.stdout)['queued'], 0)
      evidence = os.environ.get('PDF_QUEUE_TEST_ARTIFACTS')
      if evidence:
        destination = Path(evidence)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'queue-client.json').write_text(
          json.dumps(
            {
              'first': json.loads(first.stdout),
              'deduplicated': json.loads(second.stdout),
              'changed': json.loads(changed.stdout),
              'complete': json.loads(complete.stdout),
              'forbiddenSlugsRejected': 3,
            },
            indent=2,
          )
        )


if __name__ == '__main__':
  unittest.main()
