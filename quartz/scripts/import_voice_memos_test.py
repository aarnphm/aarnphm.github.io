import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path


spec = importlib.util.spec_from_file_location(
  'voice_memos', Path(__file__).with_name('import-voice-memos.py')
)
if spec is None or spec.loader is None:
  raise RuntimeError('Cannot load the voice memo importer')
memos = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memos
spec.loader.exec_module(memos)

NOW = datetime(2026, 9, 6, 18, 0, tzinfo=memos.TORONTO)
STREAM = """---
title: stream
---

## 2026-09-05

- [meta]:
  - date: 2026-09-05 18:08:48 GMT-04:00
  - tags:
    - life
    - o/training
  - description: training log 041

![[triathlon#2026-09-05#analytics]]

Training felt good today.

![[triathlon/memos/20260905.qta]]

---

## Another entry

Keep this writing.
"""


class VoiceMemoTests(unittest.TestCase):
  def test_existing_entry_and_repeat_import(self) -> None:
    updated = memos.update_stream(
      STREAM, date(2026, 9, 5), ['20260905', 'second'], NOW
    )
    expected = STREAM.replace('20260905.qta', '20260905.m4a').replace(
      '![[triathlon/memos/20260905.m4a]]',
      '![[triathlon/memos/20260905.m4a]]\n\n![[triathlon/memos/second.m4a]]',
    )
    self.assertEqual(updated, expected)
    self.assertEqual(
      memos.update_stream(
        updated, date(2026, 9, 5), ['20260905', 'second'], NOW
      ),
      updated,
    )

  def test_new_entry_has_next_number_and_preserves_previous_entries(
    self,
  ) -> None:
    updated = memos.update_stream(
      STREAM, date(2026, 9, 6), ['first', 'second'], NOW
    )
    self.assertIn('description: training log 042', updated)
    self.assertIn('![[triathlon#2026-09-06#analytics]]', updated)
    self.assertIn('date: 2026-09-06 18:00:00 GMT-04:00', updated)
    self.assertTrue(updated.endswith(STREAM.split('## 2026-09-05')[1]))
    self.assertEqual(updated.count('![[triathlon/memos/first.m4a]]'), 1)

  def test_ambiguous_training_entries_stop(self) -> None:
    with self.assertRaisesRegex(ValueError, 'Multiple training entries'):
      memos.update_stream(STREAM + STREAM, date(2026, 9, 5), ['first'], NOW)

  def test_audio_import_preserves_source_and_is_idempotent(self) -> None:
    with tempfile.TemporaryDirectory(prefix='memo-import-test-') as temporary:
      root = Path(temporary)
      source = root / '20260906_hash.qta'
      subprocess.run(
        [
          'ffmpeg',
          '-v',
          'error',
          '-f',
          'lavfi',
          '-i',
          'sine=frequency=440:duration=0.2',
          '-c:a',
          'aac',
          '-metadata',
          'creation_time=2026-09-06T02:00:00Z',
          '-f',
          'mov',
          str(source),
        ],
        check=True,
        timeout=30,
      )
      source.with_suffix('.waveform').write_bytes(b'waveform fixture')
      recording = memos.probe(source)
      self.assertEqual(recording.recorded_at.date(), date(2026, 9, 5))
      destination = root / 'content/triathlon/memos'
      name = memos.import_recording(recording, destination)
      self.assertTrue(name.startswith('20260905-220000-'))
      self.assertEqual(memos.sha256(source), recording.digest)
      self.assertEqual(
        source.with_suffix('.waveform').read_bytes(), b'waveform fixture'
      )
      self.assertEqual(
        {file.name for file in destination.iterdir()},
        {f'{name}.m4a', f'{name}.peaks.json'},
      )
      metadata = json.loads((destination / f'{name}.peaks.json').read_text())
      self.assertEqual(len(metadata['peaks']), 512)
      self.assertEqual(max(metadata['peaks']), 1)
      self.assertEqual(metadata['sourceSha256'], recording.digest)
      before = {
        file.name: file.stat().st_mtime_ns for file in destination.iterdir()
      }
      self.assertEqual(memos.import_recording(recording, destination), name)
      self.assertEqual(
        before,
        {file.name: file.stat().st_mtime_ns for file in destination.iterdir()},
      )
      stream = root / 'content/stream.md'
      stream.write_text(STREAM)
      command = [
        sys.executable,
        str(Path(memos.__file__)),
        '--date',
        '2026-09-05',
        '--repo',
        str(root),
        '--source',
        str(root),
      ]
      subprocess.run(command, check=True, capture_output=True, timeout=30)
      self.assertIn(f'![[triathlon/memos/{name}.m4a]]', stream.read_text())
      written = stream.stat().st_mtime_ns
      subprocess.run(command, check=True, capture_output=True, timeout=30)
      self.assertEqual(stream.stat().st_mtime_ns, written)
      (destination / f'{name}.m4a').write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError, 'incomplete or changed'):
        memos.import_recording(recording, destination)


if __name__ == '__main__':
  unittest.main()
