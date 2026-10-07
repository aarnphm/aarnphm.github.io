import importlib.util
import json
import os
import shutil
import sqlite3
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
  def test_recently_deleted_recording_is_excluded_from_list_and_import(
    self,
  ) -> None:
    with tempfile.TemporaryDirectory(prefix='memo-trash-test-') as temporary:
      root = Path(temporary)
      source = root / 'Recordings'
      source.mkdir()
      for name, frequency in (
        ('20260906_active.qta', 440),
        ('20260906_trash.qta', 880),
      ):
        subprocess.run(
          [
            'ffmpeg',
            '-v',
            'error',
            '-f',
            'lavfi',
            '-i',
            f'sine=frequency={frequency}:duration=0.2',
            '-c:a',
            'aac',
            '-metadata',
            'creation_time=2026-09-06T02:00:00Z',
            '-f',
            'mov',
            str(source / name),
          ],
          check=True,
          timeout=30,
        )
      with sqlite3.connect(source / 'CloudRecordings.db') as db:
        db.execute(
          'CREATE TABLE ZCLOUDRECORDING (ZPATH TEXT, ZEVICTIONDATE REAL)'
        )
        db.executemany(
          'INSERT INTO ZCLOUDRECORDING VALUES (?, ?)',
          [('20260906_active.qta', None), ('20260906_trash.qta', 1.0)],
        )
      repository = root / 'repo'
      repository.mkdir()
      (repository / 'content').mkdir()
      (repository / 'content/stream.md').write_text(STREAM)
      command = [
        sys.executable,
        str(Path(memos.__file__)),
        '--date',
        '2026-09-05',
        '--source',
        str(source),
        '--repo',
        str(repository),
      ]
      listed = subprocess.run(
        [*command, '--list'],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
      )
      self.assertIn('20260906_active.qta', listed.stdout)
      self.assertNotIn('20260906_trash.qta', listed.stdout)
      explicit = subprocess.run(
        [*command, '--file', str(source / '20260906_trash.qta'), '--list'],
        capture_output=True,
        text=True,
        timeout=30,
      )
      self.assertNotEqual(explicit.returncode, 0)
      self.assertIn('not an active Voice Memo', explicit.stderr)
      subprocess.run(command, check=True, capture_output=True, timeout=30)
      output = list(
        (repository / 'content/triathlon/memos').glob('*.peaks.json')
      )
      self.assertEqual(len(output), 1)
      self.assertEqual(
        json.loads(output[0].read_text())['source'], '20260906_active.qta'
      )

  def test_memo_attributes_replace_lfs_entries_and_preserve_other_rules(
    self,
  ) -> None:
    text = (
      '* text=auto eol=lf\n'
      '*.pdf filter=lfs diff=lfs merge=lfs -text\n'
      'content/triathlon/memos/*.m4a filter=lfs diff=lfs merge=lfs -text\n'
      'content/triathlon/memos/day.peaks.json filter=lfs diff=lfs merge=lfs -text\n'
      'content/triathlon/memos/** filter=lfs diff=lfs merge=lfs -text\n'
      'content/triathlon/memos/*.peaks.json linguist-generated=true\n'
    )
    expected = (
      '* text=auto eol=lf\n'
      '*.pdf filter=lfs diff=lfs merge=lfs -text\n'
      'content/triathlon/memos/*.peaks.json linguist-generated=true\n'
      'content/triathlon/memos/**/*.m4a filter=lfs diff=lfs merge=lfs -text\n'
      'content/triathlon/memos/**/*.peaks.json -filter diff merge text\n'
    )
    updated = memos.update_memo_attributes(text)
    self.assertEqual(updated, expected)
    self.assertEqual(memos.update_memo_attributes(updated), updated)
    self.assertEqual(
      memos.update_memo_attributes(updated + '*.json filter=lfs\n'),
      expected.split('content/triathlon/memos/**/*.m4a')[0]
      + '*.json filter=lfs\n'
      + '\n'.join(memos.MEMO_ATTRIBUTE_RULES)
      + '\n',
    )

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
      attributes = root / '.gitattributes'
      attributes.write_text(
        '*.pdf filter=lfs diff=lfs merge=lfs -text\n'
        'content/triathlon/memos/** filter=lfs diff=lfs merge=lfs -text\n'
      )
      subprocess.run(
        [*command, '--list'], check=True, capture_output=True, timeout=30
      )
      self.assertNotIn('*.peaks.json', attributes.read_text())
      subprocess.run(command, check=True, capture_output=True, timeout=30)
      self.assertIn(f'![[triathlon/memos/{name}.m4a]]', stream.read_text())
      self.assertEqual(
        attributes.read_text(),
        '*.pdf filter=lfs diff=lfs merge=lfs -text\n'
        'content/triathlon/memos/**/*.m4a filter=lfs diff=lfs merge=lfs -text\n'
        'content/triathlon/memos/**/*.peaks.json -filter diff merge text\n',
      )
      written = stream.stat().st_mtime_ns
      attributes_written = attributes.stat().st_mtime_ns
      subprocess.run(command, check=True, capture_output=True, timeout=30)
      self.assertEqual(stream.stat().st_mtime_ns, written)
      self.assertEqual(attributes.stat().st_mtime_ns, attributes_written)
      (destination / f'{name}.m4a').write_bytes(b'changed')
      with self.assertRaisesRegex(ValueError, 'incomplete or changed'):
        memos.import_recording(recording, destination)


class VoiceMemoBatchTests(unittest.TestCase):
  def setUp(self) -> None:
    temporary = tempfile.TemporaryDirectory(prefix='memo-batch-test-')
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    self.source = self.root / 'Recordings'
    self.source.mkdir()
    self.repository = self.root / 'repo'
    self.stream = self.repository / 'content/stream.md'
    self.stream.parent.mkdir(parents=True)
    self.stream.write_text(STREAM)
    self.attributes = self.repository / '.gitattributes'
    self.attributes.write_text('*.pdf filter=lfs diff=lfs merge=lfs -text\n')
    self.destination = self.repository / 'content/triathlon/memos'
    self.fixtures = [
      ('20260905_before.qta', '2026-09-05T03:59:59Z', 220),
      ('20260905_boundary.qta', '2026-09-05T04:00:00Z', 330),
      ('20260906_evening.qta', '2026-09-06T02:00:00Z', 440),
      ('20260906_next.m4a', '2026-09-06T04:00:00Z', 550),
      ('renamed.qta', '2026-09-07T16:00:00Z', 660),
      ('20260907_deleted.qta', '2026-09-07T17:00:00Z', 770),
    ]
    for name, created, frequency in self.fixtures:
      subprocess.run(
        [
          'ffmpeg',
          '-v',
          'error',
          '-f',
          'lavfi',
          '-i',
          f'sine=frequency={frequency}:duration=0.2',
          '-c:a',
          'aac',
          '-metadata',
          f'creation_time={created}',
          '-f',
          'mov',
          str(self.source / name),
        ],
        check=True,
        timeout=30,
      )
    with sqlite3.connect(self.source / 'CloudRecordings.db') as db:
      db.execute(
        'CREATE TABLE ZCLOUDRECORDING (ZPATH TEXT, ZEVICTIONDATE REAL)'
      )
      db.executemany(
        'INSERT INTO ZCLOUDRECORDING VALUES (?, ?)',
        [
          (name, 1.0 if 'deleted' in name else None)
          for name, _, _ in self.fixtures
        ],
      )
    # An exported web memo has no creation_time and must not be reimported.
    (self.source / 'exported.m4a').write_bytes(b'already exported')
    (self.source / 'exported.peaks.json').write_text('{}\n')
    self.command = [
      sys.executable,
      str(Path(memos.__file__)),
      '--source',
      str(self.source),
      '--repo',
      str(self.repository),
    ]

  def run_import(self, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
      [*self.command, *arguments], capture_output=True, text=True, timeout=30
    )

  def test_since_lists_and_imports_each_local_day_idempotently(self) -> None:
    original_sources = {
      path.name: memos.sha256(path) for path in self.source.iterdir()
    }
    listed = self.run_import('--since', '2026-09-05', '--list')
    self.assertEqual(listed.returncode, 0, listed.stderr)
    self.assertEqual(
      [line.split()[-1] for line in listed.stdout.splitlines()],
      [
        '20260905_boundary.qta',
        '20260906_evening.qta',
        '20260906_next.m4a',
        'renamed.qta',
      ],
    )
    self.assertIn('2026-09-05T00:00:00-04:00', listed.stdout)
    self.assertEqual(self.stream.read_text(), STREAM)
    self.assertFalse(self.destination.exists())
    self.assertEqual(
      self.attributes.read_text(),
      '*.pdf filter=lfs diff=lfs merge=lfs -text\n',
    )

    imported = self.run_import('--since', '2026-09-05')
    self.assertEqual(imported.returncode, 0, imported.stderr)
    updated = self.stream.read_text()
    self.assertTrue(
      updated.index('## 2026-09-07')
      < updated.index('## 2026-09-06')
      < updated.index('## 2026-09-05')
    )
    self.assertIn('description: training log 042', updated)
    self.assertIn('description: training log 043', updated)
    self.assertEqual(updated.count('description: training log'), 3)
    existing = updated[updated.index('## 2026-09-05') :]
    self.assertIn(
      STREAM.split('## 2026-09-05')[1].strip().split('\n\n---')[0], existing
    )
    self.assertTrue(updated.endswith('Keep this writing.\n'))
    self.assertEqual(len(list(self.destination.iterdir())), 8)
    for peaks in self.destination.glob('*.peaks.json'):
      metadata = json.loads(peaks.read_text())
      name = peaks.name.removesuffix('.peaks.json')
      audio = self.destination / f'{name}.m4a'
      self.assertEqual(memos.sha256(audio), metadata['audioSha256'])
      self.assertEqual(
        memos.sha256(self.source / metadata['source']),
        metadata['sourceSha256'],
      )
      self.assertEqual(len(metadata['peaks']), 512)
      day = datetime.fromisoformat(metadata['recordedAt']).date().isoformat()
      section = updated.split(f'## {day}\n')[1].split('\n## ')[0]
      self.assertEqual(section.count(f'![[triathlon/memos/{name}.m4a]]'), 1)
      self.assertIn(f'![[triathlon#{day}#analytics]]', section)
      result = subprocess.run(
        [
          'ffprobe',
          '-v',
          'error',
          '-show_entries',
          'stream=codec_name,duration',
          '-of',
          'json',
          str(audio),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
      )
      track = json.loads(result.stdout)['streams'][0]
      self.assertEqual(track['codec_name'], 'aac')
      self.assertAlmostEqual(
        float(track['duration']), metadata['duration'], places=3
      )
    before = {
      str(path.relative_to(self.repository)): (
        memos.sha256(path),
        path.stat().st_mtime_ns,
      )
      for path in self.repository.rglob('*')
      if path.is_file()
    }
    repeated = self.run_import('--since', '2026-09-05')
    self.assertEqual(repeated.returncode, 0, repeated.stderr)
    self.assertEqual(
      before,
      {
        str(path.relative_to(self.repository)): (
          memos.sha256(path),
          path.stat().st_mtime_ns,
        )
        for path in self.repository.rglob('*')
        if path.is_file()
      },
    )
    self.assertEqual(
      original_sources,
      {path.name: memos.sha256(path) for path in self.source.iterdir()},
    )
    if artifact_path := os.environ.get('VOICE_MEMO_TEST_ARTIFACTS'):
      artifact = Path(artifact_path)
      artifact.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(self.stream, artifact / 'stream.md')
      report = {
        'fixtures': [
          {'name': name, 'creationTime': created, 'frequency': frequency}
          for name, created, frequency in self.fixtures
        ],
        'list': {'command': listed.args, 'stdout': listed.stdout},
        'import': {'command': imported.args, 'stdout': imported.stdout},
        'repeat': {'command': repeated.args, 'stdout': repeated.stdout},
        'metadata': [
          json.loads(path.read_text())
          for path in sorted(self.destination.glob('*.peaks.json'))
        ],
        'sourceFilesUnchanged': True,
        'repeatPreservedBytesAndMtimes': True,
      }
      (artifact / 'report.json').write_text(
        json.dumps(report, indent=2) + '\n'
      )

  def test_since_accepts_explicit_files_across_days(self) -> None:
    result = self.run_import(
      '--since',
      '2026-09-05',
      '--file',
      str(self.source / 'renamed.qta'),
      '--file',
      str(self.source / '20260905_boundary.qta'),
      '--file',
      str(self.source / 'renamed.qta'),
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(len(list(self.destination.glob('*.m4a'))), 2)
    self.assertNotIn('## 2026-09-06', self.stream.read_text())
    self.assertEqual(self.stream.read_text().count('![[triathlon/memos/'), 3)

  def test_since_rejects_explicit_file_before_start(self) -> None:
    result = self.run_import(
      '--since',
      '2026-09-05',
      '--file',
      str(self.source / '20260905_before.qta'),
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('2026-09-04', result.stderr)
    self.assertIn('before', result.stderr)
    self.assertEqual(self.stream.read_text(), STREAM)
    self.assertFalse(self.destination.exists())

  def test_since_empty_range_leaves_repository_unchanged(self) -> None:
    result = self.run_import('--since', '2026-09-08')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('No recordings since 2026-09-08', result.stdout)
    self.assertEqual(self.stream.read_text(), STREAM)
    self.assertFalse(self.destination.exists())
    self.assertEqual(
      self.attributes.read_text(),
      '*.pdf filter=lfs diff=lfs merge=lfs -text\n',
    )

  def test_since_rejects_conflicting_and_invalid_dates(self) -> None:
    for arguments in (
      ('--since', '2026-09-05', '--date', '2026-09-06'),
      ('--since', '2026-02-30'),
    ):
      with self.subTest(arguments=arguments):
        result = self.run_import(*arguments)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.stream.read_text(), STREAM)
        self.assertFalse(self.destination.exists())

  def test_since_validates_all_training_entries_before_importing(self) -> None:
    duplicate = STREAM.replace('2026-09-05', '2026-09-07')
    original = STREAM + duplicate + duplicate
    self.stream.write_text(original)
    result = self.run_import('--since', '2026-09-05')
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(
      'Multiple training entries exist for 2026-09-07', result.stderr
    )
    self.assertEqual(self.stream.read_text(), original)
    self.assertFalse(self.destination.exists())


if __name__ == '__main__':
  unittest.main()
