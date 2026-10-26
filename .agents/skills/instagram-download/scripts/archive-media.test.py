import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('archive-media.py')
PIXEL = base64.b64decode(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='
)


class ArchiveMediaTest(unittest.TestCase):
  def setUp(self) -> None:
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.root = Path(self.temporary.name)
    self.repo = self.root / 'garden'
    self.repo.mkdir()
    self.source = self.root / 'instagram-Example1234'
    self.source.mkdir()
    (self.source / '01.png').write_bytes(PIXEL)
    self.report = {
      'source_url': 'https://www.instagram.com/p/Example1234/?tracking=test',
      'account': 'artist.one, artist.two',
      'status': 'passed',
      'outputs': [
        {'file': '01.png', 'sha256': hashlib.sha256(PIXEL).hexdigest()}
      ],
    }
    self.save_report()

  def save_report(self) -> None:
    (self.source / 'verification.json').write_text(json.dumps(self.report))

  def run_archive(self, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
      [
        sys.executable,
        str(SCRIPT),
        '--repo',
        str(self.repo),
        '--source',
        str(self.source),
        *extra,
      ],
      capture_output=True,
      text=True,
    )

  def test_preview_is_read_only_and_apply_is_idempotent(self) -> None:
    preview = self.run_archive()
    self.assertEqual(preview.returncode, 0, preview.stderr)
    self.assertFalse((self.repo / 'content').exists())
    applied = self.run_archive('--apply')
    self.assertEqual(applied.returncode, 0, applied.stderr)
    images = self.repo / 'content/thoughts/images'
    name = (
      base64
      .urlsafe_b64encode(hashlib.sha256(PIXEL).digest())
      .decode()
      .rstrip('=')
      + '.png'
    )
    self.assertEqual((images / name).read_bytes(), PIXEL)
    manifest = images / 'instagram/attribution.json'
    original = manifest.read_bytes()
    entry = json.loads(original)[f'[[thoughts/images/{name}]]']
    self.assertEqual(entry['authors'], ['artist.one', 'artist.two'])
    self.assertEqual(
      entry['source_url'], 'https://www.instagram.com/p/Example1234/'
    )
    repeated = self.run_archive('--apply')
    self.assertEqual(repeated.returncode, 0, repeated.stderr)
    self.assertEqual(manifest.read_bytes(), original)
    self.assertTrue((self.source / '01.png').exists())
    self.assertFalse(list(images.rglob('*.md')))

  def test_failed_capture_and_wrong_hash_leave_destination_untouched(
    self,
  ) -> None:
    for change in [
      {'status': 'failed'},
      {'outputs': [{'file': '01.png', 'sha256': '0' * 64}]},
    ]:
      self.report.update(change)
      self.save_report()
      result = self.run_archive('--apply')
      self.assertNotEqual(result.returncode, 0)
      self.assertFalse((self.repo / 'content').exists())

  def test_distinct_attribution_for_identical_bytes_is_preserved(self) -> None:
    self.assertEqual(self.run_archive('--apply').returncode, 0)
    self.report['account'] = 'another.artist'
    self.report['source_url'] = 'https://www.instagram.com/p/OtherPost12/'
    self.save_report()
    result = self.run_archive('--apply')
    self.assertEqual(result.returncode, 0, result.stderr)
    manifest = self.repo / 'content/thoughts/images/instagram/attribution.json'
    entries = json.loads(manifest.read_text())
    self.assertEqual(len(entries), 2)
    self.assertEqual(
      {tuple(v['authors']) for v in entries.values()},
      {('artist.one', 'artist.two'), ('another.artist',)},
    )
    self.assertEqual(self.run_archive('--apply').returncode, 0)
    self.assertEqual(len(json.loads(manifest.read_text())), 2)

  def test_move_removes_sources_only_after_verified_archive_write(
    self,
  ) -> None:
    result = self.run_archive('--apply', '--move')
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertFalse((self.source / '01.png').exists())
    self.assertFalse((self.source / 'verification.json').exists())
    archive = self.repo / 'content/thoughts/images/instagram'
    self.assertTrue((archive / 'attribution.json').is_file())
    self.assertEqual(len(list(archive.glob('*.json'))), 2)


if __name__ == '__main__':
  unittest.main()
