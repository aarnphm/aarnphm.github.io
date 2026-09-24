import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  buildBlobUrl,
  joinFileLines,
  lineRangeLabel,
  lineRangeMeta,
  parseGithubBlobUrl,
  parseGithubRepositoryUrl,
  parseGithubSourceUrl,
  parseLineRange,
} from './github-embed'

describe('github embed utilities', () => {
  test('resolves file pages and raw URLs without changing ref or path segments', () => {
    const rawUrl =
      'https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/model_runner.py'
    for (const url of [
      'https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py#L20-L30',
      'https://www.github.com/vllm-project/vllm/raw/main/vllm/v1/worker/gpu/model_runner.py?raw=true',
      rawUrl,
      rawUrl.replace('raw.githubusercontent.com', 'raw.github.com'),
    ])
      assert.deepStrictEqual(parseGithubSourceUrl(url), { rawUrl, fileName: 'model_runner.py' })
    assert.deepStrictEqual(
      parseGithubSourceUrl(
        'https://github.com/owner/repo/blob/feature/reader/src/hello%20world.ts',
      ),
      {
        rawUrl: 'https://raw.githubusercontent.com/owner/repo/feature/reader/src/hello%20world.ts',
        fileName: 'hello world.ts',
      },
    )
    assert.deepStrictEqual(
      parseGithubSourceUrl('https://github.com/owner/repo/raw/refs/heads/main/file.rs'),
      {
        rawUrl: 'https://raw.githubusercontent.com/owner/repo/refs/heads/main/file.rs',
        fileName: 'file.rs',
      },
    )
    assert.deepStrictEqual(
      parseGithubSourceUrl('https://gist.githubusercontent.com/owner/123/raw/abc/file.py'),
      {
        rawUrl: 'https://gist.githubusercontent.com/owner/123/raw/abc/file.py',
        fileName: 'file.py',
      },
    )
  })

  test('source parsing excludes repository pages, credentials, and lookalike hosts', () => {
    for (const url of [
      'https://github.com/owner/repo',
      'https://github.com/owner/repo/tree/main/src',
      'https://github.com/owner/repo/issues/123',
      'https://github.com/owner/repo/blob/main',
      'https://raw.githubusercontent.com/owner/repo/main',
      'https://raw.githubusercontent.com.evil.example/owner/repo/main/file.py',
      'https://token@github.com/owner/repo/blob/main/file.py',
      'https://github.com:8080/owner/repo/blob/main/file.py',
      'https://github.com/owner/repo/blob/main/file%00.py',
      'https://github.com/owner/repo/blob/main/%ff',
      'file:///owner/repo/blob/main/file.py',
    ])
      assert.strictEqual(parseGithubSourceUrl(url), null, url)
  })
  test('recognizes repository roots without treating files, issues, or profiles as repositories', () => {
    for (const url of [
      'https://github.com/JasonGross/guarantees-based-mechanistic-interpretability',
      'http://www.github.com/JasonGross/guarantees-based-mechanistic-interpretability.git/?tab=readme-ov-file#readme',
    ])
      assert.deepStrictEqual(parseGithubRepositoryUrl(url), {
        owner: 'JasonGross',
        repo: 'guarantees-based-mechanistic-interpretability',
        url: 'https://github.com/JasonGross/guarantees-based-mechanistic-interpretability',
      })
    for (const url of [
      'https://github.com/JasonGross',
      'https://github.com/topics/interpretability',
      'https://github.com/orgs/github',
      'https://github.com/owner/repo/issues/1',
      'https://github.com/owner/repo/blob/main/README.md',
      'https://github.com/owner/repo/tree/main',
      'https://github.com.evil.example/owner/repo',
      'https://person@github.com/owner/repo',
      'https://github.com:8443/owner/repo',
      'file://github.com/owner/repo',
      'https://github.com/owner/.git',
      'not a URL',
    ])
      assert.strictEqual(parseGithubRepositoryUrl(url), null, url)
  })

  test('parses GitHub blob URLs into raw.githubusercontent.com URLs', () => {
    const ref = parseGithubBlobUrl(
      'https://github.com/FasterDecoding/SnapKV/blob/82135ce2cc60f212a9ba918467f3d9c8134e163f/snapkv/monkeypatch/llama_hijack_4_37.py#L19',
    )

    assert.deepStrictEqual(ref, {
      owner: 'FasterDecoding',
      repo: 'SnapKV',
      ref: '82135ce2cc60f212a9ba918467f3d9c8134e163f',
      filePath: 'snapkv/monkeypatch/llama_hijack_4_37.py',
      rawUrl:
        'https://raw.githubusercontent.com/FasterDecoding/SnapKV/82135ce2cc60f212a9ba918467f3d9c8134e163f/snapkv/monkeypatch/llama_hijack_4_37.py',
    })
  })

  test('parses GitHub line anchors and renders pretty-code meta', () => {
    assert.deepStrictEqual(parseLineRange('L19'), { start: 19, end: 19 })
    assert.deepStrictEqual(parseLineRange('L19C4-L24C8'), { start: 19, end: 24 })
    assert.deepStrictEqual(parseLineRange('L24-L19'), { start: 19, end: 24 })
    assert.strictEqual(lineRangeLabel({ start: 19, end: 19 }), '19')
    assert.strictEqual(lineRangeLabel({ start: 19, end: 24 }), '19-24')
    assert.strictEqual(lineRangeMeta({ start: 19, end: 24 }), 'showLineNumbers {19-24}')
  })

  test('keeps whole-file content while trimming one terminal split line', () => {
    assert.strictEqual(joinFileLines(['one', 'two', '']), 'one\ntwo')
    assert.strictEqual(joinFileLines(['one', 'two']), 'one\ntwo')
  })

  test('rebuilds GitHub blob URLs with the original anchor text', () => {
    const ref = parseGithubBlobUrl(
      'https://github.com/aarnphm/garden/blob/main/quartz/util/path.ts',
    )
    assert.ok(ref)
    assert.strictEqual(
      buildBlobUrl(ref, 'L10-L12'),
      'https://github.com/aarnphm/garden/blob/main/quartz/util/path.ts#L10-L12',
    )
  })
})
