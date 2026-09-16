import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { getPlatformProxy } from 'wrangler'
import type { ArenaReaderArtifact } from '../quartz/util/arena-reader'
import {
  ARENA_READER_PROFILE,
  ARENA_TWITTER_PROFILE,
  arenaReaderHash,
  arenaRenderCacheDecision,
  claimArenaRenderLease,
  keepCompleteArenaSnapshot,
  loadArenaReaderSnapshot,
  parseArenaReaderArtifact,
  publishArenaRenderState,
  readArenaRenderCache,
  saveArenaReaderSnapshot,
} from './arena-reader-cache'

const articleId = `article-v1-${'a'.repeat(64)}`

async function artifact(
  quality: 'complete' | 'partial' = 'complete',
): Promise<ArenaReaderArtifact> {
  return {
    schemaVersion: 1,
    articleId,
    snapshotId: crypto.randomUUID(),
    title: 'A saved article',
    sourceUrl: 'https://example.com/article',
    finalUrl: 'https://www.example.com/article',
    capturedAt: 1000,
    profileVersion: ARENA_READER_PROFILE,
    fingerprint: await arenaReaderHash('A saved article'),
    resources: [],
    kind: 'html',
    readerHtml: '<p>A saved article.</p>',
    quality,
    diagnostics: [],
  }
}

test('artifact validation rejects incompatible policies and broken resource references', async () => {
  const saved = await artifact()
  assert.deepEqual(parseArenaReaderArtifact(saved), saved)
  for (const profileVersion of [
    'anonymous-readability-1-purify-1',
    'anonymous-defuddle-0.19.3-purify-1',
    'twitter-oembed-1',
    ARENA_TWITTER_PROFILE,
  ]) {
    assert.deepEqual(
      parseArenaReaderArtifact({
        ...saved,
        profileVersion,
        documentHtml: '<main>Old full view</main>',
      }),
      { ...saved, profileVersion },
    )
  }
  assert.equal(parseArenaReaderArtifact({ ...saved, profileVersion: 'obsolete' }), null)
  assert.equal(parseArenaReaderArtifact({ ...saved, articleId: '../other-owner' }), null)
  assert.equal(
    parseArenaReaderArtifact({ ...saved, kind: 'pdf', resourceId: `resource-${'a'.repeat(32)}` }),
    null,
  )
  assert.equal(
    parseArenaReaderArtifact({
      ...saved,
      resources: [
        { id: `resource-${'a'.repeat(32)}`, kind: 'image', url: 'https://example.com/a.png' },
        { id: `resource-${'a'.repeat(32)}`, kind: 'image', url: 'https://example.com/b.png' },
      ],
    }),
    null,
  )
})

test('real R2 leases coalesce misses, reject stale publishers, and retain saved versions', async t => {
  const directory = await mkdtemp(path.join(tmpdir(), 'arena-reader-cache-'))
  const configPath = path.join(directory, 'wrangler.json')
  await writeFile(
    configPath,
    JSON.stringify({
      name: 'arena-reader-cache-test',
      compatibility_date: '2025-01-21',
      r2_buckets: [{ binding: 'CONTENT', bucket_name: 'arena-reader-cache-test' }],
    }),
  )
  const platform = await getPlatformProxy<{ CONTENT: R2Bucket }>({
    configPath,
    persist: false,
    remoteBindings: false,
  })
  try {
    const bucket = platform.env.CONTENT
    await t.test(
      'saved copies from the previous extractor keep their state and snapshot URLs',
      async () => {
        const legacyArticleId = `article-v1-${'b'.repeat(64)}`
        const saved = {
          ...(await artifact()),
          articleId: legacyArticleId,
          profileVersion: 'anonymous-readability-1-purify-1',
        }
        const key = `arena-reader/v1/${legacyArticleId}/anonymous-readability-1-purify-1/state.json`
        await bucket.put(
          `arena-reader/v1/${legacyArticleId}/snapshots/${saved.snapshotId}.json`,
          JSON.stringify({ ...saved, documentHtml: '<main>Old full view</main>' }),
        )
        await bucket.put(
          key,
          JSON.stringify({
            schemaVersion: 1,
            generation: 1,
            snapshotId: saved.snapshotId,
            lease: null,
            failure: null,
          }),
        )
        const cached = await readArenaRenderCache(bucket, legacyArticleId)
        assert.equal(arenaRenderCacheDecision(cached.state, 1000, false), 'ready')
        assert.equal(arenaRenderCacheDecision(cached.state, 1000, true), 'render')
        assert.deepEqual(
          await loadArenaReaderSnapshot(bucket, legacyArticleId, saved.snapshotId),
          saved,
        )
        const lease = await claimArenaRenderLease(bucket, legacyArticleId, cached, 1000)
        assert.ok(lease)
        const refreshed = { ...(await artifact()), articleId: legacyArticleId }
        assert.equal(await saveArenaReaderSnapshot(bucket, refreshed), true)
        assert.equal(
          await publishArenaRenderState(
            bucket,
            legacyArticleId,
            lease,
            refreshed.snapshotId,
            null,
            1001,
          ),
          true,
        )
        assert.equal(
          (await readArenaRenderCache(bucket, legacyArticleId)).state.snapshotId,
          refreshed.snapshotId,
        )
        assert.deepEqual(
          await loadArenaReaderSnapshot(bucket, legacyArticleId, saved.snapshotId),
          saved,
        )
      },
    )
    const initial = await readArenaRenderCache(bucket, articleId)
    assert.equal(arenaRenderCacheDecision(initial.state, 1000, false), 'render')
    const claims = await Promise.all([
      claimArenaRenderLease(bucket, articleId, initial, 1000),
      claimArenaRenderLease(bucket, articleId, initial, 1000),
    ])
    assert.equal(claims.filter(Boolean).length, 1)
    const winner = claims.find(claim => claim !== null)
    assert.ok(winner)
    const active = await readArenaRenderCache(bucket, articleId)
    assert.equal(arenaRenderCacheDecision(active.state, 1001, false), 'pending')
    assert.equal(await claimArenaRenderLease(bucket, articleId, active, 1001), null)

    await t.test('an expired lease cannot overwrite or clear its successor', async () => {
      const successor = await claimArenaRenderLease(bucket, articleId, active, winner.expiresAt + 1)
      assert.ok(successor)
      assert.equal(
        await publishArenaRenderState(bucket, articleId, winner, null, null, 1002),
        false,
      )
      assert.equal(
        (await readArenaRenderCache(bucket, articleId)).state.lease?.owner,
        successor.owner,
      )
      const saved = await artifact()
      assert.equal(await saveArenaReaderSnapshot(bucket, saved), true)
      assert.equal(
        await publishArenaRenderState(
          bucket,
          articleId,
          successor,
          saved.snapshotId,
          null,
          successor.expiresAt - 1,
        ),
        true,
      )
      assert.deepEqual(await loadArenaReaderSnapshot(bucket, articleId, saved.snapshotId), saved)
      assert.equal(
        arenaRenderCacheDecision(
          (await readArenaRenderCache(bucket, articleId)).state,
          successor.expiresAt,
          false,
        ),
        'ready',
      )
    })

    await t.test(
      'a failed refresh retains its ready pointer while imposing a cooldown',
      async () => {
        const previous = await readArenaRenderCache(bucket, articleId)
        const lease = await claimArenaRenderLease(bucket, articleId, previous, 500_000)
        assert.ok(lease)
        assert.equal(
          await publishArenaRenderState(
            bucket,
            articleId,
            lease,
            previous.state.snapshotId,
            { reason: 'blocked', message: 'Publisher returned a challenge.', retryAt: 600_000 },
            500_001,
          ),
          true,
        )
        const failed = await readArenaRenderCache(bucket, articleId)
        assert.equal(failed.state.snapshotId, previous.state.snapshotId)
        assert.equal(arenaRenderCacheDecision(failed.state, 500_002, false), 'ready')
        assert.equal(arenaRenderCacheDecision(failed.state, 500_002, true), 'cooldown')
        assert.equal(arenaRenderCacheDecision(failed.state, 600_001, true), 'render')
      },
    )

    await t.test(
      'snapshot writes are immutable and partial refreshes preserve a complete copy',
      async () => {
        const complete = await artifact()
        const partial = await artifact('partial')
        assert.equal(keepCompleteArenaSnapshot(complete, partial), true)
        assert.equal(keepCompleteArenaSnapshot(partial, complete), false)
        assert.equal(await saveArenaReaderSnapshot(bucket, complete), true)
        assert.equal(await saveArenaReaderSnapshot(bucket, complete), true)
        assert.equal(
          await saveArenaReaderSnapshot(bucket, { ...complete, fingerprint: 'b'.repeat(64) }),
          false,
        )
        assert.deepEqual(
          await loadArenaReaderSnapshot(bucket, articleId, complete.snapshotId),
          complete,
        )
        assert.equal(await loadArenaReaderSnapshot(bucket, articleId, '../state'), null)
        assert.equal(
          await loadArenaReaderSnapshot(
            bucket,
            `article-v1-${'b'.repeat(64)}`,
            complete.snapshotId,
          ),
          null,
        )
      },
    )
  } finally {
    await platform.dispose()
    await rm(directory, { recursive: true, force: true })
  }
})
