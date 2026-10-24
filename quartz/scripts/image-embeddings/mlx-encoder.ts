import { execFile, spawn } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import sharp from 'sharp'
import { isRecord } from '../../util/type-guards'
import { prepareImage } from './prepare-image'
import { IMAGE_ENCODER_CONTRACT, normalizeImageVector, type ImageEncoderContract } from './store'

const workerPath = fileURLToPath(new URL('./mlx-encoder.py', import.meta.url))
const execFileAsync = promisify(execFile)

interface MlxMetadata {
  modelId: string
  modelRevision: string
  mlxVersion: string
  transformersVersion: string
  mlxVlmVersion: string
  pillowVersion: string
}

function readMetadata(value: unknown): MlxMetadata {
  if (
    !isRecord(value) ||
    typeof value.modelId !== 'string' ||
    typeof value.modelRevision !== 'string' ||
    typeof value.mlxVersion !== 'string' ||
    typeof value.transformersVersion !== 'string' ||
    typeof value.mlxVlmVersion !== 'string' ||
    typeof value.pillowVersion !== 'string'
  )
    throw new Error('Invalid local MLX encoder metadata.')
  return {
    modelId: value.modelId,
    modelRevision: value.modelRevision,
    mlxVersion: value.mlxVersion,
    transformersVersion: value.transformersVersion,
    mlxVlmVersion: value.mlxVlmVersion,
    pillowVersion: value.pillowVersion,
  }
}

export async function mlxImageEncoderContract(): Promise<ImageEncoderContract> {
  const { stdout } = await execFileAsync('uv', ['run', workerPath, '--metadata'], {
    timeout: 120000,
    maxBuffer: 2 ** 20,
  })
  const metadata = readMetadata(JSON.parse(stdout))
  return {
    ...IMAGE_ENCODER_CONTRACT,
    repo: metadata.modelId,
    modelRevision: metadata.modelRevision,
    dtype: 'bf16',
    device: 'mlx',
    preprocessing: {
      ...IMAGE_ENCODER_CONTRACT.preprocessing,
      processorRevision: metadata.modelRevision,
      transformerVersion: metadata.transformersVersion,
      mlxVersion: metadata.mlxVersion,
      mlxVlmVersion: metadata.mlxVlmVersion,
      pillowVersion: metadata.pillowVersion,
    },
  }
}

export async function createMlxImageEncoder(contract: ImageEncoderContract) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'garden-image-rgb-'))
  const process = spawn('uv', ['run', workerPath], { stdio: ['pipe', 'pipe', 'inherit'] })
  const lines = createInterface({ input: process.stdout })
  let nextId = 0
  let failure: Error | undefined
  let disposed = false
  const pending = new Map<
    number,
    {
      resolve(vector: Float32Array): void
      reject(error: Error): void
      timer: ReturnType<typeof setTimeout>
    }
  >()
  let resolveReady: () => void = () => {}
  let rejectReady: (error: Error) => void = () => {}
  const ready = new Promise<void>((resolve, reject) => {
    resolveReady = resolve
    rejectReady = reject
  })
  const startupTimer = setTimeout(
    () => fail(new Error('Local MLX image encoder startup exceeded four minutes.')),
    240000,
  )
  function fail(error: Error): void {
    if (failure || disposed) return
    failure = error
    clearTimeout(startupTimer)
    rejectReady(error)
    for (const request of pending.values()) {
      clearTimeout(request.timer)
      request.reject(error)
    }
    pending.clear()
    process.kill()
  }
  lines.on('line', line => {
    try {
      const message: unknown = JSON.parse(line)
      if (!isRecord(message)) throw new Error('Invalid local image encoder response.')
      if (message.ready === true) {
        const actual = readMetadata(message)
        if (actual.modelRevision !== contract.modelRevision || actual.modelId !== contract.repo)
          throw new Error('Local MLX encoder revision changed after cache selection.')
        clearTimeout(startupTimer)
        resolveReady()
        return
      }
      if (typeof message.id !== 'number')
        throw new Error('Local image encoder response lacks a request identity.')
      const request = pending.get(message.id)
      if (!request) return
      clearTimeout(request.timer)
      if (typeof message.error === 'string') request.reject(new Error(message.error))
      else if (
        Array.isArray(message.vector) &&
        message.vector.every((value): value is number => typeof value === 'number')
      )
        request.resolve(normalizeImageVector(message.vector))
      else request.reject(new Error('Local image encoder response lacks a vector.'))
      pending.delete(message.id)
    } catch (error) {
      fail(error instanceof Error ? error : new Error(String(error)))
    }
  })
  process.on('error', fail)
  process.stdin.on('error', fail)
  process.on('exit', (code, signal) =>
    fail(new Error(`Local image encoder exited (${code ?? signal}).`)),
  )
  try {
    await ready
  } catch (error) {
    lines.close()
    await rm(directory, { recursive: true, force: true })
    throw error
  }
  async function request(
    payload: { kind: 'image'; path: string } | { kind: 'query'; text: string },
  ): Promise<Float32Array> {
    if (failure) throw failure
    if (disposed) throw new Error('Local image encoder has been disposed.')
    const id = ++nextId
    return new Promise<Float32Array>((resolve, reject) => {
      const timer = setTimeout(
        () => fail(new Error('Local image encoding exceeded four minutes.')),
        240000,
      )
      pending.set(id, { resolve, reject, timer })
      process.stdin.write(`${JSON.stringify({ id, ...payload })}\n`)
    })
  }
  return {
    async image(source: Uint8Array): Promise<Float32Array> {
      const file = path.join(directory, `${nextId + 1}.png`)
      try {
        const { data, info } = await prepareImage(source)
        await sharp(data, { raw: { width: info.width, height: info.height, channels: 3 } })
          .png()
          .toFile(file)
        return await request({ kind: 'image', path: file })
      } finally {
        await rm(file, { force: true })
      }
    },
    query(text: string): Promise<Float32Array> {
      return request({ kind: 'query', text })
    },
    async dispose(): Promise<void> {
      disposed = true
      process.stdin.end()
      process.kill()
      lines.close()
      await rm(directory, { recursive: true, force: true })
    },
  }
}
