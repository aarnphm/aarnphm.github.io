import { constants, promises } from 'fs'
import http from 'http'
import { styleText } from 'node:util'
import path from 'path'
import serveHandler from 'serve-handler'
import { WebSocket, WebSocketServer } from 'ws'
import { handleLocalLeanVerifyNodeRequest } from '../../util/lean-local-verify.ts'

export function createReloadServer(port) {
  const server = new WebSocketServer({ port })
  server.on('connection', socket => {
    socket.on('error', () => socket.terminate())
  })
  return {
    server,
    refresh() {
      for (const socket of server.clients) {
        if (socket.readyState === WebSocket.OPEN) socket.send('rebuild')
      }
    },
  }
}

export function createPreviewServer(argv, buildMutex) {
  return http.createServer(async (req, res) => {
    if (argv.baseDir && !req.url?.startsWith(argv.baseDir)) {
      console.log(
        styleText(
          'red',
          `[404] ${req.url} (warning: link outside of site, this is likely a Quartz bug)`,
        ),
      )
      res.writeHead(404)
      res.end()
      return
    }

    // Watch builds stage full output and replace individual files atomically. Requests can keep
    // reading the last successful output while the build mutex is held.
    if (!argv.watch) {
      const release = await buildMutex.acquire()
      if (res.destroyed) {
        release()
        return
      }
      res.once('finish', release)
      res.once('close', release)
    }

    // strip baseDir prefix
    req.url = req.url?.slice(argv.baseDir.length)

    const serve = async () => {
      const snapshots = new Map()
      const handles = new Map()
      let closed = false
      const closeReads = () => {
        if (closed) return
        closed = true
        void Promise.allSettled(
          [...snapshots.values()].map(pending =>
            pending.then(snapshot => snapshot.handle?.close()),
          ),
        )
      }
      res.once('finish', closeReads)
      res.once('close', closeReads)
      const snapshot = fp => {
        let pending = snapshots.get(fp)
        if (!pending) {
          pending = (async () => {
            const info = await promises.lstat(fp)
            if (!info.isFile()) return { info }
            if (closed) throw new Error('Preview response closed')
            const handle = await promises.open(fp, constants.O_RDONLY | constants.O_NOFOLLOW)
            try {
              const info = await handle.stat()
              handles.set(fp, handle)
              return { info, handle }
            } catch (error) {
              await handle.close()
              throw error
            }
          })()
          snapshots.set(fp, pending)
        }
        return pending
      }
      // Pin metadata and bytes to the same open inode while publication replaces its pathname.
      const reads = {
        async lstat(fp) {
          return (await snapshot(fp)).info
        },
        createReadStream(fp, options) {
          const handle = handles.get(fp)
          if (!handle) throw new Error(`Missing preview read snapshot: ${fp}`)
          const stream = handle.createReadStream({ start: 0, ...options })
          res.once('close', () => stream.destroy())
          return stream
        },
      }
      await serveHandler(
        req,
        res,
        {
          public: argv.output,
          directoryListing: false,
          headers: [
            { source: '**/*.*', headers: [{ key: 'Content-Disposition', value: 'inline' }] },
            { source: '**/*.webp', headers: [{ key: 'Content-Type', value: 'image/webp' }] },
            // fixes bug where avif images are displayed as text instead of images (future proof)
            { source: '**/*.avif', headers: [{ key: 'Content-Type', value: 'image/avif' }] },
          ],
        },
        reads,
      )
      const status = res.statusCode
      const statusString =
        status >= 200 && status < 300
          ? styleText('green', `[${status}]`)
          : styleText('red', `[${status}]`)
      console.log(statusString + styleText('gray', ` ${argv.baseDir}${req.url}`))
    }

    const redirect = newFp => {
      newFp = argv.baseDir + newFp
      res.writeHead(302, { Location: newFp })
      console.log(
        styleText('yellow', '[302]') + styleText('gray', ` ${argv.baseDir}${req.url} -> ${newFp}`),
      )
      res.end()
    }

    const outputRoot = path.resolve(argv.output)
    const normalizeRequestPath = value => {
      try {
        const parsed = new URL(value ?? '/', 'http://quartz.local')
        const decoded = decodeURIComponent(parsed.pathname)
        if (decoded.includes('\0')) return '/'
        return path.posix.normalize(`/${decoded}`)
      } catch {
        return '/'
      }
    }
    const resolveOutputPath = publicPath => {
      const resolved = path.resolve(outputRoot, `.${publicPath}`)
      if (resolved === outputRoot || resolved.startsWith(`${outputRoot}${path.sep}`)) {
        return resolved
      }
    }
    const existsInOutput = async publicPath => {
      const resolved = resolveOutputPath(publicPath)
      if (resolved === undefined) return false
      try {
        await promises.access(resolved)
        return true
      } catch {
        return false
      }
    }

    let fp = normalizeRequestPath(req.url)

    if (fp === '/api/lean/verify') {
      await handleLocalLeanVerifyNodeRequest(req, res)
      return
    }

    // handle redirects
    if (fp.endsWith('/')) {
      // /trailing/
      // does /trailing/index.html exist? if so, serve it
      const indexFp = path.posix.join(fp, 'index.html')
      if (await existsInOutput(indexFp)) {
        req.url = fp
        return serve()
      }

      // does /trailing.html exist? if so, redirect to /trailing
      let base = fp.slice(0, -1)
      if (path.extname(base) === '') {
        base += '.html'
      }
      if (await existsInOutput(base)) {
        return redirect(fp.slice(0, -1))
      }
    } else {
      // /regular
      // does /regular.html exist? if so, serve it
      let base = fp
      if (path.extname(base) === '') {
        base += '.html'
      }
      if (await existsInOutput(base)) {
        req.url = fp
        return serve()
      }

      // does /regular/index.html exist? if so, redirect to /regular/
      let indexFp = path.posix.join(fp, 'index.html')
      if (await existsInOutput(indexFp)) {
        return redirect(fp + '/')
      }
    }

    return serve()
  })
}
