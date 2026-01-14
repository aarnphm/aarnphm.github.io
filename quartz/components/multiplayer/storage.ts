import type { OperationInput } from "./model"
import { parsePendingOps } from "./model"

const pendingOpsStoragePrefix = "comment-pending-ops:"

export function pendingOpsKey(pageId: string): string {
  return `${pendingOpsStoragePrefix}${pageId}`
}

export function persistPendingOps(pageId: string, pendingOps: Map<string, OperationInput>) {
  if (!pageId) return
  try {
    const key = pendingOpsKey(pageId)
    if (pendingOps.size === 0) {
      sessionStorage.removeItem(key)
      return
    }
    const payload = JSON.stringify([...pendingOps.values()])
    sessionStorage.setItem(key, payload)
  } catch {}
}

export function restorePendingOps(pageId: string): OperationInput[] {
  if (!pageId) return []
  try {
    const raw = sessionStorage.getItem(pendingOpsKey(pageId))
    if (!raw) return []
    return parsePendingOps(raw)
  } catch {
    return []
  }
}
