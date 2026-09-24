import crypto from 'node:crypto'
import type { ArenaChannel } from '../plugins/transformers/arena'
import { assertArenaFeedRouteAvailable } from './arena-feed'

export type ArenaChannelEmitState = {
  fingerprint: string
  jsonEnabled: boolean
  entryIds: string[]
}

export type ArenaEmitState = { channelStates: Map<string, ArenaChannelEmitState> }

export type ArenaPartialEmitPlan = {
  changedChannels: ArenaChannel[]
  deletedChannels: Array<[string, ArenaChannelEmitState]>
  nextState: ArenaEmitState
  hasChanges: boolean
}

export function isArenaChannelJsonEnabled(channel: ArenaChannel): boolean {
  return channel.metadata?.json === 'true' || channel.metadata?.json === true
}

function fingerprintChannel(channel: ArenaChannel): string {
  return crypto.createHash('sha256').update(JSON.stringify(channel)).digest('hex')
}

export function collectArenaEmitState(channels: ArenaChannel[]): ArenaEmitState {
  assertArenaFeedRouteAvailable(channels)
  return {
    channelStates: new Map(
      channels.map(channel => [
        channel.slug,
        {
          fingerprint: fingerprintChannel(channel),
          jsonEnabled: isArenaChannelJsonEnabled(channel),
          entryIds: channel.blocks.map(block => block.entryId).filter((id): id is string => !!id),
        },
      ]),
    ),
  }
}

export function planArenaPartialEmit(
  previousState: ArenaEmitState | undefined,
  channels: ArenaChannel[],
): ArenaPartialEmitPlan {
  const nextState = collectArenaEmitState(channels)
  if (!previousState) {
    return { changedChannels: channels, deletedChannels: [], nextState, hasChanges: true }
  }

  const changedChannels: ArenaChannel[] = []
  for (const channel of channels) {
    const previous = previousState.channelStates.get(channel.slug)
    const current = nextState.channelStates.get(channel.slug)
    if (!current) continue
    if (!previous || previous.fingerprint !== current.fingerprint) {
      changedChannels.push(channel)
    }
  }

  const deletedChannels = Array.from(previousState.channelStates.entries()).filter(
    ([slug]) => !nextState.channelStates.has(slug),
  )
  const channelOrderChanged = [...previousState.channelStates.keys()].some(
    (slug, index) => channels[index]?.slug !== slug,
  )

  return {
    changedChannels,
    deletedChannels,
    nextState,
    hasChanges: changedChannels.length > 0 || deletedChannels.length > 0 || channelOrderChanged,
  }
}
