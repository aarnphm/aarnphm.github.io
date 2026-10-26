import { getFullSlug } from '../../../util/path'
import { gazeAt } from '../pet-gaze'
import {
  center,
  clamp,
  edgeY,
  friendCenter,
  friendScale,
  friendSize,
  friendState,
  height,
  later,
  margin,
  petState,
  reducedMotion,
  transformFor,
  viewTop,
  type Edge,
  type Perch,
  type Point,
} from './actor-model'
import { friendPlay, renderFriend } from './appearance'
import {
  boxOf,
  coversText,
  edgesNear,
  floorStep,
  inFlow,
  petObstacles,
  placeOn,
  type PlacementMeasurements,
} from './perch-placement'

export const friendPage = () => getFullSlug(window) === 'friends'

// Its eyes go to the pointer when it comes close, else to the pet.
export function friendWatch() {
  const c = friendCenter()
  if (!friendState.friend || !c || friendState.friendHop) return
  const near =
    petState.pointer && Math.hypot(petState.pointer.x - c.x, petState.pointer.y - c.y) < 140
      ? petState.pointer
      : null
  const target = near ?? (petState.doc || petState.carry ? center() : null)
  if (!target) return
  friendState.friendGaze?.face(friendState.friendFace)
  friendState.friendGaze?.look(
    ...gazeAt(target.x - c.x, target.y - c.y, friendState.friendFace, petState.unit),
  )
}

// Where the friend goes when the pet holds `p` from `m` (its top-left in document coordinates):
// the same edge, shoulder to shoulder, on its usual side if that side has room and no text. With
// `far` off it stays on the pet's own edge or head, which is what a landing asks for. A search over
// many perches shares `seen`, so the text is read once, with both blobs hidden.
export function friendSpot(
  p: Perch,
  m: Point,
  far = true,
  seen: PlacementMeasurements = {},
): { perch: Perch; doc: Point } | null {
  const covered = (edge: Edge) => coversText(edge, (seen.obstacles ??= petObstacles()))
  const s = p.scale ?? 1
  const fs = s * friendSize
  const mainX = m.x - window.scrollX + (petState.width * s) / 2
  const gap = 0.38 * petState.width * (s + fs)
  const beside = (side: number, off: number) => {
    const x = mainX + side * (gap + off)
    // placeOn holds a box or floor spot inside the view but takes a foot as given, so a foot past
    // the view's side is caught here.
    const half = (petState.width * fs) / 2
    if (x < margin + half - 2 || x > window.innerWidth - margin - half + 2) return null
    let perch: Perch
    if (p.foot) perch = { ...p, foot: { x: p.foot.x + side * (gap + off), y: p.foot.y }, scale: fs }
    else if (p.el) {
      // On a box it keeps over the box: a sitter holds a corner, which leaves only the inside.
      const rect = boxOf(p.el)
      if (x < rect.left + petState.width * fs * 0.2 || x > rect.right - petState.width * fs * 0.2)
        return null
      perch = { ...p, t: (x - rect.left) / rect.width, scale: fs }
    } else perch = { ...p, t: x / window.innerWidth, scale: fs }
    const at = placeOn(perch)
    const cx = at.x - window.scrollX + (petState.width * fs) / 2
    // Held back into the view, it would crowd the pet.
    if (Math.abs(cx - x) > 2) return null
    const foot = at.y - window.scrollY + (edgeY[perch.kind] - viewTop) * petState.unit * fs
    if (covered({ ...perch, at: { x: cx, y: foot }, score: 0 })) return null
    friendState.friendSide = side
    return { perch, doc: at }
  }
  // Up on the pet's head.
  const onHead = () => {
    const head = { x: mainX, y: m.y - window.scrollY + (-0.97 - viewTop) * petState.unit * s }
    const perch: Perch = {
      el: null,
      kind: 'stand',
      t: 0,
      foot: { x: head.x + window.scrollX, y: head.y + window.scrollY },
      scale: fs,
    }
    const top = head.y - (edgeY.stand - viewTop) * petState.unit * fs
    if (top < margin || covered({ ...perch, at: head, score: 0 })) return null
    return { perch, doc: placeOn(perch) }
  }
  // Boxed in by text, it takes the nearest clear edge a short way off, out of the pet's way.
  const nearby = () => {
    const pet = { x: mainX, y: m.y - window.scrollY, w: petState.width * s, h: height(s) }
    const center = { x: pet.x, y: pet.y + pet.h / 2 }
    for (const edge of edgesNear(center, null).slice(0, 20)) {
      if (!inFlow(edge.el!)) continue
      const rect = boxOf(edge.el!)
      const offs =
        edge.kind === 'stand'
          ? [0, ...Array.from({ length: 24 }, (_, k) => (k + 1) * floorStep)]
          : [0]
      for (const off of offs)
        for (const side of off ? [friendState.friendSide, -friendState.friendSide] : [1]) {
          const x = edge.at.x + side * off
          if (
            x < rect.left + petState.width * fs * 0.2 ||
            x > rect.right - petState.width * fs * 0.2
          )
            continue
          if (Math.hypot(x - center.x, edge.at.y - center.y) > 400) continue
          const top = edge.at.y - (edgeY[edge.kind] - viewTop) * petState.unit * fs
          const overlaps =
            Math.abs(x - pet.x) < (pet.w + petState.width * fs) / 2 &&
            top < pet.y + pet.h &&
            top + height(fs) > pet.y
          if (overlaps) continue
          if (edge.kind === 'stand') {
            const hit = document
              .elementsFromPoint(x, edge.at.y + 3)
              .find(e => !petState.root?.contains(e) && !friendState.friend?.contains(e))
            if (!hit || !(edge.el === hit || edge.el!.contains(hit))) continue
          }
          const perch: Perch = {
            el: edge.el,
            kind: edge.kind,
            t: (x - rect.left) / rect.width,
            scale: fs,
          }
          if (covered({ ...perch, at: { x, y: edge.at.y }, score: 0 })) continue
          return { perch, doc: placeOn(perch) }
        }
    }
    return null
  }
  // Shoulder to shoulder, else on its head, else a few steps along the same edge, else nearby.
  const probe = () => {
    for (const side of [friendState.friendSide, -friendState.friendSide]) {
      const spot = beside(side, 0)
      if (spot) return spot
    }
    const head = onHead()
    if (head) return head
    for (let off = floorStep; off <= 12 * floorStep; off += floorStep)
      for (const side of [friendState.friendSide, -friendState.friendSide]) {
        const spot = beside(side, off)
        if (spot) return spot
      }
    return far ? nearby() : null
  }
  const before = friendState.friend?.style.visibility ?? ''
  if (friendState.friend) friendState.friend.style.visibility = 'hidden'
  try {
    return probe()
  } finally {
    if (friendState.friend) friendState.friend.style.visibility = before
  }
}

export function hideFriend() {
  if (!friendState.friend) return
  friendState.friendHop?.cancel()
  friendState.friendHop = null
  friendState.friend.hidden = true
  friendState.friendDoc = null
  friendState.friendPerch = null
}

// Follows the pet onto `p`: placed at once, walking beside it, hopping after it, or appearing once
// it has landed (after a far jump or a ride on the rail).
export function friendFollow(
  p: Perch,
  m: Point,
  how: 'place' | 'walk' | 'hop' | 'appear',
  duration = 0,
) {
  if (!friendState.friend) return
  const spot = friendSpot(p, m)
  if (!spot) return hideFriend()
  const from = friendState.friend.hidden ? null : friendState.friendDoc
  friendState.friendHop?.cancel()
  friendState.friendHop = null
  friendState.friendPerch = spot.perch
  friendState.friendDoc = spot.doc
  const fs = friendScale()
  const end = transformFor(spot.doc, fs)
  const settle = () => {
    if (!friendState.friend) return
    friendState.friend.style.transform = end
    friendWatch()
    renderFriend()
  }
  if (how === 'place' || reducedMotion()) {
    friendState.friend.hidden = false
    return settle()
  }
  if (!from || how === 'appear') {
    friendState.friend.hidden = true
    later(() => {
      if (!friendState.friend || friendState.friendDoc !== spot.doc) return
      friendState.friend.hidden = false
      friendState.friend.style.transform = end
      friendState.friend.animate(
        [
          { transform: transformFor({ x: spot.doc.x, y: spot.doc.y - 40 }, fs), opacity: 0 },
          { transform: end, opacity: 1 },
        ],
        { duration: 320, easing: 'cubic-bezier(0.2, 0, 0, 1)' },
      )
      friendPlay('land', 380)
      settle()
    }, duration)
    return
  }
  const d = Math.hypot(spot.doc.x - from.x, spot.doc.y - from.y)
  if (d < 2) return settle()
  if (Math.abs(spot.doc.x - from.x) > 2) friendState.friendFace = spot.doc.x > from.x ? -1 : 1
  const walking = how === 'walk'
  const peak = Math.min(
    140,
    40 + d * 0.3,
    Math.max(0, Math.min(from.y, spot.doc.y) - window.scrollY),
  )
  const frames: Keyframe[] = walking
    ? [{ transform: transformFor(from, fs) }, { transform: end }]
    : [
        { transform: transformFor(from, fs), easing: 'cubic-bezier(0.3, 0.6, 0.5, 1)' },
        {
          transform: transformFor(
            { x: (from.x + spot.doc.x) / 2, y: Math.min(from.y, spot.doc.y) - peak },
            fs,
          ),
          offset: 0.5,
          easing: 'cubic-bezier(0.5, 0, 0.7, 0.4)',
        },
        { transform: end },
      ]
  const anim = friendState.friend.animate(frames, {
    duration: walking ? duration : clamp(320 + d * 0.45, 380, 900),
    // A beat behind the pet, as a friend tagging along would be.
    delay: walking ? 120 : 160,
    easing: walking ? 'linear' : undefined,
    fill: 'backwards',
  })
  friendState.friendHop = anim
  friendState.friendOverride = {
    state: walking ? (petState.svg?.dataset.petState === 'walk' ? 'walk' : 'run') : 'fly',
    until: Infinity,
  }
  renderFriend()
  anim.onfinish = () => {
    if (friendState.friendHop !== anim || !friendState.friend) return
    friendState.friend.style.transform = end
    anim.cancel()
    friendState.friendHop = null
    friendState.friendOverride = null
    if (walking) settle()
    else {
      settle()
      friendPlay('land', 380)
    }
  }
}
