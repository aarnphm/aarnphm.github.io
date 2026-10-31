// The renderer's kinds, for a build that imports everything it draws up front. Upstream
// (render/src/kinds.js) fetches each module on demand by URL; here the leaf and flower families
// are registered statically, and register(m) adds a module of your own the same way upstream does.

import bell from './kinds/flower/bell.js'
import bud from './kinds/flower/bud.js'
import burst from './kinds/flower/burst.js'
import daisy from './kinds/flower/daisy.js'
import dotring from './kinds/flower/dotring.js'
import phyllo from './kinds/flower/phyllo.js'
import pomegranate from './kinds/flower/pomegranate-fruit.js'
import quatre from './kinds/flower/quatre.js'
import rose from './kinds/flower/rose.js'
import rosette from './kinds/flower/rosette.js'
import star from './kinds/flower/star.js'
import target from './kinds/flower/target.js'
import thistle from './kinds/flower/thistle.js'
import tulip from './kinds/flower/tulip.js'
import acanthus from './kinds/leaf/acanthus.js'
import cinquefoil from './kinds/leaf/cinquefoil.js'
import comma from './kinds/leaf/comma.js'
import dot from './kinds/leaf/dot.js'
import dotted from './kinds/leaf/dotted.js'
import fan from './kinds/leaf/fan.js'
import fern from './kinds/leaf/fern.js'
import heart from './kinds/leaf/heart.js'
import ivy from './kinds/leaf/ivy.js'
import lance from './kinds/leaf/lance.js'
import needles from './kinds/leaf/needles.js'
import oak from './kinds/leaf/oak.js'
import ovate from './kinds/leaf/ovate.js'
import pod from './kinds/leaf/pod.js'
import teardrop from './kinds/leaf/teardrop.js'
import trefoil from './kinds/leaf/trefoil.js'
import willow from './kinds/leaf/willow.js'

export const HEADS = new Set()
const registry = { leaf: new Map(), flower: new Map() }

export function register(m) {
  if (Array.isArray(m)) {
    m.forEach(register)
    return
  }
  if (!m || !registry[m.family])
    throw new Error(
      `INITIĀLIS: not a kind module (family must be one of ${Object.keys(registry).join(', ')})`,
    )
  registry[m.family].set(m.name, m)
  if (m.head) HEADS.add(m.name)
}

export function kind(family, name) {
  return (registry[family] && registry[family].get(name)) || null
}

register([
  acanthus,
  cinquefoil,
  comma,
  dot,
  dotted,
  fan,
  fern,
  heart,
  ivy,
  lance,
  needles,
  oak,
  ovate,
  pod,
  teardrop,
  trefoil,
  willow,
  bell,
  bud,
  burst,
  daisy,
  dotring,
  phyllo,
  pomegranate,
  quatre,
  rose,
  rosette,
  star,
  target,
  thistle,
  tulip,
])
