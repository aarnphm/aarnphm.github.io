import type { ShelfMaterial } from './style-model'
import { rgbToHex } from '../geometry'

export interface Ladder {
  top: string
  face: string
  s3: string
  s2: string
  s1: string
  s0: string
  mid: string
  faint: string
  lo: string
  hi: string
  halo: string
  /** Lit end of a face gradient; the halo is its shaded end. */
  sheen: string
  /** Stripe on thin plywood edges. */
  ply?: string
}

const ladderOf = (row: string, halo: string, sheen: string, ply?: string): Ladder => {
  const [top, face, s3, s2, s1, s0, mid, faint, lo, hi] = row.split(' ')
  return { top, face, s3, s2, s1, s0, mid, faint, lo, hi, halo, sheen, ply }
}

export const LADDERS: Record<'light' | 'dark', Record<ShelfMaterial, Ladder>> = {
  light: {
    paper: ladderOf(
      '#FFFCF0 #FFFCF0 #F9F6EA #F1EEE2 #E8E5DA #DFDCD1 #6F6E69 #D2D0C6 #9F9D96 #282726',
      '#403E3C',
      '#FFFFFF',
    ),
    walnut: ladderOf(
      '#866A53 #6E4E3A #634535 #583D2F #4C3529 #412D23 #3A2A1F #5A3F31 #46312A #100F0F',
      '#221710',
      '#B89778',
    ),
    'birch-ply': ladderOf(
      '#F3EAD6 #EADBC0 #DFCFB4 #D3C3A8 #C6B69C #BBAB90 #6E5B45 #C5B59B #AC9B83 #3A2E22',
      '#403E3C',
      '#FFFBEE',
      '#B2A189',
    ),
    'black-lacquer': ladderOf(
      '#3C3B38 #2C2B29 #1C1B1A #181716 #151413 #100F0F #8E8C85 #4E4D49 #696863 #FFFCF0',
      '#000000',
      '#8A8883',
    ),
    steel: ladderOf(
      '#D5D8DC #AEB4BA #9EA4AA #8D949A #7C8389 #6C7379 #3C454D #8C9399 #757D84 #1F262B',
      '#2A3238',
      '#F7F9FA',
    ),
    // Antiqued copper: brown-gold, the lit faces a step warmer, no orange.
    copper: ladderOf(
      '#B48D60 #9C744A #8E6941 #7F5D39 #6F5031 #5F4429 #3A2A19 #80603C #5C4229 #1A120A',
      '#2A1A0E',
      '#EBCD98',
    ),
    // Faux-concrete panel: light warm grey with a short ladder, no blue.
    concrete: ladderOf(
      '#CDCAC2 #C1BDB5 #B6B2AA #ABA79F #9F9B93 #918D85 #5B5955 #A6A29A #86837C #1F1E1C',
      '#2E2D2A',
      '#E8E5DC',
    ),
  },
  dark: {
    paper: ladderOf(
      '#403E3C #343331 #2F2E2C #292827 #242322 #1E1D1D #9F9D96 #52514D #6F6E69 #CECDC3',
      '#000000',
      '#6B6965',
    ),
    walnut: ladderOf(
      '#6A5040 #563B2C #4C3428 #432E24 #39271F #2F201A #9D8977 #6E5444 #7E6553 #FFFCF0',
      '#000000',
      '#A08468',
    ),
    'birch-ply': ladderOf(
      '#C6B599 #B8A383 #A49175 #8E7E66 #786B57 #645949 #3A2E22 #928066 #796953 #FFFCF0',
      '#000000',
      '#E6D8BC',
      '#7F6E57',
    ),
    'black-lacquer': ladderOf(
      '#454441 #33322F #252423 #1C1B1A #171616 #100F0F #A4A29A #575652 #777570 #FFFCF0',
      '#000000',
      '#7A7873',
    ),
    steel: ladderOf(
      '#AEB3B7 #7F868C #72787E #63696F #545A5F #474D51 #D5DADF #999FA5 #AAB0B6 #F4F6F8',
      '#000000',
      '#E3E7EA',
    ),
    copper: ladderOf(
      '#8E6E48 #7A5A38 #6E5033 #61462C #553D26 #483320 #D2B48C #8A6A48 #A07F5A #FFF3DC',
      '#000000',
      '#E0BE88',
    ),
    concrete: ladderOf(
      '#8E8C86 #7E7C76 #747069 #6A6660 #5F5B56 #534F4A #B8B5AC #87847D #9A978F #FFFCF0',
      '#000000',
      '#B3B0A8',
    ),
  },
}

/** Flat tone ladder of a material. Book colours never change with the material. */
export function materialLadder(material: ShelfMaterial, theme: 'light' | 'dark'): Ladder {
  return LADDERS[theme][material]
}

// A declaration: GRAINS above calls it while the module loads.
export function hexRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ]
}

/** sRGB mix of `t` of `b` into `a`. */
export function mixHex(a: string, b: string, t: number): string {
  const [ar, ag, ab] = hexRgb(a)
  const [br, bg, bb] = hexRgb(b)
  return rgbToHex(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)
}
