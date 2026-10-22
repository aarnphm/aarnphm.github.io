export type MotionTuning = { wind: number; leaves: number }

// Multipliers belong to the mounted scene. The animation owners read them without changing
// scene attributes, which would make the compositor reconsider its retained textures.
export function createMotionTuning(scene: HTMLElement) {
  const settings: MotionTuning = { wind: 100, leaves: 100 }
  return {
    settings,
    changed() {
      scene.dispatchEvent(new Event('trail:motion-tuning'))
    },
  }
}
