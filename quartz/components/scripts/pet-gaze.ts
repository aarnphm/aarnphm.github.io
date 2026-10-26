// The eyes ride an arc toward the target, from left up over the top to right, and ease back to
// centre as it comes close. They sit left of centre in the pet's own frame, so they travel further
// right than left. Hundredths of a body radius, so a still target repaints nothing.
export function gazeAt(dx: number, dy: number, face: number, unit: number): [string, string] {
  const r = Math.hypot(dx, dy)
  const k = Math.min(1, r / (3 * unit)) / Math.max(r, 1)
  // In the pet's own frame, which is mirrored when it faces right.
  const ux = face * dx * k
  const uy = dy * k
  return [(ux * (ux < 0 ? 0.2 : 0.38)).toFixed(2), (uy * (uy < 0 ? 0.14 : 0.16)).toFixed(2)]
}

export function createPetGaze(svg: SVGSVGElement) {
  const look = svg.querySelector<SVGGElement>('.pet-look')
  let eyeX = '0'
  let eyeY = '0'

  return {
    face(direction: number) {
      const scale = `${direction} 1`
      if (svg.style.scale !== scale) svg.style.scale = scale
    },
    look(x: string, y: string) {
      if (look && x !== eyeX) look.style.setProperty('--pet-eye-x', x)
      if (look && y !== eyeY) look.style.setProperty('--pet-eye-y', y)
      eyeX = x
      eyeY = y
    },
    syncTrail(target: SVGElement) {
      for (const [property, value] of [
        ['--pet-lx', eyeX],
        ['--pet-ly', eyeY],
      ]) {
        if (target.style.getPropertyValue(property) !== value)
          target.style.setProperty(property, value)
      }
    },
  }
}
