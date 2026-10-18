export function createPetGaze(svg: SVGSVGElement) {
  const look = svg.querySelector<SVGGElement>('.pet-look')
  const body = svg.querySelector<SVGGElement>('.pet-body')
  let eyeX = '0'
  let eyeY = '0'
  let bodyTilt = '0'

  return {
    face(direction: number) {
      const scale = `${direction} 1`
      if (svg.style.scale !== scale) svg.style.scale = scale
    },
    look(x: string, y: string, tilt: string) {
      if (look && x !== eyeX) look.style.setProperty('--pet-eye-x', x)
      if (look && y !== eyeY) look.style.setProperty('--pet-eye-y', y)
      if (body && tilt !== bodyTilt) body.style.setProperty('--pet-tilt', tilt)
      eyeX = x
      eyeY = y
      bodyTilt = tilt
    },
    syncTrail(target: SVGElement) {
      for (const [property, value] of [
        ['--pet-lx', eyeX],
        ['--pet-ly', eyeY],
        ['--pet-tilt', bodyTilt],
      ]) {
        if (target.style.getPropertyValue(property) !== value)
          target.style.setProperty(property, value)
      }
    },
  }
}
