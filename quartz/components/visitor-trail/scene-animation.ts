// Scripted SVG motion joins the compositor before its next frame is sampled.
export function animateScene(
  element: Element,
  frames: Keyframe[] | PropertyIndexedKeyframes | null,
  options?: number | KeyframeAnimationOptions,
) {
  const animation = element.animate(frames, options)
  const view = element.ownerDocument.defaultView
  if (view) element.dispatchEvent(new view.Event('trail:animation', { bubbles: true }))
  return animation
}
