import { computePosition, shift, flip, offset } from '@floating-ui/dom'

document.addEventListener('nav', () => {
  const slug = document.body.dataset.slug
  if (slug === '404') {
    if (window.plausible) window.plausible('404', { props: { path: slug } })
    const tooltip = document.querySelector('.home-tooltip') as HTMLElement

    const handleMouseMove = async ({ clientX, clientY }: MouseEvent) => {
      await computePosition(
        {
          getBoundingClientRect() {
            return {
              width: 0,
              height: 0,
              x: clientX,
              y: clientY,
              left: clientX,
              right: clientX,
              top: clientY,
              bottom: clientY,
            }
          },
        },
        tooltip,
        { placement: 'right-start', middleware: [offset(15), flip(), shift()] },
      ).then(({ x, y }) => {
        Object.assign(tooltip.style, { top: `${y}px`, left: `${x}px` })
      })
    }

    const handleClick = (e: MouseEvent) => {
      e.preventDefault()
      window.spaNavigate(new URL('/', window.location.toString()))
    }

    document.body.addEventListener('click', handleClick)
    document.body.addEventListener('mousemove', handleMouseMove)

    // Show/hide tooltip on mouse enter/leave
    const showTooltip = () => {
      tooltip!.classList.add('visible')
    }
    const hideTooltip = () => {
      tooltip!.classList.remove('visible')
    }

    document.body.addEventListener('mouseenter', showTooltip)
    document.body.addEventListener('mouseleave', hideTooltip)

    // Cleanup function
    window.addCleanup(() => {
      document.body.removeEventListener('click', handleClick)
      document.body.removeEventListener('mousemove', handleMouseMove)
      document.body.removeEventListener('mouseenter', showTooltip)
      document.body.removeEventListener('mouseleave', hideTooltip)
    })
  }
})
