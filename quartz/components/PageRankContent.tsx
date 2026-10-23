import type { QuartzComponent, QuartzComponentConstructor } from '../types/component'

export default (() => {
  const PageRankContent: QuartzComponent = () => (
    <article class="popover-hint main-col pagerank-page">
      <p>Explore how notes connect across the garden.</p>
      <section id="pagerank-root" aria-label="Global note ranking">
        <p>The ranking view is being built.</p>
      </section>
    </article>
  )
  return PageRankContent
}) satisfies QuartzComponentConstructor
