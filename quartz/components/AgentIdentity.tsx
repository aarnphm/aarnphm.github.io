import { JSX } from 'preact'

export function personStructuredData(baseUrl: string) {
  const origin = baseUrl.startsWith('http') ? baseUrl.replace(/\/$/, '') : `https://${baseUrl}`
  return {
    '@context': 'https://schema.org',
    '@type': 'Person',
    name: 'Aaron Pham',
    alternateName: 'aarnphm',
    description:
      'Aaron Pham writes a public digital garden about machine learning systems, compilers, mathematics, software, training, and culture.',
    url: origin,
    image: `${origin}/static/og-image.webp`,
    email: 'mailto:contact@aarnphm.xyz',
    contactPoint: {
      '@type': 'ContactPoint',
      contactType: 'personal inquiries',
      email: 'contact@aarnphm.xyz',
      url: `${origin}/contact`,
    },
    sameAs: [
      'https://github.com/aarnphm',
      'https://x.com/aarnphm',
      'https://substack.com/@aarnphm',
    ],
    knowsAbout: [
      'machine learning systems',
      'compiler construction',
      'mathematics',
      'software engineering',
      'triathlon training',
    ],
  }
}

export function PersonStructuredData({ baseUrl }: { baseUrl: string }): JSX.Element {
  const json = JSON.stringify(personStructuredData(baseUrl)).replaceAll('<', '\\u003c')
  return <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: json }} />
}

export function HomepageIdentityHeading(): JSX.Element {
  return <h1 class="agent-identity-heading">Aaron Pham's notes</h1>
}
