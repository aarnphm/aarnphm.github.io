import './globals.css'

import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: {
    default: "Aaron's notes",
    template: "%s | Aaron's notes",
  },
  description: 'A second brain, my digital garden',
  keywords: ['digital-garden', 'blog', 'technology', 'machine-learning'],
  openGraph: {
    title: "Aaron's notes",
    description: 'A second brain, my digital garden',
    url: 'https://aarnphm.xyz',
    siteName: "Aaron's notes",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang='en'>
      <body>{children}</body>
    </html>
  )
}
