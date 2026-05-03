import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Conduit Graph RAG',
  description: 'Dental healthcare knowledge graph agent',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: 'system-ui, sans-serif', background: '#0f172a', color: '#f1f5f9' }}>
        {children}
      </body>
    </html>
  )
}
