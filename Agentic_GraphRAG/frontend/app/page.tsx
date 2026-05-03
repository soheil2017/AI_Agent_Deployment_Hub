'use client'

import { useState, useRef, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://conduit-graph-rag-production.up.railway.app'

const BADGE_COLORS: Record<string, string> = {
  relational: '#3b82f6',
  semantic:   '#8b5cf6',
  hybrid:     '#06b6d4',
  visual:     '#f59e0b',
}

const EXAMPLE_QUESTIONS = [
  'Which patients have a pending referral with a pending prior auth?',
  'What does CARIN Blue Button require for patient access?',
  'What procedures has Alice Johnson had and what are the TEFCA requirements for sharing that data?',
]

interface Message {
  role: 'user' | 'assistant'
  content: string
  queryType?: string
  traceId?: string
  feedback?: 'up' | 'down'
  imagePreview?: string
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [imageMediaType, setImageMediaType] = useState<string>('image/jpeg')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setImageMediaType(file.type || 'image/jpeg')
    setImagePreview(URL.createObjectURL(file))
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // Strip "data:image/jpeg;base64," prefix
      setImageBase64(result.split(',')[1])
    }
    reader.readAsDataURL(file)
  }

  function clearImage() {
    setImageBase64(null)
    setImagePreview(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  async function sendMessage(question: string) {
    if (!question.trim() || loading) return
    setInput('')
    const sentImage = imagePreview
    clearImage()
    setMessages(prev => [...prev, { role: 'user', content: question, imagePreview: sentImage ?? undefined }])
    setLoading(true)

    try {
      const body: Record<string, string> = { question }
      if (imageBase64) {
        body.image_base64 = imageBase64
        body.media_type = imageMediaType
      }
      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        queryType: data.query_type,
        traceId: data.trace_id,
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: could not reach the API.',
      }])
    } finally {
      setLoading(false)
    }
  }

  async function sendFeedback(index: number, rating: 'thumbs_up' | 'thumbs_down') {
    const msg = messages[index]
    if (!msg.traceId || msg.feedback) return
    setMessages(prev => prev.map((m, i) =>
      i === index ? { ...m, feedback: rating === 'thumbs_up' ? 'up' : 'down' } : m
    ))
    await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trace_id: msg.traceId, rating }),
    }).catch(() => {})
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: 800, margin: '0 auto', padding: '0 16px' }}>
      {/* Header */}
      <div style={{ padding: '24px 0 16px', borderBottom: '1px solid #1e293b' }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#38bdf8' }}>
          Conduit Graph RAG
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Dental healthcare knowledge graph · Neo4j + ChromaDB + Claude Vision
        </p>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 0' }}>
        {messages.length === 0 && (
          <div>
            <p style={{ color: '#475569', fontSize: 14, marginBottom: 16 }}>Try an example:</p>
            {EXAMPLE_QUESTIONS.map(q => (
              <button key={q} onClick={() => sendMessage(q)} style={{
                display: 'block', width: '100%', textAlign: 'left', background: '#1e293b',
                border: '1px solid #334155', borderRadius: 8, padding: '10px 14px',
                color: '#94a3b8', fontSize: 14, cursor: 'pointer', marginBottom: 8,
              }}>
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: 20 }}>
            {msg.role === 'user' ? (
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{ maxWidth: '80%' }}>
                  {msg.imagePreview && (
                    <img src={msg.imagePreview} alt="uploaded" style={{
                      display: 'block', maxWidth: '100%', maxHeight: 200,
                      borderRadius: 8, marginBottom: 6,
                    }} />
                  )}
                  <div style={{
                    background: '#1d4ed8', borderRadius: '12px 12px 2px 12px',
                    padding: '10px 14px', fontSize: 14, lineHeight: 1.5,
                  }}>
                    {msg.content}
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 12, color: '#64748b' }}>Agent</span>
                  {msg.queryType && (
                    <span style={{
                      background: BADGE_COLORS[msg.queryType] || '#475569',
                      borderRadius: 4, padding: '1px 7px', fontSize: 11, fontWeight: 600,
                    }}>
                      {msg.queryType}
                    </span>
                  )}
                </div>
                <div style={{
                  background: '#1e293b', borderRadius: '2px 12px 12px 12px',
                  padding: '10px 14px', fontSize: 14, lineHeight: 1.7,
                  whiteSpace: 'pre-wrap', color: '#e2e8f0',
                }}>
                  {msg.content}
                </div>
                {msg.traceId && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                    <button onClick={() => sendFeedback(i, 'thumbs_up')} style={{
                      background: msg.feedback === 'up' ? '#16a34a' : '#1e293b',
                      border: '1px solid #334155', borderRadius: 6, padding: '3px 10px',
                      cursor: msg.feedback ? 'default' : 'pointer', fontSize: 14, color: '#94a3b8',
                    }}>
                      👍
                    </button>
                    <button onClick={() => sendFeedback(i, 'thumbs_down')} style={{
                      background: msg.feedback === 'down' ? '#dc2626' : '#1e293b',
                      border: '1px solid #334155', borderRadius: 6, padding: '3px 10px',
                      cursor: msg.feedback ? 'default' : 'pointer', fontSize: 14, color: '#94a3b8',
                    }}>
                      👎
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ color: '#475569', fontSize: 14 }}>Thinking...</div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: '12px 0 20px', borderTop: '1px solid #1e293b' }}>
        {imagePreview && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <img src={imagePreview} alt="preview" style={{ height: 48, borderRadius: 6 }} />
            <span style={{ fontSize: 12, color: '#64748b' }}>Image attached</span>
            <button onClick={clearImage} style={{
              background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 14,
            }}>✕</button>
          </div>
        )}
        <form onSubmit={e => { e.preventDefault(); sendMessage(input) }}
          style={{ display: 'flex', gap: 8 }}>
          <input type="file" accept="image/*" ref={fileInputRef}
            onChange={handleImageUpload} style={{ display: 'none' }} />
          <button type="button" onClick={() => fileInputRef.current?.click()} style={{
            background: '#1e293b', border: '1px solid #334155', borderRadius: 8,
            padding: '10px 12px', color: imageBase64 ? '#38bdf8' : '#64748b',
            cursor: 'pointer', fontSize: 18, lineHeight: 1,
          }} title="Attach image">
            📎
          </button>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about patients, referrals, prior auths, or attach a dental image..."
            style={{
              flex: 1, background: '#1e293b', border: '1px solid #334155', borderRadius: 8,
              padding: '10px 14px', color: '#f1f5f9', fontSize: 14, outline: 'none',
            }}
          />
          <button type="submit" disabled={loading || (!input.trim() && !imageBase64)} style={{
            background: '#1d4ed8', border: 'none', borderRadius: 8,
            padding: '10px 20px', color: '#fff', fontSize: 14, cursor: 'pointer',
            opacity: loading || (!input.trim() && !imageBase64) ? 0.5 : 1,
          }}>
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
