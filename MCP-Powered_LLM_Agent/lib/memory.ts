/**
 * Memory System
 * ─────────────
 * Stores conversation history per session so users can refresh the page
 * and continue where they left off.
 *
 * CURRENT IMPLEMENTATION: In-memory Map
 *  ✓ Zero setup — works immediately in local dev
 *  ✗ Resets on server restart
 *  ✗ Not shared across multiple Vercel function instances (fine for demos)
 *
 * PRODUCTION UPGRADE → Vercel KV (Redis):
 *  1. Run: vercel kv create
 *  2. Add KV env vars to your Vercel project (done automatically by CLI)
 *  3. npm install @vercel/kv
 *  4. Swap the implementation below with the KV block (see comments)
 */

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
}

// ── In-Memory Implementation ────────────────────────────────────────────────

/** sessionId → message history */
const store = new Map<string, StoredMessage[]>();

/** sessionId → expiry timestamp (ms) */
const expiry = new Map<string, number>();

const MAX_MESSAGES = 100; // keep last 100 messages per session
const SESSION_TTL_MS = 2 * 60 * 60 * 1000; // 2 hours

export const memoryStore = {
  /** Retrieve stored messages for a session. Returns [] if not found / expired. */
  get(sessionId: string): StoredMessage[] {
    const ttl = expiry.get(sessionId);
    if (ttl && Date.now() > ttl) {
      store.delete(sessionId);
      expiry.delete(sessionId);
      return [];
    }
    return store.get(sessionId) ?? [];
  },

  /** Save messages for a session (trims to MAX_MESSAGES). */
  set(sessionId: string, messages: StoredMessage[]): void {
    const trimmed = messages.slice(-MAX_MESSAGES);
    store.set(sessionId, trimmed);
    expiry.set(sessionId, Date.now() + SESSION_TTL_MS);
  },

  /** Delete a session (used when user clicks "New Chat"). */
  delete(sessionId: string): void {
    store.delete(sessionId);
    expiry.delete(sessionId);
  },
};

/* ── Vercel KV Upgrade (uncomment to use in production) ─────────────────────

import { kv } from "@vercel/kv";

export const memoryStore = {
  async get(sessionId: string): Promise<StoredMessage[]> {
    return (await kv.get<StoredMessage[]>(sessionId)) ?? [];
  },
  async set(sessionId: string, messages: StoredMessage[]): Promise<void> {
    const trimmed = messages.slice(-MAX_MESSAGES);
    await kv.set(sessionId, trimmed, { ex: 7200 }); // 2 hour TTL
  },
  async delete(sessionId: string): Promise<void> {
    await kv.del(sessionId);
  },
};

─────────────────────────────────────────────────────────────────────────── */
