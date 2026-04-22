/**
 * Memory API Route
 * ─────────────────
 * GET  /api/memory?sessionId=xxx  — Restore previous conversation history
 * DELETE /api/memory?sessionId=xxx — Clear a session (used by "New Chat" button)
 */

import { NextRequest, NextResponse } from "next/server";
import { memoryStore } from "@/lib/memory";

export async function GET(req: NextRequest) {
  const sessionId = req.nextUrl.searchParams.get("sessionId");

  if (!sessionId) {
    return NextResponse.json({ messages: [] });
  }

  const messages = memoryStore.get(sessionId);
  return NextResponse.json({ messages });
}

export async function DELETE(req: NextRequest) {
  const sessionId = req.nextUrl.searchParams.get("sessionId");

  if (sessionId) {
    memoryStore.delete(sessionId);
  }

  return NextResponse.json({ ok: true });
}
