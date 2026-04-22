/**
 * Main Chat API Route
 * ────────────────────
 * Receives messages from the frontend, connects to all MCP tool servers,
 * and streams the agent's response back using Vercel AI SDK.
 *
 * Flow:
 *  1. Read conversation history from memory (for context continuity)
 *  2. Connect to all MCP servers defined in config/mcp.config.ts
 *  3. Run the LLM with all available tools (up to maxSteps iterations)
 *  4. Stream the response to the client
 *  5. Save updated history to memory
 *  6. Close MCP connections
 */

import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";
import { NextRequest } from "next/server";
import { createMCPTools } from "@/lib/mcp-client";
import { memoryStore, type StoredMessage } from "@/lib/memory";

// Allow up to 60 seconds — needed for multi-step tool chains on Vercel
export const maxDuration = 60;

const SYSTEM_PROMPT = `You are a helpful research assistant with access to real-time tools.

TOOLS AVAILABLE:
• search_web     — Search the internet for current information, news, and facts
• get_weather    — Get weather conditions and forecasts for any city
• calculate      — Evaluate mathematical expressions with full precision

BEHAVIOR GUIDELINES:
• For any math question → always use the calculate tool (never calculate in your head)
• For weather questions → always call get_weather with the city name
• For current events, recent facts, prices, or anything time-sensitive → search_web
• You may chain multiple tool calls in one response if needed
• Briefly explain what you're doing before/after each tool call
• Synthesise tool results into a clear, concise answer

Be helpful, accurate, and concise.`;

export async function POST(req: NextRequest) {
  const { messages, sessionId } = await req.json();

  // Derive the app's origin so MCP server URLs are absolute.
  // On Vercel: this is https://your-app.vercel.app
  // Locally:   this is http://localhost:3000
  const origin =
    process.env.NEXT_PUBLIC_APP_URL ??
    `${req.headers.get("x-forwarded-proto") ?? "http"}://${req.headers.get("host")}`;

  // Connect to all registered MCP servers and merge their tools
  const { tools, clients } = await createMCPTools(origin);

  // Build the full message list:
  // The client sends all messages via useChat, so we use them directly.
  // The memory store is used to restore history across page refreshes.
  const result = streamText({
    model: openai("gpt-4o-mini"),
    system: SYSTEM_PROMPT,
    messages,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    tools: tools as any,
    maxSteps: 5, // how many tool-call → LLM iterations are allowed
    onFinish: async ({ response }) => {
      // Persist conversation so users can refresh and continue
      if (sessionId) {
        const toStore: StoredMessage[] = [
          ...messages
            .filter((m: { role: string }) => m.role === "user" || m.role === "assistant")
            .map((m: { role: string; content: unknown }) => ({
              role: m.role as "user" | "assistant",
              content: typeof m.content === "string" ? m.content : "",
            })),
          ...response.messages
            .filter((m) => m.role === "assistant")
            .map((m) => ({
              role: "assistant" as const,
              content:
                m.content
                  .filter((p): p is { type: "text"; text: string } => p.type === "text")
                  .map((p) => p.text)
                  .join("") ?? "",
            })),
        ];
        memoryStore.set(sessionId, toStore);
      }

      // Always close MCP connections to avoid resource leaks
      await Promise.all(clients.map((c) => c.close().catch(() => {})));
    },
  });

  return result.toDataStreamResponse();
}
