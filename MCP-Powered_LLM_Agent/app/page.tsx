"use client";

import { useChat, type Message } from "ai/react";
import { useEffect, useRef, useState } from "react";

// ── Session Management ───────────────────────────────────────────────────────

function getOrCreateSessionId(): string {
  const KEY = "mcp-agent-session-id";
  const existing = localStorage.getItem(KEY);
  if (existing) return existing;
  const newId = crypto.randomUUID();
  localStorage.setItem(KEY, newId);
  return newId;
}

// ── Tool Call Badge ──────────────────────────────────────────────────────────

const TOOL_ICONS: Record<string, string> = {
  search_web: "🔍",
  get_weather: "🌤️",
  calculate: "🧮",
};

type ToolInvocation = {
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
  state: "call" | "result" | "partial-call";
  result?: unknown;
};

function ToolCallBadge({ tool }: { tool: ToolInvocation }) {
  const icon = TOOL_ICONS[tool.toolName] ?? "🔧";
  const isRunning = tool.state === "call";

  return (
    <div className="inline-flex items-start gap-2 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs w-full">
      <span className="mt-0.5">{icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-mono font-semibold text-gray-200">
            {tool.toolName}
          </span>
          {isRunning ? (
            <span className="text-amber-400 animate-pulse">running...</span>
          ) : (
            <span className="text-emerald-400">✓ done</span>
          )}
        </div>
        <div className="text-gray-500 mt-1 font-mono truncate">
          {JSON.stringify(tool.args)}
        </div>
      </div>
    </div>
  );
}

// ── Message Bubble ───────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-blue-600 rounded-2xl rounded-tr-sm px-4 py-2.5">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start gap-2.5">
      {/* Agent avatar */}
      <div className="w-7 h-7 rounded-full bg-emerald-500 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
        AI
      </div>

      <div className="max-w-[78%] space-y-2">
        {/* Tool invocations — shown above the text response */}
        {message.toolInvocations?.map((tool) => (
          <ToolCallBadge key={tool.toolCallId} tool={tool as ToolInvocation} />
        ))}

        {/* Text response */}
        {message.content && (
          <p className="text-sm leading-relaxed whitespace-pre-wrap text-gray-100">
            {message.content}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Typing Indicator ─────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-7 h-7 rounded-full bg-emerald-500 flex items-center justify-center text-xs font-bold shrink-0">
        AI
      </div>
      <div className="flex gap-1 items-center bg-gray-800 rounded-2xl px-3 py-2.5">
        {[0, 150, 300].map((delay) => (
          <div
            key={delay}
            className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

// ── Chat App (renders after session is loaded) ───────────────────────────────

function ChatApp({
  sessionId,
  initialMessages,
}: {
  sessionId: string;
  initialMessages: Message[];
}) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } =
    useChat({
      api: "/api/chat",
      body: { sessionId },
      initialMessages,
    });

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function handleNewChat() {
    await fetch(`/api/memory?sessionId=${sessionId}`, { method: "DELETE" });
    window.location.reload();
  }

  return (
    <div className="flex flex-col h-screen">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between border-b border-gray-800 px-5 py-3 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center text-sm font-bold">
            M
          </div>
          <div>
            <h1 className="font-semibold text-sm">MCP Research Agent</h1>
            <p className="text-xs text-gray-500">
              Web Search · Weather · Calculator
            </p>
          </div>
        </div>

        <button
          onClick={handleNewChat}
          className="text-xs text-gray-400 hover:text-gray-200 border border-gray-700 hover:border-gray-500 rounded-lg px-3 py-1.5 transition-colors"
        >
          New Chat
        </button>
      </header>

      {/* ── Messages ───────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4 -mt-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center text-3xl">
              🤖
            </div>
            <div>
              <h2 className="text-xl font-semibold mb-1">
                Hello! I&apos;m your Research Agent.
              </h2>
              <p className="text-sm text-gray-400 max-w-sm">
                Ask me anything. I can search the web, check the weather, and
                solve math problems in real time.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2 w-full max-w-sm mt-2">
              {[
                "What's the weather in Tokyo?",
                "What is 15% of $340?",
                "What are the latest AI news today?",
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() =>
                    handleSubmit(undefined, {
                      data: { prompt: suggestion },
                    })
                  }
                  className="text-sm text-left bg-gray-800 hover:bg-gray-750 border border-gray-700 hover:border-gray-600 rounded-xl px-4 py-2.5 transition-colors text-gray-300"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isLoading && <TypingIndicator />}

        <div ref={messagesEndRef} />
      </main>

      {/* ── Input Bar ──────────────────────────────────────────────────── */}
      <footer className="border-t border-gray-800 px-4 py-3 shrink-0">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask anything…"
            disabled={isLoading}
            className="flex-1 bg-gray-900 border border-gray-700 hover:border-gray-600 focus:border-blue-500 rounded-xl px-4 py-2.5 text-sm placeholder-gray-500 focus:outline-none transition-colors disabled:opacity-60"
          />
          {isLoading ? (
            <button
              type="button"
              onClick={stop}
              className="px-4 py-2.5 bg-red-600 hover:bg-red-700 rounded-xl text-sm font-medium transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm font-medium transition-colors"
            >
              Send
            </button>
          )}
        </form>
        <p className="text-center text-xs text-gray-600 mt-2">
          Powered by OpenAI + Model Context Protocol
        </p>
      </footer>
    </div>
  );
}

// ── Root Page — handles session loading ──────────────────────────────────────

export default function Home() {
  const [sessionId, setSessionId] = useState<string>("");
  const [initialMessages, setInitialMessages] = useState<Message[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const id = getOrCreateSessionId();
    setSessionId(id);

    // Restore previous conversation from server memory
    fetch(`/api/memory?sessionId=${id}`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.messages) && data.messages.length > 0) {
          const restored: Message[] = data.messages
            .filter(
              (m: { role: string; content: unknown }) =>
                m.role === "user" || m.role === "assistant"
            )
            .map((m: { role: string; content: unknown }, i: number) => ({
              id: `restored-${i}`,
              role: m.role as "user" | "assistant",
              content: typeof m.content === "string" ? m.content : "",
            }));
          setInitialMessages(restored);
        }
      })
      .catch(() => {
        // Memory fetch failed (e.g. first run) — start fresh
      })
      .finally(() => setHydrated(true));
  }, []);

  if (!hydrated) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return <ChatApp sessionId={sessionId} initialMessages={initialMessages} />;
}
