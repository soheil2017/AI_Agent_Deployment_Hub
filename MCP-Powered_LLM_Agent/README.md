# MCP-Powered LLM Agent

A production-ready AI research assistant built with **Next.js 15**, **OpenAI**, and the **Model Context Protocol (MCP)**. The agent can search the web, check the weather, and solve math problems in real time — and you can add new tools in minutes without touching the agent logic.

---

## Table of Contents

1. [What is this?](#1-what-is-this)
2. [What is MCP?](#2-what-is-mcp)
3. [How it works](#3-how-it-works)
4. [Tech Stack](#4-tech-stack)
5. [Project Structure](#5-project-structure)
6. [Prerequisites](#6-prerequisites)
7. [Setup (Local Development)](#7-setup-local-development)
8. [Environment Variables](#8-environment-variables)
9. [Running Locally](#9-running-locally)
10. [How to Add a New Tool](#10-how-to-add-a-new-tool)
11. [Memory System](#11-memory-system)
12. [Deploying to Vercel](#12-deploying-to-vercel)
13. [Common Issues & Troubleshooting](#13-common-issues--troubleshooting)
14. [Architecture Decisions](#14-architecture-decisions)

---

## 1. What is this?

This project is a **chat-based AI agent** that uses real-world tools to answer questions. Unlike a plain ChatGPT session, this agent can:

- **Search the web** for current information (news, prices, recent events)
- **Check the weather** for any city on Earth (current + 3-day forecast)
- **Solve math problems** precisely — never guesses, always uses a real calculator

You interact with it through a chat UI that shows you exactly which tools the agent is using and when.

**Key design goal: scalability.** Adding a new tool (e.g. a news reader, a stock price checker, a database lookup) requires:
1. Creating one new file
2. Adding one line to a config file

No changes to the agent, the UI, or the chat logic.

---

## 2. What is MCP?

**Model Context Protocol (MCP)** is an open standard created by Anthropic that defines how AI models communicate with external tools and data sources.

Think of it like this:

```
Without MCP:                  With MCP:
──────────────────            ────────────────────────────────────────
LLM knows only what           LLM + tools registry (MCP)
it was trained on.            LLM can call any registered tool and
Can't search, can't           get real data back in real time.
check live data.
```

In this project, each tool (search, weather, calculator) is an **MCP Server** — a small API that the agent connects to at runtime. The agent discovers what tools are available, decides which ones to use, calls them, and uses the results to formulate its answer.

**MCP uses JSON-RPC 2.0** — a simple request/response format. Every MCP server must respond to three types of requests:

| Request method    | What it does                                        |
|-------------------|-----------------------------------------------------|
| `initialize`      | Handshake — agent says hello, server confirms ready |
| `tools/list`      | Agent asks "what can you do?"                       |
| `tools/call`      | Agent says "run this tool with these arguments"     |

---

## 3. How it works

Here is the full request flow when you send a message:

```
Browser (Chat UI)
  │
  │ 1. On page load: GET /api/memory?sessionId=xxx
  │    → Restores previous conversation (if any)
  │
  │ 2. POST /api/chat  { messages: [...], sessionId: "xxx" }
  │
  └──► app/api/chat/route.ts
            │
            │ Reads: config/mcp.config.ts (list of MCP servers)
            │
            ├──► POST /api/mcp/search      initialize → tools/list
            ├──► POST /api/mcp/weather     initialize → tools/list
            └──► POST /api/mcp/calculator  initialize → tools/list
            │
            │ Now knows all available tools.
            │
            │ Calls: OpenAI gpt-4o-mini with all tools
            │
            │ Model decides: "I need to search the web"
            │
            ├──► POST /api/mcp/search  tools/call { query: "..." }
            │    ← Returns search results
            │
            │ Model reads results, formulates answer.
            │
            │ Streams answer back to browser  ◄────────────────────
            │
            │ onFinish: saves history to lib/memory.ts
            └── closes all MCP connections
```

The browser receives the response as a **stream** — you see words appear in real time, and tool call badges appear as tools are invoked.

---

## 4. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Framework** | Next.js 15 (App Router) | API routes + React UI in one project; deploys to Vercel out of the box |
| **LLM** | OpenAI gpt-4o-mini | Fast, affordable, excellent at tool use |
| **AI SDK** | Vercel AI SDK v4 (`ai` package) | Streaming, tool call management, `useChat` hook |
| **MCP Protocol** | `@modelcontextprotocol/sdk` | Transport layer between agent and tool servers |
| **Web Search** | Tavily API | High-quality search results; free tier available |
| **Weather** | Open-Meteo API | Free, no API key needed, global coverage |
| **Calculator** | mathjs | Accurate math evaluation; handles units, trig, etc. |
| **Styling** | Tailwind CSS v4 | Utility-first CSS; dark theme |
| **Memory** | In-memory Map (→ Vercel KV) | Simple default; easy upgrade path for production |
| **Deployment** | Vercel | Free tier, serverless, automatic HTTPS |

---

## 5. Project Structure

```
MCP-Powered_LLM_Agent/
│
├── config/
│   └── mcp.config.ts          ← THE ONLY FILE YOU EDIT TO ADD NEW TOOLS
│                                  (list of all MCP server paths)
│
├── lib/
│   ├── memory.ts              ← Conversation storage (in-memory → Vercel KV)
│   └── mcp-client.ts          ← Connects to all MCP servers, merges tools
│
├── app/
│   ├── layout.tsx             ← Root HTML layout
│   ├── globals.css            ← Tailwind CSS import + scrollbar styles
│   ├── page.tsx               ← Chat UI (handles session, renders messages)
│   │
│   └── api/
│       ├── chat/
│       │   └── route.ts       ← Main agent endpoint (POST — streams response)
│       │
│       ├── memory/
│       │   └── route.ts       ← History API (GET = restore, DELETE = clear)
│       │
│       └── mcp/
│           ├── search/
│           │   └── route.ts   ← MCP Server: web search via Tavily
│           ├── weather/
│           │   └── route.ts   ← MCP Server: weather via Open-Meteo (free)
│           └── calculator/
│               └── route.ts   ← MCP Server: math via mathjs
│
├── .env.example               ← Template for environment variables
├── .env.local                 ← YOUR secrets (not committed to git)
├── next.config.ts             ← Next.js config (mathjs external package)
├── vercel.json                ← Vercel deployment config (60s timeout)
├── package.json
└── tsconfig.json
```

---

## 6. Prerequisites

Before you start, make sure you have:

| Requirement | Version | How to check |
|-------------|---------|--------------|
| **Node.js** | 18.x or higher | `node --version` |
| **npm** | 9.x or higher | `npm --version` |
| **Git** | Any | `git --version` |
| **OpenAI API key** | — | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Tavily API key** | — | [app.tavily.com](https://app.tavily.com) (free account) |

> **About API keys:** An API key is like a password that proves to the service that your app has permission to use it. Keep them secret — never put them in your code or commit them to git.

---

## 7. Setup (Local Development)

Follow these steps exactly, in order:

### Step 1 — Clone the repo

```bash
git clone <your-repo-url>
cd MCP-Powered_LLM_Agent
```

### Step 2 — Install dependencies

```bash
npm install
```

This downloads all the packages listed in `package.json` into the `node_modules/` folder.

### Step 3 — Set up your environment variables

```bash
cp .env.example .env.local
```

Now open `.env.local` in your editor and fill in your API keys:

```env
OPENAI_API_KEY=sk-...your-key-here...
TAVILY_API_KEY=tvly-...your-key-here...
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

> **Important:** `.env.local` is listed in `.gitignore` and will never be committed. This protects your secret keys.

---

## 8. Environment Variables

| Variable | Required | Description | Where to get it |
|----------|----------|-------------|----------------|
| `OPENAI_API_KEY` | Yes | Authenticates requests to OpenAI's API | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `TAVILY_API_KEY` | Yes | Authenticates requests to Tavily Search | [app.tavily.com](https://app.tavily.com) — sign up, go to API Keys |
| `NEXT_PUBLIC_APP_URL` | Yes | The app's own URL — used so `/api/chat` can call `/api/mcp/*` routes by absolute URL | `http://localhost:3000` locally; your Vercel URL in production |

> **Why does the chat route need the app's own URL?**
>
> The `/api/chat` route runs server-side and needs to call the MCP tool routes (`/api/mcp/search`, etc.) as if it were a separate client. In serverless environments (Vercel), server-side code cannot use relative URLs — it needs the full `https://your-app.vercel.app/api/mcp/search`. That's what `NEXT_PUBLIC_APP_URL` provides.

---

## 9. Running Locally

```bash
npm run dev
```

Open your browser and go to: **http://localhost:3000**

You should see the chat interface. Try asking:
- `"What's the weather in Paris?"`
- `"What is 123 * 456?"`
- `"What happened in tech news this week?"`

Watch the tool call badges appear as the agent works!

---

## 10. How to Add a New Tool

This is the core feature of the scalable architecture. Adding a new tool takes about **5 minutes** and requires changes in **exactly 2 places**.

### Example: Adding a "Current Time" tool

#### Step 1 — Create the MCP server route

Create a new file: `app/api/mcp/time/route.ts`

```typescript
import { NextRequest, NextResponse } from "next/server";

// Define what tools this server provides
const TOOLS = [
  {
    name: "get_current_time",
    description: "Get the current date and time for any timezone.",
    inputSchema: {
      type: "object",
      properties: {
        timezone: {
          type: "string",
          description: 'IANA timezone name, e.g. "America/New_York", "Europe/London"',
        },
      },
      required: ["timezone"],
    },
  },
];

// The actual tool logic
async function getCurrentTime(timezone: string): Promise<string> {
  const now = new Date();
  return now.toLocaleString("en-US", { timeZone: timezone, timeStyle: "full", dateStyle: "full" });
}

// JSON-RPC response helpers
function ok(id: unknown, result: unknown) {
  return NextResponse.json({ jsonrpc: "2.0", id, result });
}
function err(id: unknown, code: number, message: string) {
  return NextResponse.json({ jsonrpc: "2.0", id, error: { code, message } });
}

// MCP protocol handler — must handle these 4 methods
export async function POST(req: NextRequest) {
  const { method, params, id } = await req.json();

  switch (method) {
    case "initialize":
      return ok(id, {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "time-server", version: "1.0.0" },
      });

    case "notifications/initialized":
      return new NextResponse(null, { status: 204 });

    case "tools/list":
      return ok(id, { tools: TOOLS });

    case "tools/call": {
      const { name, arguments: args } = params ?? {};
      if (name === "get_current_time") {
        const result = await getCurrentTime(args.timezone);
        return ok(id, { content: [{ type: "text", text: result }] });
      }
      return err(id, -32601, `Unknown tool: ${name}`);
    }

    default:
      return err(id, -32601, `Method not found: ${method}`);
  }
}
```

#### Step 2 — Register it in the config

Open `config/mcp.config.ts` and add one line:

```typescript
export const MCP_SERVERS: MCPServerConfig[] = [
  { name: "web-search",  path: "/api/mcp/search",     description: "..." },
  { name: "weather",     path: "/api/mcp/weather",    description: "..." },
  { name: "calculator",  path: "/api/mcp/calculator", description: "..." },

  // ← Add this line:
  { name: "time", path: "/api/mcp/time", description: "Get current time for any timezone" },
];
```

#### That's it!

Restart `npm run dev` and the agent will automatically discover and use the new tool. The agent will now respond to questions like *"What time is it in Tokyo?"* using your new tool.

---

## 11. Memory System

The agent remembers your conversation so you can refresh the page and continue where you left off.

### How it works

```
Browser                     Server (lib/memory.ts)
──────                      ──────────────────────
page loads
  → fetch /api/memory        reads from Map<sessionId, messages[]>
  ← returns saved messages
  renders chat with history

user sends message
  → POST /api/chat           streamText runs...
  ← streams response         onFinish: saves updated messages to Map

user clicks "New Chat"
  → DELETE /api/memory       deletes sessionId from Map
  ← ok
  page reloads (fresh)
```

### Session ID

Your browser generates a random session ID (a UUID like `a1b2c3d4-...`) and stores it in `localStorage`. This ID is sent with every request so the server knows which conversation history to retrieve.

### Limitations of the default implementation

The default memory uses a JavaScript `Map` (an in-memory data structure). This means:

- **Memory resets when the server restarts** (e.g. after `npm run dev` restart or Vercel redeployment)
- **On Vercel, different requests may hit different server instances** — those instances don't share memory

For a demo, this is fine. For production, upgrade to Vercel KV:

### Upgrading to Vercel KV (Production)

1. Create a KV database:
   ```bash
   vercel kv create my-agent-memory
   ```

2. Install the package:
   ```bash
   npm install @vercel/kv
   ```

3. In `lib/memory.ts`, comment out the in-memory block and uncomment the KV block at the bottom of the file.

Done — all reads and writes now go to Redis, shared across all instances.

---

## 12. Deploying to Vercel

### Step 1 — Create a Vercel account

Sign up at [vercel.com](https://vercel.com) (free).

### Step 2 — Install the Vercel CLI

```bash
npm install -g vercel
```

### Step 3 — Login

```bash
vercel login
```

### Step 4 — Link your project

```bash
cd MCP-Powered_LLM_Agent
vercel link
```

Follow the prompts. This connects your local folder to a Vercel project.

### Step 5 — Set environment variables in Vercel

Go to your project in the [Vercel Dashboard](https://vercel.com/dashboard):
1. Click **Settings** → **Environment Variables**
2. Add these three variables:

| Name | Value |
|------|-------|
| `OPENAI_API_KEY` | Your OpenAI key |
| `TAVILY_API_KEY` | Your Tavily key |
| `NEXT_PUBLIC_APP_URL` | `https://your-project-name.vercel.app` |

> **Critical:** `NEXT_PUBLIC_APP_URL` must match your Vercel deployment URL exactly. You can find it in the Vercel dashboard after the first deploy.

### Step 6 — Deploy

```bash
vercel --prod
```

Your app is now live at `https://your-project-name.vercel.app`!

### Re-deploying after changes

```bash
vercel --prod
```

Or push to GitHub and connect your repo in Vercel for automatic deployments on every push.

---

## 13. Common Issues & Troubleshooting

### "OPENAI_API_KEY is not set"
Make sure your `.env.local` file exists and contains the key. It must be in the `MCP-Powered_LLM_Agent/` folder, not in a parent folder.

### "TAVILY_API_KEY is not set"
Same as above. Get your key from [app.tavily.com](https://app.tavily.com).

### Tools aren't being called
- Make sure `NEXT_PUBLIC_APP_URL` is set correctly in `.env.local`
- Check the browser console (F12) and the terminal running `npm run dev` for errors
- Try asking the agent directly: *"Use the calculator to compute 2+2"*

### "Cannot find module 'mathjs'"
This usually means the `serverExternalPackages` config isn't applied. Make sure `next.config.ts` contains:
```typescript
serverExternalPackages: ["mathjs"],
```

### Memory doesn't persist after server restart
This is expected behavior with the in-memory implementation. See [Section 11](#11-memory-system) for the Vercel KV upgrade path.

### Vercel deployment timeout
The `/api/chat` route has `maxDuration = 60` (60 seconds). If your tool chains are very long, upgrade to Vercel Pro which allows up to 300 seconds.

### The weather tool returns "Could not find location"
Try adding the country name: *"Paris, France"* instead of just *"Paris"*.

---

## 14. Architecture Decisions

### Why implement MCP protocol manually instead of using `McpServer` from the SDK?

The `@modelcontextprotocol/sdk` provides a `McpServer` class designed for standalone Node.js processes. Inside Next.js App Router route handlers, you work with Web API `Request`/`Response` objects — not Node.js `IncomingMessage`/`ServerResponse`. Adapting them adds complexity and fragility in serverless environments.

The MCP protocol itself (JSON-RPC 2.0) is simple enough to implement directly in ~40 lines per route. This approach is more readable, debuggable, and serverless-compatible.

### Why `ai@4.3.19` (pinned) instead of the latest?

`experimental_createMCPClient` carries the `experimental_` prefix — its API can change in any release. Pinning the version prevents surprise breakage. When you want to upgrade, read the changelog first.

### Why in-memory Map instead of Vercel KV from the start?

Zero setup. The in-memory store works immediately with no external services. The abstraction in `lib/memory.ts` makes the upgrade to Vercel KV a 3-line change with no impact on the rest of the code.

### Why send full message history from the client instead of loading from server on every request?

Vercel AI SDK's `useChat` hook manages the message array client-side and sends the full history with each request. This is simpler than maintaining a session state machine and works reliably across serverless function instances that don't share memory.

The server-side memory exists only for page-refresh recovery — not for per-request context.

---

## Quick Reference

```bash
# Local development
npm run dev

# Build for production (test before deploying)
npm run build

# Deploy to Vercel
vercel --prod

# Add a new tool
# 1. Create: app/api/mcp/<name>/route.ts
# 2. Register: config/mcp.config.ts
```

---

*Built with [Next.js](https://nextjs.org), [Vercel AI SDK](https://sdk.vercel.ai), and [Model Context Protocol](https://modelcontextprotocol.io).*
