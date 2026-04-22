/**
 * MCP Server — Web Search Tool
 * ─────────────────────────────
 * Implements the MCP Streamable HTTP protocol (JSON-RPC 2.0 over HTTP POST).
 * Uses the Tavily Search API for high-quality web results.
 *
 * Requires: TAVILY_API_KEY environment variable
 * Free tier: https://app.tavily.com
 */

import { NextRequest, NextResponse } from "next/server";

// ── Tool Definition ──────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "search_web",
    description:
      "Search the web for current, real-time information. Use this for recent events, " +
      "facts that may have changed, prices, news, documentation, and any topic " +
      "where up-to-date information matters.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query. Be specific for better results.",
        },
        max_results: {
          type: "number",
          description: "Number of results to return (1–10). Default is 5.",
        },
      },
      required: ["query"],
    },
  },
];

// ── Tool Implementation ──────────────────────────────────────────────────────

async function searchWeb(query: string, maxResults = 5): Promise<string> {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey) {
    return "Error: TAVILY_API_KEY is not set. Please add it to your .env.local file.";
  }

  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      query,
      max_results: Math.min(maxResults, 10),
      search_depth: "basic",
      include_answer: true,
    }),
  });

  if (!response.ok) {
    return `Search failed: ${response.status} ${response.statusText}`;
  }

  const data = await response.json();

  // Build a readable summary
  const lines: string[] = [];

  if (data.answer) {
    lines.push(`Summary: ${data.answer}\n`);
  }

  if (data.results?.length) {
    lines.push("Sources:");
    for (const result of data.results) {
      lines.push(`\n• ${result.title}`);
      lines.push(`  URL: ${result.url}`);
      lines.push(`  ${result.content}`);
    }
  }

  return lines.join("\n") || "No results found.";
}

// ── MCP Protocol Handler ─────────────────────────────────────────────────────

function ok(id: unknown, result: unknown) {
  return NextResponse.json({ jsonrpc: "2.0", id, result });
}

function err(id: unknown, code: number, message: string) {
  return NextResponse.json({ jsonrpc: "2.0", id, error: { code, message } });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { method, params, id } = body;

  switch (method) {
    case "initialize":
      return ok(id, {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "web-search-server", version: "1.0.0" },
      });

    case "notifications/initialized":
      // Notification — no response body needed
      return new NextResponse(null, { status: 204 });

    case "tools/list":
      return ok(id, { tools: TOOLS });

    case "tools/call": {
      const { name, arguments: args } = params ?? {};
      if (name === "search_web") {
        const result = await searchWeb(args?.query, args?.max_results);
        return ok(id, { content: [{ type: "text", text: result }] });
      }
      return err(id, -32601, `Unknown tool: ${name}`);
    }

    default:
      return err(id, -32601, `Method not found: ${method}`);
  }
}
