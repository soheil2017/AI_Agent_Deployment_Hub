/**
 * MCP Server — Calculator Tool
 * ─────────────────────────────
 * Implements the MCP Streamable HTTP protocol (JSON-RPC 2.0 over HTTP POST).
 * Uses mathjs for safe, accurate expression evaluation.
 *
 * Why use a tool instead of asking the LLM to calculate?
 * LLMs frequently make arithmetic mistakes. Using mathjs guarantees precision.
 *
 * Examples:
 *  "2 + 2"                 → 4
 *  "sqrt(144)"             → 12
 *  "15% of 340"            → 51
 *  "sin(pi / 2)"           → 1
 *  "3 km to m"             → 3000 m  (unit conversion)
 */

import { NextRequest, NextResponse } from "next/server";

// ── Tool Definition ──────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "calculate",
    description:
      "Evaluate a mathematical expression and return the precise result. " +
      "Supports arithmetic, algebra, trigonometry, unit conversions, and more. " +
      "Always use this tool for any calculation — never calculate mentally.",
    inputSchema: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description:
            'A math expression to evaluate. Examples: "2^10", "sqrt(256)", ' +
            '"15% * 340", "sin(pi/4)", "3 km to m".',
        },
      },
      required: ["expression"],
    },
  },
];

// ── Tool Implementation ──────────────────────────────────────────────────────

async function calculate(expression: string): Promise<string> {
  // Dynamic import avoids bundling issues — mathjs is in serverExternalPackages
  const { evaluate, format } = await import("mathjs");

  try {
    const result = evaluate(expression);
    // format() gives a clean string representation for complex numbers, units, etc.
    const formatted = format(result, { precision: 14 });
    return `${expression} = ${formatted}`;
  } catch (error) {
    return (
      `Could not evaluate: "${expression}"\n` +
      `Error: ${error instanceof Error ? error.message : String(error)}\n\n` +
      `Tip: Use standard math notation. Examples:\n` +
      `  • Powers: 2^8 or 2**8\n` +
      `  • Roots: sqrt(x) or cbrt(x)\n` +
      `  • Trig: sin(x), cos(x), tan(x) — x in radians\n` +
      `  • Constants: pi, e, phi`
    );
  }
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
        serverInfo: { name: "calculator-server", version: "1.0.0" },
      });

    case "notifications/initialized":
      return new NextResponse(null, { status: 204 });

    case "tools/list":
      return ok(id, { tools: TOOLS });

    case "tools/call": {
      const { name, arguments: args } = params ?? {};
      if (name === "calculate") {
        const result = await calculate(args?.expression);
        return ok(id, { content: [{ type: "text", text: result }] });
      }
      return err(id, -32601, `Unknown tool: ${name}`);
    }

    default:
      return err(id, -32601, `Method not found: ${method}`);
  }
}
