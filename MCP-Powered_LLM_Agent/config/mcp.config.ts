/**
 * MCP Server Registry — Single Source of Truth
 * ─────────────────────────────────────────────
 * This file controls which tools the agent can use.
 *
 * HOW TO ADD A NEW TOOL (3 steps):
 *  1. Create  app/api/mcp/<your-tool>/route.ts  (see existing examples)
 *  2. Add one entry to the array below
 *  3. Redeploy — the agent auto-discovers new tools at startup
 *
 * No changes needed to the agent logic, chat route, or frontend.
 */

export interface MCPServerConfig {
  /** Unique identifier used in logs */
  name: string;
  /** Relative path to the MCP server API route */
  path: string;
  /** Human-readable description shown in README and logs */
  description: string;
}

export const MCP_SERVERS: MCPServerConfig[] = [
  {
    name: "web-search",
    path: "/api/mcp/search",
    description: "Search the web for real-time information using Tavily",
  },
  {
    name: "weather",
    path: "/api/mcp/weather",
    description: "Get current weather and forecasts via Open-Meteo (free, no API key)",
  },
  {
    name: "calculator",
    path: "/api/mcp/calculator",
    description: "Evaluate math expressions safely using mathjs",
  },

  // ↑ Add your new MCP server here. Example:
  // {
  //   name: "news",
  //   path: "/api/mcp/news",
  //   description: "Fetch latest news headlines",
  // },
];
