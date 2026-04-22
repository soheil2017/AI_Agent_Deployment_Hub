/**
 * MCP Client Factory
 * ──────────────────
 * Creates MCP client connections to all registered servers and merges
 * their tools into a single object that Vercel AI SDK's streamText can use.
 *
 * Each call creates fresh clients (required for serverless — no persistent
 * connections between function invocations). Clients are closed in onFinish.
 */

import { experimental_createMCPClient } from "ai";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { MCP_SERVERS } from "@/config/mcp.config";

type MCPClient = Awaited<ReturnType<typeof experimental_createMCPClient>>;

export interface MCPToolsResult {
  /** Merged tools from all connected MCP servers — pass directly to streamText */
  tools: Record<string, unknown>;
  /** All open clients — call client.close() on each in onFinish */
  clients: MCPClient[];
}

/**
 * Connect to every MCP server in the registry and collect their tools.
 *
 * @param origin  The base URL of the app (e.g. "http://localhost:3000").
 *                Used to build absolute URLs for the MCP server routes.
 */
export async function createMCPTools(origin: string): Promise<MCPToolsResult> {
  const clients: MCPClient[] = [];
  const allTools: Record<string, unknown> = {};

  await Promise.all(
    MCP_SERVERS.map(async (server) => {
      try {
        const client = await experimental_createMCPClient({
          transport: new StreamableHTTPClientTransport(
            new URL(server.path, origin)
          ),
        });

        const tools = await client.tools();
        Object.assign(allTools, tools);
        clients.push(client);
      } catch (err) {
        // One server failing should not crash the whole agent.
        // The agent will simply not have that server's tools available.
        console.warn(`[MCP] Failed to connect to "${server.name}":`, err);
      }
    })
  );

  return { tools: allTools, clients };
}
