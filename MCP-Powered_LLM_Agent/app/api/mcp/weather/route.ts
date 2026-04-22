/**
 * MCP Server — Weather Tool
 * ──────────────────────────
 * Implements the MCP Streamable HTTP protocol (JSON-RPC 2.0 over HTTP POST).
 * Uses Open-Meteo API — completely free, no API key required.
 *
 * Two-step process:
 *  1. Geocode the city name → latitude/longitude (Open-Meteo Geocoding API)
 *  2. Fetch weather data for those coordinates (Open-Meteo Forecast API)
 */

import { NextRequest, NextResponse } from "next/server";

// ── Tool Definition ──────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "get_weather",
    description:
      "Get current weather conditions and a 3-day forecast for any city or location. " +
      "Returns temperature, humidity, wind speed, and precipitation.",
    inputSchema: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description:
            'City name, optionally with country (e.g. "Paris", "Tokyo, Japan", "New York").',
        },
      },
      required: ["location"],
    },
  },
];

// ── Tool Implementation ──────────────────────────────────────────────────────

async function getWeather(location: string): Promise<string> {
  // Step 1: Geocode the location name → coordinates
  const geoRes = await fetch(
    `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1&language=en&format=json`
  );
  const geoData = await geoRes.json();

  if (!geoData.results?.length) {
    return (
      `Could not find "${location}". ` +
      `Try a different spelling or add the country name (e.g. "Paris, France").`
    );
  }

  const { latitude, longitude, name, country, timezone } = geoData.results[0];

  // Step 2: Fetch current conditions + 3-day forecast
  const weatherRes = await fetch(
    `https://api.open-meteo.com/v1/forecast` +
      `?latitude=${latitude}&longitude=${longitude}` +
      `&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,apparent_temperature` +
      `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code` +
      `&timezone=${encodeURIComponent(timezone ?? "auto")}` +
      `&forecast_days=3`
  );
  const w = await weatherRes.json();

  const weatherDescription = (code: number): string => {
    if (code === 0) return "Clear sky";
    if (code <= 3) return "Partly cloudy";
    if (code <= 48) return "Foggy";
    if (code <= 57) return "Drizzle";
    if (code <= 67) return "Rain";
    if (code <= 77) return "Snow";
    if (code <= 82) return "Rain showers";
    if (code <= 86) return "Snow showers";
    return "Thunderstorm";
  };

  const c = w.current;
  const d = w.daily;

  const result = {
    location: `${name}, ${country}`,
    timezone,
    current: {
      condition: weatherDescription(c.weather_code),
      temperature: `${c.temperature_2m}°C (feels like ${c.apparent_temperature}°C)`,
      humidity: `${c.relative_humidity_2m}%`,
      wind: `${c.wind_speed_10m} km/h`,
    },
    forecast: d.time.map((date: string, i: number) => ({
      date,
      condition: weatherDescription(d.weather_code[i]),
      high: `${d.temperature_2m_max[i]}°C`,
      low: `${d.temperature_2m_min[i]}°C`,
      precipitation: `${d.precipitation_sum[i]} mm`,
    })),
  };

  return JSON.stringify(result, null, 2);
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
        serverInfo: { name: "weather-server", version: "1.0.0" },
      });

    case "notifications/initialized":
      return new NextResponse(null, { status: 204 });

    case "tools/list":
      return ok(id, { tools: TOOLS });

    case "tools/call": {
      const { name, arguments: args } = params ?? {};
      if (name === "get_weather") {
        const result = await getWeather(args?.location);
        return ok(id, { content: [{ type: "text", text: result }] });
      }
      return err(id, -32601, `Unknown tool: ${name}`);
    }

    default:
      return err(id, -32601, `Method not found: ${method}`);
  }
}
