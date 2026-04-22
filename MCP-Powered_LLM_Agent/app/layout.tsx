import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MCP Research Agent",
  description:
    "An AI research assistant powered by OpenAI and Model Context Protocol (MCP) tools.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 antialiased">{children}</body>
    </html>
  );
}
