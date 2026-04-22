import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // mathjs uses dynamic requires and native modules — exclude from Next.js
  // server bundling to prevent "Cannot find module" errors at runtime.
  serverExternalPackages: ["mathjs"],
};

export default nextConfig;
