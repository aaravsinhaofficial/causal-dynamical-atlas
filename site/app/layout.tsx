import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "CADENCE — A cross-animal causal dynamical atlas",
    template: "%s · CADENCE",
  },
  description:
    "A prospective, leakage-sealed test of whether intervention dynamics learned in donor animals transport to a new animal calibrated on normal activity only.",
  keywords: [
    "causal dynamics",
    "neuroscience",
    "cross-animal transfer",
    "neural intervention",
    "prospective evaluation",
  ],
  authors: [{ name: "CADENCE project" }],
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f0eee7" },
    { media: "(prefers-color-scheme: dark)", color: "#0a151b" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
