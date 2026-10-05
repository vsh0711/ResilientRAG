import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResilientRAG",
  description: "Self-healing Retrieval-Augmented Generation agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
