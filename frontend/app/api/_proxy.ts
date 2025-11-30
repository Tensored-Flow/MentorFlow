import { NextResponse } from "next/server";

// Point Next.js API routes to the Flask backend (demo/app.py).
// Default to the Render deployment; override via FLASK_API_BASE_URL in env.
const BACKEND_BASE_URL = process.env.FLASK_API_BASE_URL ?? "https://mentorflow-0g4c.onrender.com";

const parseJsonSafe = (payload: string) => {
  if (!payload) return null;
  try {
    return JSON.parse(payload);
  } catch (error) {
    console.error("Failed to parse backend JSON", error, payload);
    return payload;
  }
};

export async function proxyJson(path: string, init: RequestInit = {}) {
  try {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init.headers || {}),
      },
      cache: "no-store",
    });

    const text = await response.text();
    const data = parseJsonSafe(text);

    if (!response.ok) {
      const errorMessage =
        typeof data === "string"
          ? data
          : (data && (data.error || data.message)) || "Upstream request failed";
      console.error(`[proxyJson] ${path} failed:`, errorMessage);
      return NextResponse.json({ error: errorMessage }, { status: response.status });
    }

    return NextResponse.json(
      typeof data === "string" ? { result: data } : data ?? { success: true },
      { status: response.status },
    );
  } catch (error) {
    console.error(`[proxyJson] ${path} crashed:`, error);
    return NextResponse.json({ error: "Unable to reach MentorFlow backend" }, { status: 502 });
  }
}
