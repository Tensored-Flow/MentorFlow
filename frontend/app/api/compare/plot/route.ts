import { NextResponse } from "next/server";

const BACKEND_BASE_URL = process.env.FLASK_API_BASE_URL ?? "https://mentorflow-0g4c.onrender.com";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/compare/plot`, { cache: "no-store" });
    const buf = await res.arrayBuffer();
    return new NextResponse(Buffer.from(buf), {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") || "image/png",
      },
    });
  } catch (error) {
    return NextResponse.json({ error: "Unable to fetch plot" }, { status: 502 });
  }
}
