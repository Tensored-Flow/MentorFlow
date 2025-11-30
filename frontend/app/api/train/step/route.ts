import { NextRequest } from "next/server";
import { proxyJson } from "../../_proxy";

export async function POST(request: NextRequest) {
  const body = await request.text();
  return proxyJson("/api/train/step", {
    method: "POST",
    body: body || undefined,
  });
}
