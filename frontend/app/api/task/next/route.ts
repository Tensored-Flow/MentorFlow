import { NextRequest } from "next/server";
import { proxyJson } from "../../_proxy";

export async function GET(request: NextRequest) {
  const { search } = new URL(request.url);
  return proxyJson(`/api/task/next${search}`);
}
