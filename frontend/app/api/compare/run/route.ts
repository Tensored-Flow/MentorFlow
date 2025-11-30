import { proxyJson } from "../../_proxy";

export async function POST() {
  return proxyJson("/api/compare/run");
}
