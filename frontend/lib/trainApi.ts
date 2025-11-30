const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

const buildUrl = (path: string) => `${API_BASE}${path}`;

const fetchJson = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
  const response = await fetch(buildUrl(path), {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `Request failed: ${path}`);
  }

  return response.json();
};

export const startTraining = (payload: Record<string, any> = { num_steps: 40, strategy: "ucb1" }) =>
  fetchJson<{ status: string; num_steps: number }>("/api/train/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const stopTraining = () =>
  fetchJson<{ status: string }>("/api/train/stop", {
    method: "POST",
  });

export const trainingStep = () =>
  fetchJson<Record<string, any>>("/api/train/step", {
    method: "POST",
  });

export const getTrainingStatus = () => fetchJson<Record<string, any>>("/api/train/status");

export const getTrainingProgress = () => fetchJson<Record<string, any>>("/api/train/progress");
