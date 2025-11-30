const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

const buildUrl = (path: string) => `${API_BASE}${path}`;

const fetchJson = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
  const response = await fetch(buildUrl(path), {
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    ...options,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `Request failed: ${path}`);
  }

  return response.json();
};

export type DifficultyState = {
  current_level: number;
  rolling_accuracy: number;
  history: boolean[];
};

export type TaskPayload = {
  arm_id?: number;
  family?: string;
  family_id?: number;
  topic?: string;
  prompt: string;
  choices: string[];
  answer_index: number;
  difficulty_level: number;
  difficulty_label: string;
  rolling_accuracy?: number;
};

export const fetchNextTask = (subject?: string) => {
  const query = subject ? `?subject=${encodeURIComponent(subject)}` : "";
  return fetchJson<TaskPayload>(`/api/task/next${query}`);
};

export const submitAttempt = (payload: { correct: boolean }) =>
  fetchJson<{ status: string; difficulty_state: DifficultyState; level_changed: boolean; previous_level: number }>(
    "/api/user/attempt",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
