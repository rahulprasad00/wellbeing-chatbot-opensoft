// Prefer explicit env, otherwise fall back to dev localhost
export const server =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://127.0.0.1:8000";
