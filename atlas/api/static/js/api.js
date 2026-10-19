/* ATLAS — API client. One place that knows how to reach the backend,
   so workspaces deal in data rather than in fetch plumbing. */

const cache = new Map();

async function get(path, { fresh = false } = {}) {
  if (!fresh && cache.has(path)) return cache.get(path);
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  const data = await res.json();
  cache.set(path, data);
  return data;
}

async function post(path, body, { signal } = {}) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  const data = await res.json().catch(() => ({ error: 'malformed response' }));
  if (!res.ok) throw new Error(data.error || `${res.status}`);
  return data;
}

export const api = {
  meta: () => get('/api/meta'),
  registry: () => get('/api/registry'),
  compound: (key) => get(`/api/compound/${encodeURIComponent(key)}`),
  models: () => get('/api/models'),
  modelCard: () => get('/api/model-card'),
  verification: () => get('/api/verification'),
  calibration: () => get('/api/calibration'),
  pkParameters: () => get('/api/pk-parameters'),
  predict: (body, opts) => post('/api/predict', body, opts),
  simulate: (body, opts) => post('/api/simulate', body, opts),
  sensitivity: (body, opts) => post('/api/sensitivity', body, opts),
};
