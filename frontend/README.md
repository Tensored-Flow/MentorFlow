# MentorFlow Frontend (Next.js)

## Setup
```bash
cd frontend
npm install
# Configure backend URL (Flask demo/app.py, default 5050)
echo "FLASK_API_BASE_URL=http://localhost:5050" > .env.local
npm run dev   # or npm run build && npm run start
```

## Notes
- API routes proxy to `FLASK_API_BASE_URL` (default `http://localhost:5050`) via `app/api/_proxy.ts`.
- Ensure the backend is running: `FLASK_APP=demo/app.py FLASK_ENV=production python3 demo/app.py`.
- If deploying (e.g., Vercel), set `FLASK_API_BASE_URL` in project env vars to point to your public backend.
