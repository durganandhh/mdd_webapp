# PronounceRight frontend

React + Vite UI for the mispronunciation detection app. It records or uploads audio, sends it with a transcript to the backend `/score` endpoint, and shows which words were mispronounced.

## Run

```bash
cd mdd-frontend
npm install
cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env
npm run dev
```

`VITE_API_URL` in `.env` must point at the running backend (default `http://localhost:8000`). Start the backend first, from `backend/`:

```bash
uvicorn main:app --port 8000
```

## Scripts

- `npm run dev` - dev server
- `npm run build` - production build into `dist/`
- `npm run lint` - ESLint
