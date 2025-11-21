# Deploying Heart_attack_prediction to Render

This project contains a FastAPI backend (`backend/main.py`) and a static frontend (`frontend/index.html`). The backend serves the SPA index at `/` and exposes the prediction API at `/predict`.

Quick deploy steps on Render:

1. Push your repository to GitHub (if not already).
2. In the Render dashboard, create a new Web Service and connect your GitHub repo.
3. Use the following settings:
   - Environment: `Python 3` (choose a 3.10/3.11/3.12 runtime as available)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: (Render will use the `Procfile`) or set manually:
     - `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add any environment variables you need (none required by default). If you have model artifacts (`model/*.pkl`), ensure they are present in the repo or configure Render to fetch them at deploy time.

Notes and recommendations:
- `requirements.txt` includes `aiofiles` required by FastAPI `StaticFiles`.
- The `Procfile` (`web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT`) is present and will be used by Render.
- The frontend performs requests to `/predict` (relative path) so both API and SPA must run from the same origin — this layout serves SPA via FastAPI.
- If your model files are large, consider storing them in an object store (S3, DigitalOcean Spaces) and loading them at startup.

Local test:

```bash
# from project root
pip install -r requirements.txt
# run locally
python main.py
# open http://localhost:10000
```

If you want, I can also add a small GitHub Actions workflow that builds and deploys to Render's API, or add instructions to fetch model artifacts from an external storage during startup.
