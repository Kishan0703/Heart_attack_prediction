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
4. **Environment Variables**:
   - `API_KEY`: Set this to a secure string to protect your `/predict` and `/explain` endpoints.
   - `PORT`: (Managed by Render).

Notes and recommendations:
- `requirements.txt` includes `aiofiles` required by FastAPI `StaticFiles`.
- The `Procfile` (`web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT`) is present and will be used by Render.
- The frontend performs requests to `/predict` (relative path) so both API and SPA must run from the same origin — this layout serves SPA via FastAPI.
- Ensure your model artifacts (`model/*.pkl`) are committed to the repository (if small enough) or fetched during the build step.

Local test:

```bash
# from project root
pip install -r requirements.txt
# run locally
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```
