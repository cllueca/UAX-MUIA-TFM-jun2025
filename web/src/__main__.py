# Builtins
# Installed
from fastapi import (
    FastAPI,
    Request,
    Depends
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
# Local
from src.config import CONFIG
from src.core.logger import CORE_LOGGER
from src.database_info.router import database_info_bp
from src.doctors.router import doctors_bp
from src.patients.router import patients_bp
from src.utils.auth import get_csession_token_from_cookie
# Types

app = FastAPI(
    title="TFM-demo",
    description="Demo API to showcase the computer vision models created for this TFM",
    version="0.1.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add templates folder
templates = Jinja2Templates(directory="templates")

# Render HTML template when accessing the URL
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.jinja.html", {"request": request})

# Render HTML template when accessing the URL
@app.get("/home", response_class=HTMLResponse)
async def home(
    request: Request,
    session = Depends(get_csession_token_from_cookie)
):
    return templates.TemplateResponse("main.jinja.html", {"request": request, "session": session})


if __name__ == '__main__':
    app.include_router(database_info_bp, prefix="/database")
    app.include_router(doctors_bp, prefix="/doctor")
    app.include_router(patients_bp, prefix="/patient")
    CORE_LOGGER.info("Running app...")
    CORE_LOGGER.info(f"Docs can be found on {CONFIG.LOGGER_COLORS['BOLD']}http://{CONFIG.API_HOST}:{CONFIG.API_PORT}/docs{CONFIG.LOGGER_COLORS['RESET']}")
    uvicorn.run(app, host=CONFIG.API_HOST, port=CONFIG.API_PORT)