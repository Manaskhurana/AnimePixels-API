# api/main.py
from fastapi import FastAPI, HTTPException, Form, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select
from typing import Optional, List, Generator
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import random
import jwt
from datetime import datetime, timedelta
import cloudinary
import cloudinary.uploader
import logging

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Load environment
# -----------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # may be None in some envs
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Configure Cloudinary if env vars exist (no crash if missing)
if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )
    logger.info("Cloudinary configured.")
else:
    logger.warning("Cloudinary credentials not found in env. Upload endpoints will fail if used.")

# -----------------------
# Lazy DB engine
# -----------------------
_engine = None

def build_database_url(url: str) -> str:
    if not url:
        return url
    if "sslmode" not in url:
        if "?" in url:
            url += "&sslmode=require"
        else:
            url += "?sslmode=require"
    return url

def create_engine_lazy():
    """
    Create and cache the engine on first use.
    Returns None if DATABASE_URL not configured or creation fails.
    """
    global _engine
    if _engine is not None:
        return _engine
    if not DATABASE_URL:
        logger.warning("DATABASE_URL is not set. Database features disabled.")
        return None
    try:
        final_url = build_database_url(DATABASE_URL)
        logger.info("Creating DB engine (first 60 chars): %s...", final_url[:60])
        # Keep serverless-friendly defaults (avoid heavy pooling)
        _engine = create_engine(
            final_url,
            echo=False,
            connect_args={"connect_timeout": 30},
            pool_pre_ping=True,
        )
        return _engine
    except Exception as e:
        logger.exception("Failed to create DB engine: %s", e)
        _engine = None
        return None

# -----------------------
# Models & Pydantic
# -----------------------
class Media(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    title: str
    category: str
    url: str
    media_type: str
    views: int = 0
    visible: bool = True
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow)

class MediaOut(BaseModel):
    id: int
    title: str
    category: str
    url: str
    media_type: str
    views: int
    visible: bool

    model_config = ConfigDict(from_attributes=True)

class UploadedMediaItem(BaseModel):
    filename: str
    title: str
    category: str
    media: MediaOut

    model_config = ConfigDict(from_attributes=True)

class BulkUploadResponse(BaseModel):
    success: int
    failed: int
    uploaded_media: List[UploadedMediaItem]
    errors: List[dict]

# -----------------------
# Startup / Lifespan
# -----------------------
def create_db_and_tables() -> bool:
    engine = create_engine_lazy()
    if not engine:
        logger.warning("create_db_and_tables: engine not available; skipping table creation.")
        return False
    try:
        logger.info("Creating database tables...")
        SQLModel.metadata.create_all(engine)
        logger.info("✓ Database tables created successfully!")
        return True
    except Exception as e:
        logger.exception("✗ Error creating database tables: %s", e)
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting up...")
    create_engine_lazy()
    created = create_db_and_tables()
    if not created:
        logger.warning("Database tables weren't created at startup. DB endpoints will return descriptive errors.")
    yield
    logger.info("App shutting down...")

# -----------------------
# App & Middleware
# -----------------------
app = FastAPI(title="AnimePixels API", version="3.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Constants and Auth
# -----------------------
ALLOWED_CATEGORIES = {
    "naruto", "one_piece", "demon_slayer", "jujutsu_kaisen",
    "attack_on_titan", "dragon_ball", "my_hero_academia",
    "pokemon", "spy_x_family", "solo_leveling", "nature", "popular_anime"
}

security = HTTPBearer(auto_error=False)

def create_jwt_token(data: dict):
    data.update({"exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

# -----------------------
# Utility: session dependency
# -----------------------
def get_session() -> Generator:
    """
    Dependency that yields a DB session. Raises HTTPException(500) if DB not configured.
    """
    engine = create_engine_lazy()
    if not engine:
        logger.error("Attempted to get DB session but DATABASE_URL is not configured.")
        raise HTTPException(500, "Database is not configured. Check DATABASE_URL.")
    with Session(engine) as session:
        yield session

def validate_category(cat: str):
    cat = cat.lower().replace(" ", "_")
    if cat not in ALLOWED_CATEGORIES:
        raise HTTPException(400, f"Invalid category: {cat}")
    return cat

def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(401, "Missing token")
    try:
        data = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if not data.get("is_admin"):
            raise HTTPException(403, "Unauthorized")
        return data
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# -----------------------
# Routes (unchanged behavior)
# -----------------------
@app.get("/")
def home():
    return {"message": "✅ AnimePixels API is running!"}

@app.get("/favicon.ico")
def favicon():
    # Return No Content so browsers stop hitting 500s
    return JSONResponse(status_code=204, content=None)

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid login")
    return {"token": create_jwt_token({"sub": username, "is_admin": True})}

# ---------- ADMIN ENDPOINTS ----------
@app.get("/admin/init-db")
def init_db(admin: dict = Depends(verify_admin)):
    """Initialize/create database tables - Call this once"""
    success = create_db_and_tables()
    if success:
        return {"status": "success", "message": "Database tables created successfully"}
    else:
        raise HTTPException(500, "Failed to create database tables")

@app.get("/admin/stats")
def get_stats(admin: dict = Depends(verify_admin), session: Session = Depends(get_session)):
    """Get comprehensive database statistics"""
    try:
        all_media = session.exec(select(Media)).all()

        images = [m for m in all_media if m.media_type == "image"]
        gifs = [m for m in all_media if m.media_type == "gif"]

        categories = {}
        for media in all_media:
            if media.category not in categories:
                categories[media.category] = {"total": 0, "images": 0, "gifs": 0}
            categories[media.category]["total"] += 1
            if media.media_type == "image":
                categories[media.category]["images"] += 1
            else:
                categories[media.category]["gifs"] += 1

        total_views = sum([m.views for m in all_media])
        visible = len([m for m in all_media if m.visible])
        hidden = len([m for m in all_media if not m.visible])

        return {
            "status": "ok",
            "database": "connected",
            "summary": {
                "total_media": len(all_media),
                "total_images": len(images),
                "total_gifs": len(gifs),
                "visible": visible,
                "hidden": hidden,
                "total_views": total_views
            },
            "by_category": categories,
            "allowed_categories": list(ALLOWED_CATEGORIES)
        }
    except Exception as e:
        logger.exception("Error getting stats: %s", e)
        raise HTTPException(500, f"Database error: {str(e)}")

@app.get("/admin/tables")
def list_tables(admin: dict = Depends(verify_admin), session: Session = Depends(get_session)):
    """List all database tables and their status"""
    try:
        all_media = session.exec(select(Media)).all()

        return {
            "status": "ok",
            "tables": ["media"],
            "media": {
                "total_records": len(all_media),
                "images": len([m for m in all_media if m.media_type == "image"]),
                "gifs": len([m for m in all_media if m.media_type == "gif"])
            }
        }
    except Exception as e:
        logger.exception("Error listing tables: %s", e)
        raise HTTPException(500, f"Database error: {str(e)}")

# ---------- BULK UPLOAD ENDPOINT ----------
@app.post("/admin/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload(
    files: List[UploadFile] = File(...),
    titles: List[str] = Form(...),
    categories: List[str] = Form(...),
    media_type: str = Form(...),
    admin: dict = Depends(verify_admin),
    session: Session = Depends(get_session)
):
    """
    Upload multiple images or GIFs (up to 100 files) in one request.
    """
    # Validate Cloudinary config
    if not (CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET):
        raise HTTPException(500, "Cloudinary is not configured in environment variables")

    if len(files) > 100:
        raise HTTPException(400, "Maximum 100 files allowed per upload")

    if not files:
        raise HTTPException(400, "No files provided")

    if len(titles) != len(files):
        raise HTTPException(400, f"Titles count ({len(titles)}) != files count ({len(files)})")

    if len(categories) != len(files):
        raise HTTPException(400, f"Categories count ({len(categories)}) != files count ({len(files)})")

    if media_type not in ["image", "gif"]:
        raise HTTPException(400, "media_type must be 'image' or 'gif'")

    uploaded_media = []
    errors = []
    success_count = 0
    failed_count = 0

    for idx, file in enumerate(files):
        try:
            title = titles[idx].strip()
            category = categories[idx].strip()

            if not title:
                raise ValueError("Title cannot be empty")

            category = validate_category(category)

            contents = await file.read()
            logger.info("Uploading: %s (%d bytes)", file.filename, len(contents))

            resource_type = "image"

            upload_params = {
                "resource_type": resource_type,
                "folder": f"animepixels/{category}",
                "use_filename": True,
                "unique_filename": True,
                "timeout": 60,
            }

            if media_type == "gif":
                upload_params["format"] = "gif"
                upload_params["flags"] = "animated"

            logger.info("Upload params: %s", upload_params)
            upload_result = cloudinary.uploader.upload(contents, **upload_params)

            secure_url = upload_result.get("secure_url")
            if not secure_url:
                raise RuntimeError("Cloudinary did not return secure_url")

            logger.info("Cloudinary upload successful: %s", secure_url)

            media = Media(
                title=title,
                category=category,
                url=secure_url,
                media_type=media_type,
                visible=True,
                views=0
            )

            session.add(media)
            session.flush()
            session.commit()

            logger.info("Database save successful: Media ID %s", media.id)

            media_dict = {
                "id": media.id,
                "title": media.title,
                "category": media.category,
                "url": media.url,
                "media_type": media.media_type,
                "views": media.views,
                "visible": media.visible
            }

            uploaded_item = UploadedMediaItem(
                filename=file.filename,
                title=title,
                category=category,
                media=MediaOut(**media_dict)
            )
            uploaded_media.append(uploaded_item)
            success_count += 1

        except Exception as e:
            logger.exception("Error uploading %s: %s", file.filename, e)
            failed_count += 1
            errors.append({
                "filename": file.filename,
                "index": idx,
                "error": str(e)
            })
            try:
                session.rollback()
            except Exception:
                pass
        finally:
            try:
                await file.close()
            except Exception:
                pass

    return BulkUploadResponse(
        success=success_count,
        failed=failed_count,
        uploaded_media=uploaded_media,
        errors=errors
    )

# ---------- RANDOM ENDPOINTS ----------
@app.get("/random")
def random_any(session: Session = Depends(get_session)):
    try:
        items = session.exec(select(Media).where(Media.visible == True)).all()
        if not items:
            raise HTTPException(404, "No media available")
        return random.choice(items)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/random/image")
def random_image_global(session: Session = Depends(get_session)):
    try:
        items = session.exec(select(Media).where(
            Media.media_type == "image",
            Media.visible == True
        )).all()
        if not items:
            raise HTTPException(404, "No images available")
        return random.choice(items)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/random/gif")
def random_gif_global(session: Session = Depends(get_session)):
    try:
        items = session.exec(select(Media).where(
            Media.media_type == "gif",
            Media.visible == True
        )).all()
        if not items:
            raise HTTPException(404, "No gifs available")
        return random.choice(items)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/random/image/{category}")
def random_image_by_category(category: str, session: Session = Depends(get_session)):
    try:
        category = validate_category(category)
        items = session.exec(select(Media).where(
            Media.category == category,
            Media.media_type == "image",
            Media.visible == True
        )).all()
        if not items:
            raise HTTPException(404, "No images in this category")
        return random.choice(items)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/random/gif/{category}")
def random_gif_by_category(category: str, session: Session = Depends(get_session)):
    try:
        category = validate_category(category)
        items = session.exec(select(Media).where(
            Media.category == category,
            Media.media_type == "gif",
            Media.visible == True
        )).all()
        if not items:
            raise HTTPException(404, "No gifs in this category")
        return random.choice(items)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

# ---------- GET BY ID ----------
@app.get("/image/id/{media_id}", response_model=MediaOut)
def get_image_by_id(media_id: int, session: Session = Depends(get_session)):
    try:
        media = session.get(Media, media_id)
        if not media or not media.visible or media.media_type != "image":
            raise HTTPException(404, "Image not found")
        media.views += 1
        session.add(media)
        session.commit()
        return media
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/gif/id/{media_id}", response_model=MediaOut)
def get_gif_by_id(media_id: int, session: Session = Depends(get_session)):
    try:
        media = session.get(Media, media_id)
        if not media or not media.visible or media.media_type != "gif":
            raise HTTPException(404, "GIF not found")
        media.views += 1
        session.add(media)
        session.commit()
        return media
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

# ---------- SEARCH ----------
@app.get("/search/image", response_model=List[MediaOut])
def search_images(query: str, session: Session = Depends(get_session)):
    try:
        pattern = f"%{query.lower()}%"
        results = session.exec(select(Media).where(
            Media.visible == True,
            Media.media_type == "image",
            (Media.title.ilike(pattern)) | (Media.category.ilike(pattern))
        )).all()
        if not results:
            raise HTTPException(404, "No images found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/search/gif", response_model=List[MediaOut])
def search_gifs(query: str, session: Session = Depends(get_session)):
    try:
        pattern = f"%{query.lower()}%"
        results = session.exec(select(Media).where(
            Media.visible == True,
            Media.media_type == "gif",
            (Media.title.ilike(pattern)) | (Media.category.ilike(pattern))
        )).all()
        if not results:
            raise HTTPException(404, "No gifs found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

# ---------- BY CATEGORY ----------
@app.get("/image/{category}", response_model=List[MediaOut])
def get_images_by_category(category: str, session: Session = Depends(get_session)):
    try:
        category = validate_category(category)
        items = session.exec(select(Media).where(
            Media.category == category,
            Media.media_type == "image",
            Media.visible == True
        )).all()
        if not items:
            raise HTTPException(404, "No images in this category")
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

@app.get("/gif/{category}", response_model=List[MediaOut])
def get_gifs_by_category(category: str, session: Session = Depends(get_session)):
    try:
        category = validate_category(category)
        items = session.exec(select(Media).where(
            Media.category == category,
            Media.media_type == "gif",
            Media.visible == True
        )).all()
        if not items:
            raise HTTPException(404, "No gifs in this category")
        return items
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error: %s", e)
        raise HTTPException(500, "Database error")

# ---------- HEALTH CHECK ----------
@app.get("/health")
def health(session: Session = Depends(get_session)):
    try:
        session.exec(select(1))
        all_media = session.exec(select(Media)).all()
        images = len([m for m in all_media if m.media_type == "image"])
        gifs = len([m for m in all_media if m.media_type == "gif"])
        return {
            "status": "ok",
            "database": "connected",
            "total_media": len(all_media),
            "images": images,
            "gifs": gifs
        }
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        return {
            "status": "error",
            "database": "disconnected",
            "error": str(e)
        }
