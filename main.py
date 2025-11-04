# main.py
import os
import random
import jwt
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, AsyncGenerator

from fastapi import FastAPI, HTTPException, Form, Depends, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict
from sqlmodel import SQLModel, Field as SQLField, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import cloudinary
import cloudinary.uploader

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ENV ----------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# ---------- CLOUDINARY ----------
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# ---------- DATABASE ----------
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# ---------- ASYNC SESSION DEPENDENCY ----------
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

# ---------- MODELS ----------
class Media(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    title: str
    category: str
    url: str
    media_type: str  # "image" or "gif"
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

class PaginatedMedia(BaseModel):
    page: int
    limit: int
    total_items: int
    total_pages: int
    items: List[MediaOut]
    model_config = ConfigDict(from_attributes=True)

# ---------- CONSTANTS ----------
ALLOWED_CATEGORIES = {
    "naruto", "one_piece", "demon_slayer", "jujutsu_kaisen",
    "attack_on_titan", "dragon_ball", "my_hero_academia",
    "pokemon", "spy_x_family", "solo_leveling", "nature", "popular_anime"
}

# ---------- AUTH ----------
security = HTTPBearer(auto_error=False)

def create_jwt_token(data: dict):
    data = data.copy()
    data.update({"exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

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

def validate_category(cat: str):
    cat = cat.lower().replace(" ", "_")
    if cat not in ALLOWED_CATEGORIES:
        raise HTTPException(400, f"Invalid category: {cat}")
    return cat

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables created successfully!")

# ---------- APP ----------
app = FastAPI(title="AnimePixels API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HELPERS ----------
async def increment_views(media: Media, session: AsyncSession):
    media.views += 1
    media.updated_at = datetime.utcnow()
    session.add(media)
    await session.commit()

# ---------- ROUTES ----------
@app.get("/")
def home():
    return {"message": "✅ AnimePixels API is running!"}

@app.get("/health")
async def health(session: AsyncSession = Depends(get_session)):
    try:
        await session.execute(select(1))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": "disconnected", "error": str(e)}

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid login")
    return {"token": create_jwt_token({"sub": username, "is_admin": True})}

@app.get("/admin/init-db")
async def init_db(admin: dict = Depends(verify_admin)):
    await create_db_and_tables()
    return {"status": "success", "message": "Database tables created successfully"}

@app.get("/admin/stats")
async def get_stats(admin: dict = Depends(verify_admin), session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Media))
    all_media = result.scalars().all()
    total_views = sum([m.views for m in all_media])
    return {"total_media": len(all_media), "total_views": total_views}

@app.post("/admin/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload(
    files: List[UploadFile] = File(...),
    titles: List[str] = Form(...),
    categories: List[str] = Form(...),
    media_type: str = Form(...),
    admin: dict = Depends(verify_admin),
    session: AsyncSession = Depends(get_session)
):
    if len(files) > 50:
        raise HTTPException(400, "Maximum 50 files allowed per upload")
    if media_type not in ["image", "gif"]:
        raise HTTPException(400, "media_type must be 'image' or 'gif'")

    uploaded, errors = [], []

    for idx, file in enumerate(files):
        try:
            title = titles[idx].strip()
            category = validate_category(categories[idx])
            upload = await asyncio.to_thread(
                cloudinary.uploader.upload,
                file.file,
                resource_type="image",
                folder=f"animepixels/{category}",
                use_filename=True,
                unique_filename=True,
                timeout=60
            )
            media = Media(
                title=title,
                category=category,
                url=upload["secure_url"],
                media_type=media_type,
                visible=True
            )
            session.add(media)
            await session.commit()
            uploaded.append(
                UploadedMediaItem(
                    filename=file.filename,
                    title=title,
                    category=category,
                    media=MediaOut.model_validate(media)
                )
            )
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
            await session.rollback()
        finally:
            await file.close()

    return BulkUploadResponse(success=len(uploaded), failed=len(errors), uploaded_media=uploaded, errors=errors)

# ---------- RANDOM ----------
@app.get("/random/image", response_model=MediaOut)
async def random_image(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Media.id).where(Media.visible==True, Media.media_type=="image"))
    ids = result.scalars().all()
    if not ids: raise HTTPException(404, "No images available")
    media = await session.get(Media, random.choice(ids))
    await increment_views(media, session)
    return media

@app.get("/random/gif", response_model=MediaOut)
async def random_gif(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Media.id).where(Media.visible==True, Media.media_type=="gif"))
    ids = result.scalars().all()
    if not ids: raise HTTPException(404, "No gifs available")
    media = await session.get(Media, random.choice(ids))
    await increment_views(media, session)
    return media

# ---------- GET BY ID ----------
@app.get("/image/{media_id}", response_model=MediaOut)
async def get_image(media_id: int, session: AsyncSession = Depends(get_session)):
    media = await session.get(Media, media_id)
    if not media or media.media_type != "image" or not media.visible:
        raise HTTPException(404, "Image not found")
    await increment_views(media, session)
    return media

@app.get("/gif/{media_id}", response_model=MediaOut)
async def get_gif(media_id: int, session: AsyncSession = Depends(get_session)):
    media = await session.get(Media, media_id)
    if not media or media.media_type != "gif" or not media.visible:
        raise HTTPException(404, "Gif not found")
    await increment_views(media, session)
    return media

# ---------- CATEGORY & SEARCH ----------
async def paginated_query(session: AsyncSession, media_type: str, category: str, page: int, limit: int):
    result = await session.execute(select(func.count(Media.id)).where(
        Media.visible==True, Media.category==category, Media.media_type==media_type
    ))
    total_items = result.scalar_one()
    total_pages = (total_items + limit - 1)//limit
    if page > total_pages and total_pages > 0:
        raise HTTPException(404, f"Page {page} does not exist")
    offset = (page-1)*limit
    result = await session.execute(select(Media).where(
        Media.visible==True, Media.category==category, Media.media_type==media_type
    ).offset(offset).limit(limit))
    items = result.scalars().all()
    for m in items:
        await increment_views(m, session)
    return PaginatedMedia(
        page=page,
        limit=limit,
        total_items=total_items,
        total_pages=total_pages,
        items=[MediaOut.model_validate(m) for m in items]
    )

async def search_query(session: AsyncSession, media_type: str, q: str, page: int, limit: int):
    pattern = f"%{q.lower()}%"
    result = await session.execute(select(func.count(Media.id)).where(
        Media.visible==True, Media.media_type==media_type,
        (Media.title.ilike(pattern)) | (Media.category.ilike(pattern))
    ))
    total_items = result.scalar_one()
    total_pages = (total_items + limit - 1)//limit
    if page > total_pages and total_pages > 0:
        raise HTTPException(404, f"Page {page} does not exist")
    offset = (page-1)*limit
    result = await session.execute(select(Media).where(
        Media.visible==True, Media.media_type==media_type,
        (Media.title.ilike(pattern)) | (Media.category.ilike(pattern))
    ).offset(offset).limit(limit))
    items = result.scalars().all()
    for m in items:
        await increment_views(m, session)
    return PaginatedMedia(
        page=page,
        limit=limit,
        total_items=total_items,
        total_pages=total_pages,
        items=[MediaOut.model_validate(m) for m in items]
    )

@app.get("/images/category/{category}", response_model=PaginatedMedia)
async def images_by_category(category: str, page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100),
                             session: AsyncSession = Depends(get_session)):
    category = validate_category(category)
    return await paginated_query(session, "image", category, page, limit)

@app.get("/gifs/category/{category}", response_model=PaginatedMedia)
async def gifs_by_category(category: str, page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100),
                           session: AsyncSession = Depends(get_session)):
    category = validate_category(category)
    return await paginated_query(session, "gif", category, page, limit)

@app.get("/search/images", response_model=PaginatedMedia)
async def search_images(q: str = Query(...), page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100),
                        session: AsyncSession = Depends(get_session)):
    return await search_query(session, "image", q, page, limit)

@app.get("/search/gifs", response_model=PaginatedMedia)
async def search_gifs(q: str = Query(...), page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100),
                      session: AsyncSession = Depends(get_session)):
    return await search_query(session, "gif", q, page, limit)
