import os
import random
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Form, Depends, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict
from sqlmodel import SQLModel, Field as SQLField, select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import cloudinary
import cloudinary.uploader

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting AnimePixels API initialization...")
    
    # ---------- ENV ----------
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    
    logger.info(f"DATABASE_URL found: {DATABASE_URL[:50]}...")
    
    # Fix postgres:// scheme for asyncpg
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
        logger.info("Converted postgres:// to postgresql+asyncpg://")
    
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
    JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
    
    logger.info("Environment variables loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load environment variables: {str(e)}", exc_info=True)
    raise

# ---------- CLOUDINARY ----------
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )
    logger.info("Cloudinary configured successfully")
except Exception as e:
    logger.error(f"Cloudinary configuration failed: {str(e)}", exc_info=True)

# ---------- DATABASE ENGINE (No Pooling for Vercel) ----------
try:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True,
        # Disable pooling entirely for serverless
        pool_size=0,
        max_overflow=0,
        connect_args={
            "timeout": 10,
            "command_timeout": 10,
            "server_settings": {"application_name": "animepixels_api"},
        }
    )
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}", exc_info=True)
    raise

# Use sessionmaker without pooling
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

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

# ---------- ASYNC SESSION DEPENDENCY ----------
async def get_session():
    """Get database session - fresh connection per request"""
    session = None
    try:
        session = async_session()
        logger.debug("Database session created")
        yield session
    except Exception as e:
        logger.error(f"Database session error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Database error: {str(e)}")
    finally:
        if session:
            try:
                await session.close()
                logger.debug("Database session closed")
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")

# ---------- AUTH ----------
security = HTTPBearer(auto_error=False)

def create_jwt_token(data: dict):
    data = data.copy()
    data.update({"exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        logger.warning("Admin request without token")
        raise HTTPException(401, "Missing token")
    try:
        data = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if not data.get("is_admin"):
            logger.warning("Non-admin token used for admin endpoint")
            raise HTTPException(403, "Unauthorized")
        return data
    except jwt.ExpiredSignatureError:
        logger.warning("Expired token used")
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise HTTPException(401, "Invalid token")

def validate_category(cat: str):
    cat = cat.lower().replace(" ", "_")
    if cat not in ALLOWED_CATEGORIES:
        raise HTTPException(400, f"Invalid category: {cat}")
    return cat

async def create_db_and_tables():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Failed to create tables: {str(e)}", exc_info=True)
        raise

# ---------- APP ----------
app = FastAPI(title="AnimePixels API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI app initialized successfully")

# ---------- HELPERS ----------
async def increment_views(media: Media, session: AsyncSession):
    """Increment view count for media"""
    try:
        media.views += 1
        media.updated_at = datetime.utcnow()
        session.add(media)
        await session.commit()
        logger.debug(f"Views incremented for media {media.id}")
    except Exception as e:
        logger.error(f"Failed to increment views: {str(e)}")
        await session.rollback()

# ---------- ROUTES ----------
@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {
        "message": "✅ AnimePixels API is running!",
        "status": "ok",
        "version": "5.0"
    }

@app.get("/health")
async def health(session: AsyncSession = Depends(get_session)):
    """Health check endpoint"""
    try:
        logger.info("Health check initiated")
        result = await session.execute(select(func.count(Media.id)))
        count = result.scalar_one_or_none() or 0
        logger.info(f"Health check passed. Media count: {count}")
        return {
            "status": "ok",
            "database": "connected",
            "media_count": count
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    """Admin login endpoint"""
    logger.info(f"Login attempt for user: {username}")
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        logger.warning(f"Failed login attempt for user: {username}")
        raise HTTPException(401, "Invalid login")
    token = create_jwt_token({"sub": username, "is_admin": True})
    logger.info(f"Successful login for user: {username}")
    return {"token": token, "expires_in": JWT_EXPIRE_MINUTES * 60}

@app.get("/admin/init-db")
async def init_db(admin: dict = Depends(verify_admin)):
    """Initialize database tables - call this once after deployment"""
    try:
        logger.info("Database initialization requested")
        await create_db_and_tables()
        logger.info("Database initialization completed")
        return {
            "status": "success",
            "message": "Database tables created successfully"
        }
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to initialize database: {str(e)}")

@app.get("/admin/stats")
async def get_stats(admin: dict = Depends(verify_admin), session: AsyncSession = Depends(get_session)):
    """Get statistics about media"""
    try:
        logger.info("Stats endpoint accessed")
        result = await session.execute(select(Media))
        all_media = result.scalars().all()
        total_views = sum([m.views for m in all_media])
        
        result_images = await session.execute(
            select(func.count(Media.id)).where(Media.media_type == "image")
        )
        image_count = result_images.scalar_one_or_none() or 0
        
        result_gifs = await session.execute(
            select(func.count(Media.id)).where(Media.media_type == "gif")
        )
        gif_count = result_gifs.scalar_one_or_none() or 0
        
        stats = {
            "total_media": len(all_media),
            "total_images": image_count,
            "total_gifs": gif_count,
            "total_views": total_views,
            "categories": sorted(list(ALLOWED_CATEGORIES))
        }
        logger.info(f"Stats retrieved: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to retrieve stats: {str(e)}")

@app.post("/admin/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload(
    files: List[UploadFile] = File(...),
    titles: List[str] = Form(...),
    categories: List[str] = Form(...),
    media_type: str = Form(...),
    admin: dict = Depends(verify_admin),
    session: AsyncSession = Depends(get_session)
):
    """Bulk upload media files"""
    logger.info(f"Bulk upload initiated: {len(files)} files, type: {media_type}")
    
    if len(files) > 50:
        logger.warning(f"Bulk upload rejected: too many files ({len(files)})")
        raise HTTPException(400, "Maximum 50 files allowed per upload")
    if media_type not in ["image", "gif"]:
        logger.warning(f"Bulk upload rejected: invalid media_type ({media_type})")
        raise HTTPException(400, "media_type must be 'image' or 'gif'")

    uploaded, errors = [], []

    for idx, file in enumerate(files):
        try:
            if idx >= len(titles) or idx >= len(categories):
                raise ValueError("Mismatched number of files, titles, and categories")
            
            title = titles[idx].strip()
            category = validate_category(categories[idx])
            logger.info(f"Uploading file {idx+1}/{len(files)}: {file.filename}")
            
            # Read file content
            file_content = await file.read()
            
            # Upload to Cloudinary
            upload = cloudinary.uploader.upload(
                file_content,
                resource_type="image",
                folder=f"animepixels/{category}",
                use_filename=True,
                unique_filename=True,
                timeout=20
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
            await session.refresh(media)
            
            uploaded.append(
                UploadedMediaItem(
                    filename=file.filename,
                    title=title,
                    category=category,
                    media=MediaOut.model_validate(media)
                )
            )
            logger.info(f"Successfully uploaded: {file.filename}")
        except Exception as e:
            logger.error(f"Upload failed for {file.filename}: {str(e)}", exc_info=True)
            errors.append({"filename": file.filename, "error": str(e)})
            await session.rollback()
        finally:
            await file.close()

    logger.info(f"Bulk upload completed: {len(uploaded)} succeeded, {len(errors)} failed")
    return BulkUploadResponse(
        success=len(uploaded),
        failed=len(errors),
        uploaded_media=uploaded,
        errors=errors
    )

# ---------- RANDOM ----------
@app.get("/random/image", response_model=MediaOut)
async def random_image(session: AsyncSession = Depends(get_session)):
    """Get random image"""
    try:
        result = await session.execute(
            select(Media.id).where(Media.visible == True, Media.media_type == "image")
        )
        ids = result.scalars().all()
        if not ids:
            logger.warning("No images available")
            raise HTTPException(404, "No images available")
        
        media_id = random.choice(ids)
        media = await session.get(Media, media_id)
        if media:
            await increment_views(media, session)
        logger.info(f"Random image retrieved: {media_id}")
        return media
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Random image fetch failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch random image: {str(e)}")

@app.get("/random/gif", response_model=MediaOut)
async def random_gif(session: AsyncSession = Depends(get_session)):
    """Get random GIF"""
    try:
        result = await session.execute(
            select(Media.id).where(Media.visible == True, Media.media_type == "gif")
        )
        ids = result.scalars().all()
        if not ids:
            logger.warning("No gifs available")
            raise HTTPException(404, "No gifs available")
        
        media_id = random.choice(ids)
        media = await session.get(Media, media_id)
        if media:
            await increment_views(media, session)
        logger.info(f"Random gif retrieved: {media_id}")
        return media
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Random gif fetch failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch random gif: {str(e)}")

# ---------- GET BY ID ----------
@app.get("/image/{media_id}", response_model=MediaOut)
async def get_image(media_id: int, session: AsyncSession = Depends(get_session)):
    """Get image by ID"""
    try:
        media = await session.get(Media, media_id)
        if not media or media.media_type != "image" or not media.visible:
            logger.warning(f"Image not found: {media_id}")
            raise HTTPException(404, "Image not found")
        await increment_views(media, session)
        return media
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get image failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch image: {str(e)}")

@app.get("/gif/{media_id}", response_model=MediaOut)
async def get_gif(media_id: int, session: AsyncSession = Depends(get_session)):
    """Get GIF by ID"""
    try:
        media = await session.get(Media, media_id)
        if not media or media.media_type != "gif" or not media.visible:
            logger.warning(f"Gif not found: {media_id}")
            raise HTTPException(404, "Gif not found")
        await increment_views(media, session)
        return media
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get gif failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch gif: {str(e)}")

# ---------- CATEGORY & SEARCH ----------
async def paginated_query(
    session: AsyncSession,
    media_type: str,
    category: str,
    page: int,
    limit: int
):
    """Get paginated media by category"""
    try:
        result = await session.execute(
            select(func.count(Media.id)).where(
                Media.visible == True,
                Media.category == category,
                Media.media_type == media_type
            )
        )
        total_items = result.scalar_one_or_none() or 0
        total_pages = max(1, (total_items + limit - 1) // limit)
        
        if page > total_pages and total_items > 0:
            logger.warning(f"Page {page} out of range (max: {total_pages})")
            raise HTTPException(404, f"Page {page} does not exist")
        
        offset = (page - 1) * limit
        result = await session.execute(
            select(Media).where(
                Media.visible == True,
                Media.category == category,
                Media.media_type == media_type
            ).offset(offset).limit(limit)
        )
        items = result.scalars().all()
        
        # Update views for all items
        for m in items:
            m.views += 1
            m.updated_at = datetime.utcnow()
            session.add(m)
        await session.commit()
        
        logger.info(f"Paginated query: {category}, {media_type}, page {page}")
        return PaginatedMedia(
            page=page,
            limit=limit,
            total_items=total_items,
            total_pages=total_pages,
            items=[MediaOut.model_validate(m) for m in items]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paginated query failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch paginated results: {str(e)}")

async def search_query(
    session: AsyncSession,
    media_type: str,
    q: str,
    page: int,
    limit: int
):
    """Search media by title or category"""
    try:
        pattern = f"%{q.lower()}%"
        result = await session.execute(
            select(func.count(Media.id)).where(
                Media.visible == True,
                Media.media_type == media_type,
                (Media.title.ilike(pattern)) | (Media.category.ilike(pattern))
            )
        )
        total_items = result.scalar_one_or_none() or 0
        total_pages = max(1, (total_items + limit - 1) // limit)
        
        if page > total_pages and total_items > 0:
            logger.warning(f"Search page {page} out of range (max: {total_pages})")
            raise HTTPException(404, f"Page {page} does not exist")
        
        offset = (page - 1) * limit
        result = await session.execute(
            select(Media).where(
                Media.visible == True,
                Media.media_type == media_type,
                (Media.title.ilike(pattern)) | (Media.category.ilike(pattern))
            ).offset(offset).limit(limit)
        )
        items = result.scalars().all()
        
        # Update views for all items
        for m in items:
            m.views += 1
            m.updated_at = datetime.utcnow()
            session.add(m)
        await session.commit()
        
        logger.info(f"Search query: '{q}', {media_type}, page {page}, results: {len(items)}")
        return PaginatedMedia(
            page=page,
            limit=limit,
            total_items=total_items,
            total_pages=total_pages,
            items=[MediaOut.model_validate(m) for m in items]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search query failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to search: {str(e)}")

@app.get("/images/category/{category}", response_model=PaginatedMedia)
async def images_by_category(
    category: str,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session)
):
    """Get images by category"""
    category = validate_category(category)
    return await paginated_query(session, "image", category, page, limit)

@app.get("/gifs/category/{category}", response_model=PaginatedMedia)
async def gifs_by_category(
    category: str,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session)
):
    """Get gifs by category"""
    category = validate_category(category)
    return await paginated_query(session, "gif", category, page, limit)

@app.get("/search/images", response_model=PaginatedMedia)
async def search_images(
    q: str = Query(...),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session)
):
    """Search images"""
    return await search_query(session, "image", q, page, limit)

@app.get("/search/gifs", response_model=PaginatedMedia)
async def search_gifs(
    q: str = Query(...),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session)
):
    """Search gifs"""
    return await search_query(session, "gif", q, page, limit)

# Export for Vercel
handler = app

logger.info("✅ AnimePixels API ready for deployment")
