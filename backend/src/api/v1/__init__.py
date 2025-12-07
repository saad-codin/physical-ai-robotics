from fastapi import APIRouter

from . import users, modules, lessons, user_progress, chatbot_queries, personalization, translations, chatkit

router = APIRouter(prefix="/v1")

router.include_router(users.router, prefix="/users", tags=["Users"])
router.include_router(modules.router, prefix="/modules", tags=["Modules"])
router.include_router(lessons.router, prefix="/lessons", tags=["Lessons"])
router.include_router(user_progress.router, prefix="/user-progress", tags=["User Progress"])
router.include_router(chatbot_queries.router, prefix="/chatbot-queries", tags=["Chatbot Queries"])
router.include_router(personalization.router, prefix="/personalization", tags=["Personalization"])
router.include_router(translations.router, prefix="/translations", tags=["Translations"])
router.include_router(chatkit.router, prefix="/chatkit", tags=["ChatKit"])