"""Check if there are any lessons in the database."""
import asyncio
from src.db.session import async_session
from src.models.lesson import Lesson
from sqlalchemy import select

async def check_lessons():
    async with async_session() as db:
        result = await db.execute(select(Lesson))
        lessons = result.scalars().all()
        print(f'Found {len(lessons)} lessons in database')
        for lesson in lessons[:5]:  # Show first 5
            print(f'  - {lesson.title}')

if __name__ == "__main__":
    asyncio.run(check_lessons())
