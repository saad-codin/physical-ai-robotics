"""Script to import Docusaurus markdown files into the database and index them in Qdrant."""
import asyncio
from pathlib import Path
from uuid import uuid4
from sqlalchemy import select

from src.db.session import async_session
from src.models.module import Module, ModuleName
from src.models.lesson import Lesson
from src.services.chatbot import chatbot_service

# Path to Docusaurus docs
DOCS_PATH = Path(__file__).parent.parent / "frontend" / "docs"

# Map directory names to ModuleName enum
MODULE_MAPPING = {
    "ros2": ModuleName.ROS2,
    "digital-twin": ModuleName.DIGITAL_TWIN,
    "ai-brain": ModuleName.AI_ROBOT_BRAIN,
    "vla": ModuleName.VLA,
}

async def import_docs():
    """Import all markdown files from Docusaurus docs."""
    print(f"📚 Importing documentation from: {DOCS_PATH}")
    
    if not DOCS_PATH.exists():
        print(f"❌ Docs path not found: {DOCS_PATH}")
        return
    
    async with async_session() as db:
        modules_created = 0
        lessons_created = 0
        lessons_indexed = 0
        
        # Create modules based on the four-module structure
        for dir_name, module_name in MODULE_MAPPING.items():
            module_dir = DOCS_PATH / dir_name
            if not module_dir.exists():
                print(f"⚠️  Module directory not found: {module_dir}")
                continue
            
            # Check if module already exists
            result = await db.execute(
                select(Module).where(Module.name == module_name)
            )
            existing_module = result.scalar_one_or_none()
            
            if existing_module:
                print(f"📦 Module already exists: {module_name.value}")
                module = existing_module
            else:
                # Create new module
                module = Module(
                    module_id=uuid4(),
                    name=module_name,
                    description=f"Module covering {module_name.value}",
                    order_index=list(MODULE_MAPPING.values()).index(module_name)
                )
                db.add(module)
                await db.flush()
                modules_created += 1
                print(f"✅ Created module: {module_name.value}")
            
            # Import all markdown files in this module
            for md_file in module_dir.glob("*.md"):
                lesson_title = md_file.stem.replace('-', ' ').title()
                
                # Check if lesson already exists
                result = await db.execute(
                    select(Lesson).where(
                        Lesson.module_id == module.module_id,
                        Lesson.title == lesson_title
                    )
                )
                existing_lesson = result.scalar_one_or_none()
                
                if existing_lesson:
                    print(f"  📄 Lesson already exists: {lesson_title}")
                    continue
                
                # Read markdown content
                try:
                    content = md_file.read_text(encoding='utf-8')
                    
                    # Create lesson
                    lesson = Lesson(
                        lesson_id=uuid4(),
                        module_id=module.module_id,
                        title=lesson_title,
                        content=content,
                        order_index=lessons_created
                    )
                    db.add(lesson)
                    await db.flush()
                    lessons_created += 1
                    print(f"  ✅ Created lesson: {lesson.title}")
                    
                    # Index lesson in Qdrant
                    try:
                        indexed_count = await chatbot_service.index_lesson_content(db, lesson)
                        lessons_indexed += 1
                        print(f"    🔍 Indexed {indexed_count} chunks in Qdrant")
                    except Exception as e:
                        print(f"    ⚠️  Failed to index in Qdrant: {e}")
                    
                except Exception as e:
                    print(f"  ❌ Error reading {md_file.name}: {e}")
        
        await db.commit()
        
        print(f"\n📊 Import Summary:")
        print(f"  Modules created: {modules_created}")
        print(f"  Lessons created: {lessons_created}")
        print(f"  Lessons indexed in Qdrant: {lessons_indexed}")
        print(f"\n✨ Import complete!")

if __name__ == "__main__":
    asyncio.run(import_docs())
