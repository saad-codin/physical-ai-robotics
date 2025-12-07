# Physical AI & Humanoid Robotics Textbook API

This is the backend API for an AI-Native Physical AI & Humanoid Robotics Textbook platform. The system provides:

- **Structured Content**: Four-module curriculum (ROS 2, Digital Twin, AI-Robot Brain, VLA)
- **RAG-Powered Chatbot**: Semantic search and AI responses based on textbook content
- **User Authentication & Profiling**: Personalized learning experience
- **Content Personalization**: Adaptive recommendations based on user profile
- **Multi-Language Support**: Translation capabilities for global accessibility

## Architecture

The system follows a decoupled architecture:

- **Frontend**: Docusaurus-based textbook interface
- **Backend**: FastAPI with async support
- **Database**: PostgreSQL with Neon
- **Vector Store**: Qdrant for semantic search
- **Authentication**: JWT-based with user profiling

## Features

### 1. Content Management
- Module and lesson CRUD operations
- Content organization in four modules:
  - ROS 2 Fundamentals
  - Digital Twin & Simulation
  - AI-Robot Brain
  - Vision Language Action (VLA)

### 2. RAG Chatbot
- Semantic search through textbook content
- AI-powered responses using OpenAI/Claude
- Integration with vector database (Qdrant)
- Citation and source tracking

### 3. User System
- Registration with profiling data
- JWT-based authentication
- Progress tracking
- Personalization based on experience and focus areas

### 4. Personalization
- Content recommendations
- Adaptive learning paths
- Experience-level matching

### 5. Multi-Language Support
- Content translation capabilities
- Language preference support
- Translation management

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Web framework with async support
- **SQLAlchemy** - ORM for PostgreSQL
- **Alembic** - Database migrations
- **Qdrant** - Vector database for semantic search
- **Pydantic** - Data validation
- **JWT** - Authentication
- **OpenAI/Claude** - LLM integration

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run database migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Start the server**:
   ```bash
   python -m src.main
   # or with uvicorn
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Environment Variables

Required environment variables are defined in `.env.example`. Key variables include:

- `DATABASE_URL` - PostgreSQL connection string
- `QDRANT_URL` - Qdrant vector database URL
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` - LLM provider API key
- `AUTH_SECRET` - JWT secret key
- `FRONTEND_URL` - Allowed origin for CORS

## API Documentation

The API provides comprehensive documentation via Swagger UI at `/api/docs` and ReDoc at `/api/redoc` when running.

## Endpoints

### Authentication
- `POST /v1/users/signup` - User registration
- `POST /v1/users/login` - User login
- `GET /v1/users/me` - Get current user

### Content
- `GET/POST /v1/modules/` - Module management
- `GET/POST /v1/lessons/` - Lesson management with optional language parameter
- `GET /v1/lessons/{id}?language=en` - Get lesson in specific language

### Progress & Personalization
- `GET/POST /v1/user-progress/` - Track user progress
- `GET /v1/personalization/recommendations` - Get content recommendations
- `GET /v1/personalization/learning-path` - Get personalized learning path

### Chatbot
- `POST /v1/chatbot-queries/chat` - Chat with the RAG bot
- `GET/POST /v1/chatbot-queries/` - Chat history management

### Translations
- `GET /v1/translations/supported-languages` - Get supported languages
- `GET /v1/translations/lesson/{id}/{lang}` - Get specific translation
- `POST /v1/translations/` - Request new translation

## Development

### Running Tests
```bash
pytest
# or for specific test
pytest tests/test_api_endpoints.py -v
```

### Code Quality
- Code formatting with Black
- Linting with Ruff
- Type checking with mypy

### Project Structure
```
backend/
├── src/
│   ├── api/          # API routes
│   ├── models/       # SQLAlchemy models
│   ├── schemas/      # Pydantic schemas
│   ├── crud/         # Database operations
│   ├── services/     # Business logic
│   ├── utils/        # Utility functions
│   ├── db/           # Database configuration
│   └── core/         # Core utilities (auth, security)
├── tests/            # Test files
├── alembic/          # Database migrations
├── requirements.txt  # Python dependencies
└── pyproject.toml    # Project configuration
```

## Configuration

The application uses Pydantic Settings for configuration management. Settings are loaded from:
1. Environment variables
2. `.env` file
3. Default values in the Settings class

## Security

- JWT tokens for authentication
- Password hashing with bcrypt
- Input validation with Pydantic
- Rate limiting (placeholder implementation)
- CORS configured for frontend communication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.