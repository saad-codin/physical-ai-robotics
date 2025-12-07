# API Endpoints Documentation

This document provides a comprehensive overview of all API endpoints available in the Physical AI & Humanoid Robotics Textbook API.

## Authentication Endpoints

### User Registration & Login
- `POST /v1/users/signup` - Register a new user with profiling data
  - Request: `UserCreate` schema
  - Response: `UserResponse` schema
  - Description: Creates a new user account with specialization and experience level

- `POST /v1/users/login` - Authenticate user and get JWT token
  - Request: `UserCreate` schema (email and password)
  - Response: `Token` schema
  - Description: Returns access token for authenticated requests

- `GET /v1/users/me` - Get current authenticated user's profile
  - Response: `UserResponse` schema
  - Auth: Required
  - Description: Returns details of the currently authenticated user

- `PUT /v1/users/me` - Update current user's profile
  - Request: `UserUpdate` schema
  - Response: `UserResponse` schema
  - Auth: Required
  - Description: Updates user's specialization, experience level, or preferences

## Content Management Endpoints

### Modules
- `GET /v1/modules/` - List all modules
  - Response: `List[ModuleResponse]`
  - Auth: Required
  - Description: Returns all textbook modules (ROS2, DigitalTwin, AIRobotBrain, VLA)

- `POST /v1/modules/` - Create a new module
  - Request: `ModuleCreate` schema
  - Response: `ModuleResponse` schema
  - Auth: Required
  - Description: Creates a new textbook module

- `GET /v1/modules/{module_id}` - Get specific module
  - Response: `ModuleResponse` schema
  - Auth: Required
  - Description: Returns details of a specific module

### Lessons
- `GET /v1/lessons/?module_id={id}&language={code}` - List lessons with optional filtering
  - Query params:
    - `module_id`: Filter by module ID
    - `language`: Language code (default: "en")
  - Response: `List[LessonResponse]`
  - Auth: Required
  - Description: Returns lessons, optionally filtered by module and translated to specified language

- `POST /v1/lessons/` - Create a new lesson
  - Request: `LessonCreate` schema
  - Response: `LessonResponse` schema
  - Auth: Required
  - Description: Creates a new lesson within a module

- `GET /v1/lessons/{lesson_id}?language={code}` - Get specific lesson
  - Query params:
    - `language`: Language code (default: "en")
  - Response: `LessonResponse` schema
  - Auth: Required
  - Description: Returns a specific lesson, optionally translated to specified language

- `PUT /v1/lessons/{lesson_id}` - Update a lesson
  - Request: `LessonUpdate` schema
  - Response: `LessonResponse` schema
  - Auth: Required
  - Description: Updates an existing lesson

- `DELETE /v1/lessons/{lesson_id}` - Delete a lesson
  - Response: 204 No Content
  - Auth: Required
  - Description: Removes a lesson and its vector index

## User Progress & Personalization Endpoints

### User Progress
- `GET /v1/user-progress/me?lesson_id={id}&bookmarked={bool}` - Get user's progress
  - Query params:
    - `lesson_id`: Filter by lesson ID
    - `bookmarked`: Filter by bookmark status
  - Response: `List[UserProgressResponse]`
  - Auth: Required
  - Description: Returns the current user's progress through lessons

- `POST /v1/user-progress/` - Create/update user progress
  - Request: `UserProgressCreate` schema
  - Response: `UserProgressResponse` schema
  - Auth: Required
  - Description: Creates or updates progress for a lesson

- `GET /v1/user-progress/{progress_id}` - Get specific progress record
  - Response: `UserProgressResponse` schema
  - Auth: Required
  - Description: Returns details of a specific progress record

- `PUT /v1/user-progress/{progress_id}` - Update progress record
  - Request: `UserProgressUpdate` schema
  - Response: `UserProgressResponse` schema
  - Auth: Required
  - Description: Updates an existing progress record

- `DELETE /v1/user-progress/{progress_id}` - Delete progress record
  - Response: 204 No Content
  - Auth: Required
  - Description: Removes a progress record

### Personalization
- `GET /v1/personalization/recommendations` - Get lesson recommendations
  - Response: `LessonRecommendationsResponse` schema
  - Auth: Required
  - Description: Returns personalized lesson recommendations based on user profile and progress

- `GET /v1/personalization/learning-path` - Get personalized learning path
  - Response: `PersonalizedLearningPathResponse` schema
  - Auth: Required
  - Description: Returns a complete personalized learning path with module progress

- `GET /v1/personalization/profile-analysis` - Get profile analysis
  - Response: `dict` with profile information
  - Auth: Required
  - Description: Returns information about how the user's profile is used for personalization

## Chatbot & RAG Endpoints

### Chatbot Queries
- `POST /v1/chatbot-queries/chat` - Chat with the RAG bot
  - Request: `ChatbotQueryRequest` schema
  - Response: `ChatbotQueryResponse` schema
  - Auth: Required
  - Description: Processes a user query through the RAG system and returns AI-generated response

- `GET /v1/chatbot-queries/me` - Get user's chat history
  - Response: `List[ChatbotQueryResponse]` schema
  - Auth: Required
  - Description: Returns the current user's chat query history

- `POST /v1/chatbot-queries/` - Create a chatbot query record
  - Request: `ChatbotQueryCreate` schema
  - Response: `ChatbotQueryResponse` schema
  - Auth: Required
  - Description: Creates a record of a chatbot interaction

- `GET /v1/chatbot-queries/{query_id}` - Get specific query
  - Response: `ChatbotQueryResponse` schema
  - Auth: Required
  - Description: Returns details of a specific chat query

- `DELETE /v1/chatbot-queries/{query_id}` - Delete query
  - Response: 204 No Content
  - Auth: Required
  - Description: Removes a chat query record

## Translation Endpoints

### Content Translations
- `GET /v1/translations/supported-languages` - Get supported languages
  - Response: `SupportedLanguagesResponse` schema
  - Description: Returns list of all supported languages for translation

- `GET /v1/translations/lesson/{lesson_id}/{language_code}` - Get specific translation
  - Response: `TranslationResponse` schema
  - Auth: Required
  - Description: Returns translated content for a lesson in the specified language

- `GET /v1/translations/lesson/{lesson_id}/translations` - Get all translations for lesson
  - Response: `List[ContentTranslationResponse]` schema
  - Auth: Required
  - Description: Returns all available translations for a specific lesson

- `POST /v1/translations/` - Request content translation
  - Request: `TranslationRequest` schema
  - Response: `TranslationResponse` schema
  - Auth: Required
  - Description: Requests translation of lesson content to specified language

## Health & Utility Endpoints

- `GET /health` - Health check
  - Response: `{"status": "healthy", ...}`
  - Description: Returns the health status of the API

- `GET /api/docs` - API Documentation (Swagger UI)
  - Description: Interactive API documentation

- `GET /api/redoc` - API Documentation (ReDoc)
  - Description: Alternative API documentation interface

## Authentication

Most endpoints require authentication using JWT Bearer tokens. Include the token in the Authorization header:

```
Authorization: Bearer {access_token}
```

## Query Parameters

Many endpoints support optional query parameters:

- `skip` (integer): Number of records to skip (for pagination)
- `limit` (integer): Maximum number of records to return (for pagination)
- `language` (string): Language code for content translation (default: "en")

## Error Responses

The API returns standard HTTP error codes:

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Requested resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

The API includes rate limiting to prevent abuse. The specific limits are configured in the settings.