# RAG with Custom Vector Search

Implemented Vector Search from scratch as SimpleVectorStore not available in Maven Central for Spring AI 1.0.0.

## How it Works
1. Embedding: Ollama nomic-embed-text converts text -> vector[768]
2. Store: In-memory List
3. Search: Cosine similarity dot(a,b)/(|a||b|) -> Top 3 relevant docs
4. RAG: Context + User Question -> LLM llama3

## APIs
POST /api/rag/add {"text":"..."}
GET /api/rag/ask?q=What is Revit?

## Tested
DB Size 4, Cosine search returning exact match

## Tech Stack
Spring Boot 3.3.4, Spring AI 1.0.0, Ollama, Java 17

## Run with Docker
docker-compose up --build -d
docker exec -it  ollama pull nomic-embed-text
docker exec -it  ollama pull llama3

## Deployment
Dockerized + GitHub Actions CI
