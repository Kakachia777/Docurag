# DocuRAG: Production-Grade Documentation Assistant

A sophisticated Retrieval Augmented Generation (RAG) system designed for technical documentation assistance, similar to systems used at companies like Docker, OpenAI, and kapa.ai.

## Features

- Advanced RAG engine with semantic search and chunking
- State-of-the-art embedding model for document retrieval
- LLM integration with GPT-4o for enhanced response generation
- Real-time API with FastAPI
- Redis caching for performance optimization
- Comprehensive monitoring and analytics
- Production-ready deployment configurations
- Security features including API key authentication and rate limiting

## Architecture

```
docurag/
├── src/
│   ├── rag_engine/     # Core RAG implementation
│   ├── llm/            # LLM integration
│   ├── api/            # FastAPI service
│   └── utils/          # Utilities
├── tests/              # Test suite
├── config/             # Configuration
├── data/               # Data storage
├── models/             # Model artifacts
└── deployment/         # K8s configs
```

## Technical Stack

- **Embedding**: Sentence Transformers (all-mpnet-base-v2)
- **Vector Store**: ChromaDB
- **LLM**: GPT-4o
- **API**: FastAPI
- **Cache**: Redis
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Kubernetes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kakachia777/docurag.git
cd docurag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

## Usage

### Starting the Service

1. Start Redis:
```bash
docker run -d -p 6379:6379 redis
```

2. Start the API server:
```bash
uvicorn src.api.service:create_app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### Query Endpoint
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I use Docker volumes?", "conversation_id": "123"}'
```

#### Document Indexing
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"text": "Docker volumes are...", "metadata": {"source": "docker-docs", "title": "Volumes"}}'
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Monitoring

Access monitoring endpoints:
- Prometheus metrics: http://localhost:9090/metrics
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Deployment

### Local Development
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f deployment/k8s/
```

## Performance Optimization

The system includes several optimizations:
- Efficient document chunking with overlap
- Redis caching for frequent queries
- Background tasks for non-critical operations
- Batched embedding generation
- Vector store indexing for fast retrieval

## Security

- API key authentication
- Rate limiting
- CORS configuration
- Input validation
- Secure model serving

## Monitoring and Analytics

- Query latency tracking
- Token usage monitoring
- Cache hit rates
- Error rates
- Custom business metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/docurag/issues)

## Acknowledgments

- Sentence Transformers team for the embedding model
- OpenAI for GPT-3.5-turbo
- FastAPI team for the excellent web framework
- ChromaDB team for the vector store implementation 