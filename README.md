# LLM Chat Service 

A multi-model chat interface with user authentication and usage tracking. 

## Tech Stack 

- **Backend**: FastAPI 
- **Frontend**: Gradio 
- **Database**: PostgreSQL 
- **Deployment**: Docker 
- **LLM Provider**: OpenRouter (GPT-5, Claude, Gemini) 

## Project Phases 

- **Phase 1** (Current): Basic chat interface with model switching, user accounts, cost tracking 
- **Phase 2**: Billing and invoicing integration 
- **Phase 3**: OAuth, email auth, 2FA 
- **Phase 4**: Mobile app 

## Development Setup 

1. Copy the sample environment configuration:
   ```bash
   cp .env.example .env
   ```
2. Start the stack with Docker Compose:
   ```bash
   docker compose up --build
   ```
3. Visit the FastAPI docs at http://localhost:8000/docs or the Gradio UI at http://localhost:8000/ui.

To run the API directly on your machine:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Deployment 

Deploy with Docker by building the image and running it alongside PostgreSQL using the provided `docker-compose.yml`. Configure environment variables through `.env` or your orchestrator's secret manager. 

## License 
MIT License 

Copyright (c) 2025 Aleksejs Tokarenko

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.  

