from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.os import AgentOS

import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# 1. Configuração do RAG (Conhecimento Externo)
vector_db = ChromaDb(collection="pdf_agent",
                     path="tmp/chromadb", persistent_client=True)
knowledge = Knowledge(vector_db=vector_db)

# 2. Configuração do Banco de Dados e do Agente
db = SqliteDb(session_table="agent_session", db_file="tmp/agent.db")

agent = Agent(
    id="pdf_agent",
    name="Agente de PDF",
    model=OpenAIChat(id="gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY")),
    db=db,
    knowledge=knowledge,
    search_knowledge=True,
    enable_user_memories=True,
    instructions="Você deve chamar o usuário de senhor e busque as informações no PDF.",
    debug_mode=True
)

# AGENTOS ==================================================
agent_os = AgentOS(
    name="AgenteOS de PDF",
    agents=[agent],
)

app = agent_os.get_app()

# RUN ======================================================
if __name__ == "__main__":
    knowledge.add_content(
        url="https://s3.sa-east-1.amazonaws.com/static.grendene.aatb.com.br/releases/2417_2T25.pdf",
        metadata={"source": "Grendene", "type": "pdf",
                  "description": "Relatório Financeiro 2T25 da Grendene"},
        skip_if_exists=True,
        reader=PDFReader()
    )
    agent_os.serve(app="exemplo2:app", host="0.0.0.0", port=7777, reload=True)
