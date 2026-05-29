"""OrchestratorAgent — master agent with Gemini function calling."""
 
import asyncio
import json
from typing import Any
 
from google import genai
from google.genai import types
 
from agents.analyst_agent import AnalystAgent
from agents.rag_agent import RAGAgent
from core.logging import get_logger
from core.models import ToolResult
from core.settings import get_settings
from infra.vector_store import VectorStore, get_vector_store
 
logger = get_logger(__name__)
 
_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_documents",
                description="Semantic search over the ML paper corpus.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(type="STRING",
                            description="Natural language search query"),
                        "arxiv_ids": types.Schema(type="ARRAY",
                            description="Optional arXiv IDs to filter by",
                            items=types.Schema(type="STRING")),
                        "top_k": types.Schema(type="INTEGER",
                            description="Number of results"),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="extract_section",
                description="Extract a section from a specific paper by arXiv ID.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "arxiv_id": types.Schema(type="STRING",
                            description="arXiv ID e.g. '1706.03762'"),
                        "section": types.Schema(type="STRING",
                            description="Section name e.g. 'abstract'"),
                    },
                    required=["arxiv_id", "section"],
                ),
            ),
            types.FunctionDeclaration(
                name="compare_papers",
                description="Compare multiple papers on a specific aspect.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "arxiv_ids": types.Schema(type="ARRAY",
                            description="List of arXiv IDs",
                            items=types.Schema(type="STRING")),
                        "aspect": types.Schema(type="STRING",
                            description="Aspect to compare"),
                        "question_context": types.Schema(type="STRING",
                            description="Extra context"),
                    },
                    required=["arxiv_ids", "aspect"],
                ),
            ),
            types.FunctionDeclaration(
                name="summarize",
                description="Generate a summary of a specific paper.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "arxiv_id": types.Schema(type="STRING",
                            description="arXiv ID to summarize"),
                        "bullet_points": types.Schema(type="INTEGER",
                            description="Number of bullet points"),
                    },
                    required=["arxiv_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="rank_papers",
                description="Rank papers by a given criterion.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "criterion": types.Schema(type="STRING",
                            description="Ranking criterion"),
                        "arxiv_ids": types.Schema(type="ARRAY",
                            description="Papers to rank (default: all 5)",
                            items=types.Schema(type="STRING")),
                        "question_context": types.Schema(type="STRING",
                            description="Extra context"),
                    },
                    required=["criterion"],
                ),
            ),
        ]
    )
]
 
_RAG_TOOLS = {"search_documents", "extract_section"}
_ANALYST_TOOLS = {"compare_papers", "summarize", "rank_papers"}
 
_SYSTEM = """You are the Orchestrator of a multi-agent ML paper analysis system.
You have access to a knowledge base of ML papers that includes, but may not be limited to:
- 1706.03762: Attention Is All You Need
- 1810.04805: BERT: Pre-training of Deep Bidirectional Transformers
- 2005.11401: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- 2210.03629: ReAct: Synergizing Reasoning and Acting in Language Models
- 2302.04761: Toolformer: Language Models Can Teach Themselves to Use Tools

Additional papers may have been uploaded by the user and are also available in the knowledge base.
ALWAYS use the search_documents tool to find content — never refuse a question based on the list above.
If a paper is not found after searching, only then inform the user it was not found.
Respond in the same language as the user."""
 
class OrchestratorAgent:
    MAX_ROUNDS = 3
 
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        vs = vector_store or get_vector_store()
        self._rag = RAGAgent(vs)
        self._analyst = AnalystAgent(vs)
        self._settings = get_settings()
        logger.info("orchestrator_initialized")
 
    async def _generate_with_retry(
        self,
        client: genai.Client,
        model: str,
        contents: list,
        config: types.GenerateContentConfig,
        max_attempts: int = 3,
    ) -> Any:
        """Retry only on 503 (transient). Fail fast on 429 (quota exhausted)."""
        wait = 20.0
        for attempt in range(max_attempts):
            try:
                return await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err = str(e)
                is_transient = "503" in err or "UNAVAILABLE" in err
                is_quota = "429" in err or "RESOURCE_EXHAUSTED" in err

                if is_quota:
                    raise  # fail immediately — retrying won't help
                elif is_transient and attempt < max_attempts - 1:
                    logger.warning("llm_unavailable_retry",
                                attempt=attempt + 1, wait_seconds=wait)
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, 60.0)
                else:
                    raise
 
    async def _dispatch(self, name: str, params: dict[str, Any]) -> str:
        if name in _RAG_TOOLS:
            result = await self._rag.dispatch(name, params)
        elif name in _ANALYST_TOOLS:
            result = await self._analyst.dispatch(name, params)
        else:
            result = ToolResult.failure(name, f"Unknown tool: {name}")
        if result.status == "error":
            return json.dumps({"error": result.error})
        return json.dumps(result.data, ensure_ascii=False, default=str)
 
    async def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        settings = get_settings()
        client = genai.Client(
            api_key=settings.google_api_key,
            http_options={"api_version": "v1beta"},
        )
 
        contents: list[types.Content] = []
        for msg in (history or [])[-settings.max_thread_history:]:
            # Gemini API uses "model" not "assistant"
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=question)],
        ))
 
        config = types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=_SYSTEM,
            tools=_TOOLS,
        )
 
        response = await self._generate_with_retry(
            client, settings.gemini_model, contents, config
        )
 
        for _ in range(self.MAX_ROUNDS):
            function_calls = [
                p.function_call
                for p in response.candidates[0].content.parts
                if p.function_call
            ]
            if not function_calls:
                break
 
            logger.info("tool_calls", tools=[fc.name for fc in function_calls])
            contents.append(response.candidates[0].content)
 
            tool_parts = []
            for fc in function_calls:
                result_json = await self._dispatch(fc.name, dict(fc.args))
                tool_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result_json},
                    )
                ))
 
            contents.append(types.Content(role="user", parts=tool_parts))
            response = await self._generate_with_retry(
                client, settings.gemini_model, contents, config
            )
 
        final = "".join(
            p.text for p in response.candidates[0].content.parts
            if hasattr(p, "text") and p.text
        )
        return final or "Unable to generate a response. Please try again."
 