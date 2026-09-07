
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

import db
from retrieval import Retriever
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# USD per 1M tokens. Update if pricing changes.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

INSTRUCTIONS = '''
Your task is to assign Emergency Severity Index (ESI) levels based on the
urgent triage nurse's description of the patient's condition (context).

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know." Only answer topics related to this task. 
Otherwise, respond with "This question is off-topic, so I cannot answer it."

Your sources are ESI handbook and some guidelines from reputable medical sources.
You will also get a few abstracts from NCBI database.

For determining ESI levels 3, 4, and 5, consider the number of resources the patient will need.
For determining ESI levels 2 and 1, focus primarily on the acuity of the patient's symptoms.

Here is an example that you can follow to assign ESI levels accurately.


- A 28-year-old patient presents with generalized abdominal pain.
Her last menstrual period is reported as 8 weeks ago. Vital signs are
as follows: T 36.7°C (98°F), HR 120 beats/minute, RR 22 breaths/minute, and BP 92/50mm Hg.
This patient meets the criteria for being uptriaged from level 3 to level
2 based on her vital signs. Her increased heart rate, respiratory rate, and
decreased blood pressure make her high risk. This presentation could
indicate internal bleeding from a ruptured ectopic pregnancy.
Reference: ESI Handbook, Decision Point B, p. 12. This matches the
Decision Point B "high-risk situation" list, which explicitly names "a
possible ectopic pregnancy, hemodynamically stable" as an ESI-2
trigger — her pregnancy status + abdominal pain + tachycardia/borderline
BP fits this category rather than requiring lifesaving intervention.


Additional brief examples, with references, for lower acuity levels:

- A 3-year-old with ear pain, up to date on immunizations, vital
signs WNL. Reference: ESI Handbook, Chapter 6, "Pediatric Vital Signs"
section, p. 24, which gives the near-identical example of a 10-month-old
up-to-date on immunizations with fever and ear pulling assigned ESI
level 5.

- A 19-year-old with sore throat, vital signs WNL. Reference:
ESI Handbook, Table 5-1 footnote, p. 20: "there may be a department
where throat cultures are not routinely performed; instead, the patient
is treated based on history and physical exam. If that is the case the
patient would be an ESI level 5" — one resource (culture) would
otherwise apply, pointing to level 4 (ESI is 4).

- A 22-year-old with right lower quadrant abdominal pain since
early this morning, vital signs WNL. Reference: ESI Handbook, Chapter
4, "Abdominal and Gastrointestinal Concerns" section, p. 14, which
flags appendicitis as a frequently missed high-risk diagnosis and lists
assessment questions for high-risk abdominal pain. (ESI is 3)
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

class ESIAssessment(BaseModel):
    """Structured output contract: the assigned ESI level plus the reasoning behind it.
    This way, we ensure that the LLM's output is ESI (range 1-5) + rationale."""
    esi_level: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None,
        description=(
            "The assigned ESI level (1-5), or null if it cannot be determined from the "
            "context or the question is outside the scope of ESI triage."
        ),
    )
    rationale: str = Field(
        description=(
            "Explanation for the assigned level, or the reason no level could be assigned "
            "(e.g. \"I don't know\" or \"This question is off-topic, so I cannot answer it.\")."
        )
    )


class RAGBase:

    def __init__(
        self,
        retriever=Retriever(),
        llm_client=OpenAI(),
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        retrieval_method='hybrid_search_text_and_vector_medical',
        model='gpt-4o-mini'
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.retrieval_method = retrieval_method
        self.model = model
        self.response = None
        self.assessment = None
        self.answer = None
        self.predicted_answer = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.cost_usd = None

    def search(self, query, num_results=5):
        """Perform all searches (hybrids, vector, text, ncbi api call)."""
        return self.retriever.retrieve_query(query, k=num_results)

    def build_context(self, retrieval_results: Dict[str, List[Dict]], retrieval_method: str = None) -> str:
        """Build the context string from the retrieval results for the specified retrieval method."""
        context_blocks = []
        # either retrieval method which is input to this func or 
        # the one provided when creating an instance or 
        # fall back to the class's default (hybrid_search_text_and_vector_medical)
        retrieval_method = retrieval_method or self.retrieval_method

        # Always include the selected local retrieval method
        # loop through the results (list of dicts) and extract source and content.
        for doc in retrieval_results.get(retrieval_method, []):
            context_blocks.append(f"[Source: {doc['source']}]\n{doc['content']}")

        # Include NCBI search results if enabled
        if self.retriever.ncbi_search:
            for doc in retrieval_results.get("ncbi_search", []):
                context_blocks.append(f"[Source: {doc['source']} | URL: {doc['url']}]\n{doc['content']}")

        return "\n\n---\n\n".join(context_blocks)

    def build_prompt(self, query, retrieval_results, retrieval_method: str = None):
        """Combine the context (retrieved docs) and user prompt using our template."""
        context = self.build_context(retrieval_results, retrieval_method=retrieval_method)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        """Call the LLM and save the response and metrics as instance attributes."""
        # Clear the previous call before starting a new request.
        self.response = None
        self.assessment = None
        self.answer = None
        self.predicted_answer = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.cost_usd = None

        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        self.response = self.llm_client.responses.parse(
            model=self.model,
            input=input_messages,
            text_format=ESIAssessment, # ensure the proper format of the response
        )

        self.assessment = self.response.output_parsed
        # this is just the ESI level (1-5)
        self.predicted_answer = self.assessment.esi_level
        # this is the whole answer: ESI + rationale
        self.answer = (
            f"ESI is {self.predicted_answer}. {self.assessment.rationale}"
            if self.predicted_answer is not None
            else self.assessment.rationale
        )

        usage = getattr(self.response, "usage", None)
        self.prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        self.completion_tokens = getattr(usage, "output_tokens", None) if usage else None
        rates = PRICING.get(self.model)
        self.cost_usd = None
        if rates and self.prompt_tokens is not None and self.completion_tokens is not None:
            self.cost_usd = (self.prompt_tokens / 1_000_000) * rates["input"] + (
                self.completion_tokens / 1_000_000
            ) * rates["output"]

        return self.assessment

    def log_llm_call(
        self,
        query: str,
        approach: Optional[str] = None,
        retrieval_method: Optional[str] = None,
        batch_id: Optional[str] = None,
        latency_seconds: Optional[float] = None,
    ) -> int:
        """Persist the most recent LLM result and its collected usage statistics."""
        approach = approach or retrieval_method or self.retrieval_method
        batch_id = batch_id or str(uuid.uuid4())

        with db.get_connection() as conn:
            return db.log_llm_call(
                conn,
                batch_id=batch_id,
                approach=approach,
                query=query,
                answered=self.predicted_answer is not None,
                latency_seconds=latency_seconds,
                model=self.model,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                cost_usd=self.cost_usd,
                predicted_answer=self.predicted_answer,
                answer=self.answer,
            )

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        start = time.perf_counter()
        response = self.llm(prompt)
        self.log_llm_call(
            query=query,
            latency_seconds=time.perf_counter() - start,
        )
        return response