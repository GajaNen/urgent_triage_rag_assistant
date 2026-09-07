
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

INSTRUCTIONS = '''
Your task is to assign Emergency Severity Index (ESI) levels based on the
urgent triage nurse's description of the patient's condition (context).

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know." Only answer topics related to this task. 
Otherwise, respond with "This question is off-topic, so I cannot answer it."

Your sources are ESI handbook and some guidelines from reputable medical sources.
You will also get a few abstracts from NCBI database for each prompt.

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
    """Structured output contract: the assigned ESI level plus the reasoning behind it."""
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
            "(e.g. \"I don't know\" or \"This question is outside the scope of my purpose.\")."
        )
    )


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        retrieval_method='hybrid_search_text_and_vector_medical',
        model='gpt-4o-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.retrieval_method = retrieval_method
        self.model = model

    def search(self, query, num_results=5):
        return self.index.retrieve_query(query, k=num_results)

    def build_context(self, retrieval_results: Dict[str, List[Dict]], retrieval_method: str = None) -> str:
        context_blocks = []
        retrieval_method = retrieval_method or self.retrieval_method

        # 1. Local knowledge base chunks (from the selected hybrid/vector/text method)
        for doc in retrieval_results.get(retrieval_method, []):
            context_blocks.append(f"[Source: {doc['source']}]\n{doc['content']}")

        # 2. Live PubMed online search results
        for doc in retrieval_results.get("ncbi_search", []):
            context_blocks.append(f"[Source: {doc['source']} | URL: {doc['url']}]\n{doc['content']}")

        return "\n\n---\n\n".join(context_blocks)

    def build_prompt(self, query, retrieval_results, retrieval_method: str = None):
        context = self.build_context(retrieval_results, retrieval_method=retrieval_method)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        return self.llm_client.responses.parse(
            model=self.model,
            input=input_messages,
            text_format=ESIAssessment,
        )

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_parsed