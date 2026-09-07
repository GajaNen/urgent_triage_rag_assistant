
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

INSTRUCTIONS = '''
Your task is to assign Emergency Severity Index (ESI) levels based on the
urgent triage nurse's description of the patient's condition (context).

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."

Your sources are ESI handbook and some guidelines from reputable medical sources.
You will also get a few abstracts from NCBI database for each prompt.

For determining ESI levels 3, 4, and 5, consider the number of resources the patient will need.
For determining ESI levels 2 and 1, focus primarily on the acuity of the patient's symptoms.

Here are a few examples that you can follow to assign ESI levels accurately.

Example One

A 28-year-old patient presents with generalized abdominal pain.
Her last menstrual period is reported as 8 weeks ago. Vital signs are
as follows: T 36.7°C (98°F), HR 120 beats/minute, RR 22 breaths/
minute, and BP 92/50mm Hg.


This patient meets the criteria for being uptriaged from level 3 to level
2 based on her vital signs.

Reference: ESI Handbook, Decision Point B, p. 12. This matches the
Decision Point B "high-risk situation" list, which explicitly names "a
possible ectopic pregnancy, hemodynamically stable" as an ESI-2
trigger — her pregnancy status + abdominal pain + tachycardia/borderline
BP fits this category rather than requiring lifesaving intervention.


Example Two

A 15-month-old presents with their caregiver, who states the child
has had decreased appetite, a low-grade temperature, and numerous
liquid stools. The toddler is sitting quietly on the mother's lap.
They have no past medical history, no known drug allergies, and are
not on any medications. Vital signs are as follows: T 38°C (100.4° F),
HR 158 beats/minute, RR 42 breaths/minute, BP 86/50 mm Hg.
Capillary refill is 3 seconds


Prior to vital sign assessment, this patient meets the criteria for ESI
level 3. Based on vital sign assessment, the nurse should triage them to
an ESI level 2.

Reference: ESI Handbook, Decision Point D, Figure 6-1, p. 23. This is
Decision Point D, where the age-based high-risk vital sign thresholds
are applied (1-3y: HR >140, RR >40) — this toddler exceeds both,
indicating need to reassess/uptriage.


Example Three

A 57-year-old presents with cough for multiple days. The patient
tells you that they had a temperature of 101°F (38.3°C) last night.
Vital signs are as follows: T 38.5°C (101.4°F), RR 26 breaths/minute,
HR 100 beats/minute, and SpO2 90%.


At the beginning of the triage assessment, this patient presents as
though they could have pneumonia or viral illness. After assessing
vital signs, the nurse should uptriage the patient to an ESI level 2.

Reference: ESI Handbook, p. 9. Page 9 lists ESI Level-1 criteria examples,
including "SpO2 < 90% that is not the patient's norm, with other signs
of respiratory compromise" — this patient's SpO2 of exactly 90% with
fever/cough/tachypnea places him at this threshold for immediate-
intervention consideration.


Example Four

A 34-year-old patient assigned female at birth presents with generalized
abdominal pain, vomiting, and constipation. She has a history of
laminectomy and currently takes no medications. She states her
LMP was within the last 28 days. Vital signs are as follows: T 36.5°C
(97.8°F), HR 102 beats/minute, RR 16 breaths/minute, BP 132/80
mm Hg, and SpO2 99%.

The heart rate falls just outside the accepted
parameter for the age of the patient, but other vital signs are within
expected limits. In this case, the decision should be to assign the patient
to ESI level 3.

Reference: ESI Handbook, p. 14. This falls under the Abdominal/
Obstetrical-Gynecological concerns section, which flags pregnancy-capable
patients presenting with abdominal pain as needing evaluation for
pregnancy complications — the key assessment question ("Is the patient
pregnant or postpartum?") drives the acuity decision here.


Example Five

A 72-year-old patient presents to the ED with oxygen via nasal cannula
for her advanced chronic obstructive pulmonary disease. She informs
the triage nurse that she has an infected cat bite on her left hand.
The hand is red, tender, and swollen. The patient has no other medical
problems, takes daily inhaled steroids, uses albuterol as needed, and
takes an aspirin daily. Vital signs include the following: T 37.5°C
(99.6°F), HR 105 beats/minute, RR 24 breaths/minute, BP 138/80
mm Hg, and SpO2 91% (on 2 L nasal cannula per norm, states her
SpO2 is 90 to 91\% at home). She denies respiratory distress.


This patient will require two or more resources: labs and intravenous
antibiotics. She meets the criteria for ESI level 3. The triage nurse
notices that her oxygen saturation and respiratory rate are outside
the accepted parameters for the adult, but this patient has advanced
chronic obstructive pulmonary disease. However, because the patient
takes a steroid for her COPD, it is possible she will not mount a
robust immune response, so her vital signs should be considered
in the context of both immunosuppression and the possibility of
sepsis. These vital signs are not surprising given the patient’s history,
so the temptation is to attribute them to her respiratory disease.
However, given the infected bite, it is critical to consider that as a cause
of her elevated heart rate and respiratory rate and uptriage the patient
to an ESI 2.

Reference: ESI Handbook, Chapter 6 intro, p. 23. This page explains
that vital signs "must be contextualized in light of the patient's history,
medications, and presentation," and that patients may have "medication-
mediated 'normal' vital signs" — directly applicable since her SpO2 is at
her personal baseline, not a true high-risk deviation.


Additional brief examples, with references, for lower acuity levels:

score 5: A 3-year-old with ear pain, up to date on immunizations, vital
signs WNL. Reference: ESI Handbook, Chapter 6, "Pediatric Vital Signs"
section, p. 24, which gives the near-identical example of a 10-month-old
up-to-date on immunizations with fever and ear pulling assigned ESI
level 5.

score 5: A 42-year-old needing a rescue inhaler prescription refill,
asymptomatic, vital signs WNL. Reference: ESI Handbook, Table 5-1,
"Predicted ESI Resources," p. 20, which lists "Prescription refills"
under "Not Resources" — meaning no resources needed, pointing to
ESI level 5.

score 4: A 19-year-old with sore throat, vital signs WNL. Reference:
ESI Handbook, Table 5-1 footnote, p. 20: "there may be a department
where throat cultures are not routinely performed; instead, the patient
is treated based on history and physical exam. If that is the case the
patient would be an ESI level 5" — one resource (culture) would
otherwise apply, pointing to level 4.

score 4: A 29-year-old with dysuria, vital signs WNL. Reference: ESI
Handbook, Chapter 5, "Decision Point C: How Many Resources?", pp.
19-20. This presentation would typically require a urinalysis, i.e., one
resource, pointing to ESI level 4.

score 3: A 22-year-old with right lower quadrant abdominal pain since
early this morning, vital signs WNL. Reference: ESI Handbook, Chapter
4, "Abdominal and Gastrointestinal Concerns" section, p. 14, which
flags appendicitis as a frequently missed high-risk diagnosis and lists
assessment questions for high-risk abdominal pain.

score 3: A 45-year-old with left lower leg pain and swelling, started 2
days ago after a 12-hour car trip, vital signs WNL. Reference: ESI
Handbook, Chapter 5, "Decision Point C", pp. 19-20. This presentation
would likely need imaging such as ultrasound, i.e. one or more
resources, guiding an ESI level 3 or 4 determination; note the handbook
doesn't explicitly discuss DVT, so this is inferred from the
resource-counting framework.


Only answer topics related to this task. Otherwise, respond with 
"This question is outside the scope of my purpose."
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