- evaluation: ground truth could be medical exams, and then we see which answer produces
my assistant and then somehow we evaluate how similar it is to the correct answer

- taking keywords from the input prompt -- using a ml/nlp technique instead of LLM

- data: manuals, scoring stuff, The WHO Clinical Registry 

- ground truth:

 * https://quizlet.com/test-questions/emergency-severity-index-practice-test-af46b8f8-0f03-40e5-8801-8d46a51b4395
 * https://nurseslabs.com/emergency-nursing-triage-nclex-practice-quiz/
 * https://quizlet.com/test-questions/emergency-triage-practice-test-45f040fe-7935-44b8-ab42-7f0244ebc7fb
 * https://practicetestgeeks.com/urgent-care-urgent-care-triage-and-assessment-practice-test
 * https://practicetestgeeks.com/urgent-care-urgent-care-triage-and-assessment-practice-test


- i'll use uv for dependeny management

LLM: OpenAI
Knowledge base: some medical manuals and pubmed api
Monitoring: Streamlit or dash
Interface: FastAPI + ?
Ingestion pipeline: dlt, Airflow or py scripts?

- i'll use fastapi for presentation and streamlit for dashboard


## Evaluation Criteria

Use these criteria to score the project:

* Problem description
    * 0 points: The problem is not described
    * 1 point: The problem is described but briefly or unclearly
    * 2 points: The problem is well-described and it's clear what problem the project solves
* Retrieval flow
    * 0 points: No knowledge base or LLM is used
    * 1 point: No knowledge base is used, and the LLM is queried directly
    * 2 points: Both a knowledge base and an LLM are used in the flow
* Retrieval evaluation
    * 0 points: No evaluation of retrieval is provided
    * 1 point: Only one retrieval approach is evaluated
    * 2 points: Multiple retrieval approaches are evaluated, and the best one is used
* LLM evaluation
    * 0 points: No evaluation of final LLM output is provided
    * 1 point: Only one approach (e.g., one prompt) is evaluated
    * 2 points: Multiple approaches are evaluated, and the best one is used
* Interface
   * 0 points: No way to interact with the application at all
   * 1 point: Command line interface, a script, or a Jupyter notebook
   * 2 points: UI (e.g., Streamlit), web application (e.g., Django), or an API (e.g., built with FastAPI)
* Ingestion pipeline
   * 0 points: No ingestion
   * 1 point: Semi-automated ingestion of the dataset into the knowledge base, e.g., with a Jupyter notebook or a Python script 
   * 2 points: Automated ingestion with a special tool (e.g., Kestra, dlt, Airflow, Prefect)
* Monitoring
   * 0 points: No monitoring
   * 1 point: User feedback is collected OR there's a monitoring dashboard
   * 2 points: User feedback is collected and there's a dashboard with at least 5 charts
* Containerization
    * 0 points: No containerization
    * 1 point: Dockerfile is provided for the main application OR there's a docker-compose for the dependencies only
    * 2 points: Everything is in docker-compose
* Reproducibility
    * 0 points: No instructions on how to run the code, the data is missing, or it's unclear how to access it
    * 1 point: Some instructions are provided but are incomplete, OR instructions are clear and complete, the code works, but the data is missing
    * 2 points: Instructions are clear, the dataset is accessible, it's easy to run the code, and it works. The versions for all dependencies are specified.
* Best practices
    * [ ] Hybrid search: combining both text and vector search (at least evaluating it) (1 point)
    * [ ] Document re-ranking (1 point)
    * [ ] User query rewriting (1 point)
* Bonus points (not covered in the course)
    * [ ] Deployment to the cloud (2 points)
    * [ ] Up to 3 extra bonus points if you want to award for something extra (write in feedback for what)

Note that your dataset doesn't have to be in thr Q&A form. Check [etc/chunking.md](etc/chunking.md) to learn more about chunking.