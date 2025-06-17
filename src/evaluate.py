import os
import pandas as pd
import nest_asyncio
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document as LangChainDocument
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.runnables import RunnableSequence
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.index_builder import build_indexes
from src.ingest_pipeline import ingest_documents

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def generate_questions(documents):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    prompt_template = """
    Given the following document content, generate 5 relevant evaluation questions for a RAG system:
    {content}

    Questions:
    1. ...
    2. ...
    3. ...
    4. ...
    5. ...
    """
    prompt = PromptTemplate(input_variables=["content"], template=prompt_template)
    chain = prompt | llm

    questions = []
    for doc in documents[:5]:  # Limit to 5 documents
        result = chain.invoke({"content": doc.page_content})
        q_list = result.content.split("\n")[1:6]
        questions.extend([{"question": q.strip()} for q in q_list if q.strip()])

    df = pd.DataFrame(questions)
    return df


async def evaluate_async(collection, df):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings()

    # Load LangChain evaluators
    correctness_evaluator = load_evaluator(
        EvaluatorType.CRITERIA,
        llm=llm,
        criteria={"correctness": "Is the answer correct and relevant to the question?"}
    )
    relevancy_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Does the context: {context} provide relevant information to answer the query: {query}? Output 'Yes' or 'No'."
    )
    relevancy_chain = relevancy_prompt | llm

    results = []
    for _, row in df.iterrows():
        query = row["question"]
        # Query ChromaDB directly
        query_embedding = embeddings.embed_query(query)
        chroma_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Convert Chroma results to LangChain Documents
        docs = [
            LangChainDocument(
                page_content=doc,
                metadata=meta or {}
            )
            for doc, meta in zip(
                chroma_results["documents"][0],
                chroma_results["metadatas"][0]
            )
        ]
        context = " ".join([doc.page_content for doc in docs])

        # Define prompt for QA
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        # Create stuff chain
        stuff_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt,
            document_variable_name="context"
        )

        # Run QA with documents
        response = stuff_chain.invoke({
            "context": docs,
            "question": query
        })
        answer = response

        # Evaluate correctness using CriteriaEvalChain
        correctness_result = correctness_evaluator.invoke({
            "input": query,
            "output": answer  # Changed from 'prediction' to 'output'
        })
        correctness_score = 1 if correctness_result.get("score", 0) >= 0.5 else 0  # Binary score based on correctness

        # Evaluate relevancy
        relevancy_result = relevancy_chain.invoke({"query": query, "context": context})
        relevancy_score = 1 if "Yes" in relevancy_result.content else 0

        # Faithfulness
        faithfulness_prompt = PromptTemplate(
            input_variables=["answer", "context"],
            template="Is the answer: {answer} faithful to the context: {context}? Output 'Yes' or 'No'."
        )
        faithfulness_chain = faithfulness_prompt | llm
        faithfulness_result = faithfulness_chain.invoke({"answer": answer, "context": context})
        faithfulness_score = 1 if "Yes" in faithfulness_result.content else 0

        results.append({
            "question": query,
            "answer": answer,
            "correctness": correctness_score,
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score
        })

    return results


def aggregate_results(df, eval_results):
    df_result = pd.DataFrame(eval_results)
    return df_result


def print_average_scores(df_result):
    correctness_scores = df_result["correctness"].mean()
    faithfulness_scores = df_result["faithfulness"].mean()
    relevancy_scores = df_result["relevancy"].mean()
    print(f"Correctness scores: {correctness_scores}")
    print(f"Faithfulness scores: {faithfulness_scores}")
    print(f"Relevancy scores: {relevancy_scores}")
    return correctness_scores, faithfulness_scores, relevancy_scores


def evaluate():
    # Apply nested asyncio
    nest_asyncio.apply()

    # Create document and split into nodes
    documents = ingest_documents()

    # Create vector store index
    collection = build_indexes()

    # Generate evaluation questions
    df = generate_questions(documents)

    # Evaluate and aggregate results
    eval_result = asyncio.run(evaluate_async(collection=collection, df=df))
    df_result = aggregate_results(df, eval_result)

    # Print average scores
    correctness_scores, faithfulness_scores, relevancy_scores = print_average_scores(df_result)

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    df_result.to_csv("eval_results/evaluation_results.csv", index=False)
    df.to_csv("eval_results/evaluation_questions.csv", index=False)
    with open("eval_results/average_scores.txt", "w") as f:
        f.write(f"Correctness scores: {correctness_scores}\n")
        f.write(f"Faithfulness scores: {faithfulness_scores}\n")
        f.write(f"Relevancy scores: {relevancy_scores}\n")

    return df_result
