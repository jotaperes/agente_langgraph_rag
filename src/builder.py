import functools
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from agent import (
    GraphState,
    retrieve,
    grade_documents,
    transform_query,
    generate,
    decide_to_generate
)

def build_agent_graph(llm: ChatGoogleGenerativeAI, grader_llm: ChatGoogleGenerativeAI, retriever):
    """
    Constrói o grafo do agente LangGraph.
    """
    
    # LangGraph funciona melhor com funções que recebem apenas o 'state'.
    # Usamos 'functools.partial' para "pré-carregar" o LLM e o Retriever nelas.
    retrieve_node = functools.partial(retrieve, retriever=retriever)
    grade_documents_node = functools.partial(grade_documents, grader_llm=grader_llm)
    transform_query_node = functools.partial(transform_query, llm=llm)
    generate_node = functools.partial(generate, llm=llm)
    
    # --- Montagem do Grafo ---
    workflow = StateGraph(GraphState)

    # 1. Definir os Nós
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)

    # 2. Definir as Arestas (Conexões)
    workflow.set_entry_point("retrieve") # Onde o grafo começa

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("transform_query", "retrieve") # O ciclo de volta!
    workflow.add_edge("generate", END) # Onde o grafo termina

    # 3. Definir a Aresta Condicional
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate, # A função que toma a decisão
        {
            "generate": "generate",
            "transform_query": "transform_query",
        },
    )

    # 4. Compilar o Grafo
    print("Grafo compilado com sucesso!")
    return workflow.compile()