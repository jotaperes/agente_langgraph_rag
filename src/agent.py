import operator
from typing import List, Annotated, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Definição do Estado do Grafo ---

class GraphState(TypedDict):
    """
    Representa o estado do nosso agente. É o que "flui" entre os nós.

    Attributes:
        question (str): A pergunta original do usuário.
        original_question (str): Guardamos a pergunta original para a geração final.
        documents (List[Document]): A lista de documentos recuperados.
        generation (str): A resposta gerada pelo LLM.
        iterations (int): Contador para evitar loops infinitos.
    """
    question: str
    original_question: str 
    documents: List[Document]
    generation: str
    iterations: int

# --- 2. Definição dos Nós do Grafo ---

def retrieve(state: GraphState, retriever):
    """
    Nó de Recuperação (Retrieve): Busca documentos relevantes.
    """
    print(f"--- Iteração {state['iterations']}: Recuperando documentos para a pergunta: '{state['question']}' ---")
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {
        "documents": documents, 
        "question": question, 
        "iterations": state.get("iterations", 0) + 1
    }

class GradeDocuments(BaseModel):
    """Modelo de dados para a avaliação de relevância dos documentos."""
    score: str = Field(description="A pontuação de relevância: 'sim' ou 'não'.")

def grade_documents(state: GraphState, grader_llm: ChatGoogleGenerativeAI):
    """
    Nó de Avaliação (Grade): Avalia se os documentos recuperados são úteis.
    Usa a função 'with_structured_output' do LLM para forçar uma saída "sim" ou "não".
    """
    print("--- Avaliando relevância dos documentos ---")
    question = state["question"]
    documents = state["documents"]
    
    # LLM com 'structured_output' para garantir a resposta no formato que queremos
    structured_grader = grader_llm.with_structured_output(GradeDocuments)
    
    docs_text = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
    Você é um avaliador que determina se um conjunto de documentos é relevante para responder a uma pergunta.
    Responda 'sim' se os documentos contêm informações que podem responder à pergunta.
    Responda 'não' se os documentos não são relevantes.

    Pergunta: {question}

    Documentos Recuperados:
    {docs_text}
    """

    
    response = structured_grader.invoke(prompt)
    
    print(f"--- Resultado da Avaliação: {response.score} ---")
    return {"relevance_grade": response.score}

def transform_query(state: GraphState, llm: ChatGoogleGenerativeAI):
    """
    Nó de Transformação (Transform Query): Se os documentos não forem úteis,
    reformula a pergunta para tentar uma busca melhor.
    """
    print("--- Documentos irrelevantes. Reformulando a pergunta. ---")
    question = state["question"]
    
    prompt = f"""
    Você é um assistente de pesquisa. Sua tarefa é reformular uma pergunta para torná-la
    mais clara ou específica, para que um sistema de busca possa encontrar resultados melhores.
    Mantenha o sentido original da pergunta.
    
    Pergunta Original: {question}
    
    Nova Pergunta Reformulada:
    """
    
    response = llm.invoke(prompt)
    new_question = response.content
    
    print(f"--- Nova Pergunta: {new_question} ---")
    
    return {"question": new_question}

def generate(state: GraphState, llm: ChatGoogleGenerativeAI):
    """
    Nó de Geração (Generate): Gera a resposta final com base nos documentos.
    """
    print("--- Gerando resposta ---")
    # Usa a pergunta original para a resposta final
    question = state["original_question"] 
    documents = state["documents"]

    prompt_template = """
    Você é um assistente especializado em responder perguntas.
    Use os trechos de contexto a seguir para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Não tente inventar uma resposta.
    Seja conciso e direto ao ponto.

    Pergunta: {question}

    Contexto:
    {context}

    Resposta:
    """
    
    context_str = "\n\n".join([doc.page_content for doc in documents])
    prompt = prompt_template.format(question=question, context=context_str)
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- 3. Definição das Arestas Condicionais ---

MAX_ITERATIONS = 3

def decide_to_generate(state: GraphState):
    """
    Aresta Condicional: Decide se deve gerar uma resposta ou reformular a pergunta.
    Verifica também o limite de iterações.
    """
    print("--- Decidindo próximo passo ---")
    relevance_grade = state.get("relevance_grade", "não") # Padrão "não" se não houver
    
    if relevance_grade == "sim":
        print("--- Decisão: Gerar resposta. ---")
        return "generate"
    
    if state["iterations"] >= MAX_ITERATIONS:
        print(f"--- Decisão: Limite de {MAX_ITERATIONS} iterações atingido. Gerando resposta mesmo assim. ---")
        return "generate"
    else:
        print("--- Decisão: Documentos não relevantes. Tentar reformular. ---")
        return "transform_query"