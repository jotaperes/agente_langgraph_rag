"""
Agente RAG com LangChain + Gemini (Google Generative AI)
Autenticação via conta de serviço (GOOGLE_APPLICATION_CREDENTIALS)
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from builder import build_agent_graph


# --- 1. Setup inicial e verificação de ambiente ---
print("Carregando variáveis de ambiente...")
load_dotenv()

# Verifica se a credencial foi configurada corretamente
cred_path = "C:/Users/jpedr/Codigos/agente_rag_langgraph/gen-lang-client.json" #os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_path or not os.path.exists(cred_path):
    raise EnvironmentError(
        "GOOGLE_APPLICATION_CREDENTIALS não encontrada.\n"
        "Defina a variável apontando para o JSON da conta de serviço.\n"
        "Exemplo (PowerShell):\n"
        '  $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\\caminho\\service-account.json"'
    )

print(f"Credenciais encontradas: {cred_path}")

# Caminho do documento PDF
DOC_PATH = "data/TAESA-Release-1T25.pdf"
if not os.path.exists(DOC_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado em {DOC_PATH}. Coloque o PDF na pasta 'data'.")


# --- 2. Função de preparação do vetor (VectorStore) ---
def setup_vectorstore(file_path: str):
    """
    Carrega, divide e indexa o PDF em vetores.
    """
    print(f"Processando o documento: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print("Dividindo o texto em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    print("Gerando embeddings e construindo o índice FAISS...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    print("Documento processado e indexado com sucesso.")
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# --- 3. Inicialização dos componentes ---
try:
    retriever = setup_vectorstore(DOC_PATH)

    print("Inicializando modelos Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        convert_system_message_to_human=True,
    )

    grader_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        convert_system_message_to_human=True,
    )

    print("Construindo grafo do agente RAG...")
    app = build_agent_graph(llm, grader_llm, retriever)

except Exception as e:
    print(f"Erro durante a inicialização: {e}")
    exit()


# --- 4. Loop de interação com o usuário ---
print("\nAgente RAG Cíclico pronto!")
print("Faça sua pergunta sobre o documento. Digite 'sair' para encerrar.")

while True:
    try:
        user_question = input("\nPergunta: ").strip()
        if user_question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando execução.")
            break
        if not user_question:
            continue

        inputs = {
            "question": user_question,
            "original_question": user_question,
            "iterations": 0,
        }

        print("Consultando o agente...")
        final_state = app.invoke(inputs)

        print("\nResposta do Agente:")
        print(final_state.get("generation", "Nenhuma resposta gerada."))

    except KeyboardInterrupt:
        print("\nExecução interrompida.")
        break
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        # import traceback; traceback.print_exc()
