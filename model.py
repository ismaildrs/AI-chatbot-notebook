from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"], temperature=0.5)

def AiResponse(question):
    prompt = PromptTemplate(
        input_variables=["question", "documents", "language"],
        template="""Vous êtes Omnidoc AI assistant, un assistant IA spécialisé en santé.

        Instructions:
        1. Répondez uniquement aux questions relatives à la santé ou à la société Omnidoc.
        2. Pour les questions générales sur la santé, répondez directement sans utiliser les documents fournis.
        3. Pour les questions spécifiques à Omnidoc, basez-vous sur les documents fournis.
        4. Recommandez toujours de consulter un professionnel de santé pour des conseils médicaux personnalisés.
        5. Ne fournissez pas de diagnostic ni de plan de traitement.
        6. Abstenez-vous de discuter ou de donner des opinions sur des sujets non liés à la santé.
        7. Répondez de manière concise et directe.
        8. Si vous ne pouvez pas aider, dites simplement : "Je ne peux pas répondre à cette question."

        Question : {question}

        Documents pertinents :
        {documents}

        Répondez en {language}.

        Réponse :
        """
    )

    prompt_chain = prompt | llm | StrOutputParser()

    analyse_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Vous êtes un assistant conçu pour aider les utilisateurs à récupérer les documents les plus pertinents d'une base de données vectorielle.
        Votre tâche est de générer cinq variations distinctes de la requête de l'utilisateur pour améliorer les résultats de recherche.
        Ces variations doivent capturer différentes perspectives et nuances de la question originale afin d'atténuer les limitations des recherches basées sur la similarité de distance.
        Vous fournissez toutes ces variations sous le champ "variations" et la langue de la question sous le champ "langue", n'utilise pas les abréviations.
        La question doit être au format JSON.
        N'ajoutez rien.
        
        Question originale : {question}
        """
    )

    persist_directory = "./vectorstore/"
    
    embd = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embd)
    
    analysis_chain = analyse_prompt | llm | JsonOutputParser()
    
    analysis_result = analysis_chain.invoke({"question": question})
    
    language = analysis_result["langue"]
        
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=analyse_prompt
    )
    
    documents = retriever.invoke(question)

    response = prompt_chain.invoke({"question": question, "documents": documents, "language": language})
    
    return response

#Example usage
if __name__ == "__main__":
    response = AiResponse("Quand Omnidoc a-t-elle été créée ?")
    print(response)
