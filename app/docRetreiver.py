

def docRetreiver(vectorStore, queries):
    retreiver = vectorStore.as_retriever()
    
    docs = set()
    
    for i in queries:
        for doc in retreiver.invoke(i):
            docs.add(doc.page_content)
    
    
    return formatter(docs)
    # return context

def formatter(docs):
    if len(docs)==0:
        return ""
    
    context = "#### Personalized Relevant User's Information \n"
    for d in docs:
        context += "- "+ d + "\n"
    
    print(context)
    return context