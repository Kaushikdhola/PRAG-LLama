#Cosine similarity (-1 to 1)
THRESHOLD = 0

def docRetreiver(vectorStore, queries):
    retreiver = vectorStore
    
    docs = set()
    
    for i in queries:
        for doc, score in retreiver.search(i):
            if score > THRESHOLD:
                docs.add(doc.page_content)
    
    return formatter(docs)

def formatter(docs):
    if len(docs)==0:
        return ""
    
    context = ""
    for d in docs:
        context += "- "+ d + "\n"
    
    return context