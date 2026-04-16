from langchain_community.document_loaders import PyPDFLoader    

loader = PyPDFLoader('sample.pdf')

docs = loader.load()
print(docs)
print(type(docs))
print(len(docs))
print(type(docs[0]))
print(docs[0].page_content) 
print(docs[0].metadata)