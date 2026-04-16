from langchain_community.document_loaders import CSVLoader, DirectoryLoader, PyPDFLoader    

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    # glob="data/*.csv",
    loader_cls=PyPDFLoader
)

csv_loader = DirectoryLoader(
    path='books',
    glob='*.csv',
    loader_cls=CSVLoader
)

docs = list(loader.lazy_load()) + list(csv_loader.lazy_load())

for doc in docs:
    print(doc.metadata)
# print(len(docs))
# print(docs[325].page_content)
# print(docs[0].metadata)