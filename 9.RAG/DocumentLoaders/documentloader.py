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

docs = loader.load() + csv_loader.load()
print(len(docs))
print(docs[325].page_content)
print(docs[0].metadata)