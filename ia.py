from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import UnstructuredURLLoader
from pathlib import Path

path_file = Path("C:/Users/esantos/Downloads/905014655.xlsx")
path_file2 = Path("C:/Users/esantos/Desktop/tbemail.csv")

converter = DocumentConverter()
result = converter.convert(path_file2)
output = result.document.export_to_text()

print(output)


loader = UnstructuredURLLoader(urls=output)
docs = loader.load()




# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# chunks = text_splitter.split_documents(docs)