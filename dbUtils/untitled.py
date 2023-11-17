import firebase_admin
from firebase_admin import credentials, firestore

# Carregar credenciais
cred = credentials.Certificate("Documentos/GitHub/project_dataScience_POSCOMP/dbUtils/poscomp-project-firebase-adminsdk-i8jn5-5d70fb0009.json")
firebase_admin.initialize_app(cred)

# Referência para o Firestore
db = firestore.client()

# Referência para a coleção
collection_ref = db.collection('nome_da_colecao')

# Adicionar documento
doc_ref = collection_ref.add({'chave': 'valor'})
print(f'Documento adicionado com ID: {doc_ref.id}')

# Recuperar dados
docs = collection_ref.stream()
for doc in docs:
    print(f'{doc.id} => {doc.to_dict()}')
