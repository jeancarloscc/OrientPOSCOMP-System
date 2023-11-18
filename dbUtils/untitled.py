import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import asyncio
import nest_asyncio
from concurrent.futures import ProcessPoolExecutor

class FirebaseBase:

    def __
    # Inicialize o SDK do Firebase
    cred = credentials.Certificate("poscomp_db.json")
    firebase_admin.initialize_app(cred)

    # Inicialize o Firestore
    db = firestore.client()
