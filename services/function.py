import pandas as pd
import re
from datetime import date

class Function:
    def substituir_palavras(df, palavras_a_substituir, palavra_substituta):
        # Para cada palavra a ser substituída
        for palavra in palavras_a_substituir:
            # Use uma expressão regular para encontrar a palavra inteira e aplicar a substituição
            padrao = r'\b' + re.escape(palavra) + r'\b'
            df['ESPECIALIDADE'] = df['ESPECIALIDADE'].str.replace(padrao, palavra_substituta, regex=True)
    
        # Retorna o novo DataFrame com as palavras substituídas
        return df
    
    def calculate_age(row):
        birth_year = int(row["idade"].split("-")[0])  # Extrai o ano de nascimento a partir da data de nascimento
        exam_year = row["ano"]  # Obtém o ano da realização da prova
        age = exam_year - birth_year
        return age