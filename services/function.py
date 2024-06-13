import pandas as pd
import re

class Function():
    def substituir_palavras(df, palavras_a_substituir, palavra_substituta):
        # Para cada palavra a ser substituída
        for palavra in palavras_a_substituir:
            # Use uma expressão regular para encontrar a palavra inteira e aplicar a substituição
            padrao = r'\b' + re.escape(palavra) + r'\b'
            df['ESPECIALIDADE'] = df['ESPECIALIDADE'].str.replace(padrao, palavra_substituta, regex=True)
    
        # Retorna o novo DataFrame com as palavras substituídas
        return df

