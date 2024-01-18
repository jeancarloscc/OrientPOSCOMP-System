import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizacao:
    def __init__(self):
        sns.set_theme(style="white")
        self.paleta = ["#E06141", "#4169E0"]
        sns.set_palette(self.paleta)
        
    def scatter3D_plot(self, dataframe, atributos, hue, figsize=(12, 10), fontsize=12, title_legend='', legend_mapping=None, save_path=None,title='Scatter 3D Plot'):
        sns.set_theme(style="white")
    
        # Criar figura e eixos 3D
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})
    
        # Plotagem do gráfico de dispersão 3D com cores distintas para cada grupo
        scatter = ax.scatter3D(dataframe[atributos[0]], dataframe[atributos[1]], dataframe[atributos[2]],
                               c=dataframe[hue], cmap='seismic', label=dataframe[hue].unique())
    
        # Rótulos dos eixos
        ax.set_xlabel(atributos[0], labelpad=10)
        ax.set_ylabel(atributos[1], labelpad=10)
        ax.set_zlabel(atributos[2], labelpad=10)
    
        # Adicionar legenda com mapeamento de rótulos
        if legend_mapping:
            unique_labels = dataframe[hue].unique()
            legend_labels = [legend_mapping.get(label, label) for label in unique_labels]
            legend1 = ax.legend(*scatter.legend_elements(), title=title_legend, labels=legend_labels, loc='upper right')
            ax.add_artist(legend1)
        else:
            ax.legend(*scatter.legend_elements(), title=hue, loc='upper right')
    
        # Adicionar título
        ax.set_title(title, fontsize=fontsize)
    
        # Adicionar texto à direita
        fig.text(0.85, 0.5, atributos[2], va='center', rotation='vertical', fontsize=fontsize)
    
        # Salvar figura se especificado
        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
        plt.show()


        
    def barplot_view(self, dataframe, x, y, hue=None, fontsize=14, save_path=None, figsize=(8, 5), title_legend='', xlabel='', ylabel='', dodge=True):
        sns.set_theme(style="white")
    
        sns.set_palette("tab10")
    
        plt.figure(figsize=figsize)
    
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, dodge=dodge);
    
        for p in ax.containers:
            ax.bar_label(p, label_type='edge', fontsize=fontsize, padding=2)
    
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        legend = plt.legend(title=title_legend, fontsize=fontsize)
        legend.get_title().set_fontsize(fontsize)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
        plt.tight_layout()
    
        plt.show()

    def barplot_view_procents(self, dataframe, x, y, hue=None, fontsize=14, save_path=None, figsize=(8, 5), title_legend='', xlabel='', ylabel='', dodge=True):
        sns.set_theme(style="white")
        sns.set_palette("tab10")
    
        plt.figure(figsize=figsize)
    
        ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe, dodge=dodge);
    
        for p in ax.containers:
            # labels = [f'{float(val):.2f}%' for val in dataframe[y]]
            ax.bar_label(p, label_type='edge', fontsize=fontsize, padding=3, fmt=lambda x: f'{x :.2f}%')
    
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        legend = plt.legend(title=title_legend, fontsize=fontsize)
        legend.get_title().set_fontsize(fontsize)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
        plt.tight_layout()
    
        plt.show()
    
    def histplot(self, dataframe, x, kde=False, bins=0, save_path=None):
        sns.set_style(style="white")
        sns.histplot(data=dataframe, x=x, bins=bins, kde=kde)
        plt.tight_layout()
    
        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        plt.show()
    
    def distplot(self, dataframe, x, kde=False):
        sns.distplot(dataframe[x],
                     kde=kde,
                     kde_kws={"color": "g", "alpha": 0.3, "linewidth": 5, "shade": True})
        plt.show()
    
    def plot_top_especialidades_por_sexo(self, data, sexo, top_n=10, save_path=None):
        genero_map = {1: 'Masculino', 0: 'Feminino'}
        
        data_sexo = data[data['SEXO'] == sexo]
    
        counts_esp = data_sexo['ESPECIALIDADE'].value_counts().reset_index()
        counts_esp.columns = ['ESPECIALIDADE', 'counts']
    
        top_especialidades = counts_esp.head(top_n)
    
        plt.figure(figsize=(10, 6))
    
        sns.barplot(data=top_especialidades, x='counts', y='ESPECIALIDADE', color='blue')
    
        for p in plt.gca().patches:
            plt.gca().annotate(format(p.get_width(), '.0f'), (p.get_width(), p.get_y() + p.get_height() / 2),
                               ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=20)
    
        
        plt.xlabel('')
        plt.yticks(fontsize=20)
        plt.ylabel('')
        plt.tight_layout() 
        
        sns.despine()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def line_plot(self, data, title_plot='', xlabel='', ylabel='', legend_title='', save_path=None, fontsize=16):
        plt.figure(figsize=(10,6))
        ax = sns.lineplot(data=data, marker='o')  # 'T' transpõe as linhas para as colunas
        # ax.title(title_plot)
        ax.set_title(title_plot)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.xticks(range(10), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.ylim(0.5,0.8)
        plt.legend(title=legend_title, loc='lower right')
        plt.tight_layout() 
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
