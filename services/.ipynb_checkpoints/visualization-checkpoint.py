import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def scatter3D_plot(dataframe, atributos, encoded_specialties, figsize=(12, 10), fontsize=12, save_path=None):
    sns.set_theme(style="white")
    sns.set_palette(paleta)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter3D(dataframe[atributos[0]], dataframe[atributos[1]], dataframe[atributos[2]], 
                 c=encoded_specialties, cmap='seismic')

    ax.set_xlabel(atributos[0], labelpad=10)
    ax.set_ylabel(atributos[1], labelpad=10)
    ax.set_zlabel(atributos[2], labelpad=10)
    
    fig.text(0.85, 0.5, atributos[2], va='center', rotation='vertical', fontsize=fontsize)

    legend1 = ax.legend(*sctt.legend_elements(), title='Grupo')
    ax.add_artist(legend1)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
    
paleta = ["#E06141", "#4169E0"]
def barplot_view(dataframe, x, y, hue=None, fontsize=14, save_path=None, figsize=(8, 5), paleta=paleta, title_legend='', xlabel='',ylabel=''):
    sns.set_theme(style="white")

    sns.set_palette(paleta)

    plt.figure(figsize=figsize)

    ax = sns.barplot(x=x, y=y, hue=hue, data=dataframe);

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

    plt.show();

def histplot(dataframe, x, kde=False, bins=0, save_path=None):
    sns.set_style(style="white")
    sns.histplot(data=dataframe, x=x, bins=bins, kde=kde)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    plt.show()

def distplot(dataframe, x, kde=False):
    sns.distplot(df["sepal_length"],
             kde=True,
             kde_kws={"color": "g", "alpha": 0.3, "linewidth": 5, "shade": True})
    plt.show();

def plot_top_especialidades_por_sexo(data, sexo, top_n=10, save_path=None):
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
    
    # plt.title(f'Top {top_n} Especialidades - {genero_map.get(sexo, "Desconhecido")}')
    sns.despine()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
    plt.show()

    

