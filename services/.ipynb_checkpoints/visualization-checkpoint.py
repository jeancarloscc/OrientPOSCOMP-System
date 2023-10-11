import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def scatter3D_plot(dataframe, atributos, encoded_specialties, figsize=(12, 10), fontsize=12, save_path=None):
    sns.set_theme(style="whitegrid")
    
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
    

def barplot_view(dataframe, x, y, hue=None, fontsize=14, save_path=None, figsize=(8, 5), paleta=None, title_legend='', xlabel='',ylabel=''):
    sns.set_style(style="ticks")

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
    

