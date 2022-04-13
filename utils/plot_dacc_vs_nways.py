import os
import pandas as pd
from pathlib import Path
from plot_utils import plot_stat_combined

from cfg import smry_tbls_dir
from cfg import main_acc, paper_plots_dir

pd.options.mode.chained_assignment = None

summ_file_dir = f'{smry_tbls_dir}/val2test.csv'
df_summ = pd.read_csv(summ_file_dir, sep=',')

for dataset in set(df_summ['dataset']):
    print(f'  * Plotting the firth improvements vs. the number of ways for the {dataset} dataset.')
    save_dir = f'{paper_plots_dir}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_dict = {'do_save': True,
                 'save_dir': save_dir,
                 'figname': f'dacc_vs_nways_{dataset}'}
    cond_dict = {'dataset': dataset}
    rename_dict = {'cifar': 'CIFAR-FS', 'miniImagenet': 'mini-ImageNet',
                   'tieredImagenet': 'tiered-ImageNet'}
    fig_dict = {'ylabel': r'$\Delta(ACC)$%',
                'xlabel': 'Number of Classes',
                'title': f'{rename_dict.get(dataset,dataset)}',
                'x_relabel': True,
                'width': 2.5, 'ratio': 1.5}
    if dataset.startswith('tiered'):
        fig_dict['ratio'] *= 2
    fig, axes = plot_stat_combined(df_summ, y=f'delta_{main_acc}', x='test_n_way',
                                   yerr_col=f'delta_{main_acc}_ci', hue_name='n_shot',
                                   fig_dict=fig_dict, cond_dict=cond_dict)

    from matplotlib.ticker import FormatStrFormatter
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # save settings:
    if save_dict['do_save']:
        save_dir = save_dict.get('save_dir', './')
        figname = save_dict.get('figname', 'template')
        dpi = save_dict.get('dpi', 300)
        os.makedirs(save_dir, exist_ok=True)
        fig.tight_layout()
        save_path = f'{save_dir}/{figname}.pdf'
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(f'  *   --> Figure saved at {save_path}.\n  ' + '-' * 80)
