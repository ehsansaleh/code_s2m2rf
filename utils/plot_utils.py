import os
import matplotlib.pyplot as plt
import seaborn as sns


class ColorGiver:
    def __init__(self):
        sns_pal = sns.color_palette("bright", 12)
        self.pal = sns_pal
        self.c_blue, self.c_orange, self.c_green = sns_pal[0:3]
        self.c_red, self.c_purple, self.c_brown = sns_pal[3:6]
        self.c_pink, self.c_gray, self.c_yellow = sns_pal[6:9]
        self.c_cyan = sns_pal[10]

        self.ptr = 0
        self.memory = dict()

    def __getitem__(self, hue_var):
        color = None
        c_dict = {1: self.c_blue, 10: self.c_red, 5: self.c_purple}
        color = c_dict.get(hue_var, color)
        if color is None:
            self.memory[hue_var] = self.pal[self.ptr]
            self.ptr = (self.ptr + 1) % len(self.pal)
            color = self.memory[hue_var]
        return color


def plot_stat_combined(df_, y, x, hue_name, yerr_col=None, xerr_col=None, cond_dict=None,
                       fig_dict=None, save_dict=None, legend_namer=None, fig_axes=None, plt_kwargs=None):
    if cond_dict is None:
        cond_dict = {}
    if fig_dict is None:
        fig_dict = {'x_relabel': False}
    if save_dict is None:
        save_dict = {'do_save': False}
    if legend_namer is None:
        legend_namer = lambda nshot: f'{nshot}-shot'
    if plt_kwargs is None:
        plt_kwargs = dict()

    # figure settings:
    nrows = 1
    ncols = 1
    width = fig_dict.get('width', 3)
    ratio = fig_dict.get('ratio', 1.25)

    if fig_axes is None:
        figsize = fig_dict.get('figsize', (nrows*width*ratio, ncols*width))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=144)
    else:
        fig, axes = fig_axes

    legends = []
    for i_, nshot in enumerate(sorted(list(set(df_[hue_name])))):
        cond_dict[hue_name] = nshot
        # conditioning on dataset, shots, ...
        df = df_.copy(deep=True)
        for var, val in cond_dict.items():
            df = df[df[var] == val]

        df.sort_values(by=[x], inplace=True)
        if (not yerr_col) and (yerr_col in df.columns):
            yerr = df[yerr_col]
        else:
            yerr = None

        if (not xerr_col) and (xerr_col in df.columns):
            xerr = df[xerr_col]
        else:
            xerr = None
        axes.errorbar(x=df[x], y=df[y], yerr=yerr, xerr=xerr,
                      marker='o', color=color_dict[nshot], **plt_kwargs)
        legends.append(legend_namer(nshot))

    axes.legend(legends)
    axes.set_ylabel(fig_dict.get('ylabel', ''))
    axes.set_xlabel(fig_dict.get('xlabel', ''))
    axes.set_title(fig_dict.get('title', ''))

    if fig_dict.get('x_relabel', False):
        xtick_labels = df[x].tolist()
        axes.set_xticks(xtick_labels)

    # save settings:
    if save_dict['do_save']:
        save_dir = save_dict.get('save_dir', './')
        figname = save_dict.get('figname', 'template')
        dpi = save_dict.get('dpi', 300)
        os.makedirs(save_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f'{save_dir}/{figname}.pdf', bbox_inches='tight', pad_inches=0, dpi=dpi)

    return fig, axes


color_dict = ColorGiver()
