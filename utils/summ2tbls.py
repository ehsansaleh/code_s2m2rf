import pandas as pd
import numpy as np
from pathlib import Path

from summ_utils import gen_table
from cfg import y_list as proj_y_list
from cfg import smry_tbls_dir, paper_tbls_dir, reg_sources
from cfg import table_sep_cols, row_tree, col_tree, naming_scheme, scale_percent

y_list = proj_y_list
enable_bf = False
###########################################
#############   Tri-tables   ##############
###########################################
for reg_src in reg_sources:
    no_ci = False
    Path(paper_tbls_dir).mkdir(parents=True, exist_ok=True)

    # Generating the Summarized DataFrame
    summary_df = pd.read_csv(f'{smry_tbls_dir}/{reg_src}2test.csv')

    summary_df.loc[(summary_df['dataset'] == 'tieredImagenet') &
                   (summary_df['test_n_way'] > 20), 'dataset'] = 'tieredImagenet2'

    # Creating the latex tables
    tables_dict = {}
    if len(table_sep_cols):
        grps = summary_df.groupby(table_sep_cols)
    else:
        grps = [([''], summary_df)]

    if no_ci:
        def str_maker(flt_mean, flt_ci=None, is_bold=False):
            return f'%0.2f' % (scale_percent*flt_mean) + '%'
    else:
        def str_maker(flt_mean, flt_ci=None, is_bold=False):
            pm = '+/-'
            bf_st = 'BFS' if is_bold and enable_bf else ''
            bf_end = 'BFE' if is_bold and enable_bf else ''
            if flt_ci is not None:
                out_str = f'{bf_st}%0.2f{bf_end} {pm} %.2f' % (scale_percent*flt_mean, scale_percent*flt_ci)
            else:
                out_str = f'{bf_st}%0.2f{bf_end}' % (scale_percent*flt_mean)
            return out_str

    for tbl_id, summary_table in grps:
        if naming_scheme == 's2m2rf':
            assert 'n_shot' in summary_table.columns, 'Are you sure you are using the right naming scheme?'
            summary_table = summary_table.sort_values(by=['test_n_way', 'n_shot'])
            shots_col = 'n_shot'
            ways_col = 'test_n_way'
        elif naming_scheme == 's2m2rf':
            assert 'few_shots' in summary_table.columns, 'Are you sure you are using the right naming scheme?'
            summary_table = summary_table.sort_values(by=['few_shots'])
            shots_col = 'few_shots'
            ways_col = None
        else:
            raise Exception('Not Implemented Error')

        dfcat_list = []
        out_ycol = 'print_val'
        split_colname = 'print_split'
        sort_key = lambda k: {'delta': 2, 'firth': 1, 'base': 0}.get(k[2], k[2])
        for y_col, y_col_ci, y_name in sorted(y_list, key=sort_key):
            df_ = summary_table.copy(deep=True)
            df_[out_ycol] = df_[y_col]
            df_[out_ycol+'_ci'] = df_[y_col_ci]
            df_[split_colname] = {'delta': 'Improvement', 'base': 'Before', 'firth': 'After'}.get(y_name, y_name)
            dfcat_list.append(df_)
        mysummary_tbl = pd.concat(dfcat_list, axis=0, ignore_index=True)

        mysummary_tbl[shots_col] = [f'{x}-shot' + ('s' if x > 1 else '')[:0] for x in mysummary_tbl[shots_col]]
        if ways_col is not None:
            mysummary_tbl[ways_col] = [f'{x}-way' for x in mysummary_tbl[ways_col]]

        def is_bold_maker(np_ndarr):
            a = np_ndarr.reshape(-1, 3)
            b = (a[:, 1] > a[:, 0]).astype(np.int32)
            c = np.full(a.shape, False)
            c[:, b] = True
            return c.reshape(*np_ndarr.shape)

        myrowtree = row_tree
        mycoltree = col_tree+[split_colname]
        tbl = gen_table(mysummary_tbl, myrowtree, mycoltree, out_ycol,
                        y_col_ci=out_ycol+'_ci', str_maker=str_maker,
                        is_bold_maker=is_bold_maker)
        tables_dict[tbl_id] = tbl

    for tbl_id, tbl in tables_dict.items():
        ltx_tbl_str = tbl.to_latex(multicolumn=True, escape=True,
                                   multicolumn_format='c|',
                                   column_format='|c'*20)
        ltx_tbl_str = ltx_tbl_str.replace('+/-', '$\pm$')
        ltx_tbl_str = ltx_tbl_str.replace('BFS', '')
        ltx_tbl_str = ltx_tbl_str.replace('BFE', '')
        ltx_tbl_str = ltx_tbl_str.replace('\\\n', '\\\midrule\n')

        if isinstance(tbl_id, (list, tuple)):
            list_tbl_id = tbl_id
        else:
            list_tbl_id = [tbl_id]

        tbl_name = '_'.join([str(x) for x in list_tbl_id])
        if reg_src != 'val':
            tbl_name = tbl_name + f'_{reg_src}2test'
        if no_ci:
            tbl_name = tbl_name + '_noci'

        ltx_save_path = f'{paper_tbls_dir}/{tbl_name}.tex'
        with open(ltx_save_path, 'w') as f_ptr:
            f_ptr.write(ltx_tbl_str)
            print(f'  *   --> Latex table saved at {ltx_save_path}.')

        csv_save_path = f'{paper_tbls_dir}/{tbl_name}.csv'
        tbl.to_csv(csv_save_path)
        print(f'  *   --> CSV table saved at {csv_save_path}.\n  ' + '-' * 80)