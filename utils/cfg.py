from os.path import dirname

####################################################
########          global configs           #########
####################################################

naming_scheme = 's2m2rf'
main_acc = 'acc_300'
firth_reg_col = 'firth_c'
PROJPATH = f'{dirname(dirname(__file__))}'
smry_tbls_dir = f'{PROJPATH}/summary'
paper_plots_dir = f'{PROJPATH}/figures'
paper_tbls_dir = f'{PROJPATH}/tables'

####################################################
########        csv2summ configs           #########
####################################################

reg_sources = ['val']
summ_cond_vars = ['dataset', 'test_n_way', 'n_shot', 'use_cosine_FIM_logdet']
results_csv_dir = f'{PROJPATH}/results'
# generating the path to the files to be read
# provided by a seperate .py file
fldr_mini = [f'1_mini_co_part{i}' for i in range(6)]
fldr_cifar = [f'2_cifar_co_part{i}' for i in range(6)]
fldr_tiered = [f'3_tiered_co_part{i}' for i in range(11)]
specific_csv_fldrs = fldr_mini + fldr_cifar + fldr_tiered
deprecated_cols = ['use_cosine_FIM_logdet', 'firth_alpha', 'firth_gamma']
prop_cols = ['n_shot', 'train_n_way', 'test_n_way', 'dataset',
             'method', 'model', 'split', 'fine_tune_epochs']
prop_cols = prop_cols + deprecated_cols
crn_cols = ['seed', 'iter']
dfltvals_dict = dict(use_cosine_FIM_logdet=False, firth_alpha=0.0,
                     firth_gamma=1.0)

####################################################
########        summ2tables configs        #########
####################################################

table_sep_cols = ['dataset']
row_tree = ['n_shot']
col_tree = ['test_n_way']

y_list = [(f'delta_{main_acc}', f'delta_{main_acc}_ci', 'delta'),
          (f'base_{main_acc}', f'base_{main_acc}_ci', 'base'),
          (main_acc, f'{main_acc}_ci', 'firth'),]
scale_percent = 1

####################################################
########    plot_acc_vs_nways configs      #########
####################################################
