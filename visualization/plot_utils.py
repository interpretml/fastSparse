import itertools
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import string
import numpy as np
import sys

def create_fig_object(nrows, ncols, figsize):
    fig, ax_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, ax_list

def line_plot_with_std(ax, df, x_axis, y_axis, typeKey, palette_dict, capsize=2.0, compareDict=None):
    typeValues = list(set(df[typeKey]))
    # typeValues = [tmp for tmp in compareDict.keys()]
    if compareDict:
        typeValues = sorted(typeValues, key=lambda t: compareDict[t])
    unitKey = x_axis
    for i, typeValue in enumerate(typeValues):
        x, y_mean, y_std = [], [], []
        condition_dict = {typeKey: typeValue}
        df_tmp_i = extract_df(df, condition_dict)
        unitValues = sorted(list(set(df_tmp_i[unitKey])))
        for j, unitValue in enumerate(unitValues):
            condition_dict = {unitKey: unitValue}
            df_tmp_i_j = extract_df(df, condition_dict)
            y_j_mean, y_j_std = get_avg_of_list(list(df_tmp_i_j[y_axis])), get_std_of_list(list(df_tmp_i_j[y_axis]))
            x.append(unitValue)
            y_mean.append(y_j_mean)
            y_std.append(y_j_std)
        ax.errorbar(x, y_mean, yerr=y_std, capsize=capsize, ecolor=palette_dict[typeValue], color=palette_dict[typeValue], label=typeValue)

def add_bars_to_1d(ax, df, x_axis, y_axis, typeKey, palette_dict, capsize=2.0, compareDict=None):
    typeValues = list(set(df[typeKey]))
    if compareDict:
        typeValues = sorted(typeValues, key=lambda t: compareDict[t])
    print(typeValues)
    unitKey = x_axis
    unitValues = set(df[unitKey])
    width = 1/(1+len(typeValues))
    x = np.arange(len(unitValues)) - len(typeValues)/2*width
    
    for i, typeValue in enumerate(typeValues):
        for j, unitValue in enumerate(unitValues):
            condition_dict = {typeKey: typeValue, unitKey: unitValue}
            df_tmp = extract_df(df, condition_dict)
            x_mean, y_mean, y_std = get_avg_of_list(list(df_tmp[x_axis])), get_avg_of_list(list(df_tmp[y_axis])), get_std_of_list(list(df_tmp[y_axis]))
            ax.bar(x=x[j]-((len(unitValues)-1)/2-i)*width, width=width, height=y_mean, yerr=y_std, capsize=capsize, ecolor=palette_dict[typeValue], color=palette_dict[typeValue], label=typeValue)
    x_ticks = x - ((len(unitValues)-1)/2-len(typeValues)//2)*width
    x_labels = [str(unitValue) for unitValue in unitValues]
    ax.set_xticks(ticks=x_ticks, labels=x_labels)
    return x_ticks, x_labels
            
def add_bars_to_2d(ax, df, x_axis, y_axis, typeKey, unitKey, palette_dict, capsize=2.0, fmt='o'):
    typeValues = set(df[typeKey])
    unitValues = set(df[unitKey])
    for typeValue in typeValues:
        for unitValue in unitValues:
            condition_dict = {typeKey: typeValue, unitKey: unitValue}
            df_tmp = extract_df(df, condition_dict)
            x_mean, x_std, y_mean, y_std = get_avg_of_list(list(df_tmp[x_axis])), get_std_of_list(list(df_tmp[x_axis])), get_avg_of_list(list(df_tmp[y_axis])), get_std_of_list(list(df_tmp[y_axis]))
            ax.errorbar(fmt=fmt, x=x_mean, xerr=x_std, y=y_mean, yerr=y_std, capsize=capsize, ecolor=palette_dict[typeValue], c=palette_dict[typeValue], label=typeValue)

def get_avg_of_list(list_of_values):
    if len(list_of_values) > 0:
        return np.mean(list_of_values)
    return -1

def get_std_of_list(list_of_values):
    if len(list_of_values) > 1:
        return np.std(list_of_values)
    return 0

def read_into_df(fileName, delimiter=";", header="infer"):
    return pd.read_csv(fileName, delimiter=delimiter, header=header)

def insert_to_stringTemplate(stringTemplate, value):
    return stringTemplate.format(value)

def read_multipleFiles_into_df(fileNameTemplate, templateValues):
    dfs = pd.DataFrame()
    for fold in templateValues:
        fileName = insert_to_stringTemplate(fileNameTemplate, fold)
        df = read_into_df(fileName)
        dfs = pd.concat([dfs, df])
    return dfs

def remove_sub_df(df, condition_dict):
    initial_key = df.keys()[0]
    df_column_filters = (df[initial_key] == df[initial_key]).values
    for col_name, col_val in condition_dict.items():
        df_column_filters = df_column_filters & (df[col_name] != col_val).values
        if sum(df_column_filters) == 0:
            print("{} and {} does not exist in dataframe!".format(col_name, col_val))
            sys.exit()
    if df[df_column_filters].empty:
        print("extracted dataframe is empty!")
        sys.exit()
    return (df[df_column_filters])

def extract_df(df, condition_dict):
    initial_key = df.keys()[0]
    df_column_filters = (df[initial_key] == df[initial_key]).values
    for col_name, col_val in condition_dict.items():
        df_column_filters = df_column_filters & (df[col_name] == col_val).values
        if sum(df_column_filters) == 0:
            print("{} and {} does not exist in dataframe!".format(col_name, col_val))
            sys.exit()
    if df[df_column_filters].empty:
        print("extracted dataframe is empty!")
        sys.exit()
    return (df[df_column_filters])

def get_file_prefix(method, dataset, fold):
    if method == "exp":
        # prefix = "results_02_08_2022/{}_{}_exp_time.txt".format(dataset, fold)
        prefix = "results_02_07_2022/{}_{}_exp_loss_comp_time.txt".format(dataset, fold)
    elif method == "logistic":
        prefix = "results_02_08_2022/{}_{}_log_time_adtree.txt".format(dataset, fold)
        # prefix = "results_02_07_2022/{}_{}_log_loss_comp_time_adtree.txt".format(dataset, fold)
    return prefix

def get_file_suffix(lambda0, lambda2):
    suffix = "_{}_{}_".format(lambda0, lambda2)
    return suffix

def read_oneLine_into_list(fileName):
    with open(fileName) as file:
        items = [line.split(";") for line in file]
    if len(items) > 1:
        print("there are more than 1 lines!!!")
    return list(itertools.chain.from_iterable(items))

def get_coeffs_and_indices(method, dataset, fold, lambda0, lambda2):
    prefix, suffix = get_file_prefix(method, dataset, fold), get_file_suffix(lambda0, lambda2)
    coeff_file, index_file = prefix + suffix + "coeff", prefix + suffix + "index"
    coeffs, indices = read_oneLine_into_list(coeff_file), read_oneLine_into_list(index_file)
    
    intercept, coeffs = coeffs[0], coeffs[1:]
    return coeffs, intercept, indices

def get_coeffs_and_indices_from_fileName(coeff_fileName, index_fileName):
    coeffs, indices = read_oneLine_into_list(coeff_fileName), read_oneLine_into_list(index_fileName)
    
    intercept, coeffs = coeffs[0], coeffs[1:]
    return coeffs, intercept, indices

def read_csv_columns(fileName):
    df = pd.read_csv(fileName, delimiter=";", header='infer')
    # print("length of columns is", len(list(df.columns)))
    # print(list(df.columns)[-5:])
    return list(df.columns)

def get_key_and_threshold(colName):
    key, threshold = "", "0"
    if "<=" in colName:
        splits = colName.split("<=")
        if len(splits) == 3:
            value_tmp, key, threshold = splits[0], splits[1], splits[2]
        elif len(splits) == 2:
            key, threshold = splits[0], splits[1]
    elif "=" in colName:
        splits = colName.split("=")
        key, threshold = splits[0], splits[1]
    else:
        print("The colName '{}' is ignored".format(colName))

    return key, threshold

def extract_key_and_values_from_colNames(colNames):
    colNames_dict = defaultdict(list)
    for colName in colNames:
        key, threshold = get_key_and_threshold(colName)
        try:
            colNames_dict[key].append(float(threshold))
        except:
            colNames_dict[key].append(threshold)
    return colNames_dict

def print_coeffs_and_thresholds(indices, coeffs, colNames):
    for index_tmp, coeff_tmp in zip(indices, coeffs):
        print(coeff_tmp, colNames[int(index_tmp)-1])
    print("total sparsity is", len(indices))

def fix_irregular_keys(summary_dict):
    # sex is a categorical variable
    if "sex" in summary_dict and len(summary_dict["sex"]) != 2:
        if summary_dict["sex"][0][0] == "female":
            summary_dict["sex"].append(("male", 0))
        else:
            summary_dict["sex"].append(("female", 0))

    # combine the feature ">20 previous case" and "11-20 previous case" into
    # a new feature called "previous case"
    if ">20 previous case" in summary_dict:
        if "11-20 previous case" in summary_dict:
            summary_dict["previous case"].append((10, float(summary_dict["11-20 previous case"][0][1])+float(summary_dict[">20 previous case"][0][1])))
            summary_dict["previous case"].append((20, float(summary_dict[">20 previous case"][0][1])))
            summary_dict["previous case"].append((25, float(summary_dict["11-20 previous case"][0][1])))
            summary_dict.pop("11-20 previous case")
            summary_dict.pop(">20 previous case")
        else:
            summary_dict["previous case"].append((10, float(summary_dict[">20 previous case"][0][1])))
            summary_dict["previous case"].append((20, float(summary_dict[">20 previous case"][0][1])))
            summary_dict["previous case"].append((25, 0))
            summary_dict.pop(">20 previous case")

    elif "11-20 previous case" in summary_dict:
        summary_dict["previous case"].append((10, float(summary_dict["11-20 previous case"][0][1])+float(summary_dict[">20 previous case"][0][1])))
        summary_dict["previous case"].append((20, 0))
        summary_dict["previous case"].append((25, float(summary_dict["11-20 previous case"][0][1])))
        summary_dict.pop("11-20 previous case")
    return summary_dict

def get_ylim(dataset, method):
    if dataset == "netherlands" and method == "exp":
        ylim = [-4.01, 2.0]
    if dataset == "netherlands" and method == "logistic":
        ylim = [-8, 6.0]
    if dataset == "fico" and method == "exp":
        ylim = [-1, 1.0]
    if dataset == "fico_new" and method == "exp":
        ylim = [-1, 1.0]
    if dataset == "fico_new" and method == "logistic":
        ylim = [-2, 2.5]
    return ylim

def print_threshold_in_Latex_helper(coeff_tmp, letters, letter_index, threshold, printLatex=True):
    try:
        if float(threshold) < 0:
            # print(coeff_tmp, "\\times \\bm{1}_{", letters[letter_index+1], " ==", threshold, "} ", end='')
            # return str(coeff_tmp) + " \\times \\bm{1}_{ " + letters[letter_index+1] + "  == " + threshold + " } " # FICO NEW
            return str(coeff_tmp) + " \\times \\bm{1}_{ " + letters[letter_index+1] + "  \\leq " + threshold + " } " # FICO OLD
        else:
            # print(coeff_tmp, "\\times \\bm{1}_{", letters[letter_index+1], " \\leq", threshold, "} ", end='')
            return str(coeff_tmp) + " \\times \\bm{1}_{ " + letters[letter_index+1] + "  \\leq " + threshold + " } "
    except: # threshold is a string
        return str(coeff_tmp) + " \\times \\bm{1}_{ " + letters[letter_index+1] + "  == " + threshold + " } "

def print_thresholds_in_LaTex(intercept, indices, coeffs, colNames, printLatex=True):
    letters = list(string.ascii_uppercase[0:len(indices)])
    categories = set()
    letter_index = -2
    last_category = ""
    summary_dict = defaultdict(list)
    
    print_str = ""

    # print("score = &", intercept, "\\\\")
    # print("&", end='')
    print_str += "score = & " + str(intercept) + " \\\\" + "\n" + "&"
    for index_tmp, coeff_tmp in zip(indices, coeffs):
        category_tmp, threshold = get_key_and_threshold(colNames[int(index_tmp)-1])
        
        if category_tmp not in categories:
            categories.add(category_tmp)
            letter_index += 1
            if letter_index >= 0:
                # print(" && \\textit{\\#", letters[letter_index], ":{}".format(last_category), "} \\\\")
                # print("&", end='')
                print_str += " && \\textit{\\# " + str(letters[letter_index]) + " :{}".format(last_category) + " } \\\\" + "\n" + "&"
            last_category = category_tmp
        if float(coeff_tmp) > 0:
            # print("+", end='')
            print_str += "+"
        
        tmp_print_str = print_threshold_in_Latex_helper(coeff_tmp, letters, letter_index, threshold, printLatex)
        print_str += tmp_print_str
        summary_dict[category_tmp].append((threshold, coeff_tmp))

    # print(" && \\textit{\\#", letters[letter_index+1], ":{}".format(last_category), "}")
    print_str += " && \\textit{\\# " + letters[letter_index+1] + " :{}".format(last_category) + " }"
    # print("******************\n")
    if printLatex:
        print(print_str)
    return summary_dict

def plot_sex_thresholds(ax, summary_dict, key, fontsize=30, labelsize=30, marginsize=5.0):
    sex_categories, values_str = zip(*summary_dict[key])
    values = [float(tmp) for tmp in values_str]
    ax.bar(sex_categories, values, width = 1./16)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    x_axis, y_axis = key, "Score"
    ax.set_xlabel(x_axis, fontsize=fontsize)
    # ax.set_ylabel(y_axis, fontsize=fontsize)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(marginsize)
    
def get_step_pairs(key, colNames_dict, threshold_coeff_pairs):
    # print(key)
    continuous_pairs = []
    special_pairs = []
    if key in colNames_dict:
        continuous_pairs.append((max(colNames_dict[key])+2, 0))
        for i in range(len(threshold_coeff_pairs)-1, -1, -1):
            if float(threshold_coeff_pairs[i][0]) < 0: # special value in fico
                special_pairs.append((float(threshold_coeff_pairs[i][0]), continuous_pairs[-1][1]+float(threshold_coeff_pairs[i][1])))
                # special_pairs.append((float(threshold_coeff_pairs[i][0]), float(threshold_coeff_pairs[i][1])))
            else:
                continuous_pairs.append((float(threshold_coeff_pairs[i][0]), continuous_pairs[-1][1]+float(threshold_coeff_pairs[i][1])))
    else:
        print("{} is not in ColNames_dict!".format(key))
        # sys.exit()
        # handle previous case in netherlands
        for i in range(len(threshold_coeff_pairs)-1, -1, -1):
            continuous_pairs.append((float(threshold_coeff_pairs[i][0]), float(threshold_coeff_pairs[i][1])))

    continuous_pairs = [continuous_pairs[i] for i in range(len(continuous_pairs)-1, -1, -1)]
    return continuous_pairs, special_pairs

def get_xs_and_ys(continuous_pairs):
    xmin, xmax = min(continuous_pairs[0][0], 0.0001), continuous_pairs[-1][0]
    dx = (xmax - xmin)/98
    xmin, xmax = xmin-dx, xmax+dx
    xs, ys = [xmin], [continuous_pairs[0][1]]

    tmp_index = 0
    for i in range(100):
        xs.append(xs[-1] + dx)
        if xs[-1] > continuous_pairs[tmp_index][0] and tmp_index < len(continuous_pairs)-1:
            tmp_index += 1
        ys.append(continuous_pairs[tmp_index][1])
    return xs, ys, dx
        
def plot_continuous_thresholds(ax, summary_dict, colNames_dict, key, fontsize=30, ylim=None, labelsize=23, print_threshold_pairs=True, marginsize=5.0, stepLineWidth=8):
    threshold_coeff_pairs = summary_dict[key]
    if key == "sex":
        plot_sex_thresholds(ax, summary_dict, key, fontsize=fontsize, labelsize=labelsize, marginsize=marginsize)
    else:
        continuous_pairs, special_pairs = get_step_pairs(key, colNames_dict, threshold_coeff_pairs)
        if print_threshold_pairs:
            print("The key is", key)
            print(continuous_pairs, special_pairs)
        xs, ys, dx = get_xs_and_ys(continuous_pairs)

        ax.step(xs, ys, linewidth=stepLineWidth)
        for (x_special, y_special) in special_pairs:
            ax.bar(x_special, y_special, width=dx, color=ax.get_lines()[-1].get_c())
        
        # if key == "MSinceMostRecentInqexcl7days":
        #     ratio = 1.3
        #     # ax.bar(xs[0]+0.5*dx, ys[0], width=dx, hatch="/")
        #     # ax.arrow(x=xs[0], y=ys[0]/2, dx=-5*dx, dy=0, width=0.03*ratio, head_width=0.1*ratio, head_length=dx*ratio)
            
        #     xs.insert(0, jump_xy[0])
        #     ys.insert(0, jump_xy[1])
        #     ax.bar(xs[0]+0.5*dx, ys[0], width=dx, hatch="/")
        #     ax.arrow(x=xs[0], y=ys[0]/2, dx=-5*dx, dy=0, width=0.03*ratio, head_width=0.1*ratio, head_length=dx*ratio)
            # ax.annotate("", xy=(xs[0], ys[0]/2), xytext=(0, 0), arrowprops=dict(arrowstyle="<-"))
            
        # elif key == "NetFractionRevolvingBurden":
        #     ax.bar(xs[0]+0.5*dx, ys[0], width=dx, hatch="/")
        #     ax.arrow(x=xs[0], y=ys[0]/2, dx=-5*dx, dy=0, width=0.03, head_width=0.1, head_length=dx)


        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        # if key == "MSinceMostRecentInqexcl7days" or key == "NetFractionRevolvingBurden":
            # a=ax.get_xticks().tolist()
            # a[1]= str(a[1]) + '\nmissing'
            # ax.set_xticklabels(a)
            
            # a=ax.get_xticks().tolist()
            # a[1]= 'missing'
            # ax.set_xticklabels(a)
            
            # ax.text(xs[0]-15*dx, ys[0]-0.1, "missing", fontsize=labelsize)
            # ax.set_xlim(xs[0] - 20*dx, max(xs)+ 6*dx)
            # pass
            
        x_axis, y_axis = key, "Score"
        ax.set_xlabel(x_axis, fontsize=fontsize)
        # ax.set_ylabel(y_axis, fontsize=fontsize)
        if ylim == "autoMargin":
            ymin, ymax = min(ys), max(ys)
            if len(special_pairs) > 0:
                x_specials, y_specials = zip(*special_pairs)
                ymin = min(ymin, min(y_specials))
                ymax = max(ymax, max(y_specials))
            rangey = ymax-ymin
            ylim = [ymin - 0.309*rangey, ymax + 0.309*rangey]
            ax.set_ylim(ylim)
        elif ylim != None:
            ax.set_ylim(ylim)
        key = key.title()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(marginsize)
    
    title="Contributions of {}".format(key)