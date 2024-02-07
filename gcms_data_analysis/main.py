# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:45:31 2023

@author: mp933
"""


#%%
import pathlib as plib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import ele
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles

def figure_create(rows=1, cols=1, plot_type=0, paper_col=1,
    hgt_mltp=1, font='Dejavu Sans',
    sns_style='ticks'):
    """
    This function creates all the necessary objects to produce plots with
    replicable characteristics.

    Parameters
    ----------
    rows : int, optional
        Number of plot rows in the grid. The default is 1.
    cols : int, optional
        Number of plot columns in the grid. The default is 1.
    plot_type : int, optional
        One of the different plot types available. The default is 0.
        Plot types and their labels:
        0. Std: standard plot (single or grid rows x cols)
        1. Twin-x: secondary axis plot (single or grid rows x cols)
        5. Subplots with different heights
        6. Multiplot without internal x and y tick labels
        7. Multiplot without internal x tick labels
        8. Plot with specific distances between subplots and different heights
    paper_col : int, optional
        Single or double column size for the plot, meaning the actual space
        it will fit in a paper. The default is 1.
    hgt_mltp: float, optional
        Multiplies the figure height. Default is 1. Best using values between
        0.65 and 2. May not work with multiplot and paper_col=1 or out of the
        specified range.
    font: str, optional
        If the string 'Times' is given, it sets Times New Roman as the default
        font for the plot, otherwise the default Dejavu Sans is maintained.
        Default is 'Dejavu Sans'.
    sns_style: str, optional
        The style of the seaborn plot. The default is 'ticks'.

    Returns
    -------
    fig : object
        The figure object to be passed to figure_save.
    lst_ax : list of axis
        List of axis (it is a list even with 1 axis) on which to plot.
    lst_axt : list of axis
        List of secondary axis (it is a list even with 1 axis).
    fig_par : list of float
        List of parameters to reserve space around the plot canvas.

    Raises
    ------
    ValueError
        If cols > 2, which is not supported.

    """
    sns.set_palette("deep")
    # set Times New Roman as the plot font fot text
    if font == 'Times' or font == 'Times New Roman':
        # this may require the installation of the font package
        sns.set_style(sns_style, {'font.family': 'Times New Roman'})
    else:  # leave Dejavu Sans (default) as the plot font fot text
        sns.set_style(sns_style)
    # single or double column in paperthat the figure will occupy
    if cols > 2:  # numer of columns (thus of plots in the figure)
        raise ValueError('\n figure_create: cols>2 not supported')

    # width of the figure in inches, it's fixed to keep the same text size
    # is 6, 9, 12 for 1, 1.5, and 3 paper_col (columns in paper)
    fig_wdt = 6*paper_col  # width of the plot in inches
    fig_hgt = 4*paper_col*rows/cols*hgt_mltp  # heigth of the figure in inches
    px = 0.06*(6/fig_wdt)*cols  # set px so that (A) fits the square
    py = px*fig_wdt/fig_hgt/cols*rows/hgt_mltp  # set py so that (A) fits
    # if more rows are added, it increases, but if cols areadded it decreases
    # to maintain the plot ratio
    # set plot margins
    sp_lab_wdt = 0.156/paper_col  # hor. space for labels
    sp_nar_wdt = 0.02294/paper_col  # space narrow no labels (horiz)
    sp_lab_hgt = 0.147/paper_col/rows*cols/hgt_mltp  # space for labels (vert)
    sp_nar_hgt = 0.02/paper_col/rows*cols/hgt_mltp  # space narrow no labels
    # (vert)
    # =========================================================================
    # # 0. Std: standard plot (single or grid rows x cols)
    # =========================================================================
    if plot_type == 0:
        fig, ax = plt.subplots(rows, cols, figsize=(fig_wdt, fig_hgt))
        if rows*cols == 1:  # only 1 plot
            lst_ax = [ax]  # create ax list for uniform iterations over 1 obj.
        elif rows*cols > 1:  # more than one plot
            lst_ax = [axs for axs in ax.flatten()]  # create list of axis
        lst_axt = None  # no secondary axis in this plot_type
        # horizontal space between plot in percentage
        sp_btp_wdt = 0.26*paper_col**2 - 1.09*paper_col + 1.35
        # vertical space between plot in percentage !!! needs DEBUG
        sp_btp_hgt = .2/paper_col*cols/hgt_mltp
        # left, bottom, right, top, widthspace, heightspace
        fig_par = [sp_lab_wdt, sp_lab_hgt, 1-sp_nar_wdt, 1-sp_nar_hgt,
                   sp_btp_wdt, sp_btp_hgt, px, py]
    # =========================================================================
    # # 1. Twin-x: secondary axis plot (single or grid rows x cols)
    # =========================================================================
    elif plot_type == 1:
        fig, ax = plt.subplots(rows, cols, figsize=(fig_wdt, fig_hgt))
        if rows*cols == 1:  # only 1 plot
            lst_ax = [ax]  # create ax list for uniform iterations over 1 obj.
            lst_axt = [ax.twinx()]  # create a list with secondary axis object
        elif rows*cols > 1:  # more than one plot
            lst_ax = [axs for axs in ax.flatten()]  # create list of axis
            # create list of secondary twin axis
            lst_axt = [axs.twinx() for axs in ax.flatten()]
        # horizontal space between plot in percentage !!! needs DEBUG
        sp_btp_wdt = 1.36*paper_col**2 - 5.28*paper_col + 5.57
        # vertical space between plot in percentage !!! needs DEBUG
        sp_btp_hgt = .2/paper_col*cols/hgt_mltp
        # left, bottom, right(DIFFERENT FROM STD), top, widthspace, heightspace
        fig_par = [sp_lab_wdt, sp_lab_hgt, 1-sp_lab_wdt, 1-sp_nar_hgt,
                   sp_btp_wdt, sp_btp_hgt, px, py]

    return fig, lst_ax, lst_axt, fig_par


def figure_save(filename, out_path, fig, lst_ax, lst_axt, fig_par,
    x_lab=None, y_lab=None, yt_lab=None,
            x_lim=None, y_lim=None, yt_lim=None,
            x_ticks=None, y_ticks=None, yt_ticks=None,
            x_tick_labels=None, y_tick_labels=None, yt_tick_labels=None,
            legend=None, ncol_leg=1,
            annotate_lttrs=False, annotate_lttrs_loc='down',
            pdf=False, svg=False, eps=False, transparency=False,
            subfolder=None, tight_layout=False, grid=False, title=False,
            set_size_inches=None):
    '''
    This function takes the obects created in figure_create and allows to modify
    their appeareance and saving the results.

    Parameters
    ----------
    filename : str
        name of figure. It is the name of the png od pfd file to be saved
    out_path : pathlib.Path object. path to the output folder.
    fig : figure object. created in figure_save.
    lst_ax : list of axis. Created in figure_create
    lst_axt : list of twin (secondary) axis. Created in figure_create
    fig_par : list of figure parameters for space settings
        left, bottom, right, top, widthspace, heightspace, px, py.
        Created in figure_create
    tight_layout : bool
        If True, ignore fig_par[0:6] and fit the figure to the tightest layout
        possible. Avoids to lose part of figure, but loses control of margins
    x_lab : str.list, optional
        label of the x axis. The default is None.
        can be given as
        0. x_lab=None: no axis gets an xlabel
        1. x_lab='label': only one str, all axis get the same xlabel
        2. x_lab=['label1', None, Label2, ...]: the list must have the size of
            lst_ax and contain labels and or None values. Each axis is
            assigned its label, where None is given, no label is set.
    y_lab : str, optional
        label of the y axis. The default is None. Same options as x_lab
    yt_lab : str, optional
        label of the secondary y-axis. The default is None.
        Same options as x_lab
    x_lim : list of two values, list of lists, optional
        limits of x axis. The default is None.
        can be given as
        0. x_lim=None: no axis gets a xlim
        1. x_lab=[a,b]: all axis get the same xlim
        2. x_lab=[[a,b], None, [c,d], ...]: the list must have the size of
            lst_ax and contain [a,b] and or None values. Each axis is
            assigned its limit, where None is given, no llimit is set.
    y_lim : list of two values, optional
        limits of y axis. The default is None. Same options as x_lim
    yt_lim : list of two values, optional
        limits of secondary y axis. The default is None.
        Same options as x_lim
    x_ticks : list of int or float, optional
        list of tiks value to be shown on the axis. The default is None.
    y_ticks : list of int or float, optional
        list of tiks value to be shown on the axis. The default is None.
    yt_ticks : TYPE, optional
        list of tiks value to be shown on the axis. The default is None.
    legend : str, optional
        contains info on the legend location. To avoid printing the legend
        (also in case it is empty) set it to None.
        The default is 'best'.
    ncol_leg : int, optional
        number of columns in the legend. The default is 1.
    annotate_lttrs : bool, optional
        if True, each plot is assigned a letter between () in the lower left
        corner. The default is False. If a string is given, the string is used
        as the letter in the plot even for single plots.
    annotate_lttrs_loc: str.
        default is 'down', if 'up' is given, the letters are placed on the left
        top corner.
    pdf : bool, optional
        if True, the figure is saved also in pdf in the output folder.
        The default is False, so only a png file with 300dpi is saved
    transparency : bool, optional
        if True, background of PNG figure is transparent, defautls is False.
    subfolder : str, optional
        name of the subfolder inside the output folder where the output will
        be saved. If the folder does not exists, it is created.
        The default is None.
    '''

    fig_adj_par = fig_par[0:6]
    if not any(fig_par[0:6]):  # True if all element in fig_par[0:6] are False
        tight_layout = True
    px = fig_par[6]
    py = fig_par[7]
    n_ax = len(lst_ax)  # number of ax objects
    # for x_lab, y_lab, yt_lab creates a list with same length as n_ax.
    # only one value is given all axis are given the same label
    # if a list is given, each axis is given a different value, where False
    # is specified, no value is given to that particular axis
    vrbls = [x_lab, y_lab, yt_lab, legend]  # collect variables for iteration
    lst_x_lab, lst_y_lab, lst_yt_lab, lst_legend \
        = [], [], [], []  # create lists for iteration
    lst_vrbls = [lst_x_lab, lst_y_lab, lst_yt_lab, lst_legend]  # collect lists
    for vrbl, lst_vrbl in zip(vrbls, lst_vrbls):
        if vrbl is None:  # label is not given for any axis
            lst_vrbl[:] = [None]*n_ax
        else:  # label is given
            if np.size(vrbl) == 1:  # only one value is given
                if isinstance(vrbl, str):  # create a list before replicating it
                    lst_vrbl[:] = [vrbl]*n_ax  # each axis gets same label
                elif isinstance(vrbl, list):  # replicate the list
                    lst_vrbl[:] = vrbl*n_ax  # each axis gets same label
            elif np.size(vrbl) == n_ax:  # each axis has been assigned its lab
                lst_vrbl[:] = vrbl  # copy the label inside the list
            else:
                print(vrbl)
                print('Labels/legend size does not match axes number')
    # for x_lim, y_lim, yt_lim creates a list with same length as n_ax.
    # If one list like [a,b] is given, all axis have the same limits, if a list
    # of the same length of the axis is given, each axis has its lim. Where
    # None is given, no lim is set on that axis
    vrbls = [x_lim, y_lim, yt_lim, x_ticks, y_ticks, yt_ticks, x_tick_labels,
             y_tick_labels, yt_tick_labels]  # collect variables for iteration
    lst_x_lim, lst_y_lim, lst_yt_lim, lst_x_ticks, lst_y_ticks, lst_yt_ticks, \
        lst_x_tick_labels, lst_y_tick_labels, lst_yt_tick_labels = \
            [], [], [], [], [], [], [], [], [] # create lists for iteration
    lst_vrbls = [lst_x_lim, lst_y_lim, lst_yt_lim, lst_x_ticks, lst_y_ticks,
                 lst_yt_ticks, lst_x_tick_labels, lst_y_tick_labels,
                 lst_yt_tick_labels]  # collect lists
    for vrbl, lst_vrbl in zip(vrbls, lst_vrbls):
        if vrbl is None:  # limit is not given for any axis
            lst_vrbl[:] = [None]*n_ax
        else:
            # if only list and None are in vrbl, it is [[], None, [], ..]
            # each axis has been assigned its limits
            if any([isinstance(v, (int, float, np.int32, str))
                    for v in vrbl]):
                temporary = []  # necessary to allow append on [:]
                for i in range(n_ax):
                    temporary.append(vrbl)  # give it to all axis
                lst_vrbl[:] = temporary
            else:  # x_lim=[[a,b], None, ...] = [list, bool] # no float
                lst_vrbl[:] = vrbl  # a lim for each axis is already given
    # loops over each axs in the ax array and set the different properties
    for i, axs in enumerate(lst_ax):
        # for each property, if the variable is not false, it is set
        if lst_x_lab[i] is not None:
            axs.set_xlabel(lst_x_lab[i])
        if lst_y_lab[i] is not None:
            axs.set_ylabel(lst_y_lab[i])
        if lst_x_lim[i] is not None:
            axs.set_xlim([lst_x_lim[i][0]*(1 + px) - px*lst_x_lim[i][1],
                          lst_x_lim[i][1]*(1 + px) - px*lst_x_lim[i][0]])
        if lst_y_lim[i] is not None:
            axs.set_ylim([lst_y_lim[i][0]*(1 + py) - py*lst_y_lim[i][1],
                          lst_y_lim[i][1]*(1 + py) - py*lst_y_lim[i][0]])
        if lst_x_ticks[i] is not None:
            axs.set_xticks(lst_x_ticks[i])
        if lst_y_ticks[i] is not None:
            axs.set_yticks(lst_y_ticks[i])
        if lst_x_tick_labels[i] is not None:
            axs.set_xticklabels(lst_x_tick_labels[i])
        if lst_y_tick_labels[i] is not None:
            axs.set_yticklabels(lst_y_tick_labels[i])
        if grid:
            axs.grid(True)
        if annotate_lttrs is not False:
            if annotate_lttrs_loc == 'down':
                y_lttrs = py/px*.02
            elif annotate_lttrs_loc == 'up':
                y_lttrs = 1 - py
            if n_ax == 1:  # if only one plot is given, do not put the letters
                axs.annotate('(' + annotate_lttrs + ')',
                              xycoords='axes fraction',
                              xy=(0, 0), rotation=0, size='large',
                              xytext=(0, y_lttrs), weight='bold')
            elif n_ax > 1:  # if only one plot is given, do not put the letters
                try:  # if specific letters are provided
                    axs.annotate('(' + annotate_lttrs[i] + ')',
                                 xycoords='axes fraction',
                                 xy=(0, 0), rotation=0, size='large',
                                 xytext=(0, y_lttrs), weight='bold')
                except TypeError:  # if no specific letters, use lttrs
                    lttrs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
                    axs.annotate('(' + lttrs[i] + ')', xycoords='axes fraction',
                                 xy=(0, 0), rotation=0, size='large',
                                 xytext=(0, y_lttrs), weight='bold')

    # if secondary (twin) axis are given, set thier properties
    if lst_axt is not None:
        for i, axst in enumerate(lst_axt):
            axst.grid(False)  # grid is always false on secondaty axis
            # for each property, if the variable is not false, it is set
            if lst_yt_lab[i] is not None:
                axst.set_ylabel(lst_yt_lab[i])
            if lst_yt_lim[i] is not None:
                axst.set_ylim([lst_yt_lim[i][0]*(1 + py) - py*lst_yt_lim[i][1],
                              lst_yt_lim[i][1]*(1 + py) - py*lst_yt_lim[i][0]])
            if lst_yt_ticks[i] is not None:
                axst.set_yticks(lst_yt_ticks[i])
            if lst_yt_tick_labels[i] is not None:
                axst.set_yticklabels(lst_yt_tick_labels[i])
    # create a legend merging the entries for each couple of ax and axt
    if any(lst_legend):
        if lst_axt is None:  # with no axt, only axs in ax needs a legend
            for i, axs in enumerate(lst_ax):
                axs.legend(loc=lst_legend[i], ncol=ncol_leg)
        else:  # merge the legend for each couple of ax and axt
            i = 0
            for axs, axst in zip(lst_ax, lst_axt):
                hnd_ax, lab_ax = axs.get_legend_handles_labels()
                hnd_axt, lab_axt = axst.get_legend_handles_labels()
                axs.legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc=lst_legend[i],
                           ncol=ncol_leg)
                i += 1
    try:
        fig.align_labels()  # align labels of subplots, needed only for multi plot
    except AttributeError:
        print('align_labels not performed')
    # if a subfolder is specified, create the subfolder inside the output
    # folder if not already there and save the figure in it
    if subfolder is not None:
        out_path = plib.Path(out_path, subfolder)  # update out_path
        plib.Path(out_path).mkdir(parents=True, exist_ok=True)  # check if
        # folder is there, if not create it
    # set figure margins and save the figure in the output folder
    if set_size_inches:
        fig.set_size_inches(set_size_inches)
    if tight_layout is False:  # if margins are given sets margins and save
        fig.subplots_adjust(*fig_adj_par[0:6])  # set margins
        plt.savefig(plib.Path(out_path, filename + '.png'), dpi=300,
                    transparent=transparency)
        if pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.pdf'))
        if svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.svg'))
        if eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.eps'))
    else:  # margins are not given, use a tight layout option and save
        plt.savefig(plib.Path(out_path, filename + '.png'),
                    bbox_inches="tight", dpi=300, transparent=transparency)
        if pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.pdf'),
                        bbox_inches="tight")
        if svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.svg'),
                        bbox_inches="tight")
        if eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.eps'),
                        bbox_inches="tight")
    # add the title after saving, so it's only visible in the console
    if title is True:
        lst_ax[0].annotate(filename, xycoords='axes fraction', size='small',
                            xy=(0, 0), xytext=(0.05, .95), clip_on=True)


def name_to_properties(comp_name, df, df_class_code_frac):
    """
    used to retrieve chemical properties of the compound indicated by the
    comp_name and to store those properties in the df

    Parameters
    ----------
    GCname : str
        name from GC, used as a unique key.
    search_name : str
        name to be used to search on pubchem.
    df : pd.DataFrame
        that contains all searched compounds.
    df_class_code_frac : pd.DataFrame
        contains the list of functional group names, codes to be searched
        and the weight fraction of each one to automatically calculate the
        mass fraction of each compounds for each functional group.
        Classes are given as smarts and are looked into the smiles of the comp.

    Returns
    -------
    df : pd.DataFrame
        updated dataframe with the searched compound.
    CompNotFound : str
        if GCname did not yield anything CompNotFound=GCname.

    """

    # classes used to split compounds into functional groups
    all_classes = df_class_code_frac.classes.tolist()
    codes = df_class_code_frac.codes.tolist()  # list of code for each class
    mfs = df_class_code_frac.mfs.tolist()  # list of mass fraction of each class
    classes2codes = dict(zip(all_classes, codes))  # dictionaries
    classes2mfs = dict(zip(all_classes, mfs))  # dictionaries
    cond = True
    while cond:  # to deal with HTML issues on server sides (timeouts)
        try:
            # comp contains all info about the chemical from pubchem
            comp_inside_list = pcp.get_compounds(comp_name, 'name')
            if comp_inside_list:
                comp = comp_inside_list[0]
            else:
                print('WARNING: name_to_properties ', comp_name,
                      ' does not find an entry in pcp')
                df.loc[comp_name, 'iupac_name'] = 'unidentified'
                return df
            cond = False
        except pcp.PubChemHTTPError:  # timeout error, simply try again
            print('Caught: pcp.PubChemHTTPError')
    # fill the df with the data
    if df is None:
        df = pd.DataFrame(dtype=float)
    try:
        df.loc[comp_name, 'iupac_name'] = comp.iupac_name.lower()
    except AttributeError: # iupac_name not give
        df.loc[comp_name, 'iupac_name'] = comp_name.lower()
    df.loc[comp_name, 'molecular_formula'] = comp.molecular_formula
    df.loc[comp_name, 'canonical_smiles'] = comp.canonical_smiles
    df.loc[comp_name, 'molecular_weight'] = float(comp.molecular_weight)
    try:
        df.loc[comp_name, 'xlogp'] = float(comp.xlogp)
    except TypeError: # float() argument must be a string or a real number, not 'NoneType'
        df.loc[comp_name, 'xlogp'] = np.nan
    # count all atoms presence and compoute mass percentage
    elements = set(comp.to_dict()['elements'])
    for el in elements:
        el_count = comp.to_dict()['elements'].count(el)
        el_mass = ele.element_from_symbol(el).mass
        if not 'el_' + el in df:
            df['el_' + el] = 0
            df['el_mf_' + el] = 0.
        df.loc[comp_name, 'el_' + el] = int(el_count)
        df.loc[comp_name, 'el_mf_' + el] = \
            float(el_count)*float(el_mass)/float(comp.molecular_weight)
    # apply fragmentation using the Fragmenter class (thanks simonmb)
    frg = Fragmenter(classes2codes,
                    fragmentation_scheme_order=classes2codes.keys(),
                    algorithm='simple')
    fragmentation, _, _ = frg.fragment(comp.canonical_smiles)
    classes = list(fragmentation.keys())
    classes_mf = ['mf_' + cl for cl in classes]
    # df is the intermediate df for classes that helps with sums of
    # similar classes (ex. there are 27 different configs for ketones that
    # go in the same final class)
    newdf = pd.DataFrame(0, columns=classes + classes_mf, index=[comp_name],
                         dtype=float)
    for cl in classes: # get counts and mf of each class in compound
        newdf.loc[comp_name, cl] = fragmentation[cl]  # counts in
        newdf.loc[comp_name, 'mf_'+ cl] = \
            float(fragmentation[cl])*float(classes2mfs[cl])\
                /float(df.loc[comp_name, 'molecular_weight'])  # mass fraction of total
    # classes that must be summed and considered a single one are identified
    # by the same name followed by _#. if _ is in a class, its not unique
    unique_classes = [c if '_' not in c else c.split('_')[0] for c in classes]
    for unique_cl in unique_classes: # sum classes that must be merged
        sum_cls = [k for k in classes if unique_cl in k]  # classes to be summed
        occurr = 0.  # counts, or occurrencies
        cl_mf = 0.  # class mass fracations
        for cl in sum_cls: # for each class that must be summed
            occurr += newdf.loc[comp_name, cl].astype(int)  # sum counts
            cl_mf += newdf.loc[comp_name, 'mf_' + cl].astype(float)  # sum mass fractions
        if not 'fg_' + unique_cl in df:  # create columns if missing
            df['fg_' + unique_cl] = 0
            df['fg_mf_'+ unique_cl] = 0.
        df.loc[comp_name, 'fg_' + unique_cl] = occurr  # put values in DF
        df.loc[comp_name, 'fg_mf_' + unique_cl] = float(cl_mf)
    # heteroatoms and Si are considered functional groups as they usually
    # enter the discussion in a similar way. The atom count is used here
    hetero_atoms = [e for e in elements if e not in ['H', 'C', 'O', 'N', 'Si']]
    hetero_atoms_symb = ['el_' + e for e in hetero_atoms]
    hetero_atoms_mf = ['el_mf_' + e for e in hetero_atoms]
    if hetero_atoms is not None:
        df.loc[comp_name, 'fg_hetero_atoms'] = \
            df.loc[comp_name, hetero_atoms_symb].sum()
        df.loc[comp_name, 'fg_mf_hetero_atoms'] = \
            df.loc[comp_name, hetero_atoms_mf].sum()
    if 'Si' in df:
        df.loc[comp_name, 'fg_Si'] = df.loc[comp_name, 'el_Si'].sum()
        df.loc[comp_name, 'fg_mf_Si'] = df.loc[comp_name, 'el_mf_Si'].sum()
    print('\tInfo: name_to_properties ', comp_name)
    return df


def report_difference(rep1, rep2, diff_type='absolute'):
    """
    calculates the ave, std and p percentage of the differnece between
    two reports where columns and index are the same.
    Replicates (indicated as XX_1, XX_2) are used for std.

    Parameters
    ----------
    rep1 : pd.DataFrame
        report that is conisdered the reference to compute differences from.
    rep2 : pd.DataFrame
        report with the data to compute the difference.
    diff_type : str, optional
        type of difference, absolute vs relative (to rep1)
        . The default is 'absolute'.

    Returns
    -------
    dif_ave : pd.DataFrame
        contains the average difference.
    dif_std : pd.DataFrame
        contains the std, same units as dif_ave.
    dif_stdp : pd.DataFrame
        contains the percentage std compared to ref1.

    """
    idx_name = rep1.index.name
    rep1 = rep1.transpose()
    rep2 = rep2.transpose()

    # put the exact same name on files (by removing the '_#' at end)
    repl_idx1 = [i if '_' not in i else i.split('_')[0] for i in
                 rep1.index.tolist()]
    repl_idx2 = [i if '_' not in i else i.split('_')[0] for i in
                 rep2.index.tolist()]
    rep1.loc[:, idx_name] = repl_idx1
    rep2.loc[:, idx_name] = repl_idx2
    # compute files and std of files and update the index
    rep_ave1 = rep1.groupby(idx_name, sort=False).mean().reset_index()
    rep_std1 = rep1.groupby(idx_name, sort=False).std().reset_index()
    rep_ave1.set_index(idx_name, inplace=True)
    rep_std1.set_index(idx_name, inplace=True)
    rep_ave2 = rep2.groupby(idx_name, sort=False).mean().reset_index()
    rep_std2 = rep2.groupby(idx_name, sort=False).std().reset_index()
    rep_ave2.set_index(idx_name, inplace=True)
    rep_std2.set_index(idx_name, inplace=True)

    if diff_type == 'absolute':
        dif_ave = rep_ave1 - rep_ave2
        dif_std = np.sqrt(rep_std1**2 + rep_std2**2)
        dif_stdp = np.sqrt(rep_std1**2 + rep_std2**2)/dif_ave*100
    if diff_type == 'relative':
        dif_ave = (rep_ave1 - rep_ave2)/rep_ave1
        dif_std = np.sqrt(rep_std1**2 + rep_std2**2)/rep_ave1
        dif_stdp = np.sqrt(rep_std1**2 + rep_std2**2)/rep_ave1/dif_ave*100

    return dif_ave, dif_std, dif_stdp


def _annotate_outliers_in_plot(ax, df_ave, df_std, y_lim):
    """
    Annotates the bars in a bar plot with their average value and standard
    deviation if these values exceed the specified y-axis limits.
    The function iterates over the bars in the plot and checks if their average
    values, considering their standard deviations, are outside the provided
    y-axis limits. For such bars, it annotates the average and standard
    deviation on the
    plot, using a specific format for better visualization and understanding.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the plot is drawn.
    df_ave : pandas.DataFrame
        DataFrame containing the average values used in the plot.
    df_std : pandas.DataFrame
        DataFrame containing the standard deviation values corresponding
        to `df_ave`.
    y_lim : list of [float, float]
        A list of two floats representing the minimum (y_lim[0]) and
        maximum (y_lim[1]) limits of the y-axis.

    Returns
    -------
    None
        Modifies the provided Axes object (`ax`) by adding annotations.

    """
    dx = 0.15 * len(df_ave.index)
    dy = 0.04
    tform = blended_transform_factory(ax.transData, ax.transAxes)
    dfao = pd.DataFrame(columns=['H/L', 'xpos', 'ypos', 'ave', 'std', 'text'])
    dfao['ave'] = df_ave.transpose().to_numpy().flatten().tolist()
    dfao['std'] = df_std.transpose().to_numpy().flatten().tolist()
    try:
        dfao['xpos'] = [p.get_x() + p.get_width()/2 for p in ax.patches]
    except ValueError:  # otherwise the masking adds twice the columns
        dfao['xpos'] = [p.get_x() + p.get_width()/2 for p in
                      ax.patches[:len(ax.patches)//2]]
    cond = (dfao['ave'] < y_lim[0]) | (dfao['ave'] > y_lim[1])
    dfao = dfao.drop(dfao[~cond].index)
    for ao in dfao.index.tolist():  # loop through bars
        if dfao.loc[ao, 'ave'] == float('inf'):
            dfao.loc[ao, 'text'] = 'inf'
            dfao.loc[ao, 'H/L'] = 'H'
        elif dfao.loc[ao, 'ave'] == float('-inf'):
            dfao.loc[ao, 'text'] = '-inf'
            dfao.loc[ao, 'H/L'] = 'L'
        elif dfao.loc[ao, 'ave'] > y_lim[1]:
            dfao.loc[ao, 'H/L'] = 'H'
            dfao.loc[ao, 'text'] = \
                '{:.2f}'.format(round(dfao.loc[ao, 'ave'], 2)).strip()
            if (dfao.loc[ao, 'std'] != 0) & (~np.isnan(dfao.loc[ao, 'std'])):
                dfao.loc[ao, 'text'] += r"$\pm$" + \
                    '{:.2f}'.format(round(dfao.loc[ao, 'std'], 2))
        elif dfao.loc[ao, 'ave'] < y_lim[0]:
            dfao.loc[ao, 'H/L'] = 'L'
            dfao.loc[ao, 'text'] = str(round(dfao.loc[ao, 'ave'], 2)).strip()
            if dfao.loc[ao, 'std'] != 0:
                dfao.loc[ao, 'text'] += r"$\pm$" + \
                    '{:.2f}'.format(round(dfao.loc[ao, 'std'], 2))
        else:
            print('Something is wrong', dfao.loc[ao, 'ave'])
    for hl, ypos, dy in zip(['L', 'H'], [0.02, 0.98], [0.04, -0.04]):
        dfao1 = dfao[dfao['H/L'] == hl]
        dfao1['ypos'] = ypos
        if not dfao1.empty:
            dfao1 = dfao1.sort_values('xpos', ascending=True)
            dfao1['diffx'] = np.diff(dfao1['xpos'].values,
                                   prepend=dfao1['xpos'].values[0]) < dx
            dfao1.reset_index(inplace=True)

            for i in dfao1.index.tolist()[1:]:
                dfao1.loc[i, 'ypos'] = ypos
                for e in range(i, 0, -1):
                    if dfao1.loc[e, 'diffx']:
                        dfao1.loc[e, 'ypos'] += dy
                    else:
                        break
            for ao in dfao1.index.tolist():
                ax.annotate(dfao1.loc[ao, 'text'], xy=(dfao1.loc[ao, 'xpos'], 0),
                            xycoords=tform, textcoords=tform,
                            xytext=(dfao1.loc[ao, 'xpos'], dfao1.loc[ao, 'ypos']),
                            fontsize=9, ha='center', va='center',
                            bbox={"boxstyle": 'square,pad=0', "edgecolor": None,
                                "facecolor": 'white', "alpha": 0.7})


class Fragmenter:
    """
    taken from https://github.com/simonmb/fragmentation_algorithm
    The original version of this algorithm was published in:
        Flexible Heuristic Algorithm for Automatic Molecule Fragmentation:
        Application to the UNIFAC Group Contribution Model
    DOI: 10.1186/s13321-019-0382-39
    MIT License

    Copyright (c) 2019 Simon

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.


    """

    # tested with Python 3.8.8 and RDKit version 2021.09.4

    from rdkit import Chem
    import marshal as marshal
    from rdkit.Chem import rdmolops

    # does a substructure match and then checks whether the match
    # is adjacent to previous matches
    @classmethod
    def get_substruct_matches(cls, mol_searched_for, mol_searched_in, atomIdxs_to_which_new_matches_have_to_be_adjacent):

        valid_matches = []

        if mol_searched_in.GetNumAtoms() >= mol_searched_for.GetNumAtoms():
            matches = mol_searched_in.GetSubstructMatches(mol_searched_for)

            if matches:
                for match in matches:
                        add_this_match = True
                        if len(atomIdxs_to_which_new_matches_have_to_be_adjacent) > 0:
                            add_this_match = False

                            for i in match:
                                for neighbor in mol_searched_in.GetAtomWithIdx(i).GetNeighbors():
                                    if neighbor.GetIdx() in atomIdxs_to_which_new_matches_have_to_be_adjacent:
                                        add_this_match = True
                                        break

                        if add_this_match:
                            valid_matches.append(match)

        return valid_matches

    # count heavier isotopes of hydrogen correctly
    @classmethod
    def get_heavy_atom_count(cls, mol):
        heavy_atom_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                heavy_atom_count += 1

        return heavy_atom_count

    def __init__(self, fragmentation_scheme = {}, fragmentation_scheme_order = None, match_hydrogens = False, algorithm = '', n_atoms_cuttoff = -1, function_to_choose_fragmentation = False, n_max_fragmentations_to_find = -1):

        if not type(fragmentation_scheme) is dict:
            raise TypeError('fragmentation_scheme must be a dctionary with integers as keys and either strings or list of strings as values.')

        if len(fragmentation_scheme) == 0:
            raise ValueError('fragmentation_scheme must be provided.')

        if not algorithm in ['simple', 'complete', 'combined']:
            raise ValueError('Algorithm must be either simple ,complete or combined.')

        if algorithm == 'simple':
            if n_max_fragmentations_to_find != -1:
                raise ValueError('Setting n_max_fragmentations_to_find only makes sense with complete or combined algorithm.')

        self.algorithm = algorithm

        if algorithm in ['combined', 'complete']:
            if n_atoms_cuttoff == -1:
                raise ValueError('n_atoms_cuttoff needs to be specified for complete or combined algorithms.')

            if function_to_choose_fragmentation == False:
                raise ValueError('function_to_choose_fragmentation needs to be specified for complete or combined algorithms.')

            if not callable(function_to_choose_fragmentation):
                raise TypeError('function_to_choose_fragmentation needs to be a function.')
            else:
                if type(function_to_choose_fragmentation([{}, {}])) != dict:
                    raise TypeError('function_to_choose_fragmentation needs to take a list of fragmentations and choose one of it')

            if n_max_fragmentations_to_find != -1:
                if n_max_fragmentations_to_find < 1:
                    raise ValueError('n_max_fragmentations_to_find has to be 1 or higher.')

        if fragmentation_scheme_order is None:
            fragmentation_scheme_order = []

        if algorithm in ['simple', 'combined']:
            assert len(fragmentation_scheme) == len(fragmentation_scheme_order)
        else:
            fragmentation_scheme_order = [key for key in fragmentation_scheme.keys()]

        self.n_max_fragmentations_to_find = n_max_fragmentations_to_find

        self.n_atoms_cuttoff = n_atoms_cuttoff

        self.match_hydrogens = match_hydrogens

        self.fragmentation_scheme = fragmentation_scheme

        self.function_to_choose_fragmentation = function_to_choose_fragmentation

        # create a lookup dictionaries to faster finding a group number
        self._fragmentation_scheme_group_number_lookup = {}
        self._fragmentation_scheme_pattern_lookup = {}
        self.fragmentation_scheme_order = fragmentation_scheme_order

        for group_number, list_SMARTS in fragmentation_scheme.items():

            if type(list_SMARTS) is not list:
                list_SMARTS = [list_SMARTS]

            for SMARTS in list_SMARTS:
                if SMARTS != '':
                    self._fragmentation_scheme_group_number_lookup[SMARTS] = group_number

                    mol_SMARTS = Fragmenter.Chem.MolFromSmarts(SMARTS)
                    self._fragmentation_scheme_pattern_lookup[SMARTS] = mol_SMARTS

    def fragment(self, SMILES_or_molecule):

        if type(SMILES_or_molecule) is str:
            mol_SMILES = Fragmenter.Chem.MolFromSmiles(SMILES_or_molecule)
            mol_SMILES = Fragmenter.Chem.AddHs(mol_SMILES) if self.match_hydrogens else mol_SMILES
            is_valid_SMILES = mol_SMILES is not None

            if not is_valid_SMILES:
                raise ValueError('Following SMILES is not valid: ' + SMILES_or_molecule)

        else:
            mol_SMILES = SMILES_or_molecule

        # iterate over all separated molecules
        success = []
        fragmentation = {}
        fragmentation_matches = {}
        for mol in Fragmenter.rdmolops.GetMolFrags(mol_SMILES, asMols = True):

            this_mol_fragmentation, this_mol_success = self.__get_fragmentation(mol)

            for SMARTS, matches in this_mol_fragmentation.items():
                group_number = self._fragmentation_scheme_group_number_lookup[SMARTS]

                if not group_number in fragmentation:
                    fragmentation[group_number] = 0
                    fragmentation_matches[group_number] = []

                fragmentation[group_number] += len(matches)
                fragmentation_matches[group_number].extend(matches)

            success.append(this_mol_success)

        return fragmentation, all(success), fragmentation_matches

    def fragment_complete(self, SMILES_or_molecule):

        if type(SMILES_or_molecule) is str:
            mol_SMILES = Fragmenter.Chem.MolFromSmiles(SMILES_or_molecule)
            mol_SMILES = Fragmenter.Chem.AddHs(mol_SMILES) if self.match_hydrogens else mol_SMILES
            is_valid_SMILES = mol_SMILES is not None

            if not is_valid_SMILES:
                raise ValueError('Following SMILES is not valid: ' + SMILES_or_molecule)

        else:
            mol_SMILES = SMILES_or_molecule

        if len(Fragmenter.rdmolops.GetMolFrags(mol_SMILES)) != 1:
            raise ValueError('fragment_complete does not accept multifragment molecules.')

        temp_fragmentations, success = self.__complete_fragmentation(mol_SMILES)

        fragmentations = []
        fragmentations_matches = []
        for temp_fragmentation in temp_fragmentations:
            fragmentation = {}
            fragmentation_matches = {}
            for SMARTS, matches in temp_fragmentation.items():
                group_number = self._fragmentation_scheme_group_number_lookup[SMARTS]

                fragmentation[group_number] = len(matches)
                fragmentation_matches[group_number] = matches

            fragmentations.append(fragmentation)
            fragmentations_matches.append(fragmentation_matches)

        return fragmentations, success, fragmentations_matches


    def __get_fragmentation(self, mol_SMILES):

        success = False
        fragmentation = {}
        if self.algorithm in ['simple', 'combined']:
            fragmentation, success = self.__simple_fragmentation(mol_SMILES)

        if success:
            return fragmentation, success

        if self.algorithm in ['combined', 'complete']:
            fragmentations, success = self.__complete_fragmentation(mol_SMILES)

            if success:
                fragmentation = self.function_to_choose_fragmentation(fragmentations)

        return fragmentation, success

    def __simple_fragmentation(self, mol_SMILES):

        if self.match_hydrogens:
            target_atom_count = len(mol_SMILES.GetAtoms())
        else:
            target_atom_count = Fragmenter.get_heavy_atom_count(mol_SMILES)

        success = False
        fragmentation = {}

        fragmentation, atomIdxs_included_in_fragmentation = self.__search_non_overlapping_solution(mol_SMILES, {}, set(), set())
        success = len(atomIdxs_included_in_fragmentation) == target_atom_count

        # if not successful, clean up molecule and search again
        level = 1
        while not success:
            fragmentation_so_far , atomIdxs_included_in_fragmentation_so_far = Fragmenter.__clean_molecule_surrounding_unmatched_atoms(mol_SMILES, fragmentation, atomIdxs_included_in_fragmentation, level)
            level += 1

            if len(atomIdxs_included_in_fragmentation_so_far) == 0:
                break

            fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far = self.__search_non_overlapping_solution(mol_SMILES, fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far)

            success = len(atomIdxs_included_in_fragmentation_so_far) == target_atom_count

            if success:
                fragmentation = fragmentation_so_far

        return fragmentation, success

    def __search_non_overlapping_solution(self, mol_searched_in, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent):

        n_atomIdxs_included_in_fragmentation = len(atomIdxs_included_in_fragmentation) - 1

        while n_atomIdxs_included_in_fragmentation != len(atomIdxs_included_in_fragmentation):
            n_atomIdxs_included_in_fragmentation = len(atomIdxs_included_in_fragmentation)


            for group_number in self.fragmentation_scheme_order:
                list_SMARTS = self.fragmentation_scheme[group_number]
                if type(list_SMARTS) is not list:
                    list_SMARTS = [list_SMARTS]

                for SMARTS in list_SMARTS:
                    if SMARTS != "":
                        fragmentation, atomIdxs_included_in_fragmentation = self.__get_next_non_overlapping_match(mol_searched_in, SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent)

        return fragmentation, atomIdxs_included_in_fragmentation

    def __get_next_non_overlapping_match(self, mol_searched_in, SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent):

        mol_searched_for = self._fragmentation_scheme_pattern_lookup[SMARTS]

        if atomIdxs_to_which_new_matches_have_to_be_adjacent:
            matches = Fragmenter.get_substruct_matches(mol_searched_for, mol_searched_in, atomIdxs_to_which_new_matches_have_to_be_adjacent)
        else:
            matches = Fragmenter.get_substruct_matches(mol_searched_for, mol_searched_in, set())

        if matches:
            for match in matches:
                all_atoms_of_new_match_are_unassigned = atomIdxs_included_in_fragmentation.isdisjoint(match)

                if all_atoms_of_new_match_are_unassigned:
                    if not SMARTS in fragmentation:
                        fragmentation[SMARTS] = []

                    fragmentation[SMARTS].append(match)
                    atomIdxs_included_in_fragmentation.update(match)

        return fragmentation, atomIdxs_included_in_fragmentation

    @classmethod
    def __clean_molecule_surrounding_unmatched_atoms(cls, mol_searched_in, fragmentation, atomIdxs_included_in_fragmentation, level):

        for i in range(0, level):

            atoms_missing = set(range(0, Fragmenter.get_heavy_atom_count(mol_searched_in))).difference(atomIdxs_included_in_fragmentation)

            new_fragmentation = Fragmenter.marshal.loads(Fragmenter.marshal.dumps(fragmentation))

            for atomIdx in atoms_missing:
                for neighbor in mol_searched_in.GetAtomWithIdx(atomIdx).GetNeighbors():
                    for smart, atoms_found in fragmentation.items():
                        for atoms in atoms_found:
                            if neighbor.GetIdx() in atoms:
                                if smart in new_fragmentation:
                                    if new_fragmentation[smart].count(atoms) > 0:
                                        new_fragmentation[smart].remove(atoms)

                        if smart in new_fragmentation:
                            if len(new_fragmentation[smart]) == 0:
                                new_fragmentation.pop(smart)


            new_atomIdxs_included_in_fragmentation = set()
            for i in new_fragmentation.values():
                for j in i:
                    new_atomIdxs_included_in_fragmentation.update(j)

            atomIdxs_included_in_fragmentation = new_atomIdxs_included_in_fragmentation
            fragmentation = new_fragmentation

        return fragmentation, atomIdxs_included_in_fragmentation

    def __complete_fragmentation(self, mol_SMILES):

        heavy_atom_count = Fragmenter.get_heavy_atom_count(mol_SMILES)

        if heavy_atom_count > self.n_atoms_cuttoff:
            return {}, False

        completed_fragmentations = []
        groups_leading_to_incomplete_fragmentations = []
        completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found = self.__get_next_non_overlapping_adjacent_match_recursively(mol_SMILES, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, {}, set(), set(), self.n_max_fragmentations_to_find)
        success = len(completed_fragmentations) > 0

        return completed_fragmentations, success

    def __get_next_non_overlapping_adjacent_match_recursively(self, mol_searched_in, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far, atomIdxs_to_which_new_matches_have_to_be_adjacent, n_max_fragmentations_to_find = -1):

        n_completed_fragmentations = len(completed_fragmentations)
        incomplete_fragmentation_found = False
        complete_fragmentation_found = False

        if len(completed_fragmentations) == n_max_fragmentations_to_find:
            return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found


        for group_number in self.fragmentation_scheme_order:
            list_SMARTS = self.fragmentation_scheme[group_number]

            if complete_fragmentation_found:
                break

            if type(list_SMARTS) is not list:
                list_SMARTS = [list_SMARTS]

            for SMARTS in list_SMARTS:
                if complete_fragmentation_found:
                    break

                if SMARTS != "":
                    matches = Fragmenter.get_substruct_matches(self._fragmentation_scheme_pattern_lookup[SMARTS], mol_searched_in, atomIdxs_included_in_fragmentation_so_far)

                    for match in matches:

                        # only allow non-overlapping matches
                        all_atoms_are_unassigned = atomIdxs_included_in_fragmentation_so_far.isdisjoint(match)
                        if not all_atoms_are_unassigned:
                            continue

                        # only allow matches that do not contain groups leading to incomplete matches
                        for groups_leading_to_incomplete_fragmentation in groups_leading_to_incomplete_fragmentations:
                            if Fragmenter.__is_fragmentation_subset_of_other_fragmentation(groups_leading_to_incomplete_fragmentation, fragmentation_so_far):
                                return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found

                        # only allow matches that will lead to new fragmentations
                        use_this_match = True
                        n_found_groups = len(fragmentation_so_far)

                        for completed_fragmentation in completed_fragmentations:

                            if not SMARTS in completed_fragmentation:
                                continue

                            if n_found_groups == 0:
                                use_this_match = not Fragmenter.__is_match_contained_in_fragmentation(match, SMARTS, completed_fragmentation)
                            else:
                                if Fragmenter.__is_fragmentation_subset_of_other_fragmentation(fragmentation_so_far, completed_fragmentation):
                                    use_this_match = not Fragmenter.__is_match_contained_in_fragmentation(match, SMARTS, completed_fragmentation)

                            if not use_this_match:
                                break

                        if not use_this_match:
                            continue

                        # make a deepcopy here, otherwise the variables are modified down the road
                        # marshal is used here because it works faster than copy.deepcopy
                        this_SMARTS_fragmentation_so_far = Fragmenter.marshal.loads(Fragmenter.marshal.dumps(fragmentation_so_far))
                        this_SMARTS_atomIdxs_included_in_fragmentation_so_far = atomIdxs_included_in_fragmentation_so_far.copy()

                        if not SMARTS in this_SMARTS_fragmentation_so_far:
                            this_SMARTS_fragmentation_so_far[SMARTS] = []

                        this_SMARTS_fragmentation_so_far[SMARTS].append(match)
                        this_SMARTS_atomIdxs_included_in_fragmentation_so_far.update(match)

                        # only allow matches that do not contain groups leading to incomplete matches
                        for groups_leading_to_incomplete_match in groups_leading_to_incomplete_fragmentations:
                            if Fragmenter.__is_fragmentation_subset_of_other_fragmentation(groups_leading_to_incomplete_match, this_SMARTS_fragmentation_so_far):
                                use_this_match = False
                                break

                        if not use_this_match:
                            continue

                        # if the complete molecule has not been fragmented, continue to do so
                        if len(this_SMARTS_atomIdxs_included_in_fragmentation_so_far) < heavy_atom_count:
                            completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found = self.__get_next_non_overlapping_adjacent_match_recursively(mol_searched_in, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, this_SMARTS_fragmentation_so_far, this_SMARTS_atomIdxs_included_in_fragmentation_so_far, this_SMARTS_atomIdxs_included_in_fragmentation_so_far, n_max_fragmentations_to_find)
                            break

                        # if the complete molecule has been fragmented, save and return
                        if len(this_SMARTS_atomIdxs_included_in_fragmentation_so_far) == heavy_atom_count:
                            completed_fragmentations.append(this_SMARTS_fragmentation_so_far)
                            complete_fragmentation_found = True
                            break

        # if until here no new fragmentation was found check whether an incomplete fragmentation was found
        if n_completed_fragmentations == len(completed_fragmentations):

            if not incomplete_fragmentation_found:

                incomplete_matched_groups = {}

                if len(atomIdxs_included_in_fragmentation_so_far) > 0:
                    unassignes_atom_idx = set(range(0, heavy_atom_count)).difference(atomIdxs_included_in_fragmentation_so_far)
                    for atom_idx in unassignes_atom_idx:
                        neighbor_atoms_idx = [i.GetIdx() for i in mol_searched_in.GetAtomWithIdx(atom_idx).GetNeighbors()]

                        for neighbor_atom_idx in neighbor_atoms_idx:
                            for found_smarts, found_matches in fragmentation_so_far.items():
                                for found_match in found_matches:
                                    if neighbor_atom_idx in found_match:
                                        if not found_smarts in incomplete_matched_groups:
                                            incomplete_matched_groups[found_smarts] = []

                                        if found_match not in incomplete_matched_groups[found_smarts]:
                                            incomplete_matched_groups[found_smarts].append(found_match)

                    is_subset_of_groups_already_found = False
                    indexes_to_remove = []

                    for idx, groups_leading_to_incomplete_match in enumerate(groups_leading_to_incomplete_fragmentations):
                        is_subset_of_groups_already_found = Fragmenter.__is_fragmentation_subset_of_other_fragmentation(incomplete_matched_groups, groups_leading_to_incomplete_match)
                        if is_subset_of_groups_already_found:
                            indexes_to_remove.append(idx)

                    for index in sorted(indexes_to_remove, reverse=True):
                        del groups_leading_to_incomplete_fragmentations[index]

                    groups_leading_to_incomplete_fragmentations.append(incomplete_matched_groups)
                    groups_leading_to_incomplete_fragmentations = sorted(groups_leading_to_incomplete_fragmentations, key = len)

                    incomplete_fragmentation_found =  True

        return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found

    @classmethod
    def __is_fragmentation_subset_of_other_fragmentation(cls, fragmentation, other_fragmentation):
        n_found_groups = len(fragmentation)
        n_found_other_groups = len(other_fragmentation)

        if n_found_groups == 0:
            return False

        if n_found_other_groups < n_found_groups:
            return False

        n_found_SMARTS_that_are_subset = 0
        for found_SMARTS, _ in fragmentation.items():
            if found_SMARTS in other_fragmentation:
                found_matches_set = set(frozenset(i) for i in fragmentation[found_SMARTS])
                found_other_matches_set =  set(frozenset(i) for i in other_fragmentation[found_SMARTS])

                if found_matches_set.issubset(found_other_matches_set):
                    n_found_SMARTS_that_are_subset += 1
            else:
                return False

        return n_found_SMARTS_that_are_subset == n_found_groups

    @classmethod
    def __is_match_contained_in_fragmentation(cls, match, SMARTS, fragmentation):
        if not SMARTS in fragmentation:
            return False

        found_matches_set = set(frozenset(i) for i in fragmentation[SMARTS])
        match_set = set(match)

        return match_set in found_matches_set


class Project:
    """ the class that contains all method and info to analyze
    the project (intended as a collection of GCMS files, calibrations, etc)
    """

    folder_path = plib.Path.cwd()
    in_path = folder_path
    out_path = plib.Path(in_path, 'Output')
    shared_path = in_path.parents[0]
    plot_font='Dejavu Sans'
    plot_grid=False
    load_delimiter = '\t'
    load_skiprows = 8
    columns_to_keep_in_files = ['Ret.Time', 'Height', 'Area', 'Name']
    columns_to_rename_in_files = {'Ret.Time': 'retention_time', 'Height': 'height',
                                  'Area': 'area', 'Name': 'comp_name'}

    compounds_to_rename = {}
    param_to_axis_label = {'area': 'Peak Area [-]',
        'area_if_undiluted': 'Peak Area [-]',
        'conc_vial_mg_L':'vial conc. [mg/L] (ppm)',
        'fraction_of_sample_fr':'mass fraction [g/g$_{sample}$]',
        'fraction_of_feedstock_fr': 'mass fraction [g/g$_{feedstock}$]'}
    string_in_deriv_names = ['deriv.', 'derivative', 'TMS', 'TBDMS', 'trimethylsilyl']
    string_in_deriv_names = [s.lower() for s in string_in_deriv_names]
    files_info_defauls_columns = ['dilution_factor', 'total_sample_conc_in_vial_mg_L',
                                  'sample_yield_on_feedstock_basis_fr']
    semi_calibration = True
    tanimoto_similarity_threshold = 0.4
    delta_mol_weigth_threshold = 100

    @classmethod
    def set_folder_path(cls, path):
        """ necessary to specify the folder path with files """
        cls.folder_path = plib.Path(path).resolve()
        cls.in_path = cls.folder_path
        cls.out_path = plib.Path(cls.in_path, 'Output')
        plib.Path(cls.out_path).mkdir(parents=True, exist_ok=True)
        cls.shared_path = cls.in_path.parents[0]

    @classmethod
    def set_plot_font(cls, new_plot_font):
        """ Update plot font """
        cls.plot_font = new_plot_font

    @classmethod
    def set_tanimoto_similarity_threshold(cls, new_tanimoto_similarity_threshold):
        """ Update plot font """
        cls.tanimoto_similarity_threshold = new_tanimoto_similarity_threshold

    @classmethod
    def set_delta_mol_weigth_threshold(cls, new_delta_mol_weigth_threshold):
        """ Update plot font """
        cls.delta_mol_weigth_threshold = new_delta_mol_weigth_threshold

    @classmethod
    def set_plot_grid(cls, new_plot_grid):
        """ Update plot grid setting """
        cls.plot_grid = new_plot_grid

    @classmethod
    def set_load_skiprows(cls, new_load_skiprows):
        """ Update number of rows to skip when loading data """
        cls.load_skiprows = new_load_skiprows

    @classmethod
    def set_load_delimiter(cls, new_load_delimiter):
        """ Update delimiter used for loading data """
        cls.load_delimiter = new_load_delimiter

    @classmethod
    def set_columns_to_keep_in_files(cls, new_columns_to_keep_in_files):
        """ Update list of columns to drop """
        cls.columns_to_keep_in_files = new_columns_to_keep_in_files

    @classmethod
    def set_string_in_deriv_names(cls, new_string_in_deriv_names):
        """ Update list of columns to drop """
        cls.string_in_deriv_names = new_string_in_deriv_names

    @classmethod
    def set_compounds_to_rename(cls, new_compounds_to_rename):
        """ Update dictionary of compounds to rename """
        cls.compounds_to_rename = new_compounds_to_rename

    @classmethod
    def set_param_to_axis_label(cls, new_param_to_axis_label):
        """ Update parameter to axis label mapping """
        cls.param_to_axis_label = new_param_to_axis_label


    def __init__(self, rebuild_compounds_properties=False):
        """
        """
        self.files_info = None
        self.files_info_loaded = False
        self.samples_info = None
        self.samples_info_created = False
        self.samples_created = False
        self.class_code_frac_loaded = False
        self.list_of_all_compounds_created = False
        self.list_of_all_deriv_compounds_created = False
        self.calibrations_loaded = False
        self.calibrations_not_present = False
        self.calibrations = {}
        self.is_calibrations_deriv = {}
        self.list_of_all_compounds = []
        self.list_of_all_deriv_compounds = []


        self.class_code_frac = None
        self.compounds_properties = None
        self.deriv_compounds_properties = None
        self.compounds_properties_loaded = False
        self.deriv_compounds_properties_loaded = False
        self.iupac_to_files_added = False
        self.iupac_to_calibrations_added = False
        self.calibration_to_files_applied = False
        self.stats_to_files_info_added = False
        self.stats_to_samples_info_added = False
        self.samples_info = None
        self.list_of_created_param_reports = []
        self.list_of_created_param_aggrreps = []
        self.files = {}
        self.files_loaded = False
        self.is_files_deriv = {}
        self.samples = {}
        self.samples_std = {}
        self.reports = {}
        self.reports_std = {}
        self.aggrreps = {}
        self.aggrreps_std = {}
        self.deriv_samples_present = False

        self.load_files_info()

        if any(self.files_info['derivatized']):
            self.deriv_samples_present = True
            print('Info: derivatized samples are present')

        self.load_calibrations()

        if rebuild_compounds_properties:
            self.create_compounds_properties()
            if self.deriv_samples_present:
                self.create_deriv_compounds_properties()
        else:
            self.load_compounds_properties()
            if self.deriv_samples_present:
                self.load_deriv_compounds_properties()
        # if apply_calibration:
        #     self.apply_calibration_to_files()
        #     print('Info: calibarions applied')

        # self.create_samples_info()
    # class function definitions
    def load_files_info(self):
        """ tries to load or creates files_info if not found """
        try:
            self.files_info = pd.read_excel(plib.Path(Project.in_path, 'files_info.xlsx'),
                engine='openpyxl', index_col='filename')
            self._add_default_to_files_info()
            print('Info: files_info loaded')
        except FileNotFoundError:
            print('Info: files_info not found')
            self.files_info = pd.DataFrame()
            self.create_files_info()
            self._add_default_to_files_info()
        self.files_info.to_excel(plib.Path(Project.in_path, 'files_info.xlsx'))
        self.files_info_loaded = True
        return self.files_info

    def create_files_info(self):
        """ if files_info with provided data is not found, create default one
        using gcms files in the folder and default values
        """
        print('Info: autocreating files_info')
        filename = [a.parts[-1].split('.')[0]
            for a in list(Project.in_path.glob('**/*.txt'))]
        sample = [f.split('_')[0] for f in filename]
        replicate = [f.split('_')[1] for f in filename]
        self.files_info = pd.DataFrame({'filename':filename,
                                'sample':sample, 'replicate':replicate})
        self.files_info.set_index('filename', drop=True, inplace=True)
        self.files_info.to_excel(plib.Path(Project.in_path, 'files_info.xlsx'))

    def _add_default_to_files_info(self):
        """ adds default values if missign
        from both user provided or automatic created files_info
        """
        if 'sample' not in list(self.files_info):
            self.files_info['sample'] = \
                [f.split('_')[0] for f in self.files_info.index.tolist()]
        if 'derivatized' not in list(self.files_info):
            self.files_info['derivatized'] = False
        if 'calibration_file' not in list(self.files_info):
            self.files_info['calibration_file'] = False

        for col in Project.files_info_defauls_columns:
            if col not in list(self.files_info):
                self.files_info[col] = 1

    def load_calibrations(self):
        """ load calibration files and fix non numerical numbers,
        only load each calibration file ones"""
        if any(self.files_info['calibration_file']):
            _files_info = self.files_info.drop_duplicates(subset='calibration_file')
            for cal_name, is_cal_deriv in zip(_files_info['calibration_file'],
                                            _files_info['derivatized']):
                try:
                    cal_file = pd.read_excel(plib.Path(Project.in_path,
                        cal_name + '.xlsx'), index_col=0)
                except FileNotFoundError:
                    try:
                        cal_file = pd.read_excel(plib.Path(Project.shared_path,
                            cal_name + '.xlsx'), index_col=0)
                    except FileNotFoundError:
                        print('ERROR: ', cal_name , '.xlsx not found in project nor shared path')
                cal_file.index.name = 'comp_name'
                cols_cal_area = [c for c in list(cal_file) if 'Area' in c]
                cols_cal_ppms = [c for c in list(cal_file) if 'PPM' in c]
                cal_file[cols_cal_area + cols_cal_ppms] = \
                    cal_file[cols_cal_area + cols_cal_ppms].apply(pd.to_numeric, errors='coerce')
                cal_file['iupac_name'] = 'n.a.'
                new_cols_order = ['iupac_name'] + \
                    [col for col in cal_file.columns if col != 'iupac_name']
                cal_file = cal_file[new_cols_order]
                cal_file.index = cal_file.index.map(lambda x: x.lower())
                self.calibrations[cal_name] = cal_file
                self.is_calibrations_deriv[cal_name] = is_cal_deriv
            self.calibrations_loaded = True
            self.calibrations_not_present = False
            print('Info: load_calibrations: calibarions loaded')
        else:
            self.calibrations_loaded = True
            self.calibrations_not_present = True
            print('Info: load_calibrations: no calibarions specified')

        return self.calibrations, self.is_calibrations_deriv

    def load_class_code_frac(self):
        """ load the excel file as a df with information on SMARTS """
        try:  # first try to find the file in the folder
            self.class_code_frac = pd.read_excel(plib.Path(Project.in_path,
                'classifications_codes_fractions.xlsx'))
            print('Info: load_class_code_frac: classifications_codes_fractions loaded')
        except FileNotFoundError:  # then try in the common input folder
            try:
                self.class_code_frac = pd.read_excel(plib.Path(Project.shared_path,
                    'classifications_codes_fractions.xlsx'))
                print('Info: load_class_code_frac: classifications_codes_fractions loaded from' +
                      'shared folder (up one level)')
            except FileNotFoundError:
                print('ERROR: the file "classifications_codes_fractions.xlsx" was not found ',
                      'look in example/data for a template')
        return self.class_code_frac

    def load_all_files(self):
        """ load all files from files_info index in a dict, keys are filenames
        """
        print('Info: load_all_files: loop started')
        for filename, is_deriv in zip(self.files_info.index,
                                    self.files_info['derivatized']):
            file = self.load_single_file(filename)
            self.files[filename] = file
            self.is_files_deriv[filename] = is_deriv
        self.files_loaded = True
        print('Info: load_all_files: files loaded')
        return self.files, self.is_files_deriv

    def load_single_file(self, filename):
        """ load a single file given its name and clean data
        """
        file = pd.read_csv(plib.Path(Project.in_path, filename + '.txt'),
            delimiter=Project.load_delimiter, index_col=0, skiprows=Project.load_skiprows)
        columns_to_drop = [cl for cl in file.columns if cl not in Project.columns_to_keep_in_files]
        file.drop(columns_to_drop, axis=1, inplace=True)
        file.rename(Project.columns_to_rename_in_files, inplace=True, axis='columns')

        file['comp_name'] = file['comp_name'].fillna('unidentified')
        sum_areas_in_file = file.groupby('comp_name')['area'].sum()
        # the first ret time is kept for each duplicated Name
        file.drop_duplicates(subset='comp_name', keep='first', inplace=True)
        file.set_index('comp_name', inplace=True)  # set the cas as the index
        file['area'] = sum_areas_in_file  # used summed areas as areas

        file['area_if_undiluted'] = file['area'] * \
            self.files_info.loc[filename, 'dilution_factor']
        file['iupac_name'] = 'n.a.'
        new_cols_order = ['iupac_name'] + \
            [col for col in file.columns if col != 'iupac_name']
        file = file[new_cols_order]
        file.index.name = filename
        file.index = file.index.map(lambda x: x.lower())
        file.rename(Project.compounds_to_rename, inplace=True)
        print('\tInfo: load_single_file ', filename)
        return file

    def create_list_of_all_compounds(self):
        """ takes all compounds in all files and all calibrations and makes a
        list of with every unique compound, for underivatized and derivatized"""
        if not self.files_loaded:
            self.load_all_files()
        _dfs = []
        for filename, file in self.files.items():
            if not self.is_files_deriv[filename]:
                _dfs.append(file)
        for filename, file in self.calibrations.items():
            if not self.is_calibrations_deriv[filename]:
                _dfs.append(file)
        # non-derivatized compounds
        all_compounds = pd.concat(_dfs)
        set_of_all_compounds = pd.Index(all_compounds.index.unique())
        self.list_of_all_compounds = set_of_all_compounds.drop('unidentified')
        self.list_of_all_compounds_created = True
        print('Info: create_list_of_all_compounds: list_of_all_compounds created')
        return self.list_of_all_compounds

    def create_list_of_all_deriv_compounds(self):
        """ takes all compounds in all files and all calibrations and makes a
        list of with every unique compound, for underivatized and derivatized """
        if not self.files_loaded:
            self.load_all_files()
        _dfs_deriv = []
        for filename, file in self.files.items():
            if self.is_files_deriv[filename]:
                _dfs_deriv.append(file)
        add_to_idx = ', ' + Project.string_in_deriv_names[0]
        for filename, file in self.calibrations.items():
            if self.is_calibrations_deriv[filename]:
                # need to add to calib index to match file names
                file.index = file.index.map(lambda x: x + add_to_idx)
                _dfs_deriv.append(file)
        all_deriv_compounds = pd.concat(_dfs_deriv)
        self.list_of_all_deriv_compounds = pd.Index(
            all_deriv_compounds.index.unique()).drop('unidentified')
        self.list_of_all_deriv_compounds_created = True
        print('Info: create_list_of_all_deriv_compounds: list_of_all_deriv_compounds created')
        return self.list_of_all_deriv_compounds

    def load_compounds_properties(self):
        """ tries to load or creates compounds_properties if not found """
        try:
            self.compounds_properties = pd.read_excel(plib.Path(Project.in_path,
                'compounds_properties.xlsx'), index_col='comp_name')
            print('Info: compounds_properties loaded')
        except FileNotFoundError:
            print('Warning: compounds_properties.xlsx not found, creating it')
            self.create_compounds_properties()
        self.compounds_properties_loaded = True
        return self.compounds_properties

    def load_deriv_compounds_properties(self):
        """ tries to load or creates compounds_properties if not found """
        try:
            self.deriv_compounds_properties = pd.read_excel(plib.Path(Project.in_path,
                'deriv_compounds_properties.xlsx'), index_col='comp_name')
            print('Info: deriv_compounds_properties loaded')
        except FileNotFoundError:
            print('Warning: deriv_compounds_properties.xlsx not found, creating it')
            self.create_deriv_compounds_properties()
        self.deriv_compounds_properties_loaded = True
        return self.deriv_compounds_properties

    def create_compounds_properties(self):
        """ used to retrieve and organize properties for compounds using pubchempy
        """
        print('Info: create_compounds_properties: started')

        if not self.class_code_frac_loaded:
            self.load_class_code_frac()
        if not self.list_of_all_compounds_created:
            self.create_list_of_all_compounds()
        cpdf = pd.DataFrame(index=self.list_of_all_compounds)
        cpdf.index.name = 'comp_name'
        print('Info: create_compounds_properties: looping over names')
        for name in cpdf.index:
            cpdf = name_to_properties(name,
                cpdf, self.class_code_frac)
        # cpdf = cpdf.drop_duplicates(subset='iupac_name')
        fg_mf_cols = [c for c in list(cpdf) if 'fg_mf' in c]
        cpdf['fg_total'] = cpdf.loc[:, fg_mf_cols].sum(axis=1)
        # order columns for better visualization
        ordered_cols = list(cpdf)[:5] + \
            list(cpdf)[-1:] + \
            sorted(list(cpdf)[5:-1])
        cpdf = cpdf[ordered_cols]
        cpdf = cpdf.fillna(0)
        # save db in the project folder in the input
        cpdf.to_excel(plib.Path(Project.in_path, 'compounds_properties.xlsx'))
        self.compounds_properties = cpdf
        print('Info: create_compounds_properties: compounds_properties created')
        return self.compounds_properties

    def create_deriv_compounds_properties(self):
        """ used to retrieve and organize properties for compounds using pubchempy,
        but using the underivatized form of the chemical, while keeping the original
        name as the index
        """
        if not self.class_code_frac_loaded:
            self.load_class_code_frac()
        if not self.list_of_all_compounds_created:
            self.create_list_of_all_deriv_compounds()

        unique_deriv_compounds = self.list_of_all_deriv_compounds
        unique_underiv_compounds = [",".join(name.split(',')[:-1])
                                    for name in unique_deriv_compounds]
        dcpdf = pd.DataFrame(index=unique_underiv_compounds)
        dcpdf.index.name = 'comp_name'
        dcpdf['deriv_comp_name'] = unique_deriv_compounds
        print('Info: create_deriv_compounds_properties: looping over names')
        for name in dcpdf.index:
            dcpdf = name_to_properties(name, dcpdf, self.class_code_frac)
        # remove duplicates that may come from the "made up" name in calibration
        # dcpdf = dcpdf.drop_duplicates(subset='iupac_name')
        dcpdf['underiv_comp_name'] = dcpdf.index
        dcpdf.set_index('deriv_comp_name', inplace=True)
        dcpdf.index.name = 'comp_name'
        # self.compounds_properties.set_index('comp_name', drop=True, inplace=True)
        # the sum of all the fg_mf is 1 (or close enough)
        fg_mf_cols = [c for c in list(dcpdf) if 'fg_mf' in c]
        dcpdf['fg_total'] = dcpdf.loc[:, fg_mf_cols].sum(axis=1)
        # order columns for better visualization
        ordered_cols = list(dcpdf)[:5] + list(dcpdf)[-1:] + sorted(list(dcpdf)[5:-1])
        dcpdf = dcpdf[ordered_cols]
        dcpdf = dcpdf.fillna(0)
        # save db in the project folder in the input
        dcpdf.to_excel(plib.Path(Project.in_path, 'deriv_compounds_properties.xlsx'))
        self.deriv_compounds_properties = dcpdf
        print('Info: create_deriv_compounds_properties: deriv_compounds_properties created')
        return self.deriv_compounds_properties

    def add_iupac_to_files(self):
        ''' add iupac name as a column in each file
        for derviatized samples is the iupac name of the
        underivatized compounds'''
        if not self.compounds_properties_loaded:
            self.load_compounds_properties()
        if self.deriv_samples_present:
            if not self.deriv_compounds_properties_loaded:
                self.load_deriv_compounds_properties()
        for filename, file in self.files.items():
            if not self.is_files_deriv[filename]:
                df_comps = self.compounds_properties
            else:
                df_comps = self.deriv_compounds_properties
            for c in file.index.tolist():
                if c == 'unidentified':
                    file.loc[c, 'iupac_name'] = 'unidentified'
                else:
                    iup = df_comps.loc[c, 'iupac_name']
                    file.loc[c, 'iupac_name'] = iup
            # new_cols_order = ['iupac_name'] + [col for col in file.columns
            #                                    if col != 'iupac_name']
            # self.files[filename] = file[new_cols_order]
        self.iupac_to_files_added = True

    def add_iupac_to_calibrations(self):
        ''' add iupac name as a column in each calibration
        for derviatized calibrations is the iupac name of the
        underivatized compounds'''
        if not self.compounds_properties_loaded:
            self.load_compounds_properties()
        if self.deriv_samples_present:
            if not self.deriv_compounds_properties_loaded:
                self.load_deriv_compounds_properties()
        for calibname, calib in self.calibrations.items():
            if not self.is_calibrations_deriv[calibname]:
                df_comps = self.compounds_properties
                for c in calib.index.tolist():
                    iup = df_comps.loc[c, 'iupac_name']
                    calib.loc[c, 'iupac_name'] = iup
            else:
                df_comps = self.deriv_compounds_properties
                for c in calib.index.tolist():
                    iup = df_comps.loc[c, 'iupac_name']
                    calib.loc[c, 'iupac_name'] = iup
        self.iupac_to_calibrations_added = True

    def apply_calibration_to_files(self, save_to_excel=True):
        """ applies the correct calibration to each file"""

        print('Info: apply_calibration_to_files: loop started')
        if not self.files_loaded:
            self.load_all_files()
        if not self.calibrations_loaded:
            self.load_calibrations()
        if self.calibrations_not_present:
            print('WARNING: apply_calibration_to_files, no calibration is available',
                  'files are unchanged')
            return self.files, self.is_files_deriv
        if not self.iupac_to_files_added:
            self.add_iupac_to_files()
        if not self.iupac_to_calibrations_added:
            self.add_iupac_to_calibrations()
        for filename, _ in self.files.items():
            calibration_name = self.files_info.loc[filename, 'calibration_file']
            calibration = self.calibrations[calibration_name]
            if not self.is_files_deriv[filename]:
                df_comps = self.compounds_properties
            else:
                df_comps = self.deriv_compounds_properties
            file = self._apply_calib_to_file(filename, calibration,
                df_comps)
            if save_to_excel:
                self.save_file(file, filename)
        self.calibration_to_files_applied = True
        return self.files, self.is_files_deriv

    def _apply_calib_to_file(self, filename, calibration, df_comps):
            # """calibration.rename(Project.compounds_to_rename, inplace=True)"""
                # print(file)
        print('\tInfo: _apply_calib_to_file ', filename)
        clbrtn = calibration.set_index('iupac_name')
        cpmnds = df_comps.set_index('iupac_name')
        cpmnds = cpmnds[~cpmnds.index.duplicated(keep='first')].copy()
        cols_cal_area = [c for c in list(calibration) if 'Area' in c]
        cols_cal_ppms = [c for c in list(calibration) if 'PPM' in c]
        tot_sample_conc = \
            self.files_info.loc[filename, 'total_sample_conc_in_vial_mg_L']
        sample_yield_feed_basis = \
            self.files_info.loc[filename, 'sample_yield_on_feedstock_basis_fr']
        for comp, iupac in zip(self.files[filename].index.tolist(),
                               self.files[filename]['iupac_name'].tolist()):
            if comp == 'unidentified' or iupac == 'unidentified':
                conc_mg_l = np.nan
                comps_for_calib = 'n.a.'
            else:
                if iupac in clbrtn.index.tolist():
                    # areas and ppms for the calibration are taken from df_clbr
                    cal_areas = \
                        clbrtn.loc[iupac, cols_cal_area].to_numpy(dtype=float)
                    cal_ppms = \
                        clbrtn.loc[iupac, cols_cal_ppms].to_numpy(dtype=float)
                    # linear fit of calibration curve (exclude nan),
                    # get ppm from area
                    fit = np.polyfit(cal_areas[~np.isnan(cal_areas)],
                        cal_ppms[~np.isnan(cal_ppms)], 1)
                    # concentration at the injection solution (GC vial)
                    # ppp = mg/L
                    conc_mg_l = np.poly1d(fit)(self.files[filename].loc[comp, 'area'])
                    if conc_mg_l < 0:
                        conc_mg_l = 0
                    comps_for_calib = 'self'
                else:
                    if not Project.semi_calibration:
                        conc_mg_l = np.nan
                        comps_for_calib = 'n.a.'
                        continue
                    # get property of the compound as first elements
                    mws = [cpmnds.loc[iupac, 'molecular_weight']]
                    smis = [cpmnds.loc[iupac, 'canonical_smiles']]
                    names_cal = [iupac]
                    # then add all properties for all calibrated compounds
                    # if the sample was not derivatized (default)
                    # if not self.is_files_deriv[filename]:
                    for c in clbrtn.index.tolist():
                        names_cal.append(c)
                        # print(df_comps.index)
                        smis.append(cpmnds.loc[c, 'canonical_smiles'])
                        mws.append(cpmnds.loc[c, 'molecular_weight'])
                    # calculate the delta mw with all calib compounds
                    delta_mw = np.abs(np.asarray(mws)[0]
                                    - np.asarray(mws)[1:])
                    # get mols and fingerprints from rdkit for each comp
                    mols = [Chem.MolFromSmiles(smi) for smi in smis]
                    fps = [GetMorganFingerprintAsBitVect(ml, 2, nBits=1024)
                        for ml in mols]
                    # perform Tanimoto similarity betwenn the first and all
                    # other compounds
                    s = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
                    # create a df with results
                    df_sim = pd.DataFrame(data={'name': names_cal[1:],
                        'smiles': smis[1:], 'Similarity': s, 'delta_mw': delta_mw})
                    # put the index title as the comp
                    df_sim.index.name = iupac
                    # sort values based on similarity and delta mw
                    df_sim = df_sim.sort_values(['Similarity', 'delta_mw'],
                                                ascending=[False, True])
                    # remove values below thresholds
                    df_sim = df_sim[df_sim.Similarity >=
                                    Project.tanimoto_similarity_threshold]
                    df_sim = df_sim[df_sim.delta_mw <
                                    Project.delta_mol_weigth_threshold]
                    # if a compound matches the requirements
                    if not df_sim.empty:  # assign the calibration
                        name_clbr = df_sim.name.tolist()[0]

                        # areas and ppms are taken from df_clbr
                        cal_areas = clbrtn.loc[name_clbr, cols_cal_area
                                            ].to_numpy(dtype=float)
                        cal_ppms = clbrtn.loc[name_clbr, cols_cal_ppms
                                            ].to_numpy(dtype=float)
                        # linear fit of calibration curve (exclude nan),
                        # get ppm from area
                        fit = np.polyfit(cal_areas[~np.isnan(cal_areas)],
                                        cal_ppms[~np.isnan(cal_ppms)], 1)
                        # concentration at the injection solution (GC vial)
                        # ppm = mg/L
                        conc_mg_l = np.poly1d(fit)(self.files[filename].loc[comp, 'area'])
                        if conc_mg_l < 0:
                            conc_mg_l = 0
                        # note type of calibration and compound used
                        comps_for_calib = name_clbr + ' (sim=' + \
                            str(round(df_sim.Similarity.values[0], 2)) + \
                            '; dwt=' + str(int(df_sim.delta_mw.values[0])) + ')'
                    else:  # put concentrations to nan
                        conc_mg_l = np.nan
                        comps_for_calib = 'n.a.'
            self.files[filename].loc[comp, 'conc_vial_mg_L'] = conc_mg_l

            self.files[filename].loc[comp,
                'fraction_of_sample_fr'] = conc_mg_l/tot_sample_conc
            self.files[filename].loc[comp,
                'fraction_of_feedstock_fr'] = \
                conc_mg_l/tot_sample_conc*sample_yield_feed_basis
            self.files[filename].loc[comp,
                'compound_used_for_calibration'] = comps_for_calib
        return self.files[filename]

    def add_stats_to_files_info(self, save_to_excel=True):
        """ compute single file statistics and
        """
        print('Info: add_stats_to_files_info: started')
        if not self.calibration_to_files_applied:
            self.apply_calibration_to_files()
        for name, df in self.files.items():
            self.files_info.loc[name, 'max_height'] = df['height'].max()
            self.files_info.loc[name, 'max_area'] = df['area'].max()
            #
            self.files_info.loc[name, 'max_conc_vial_mg_L'] = \
                df['conc_vial_mg_L'].max()
            self.files_info.loc[name, 'max_fraction_of_sample_fr'] = \
                df['fraction_of_sample_fr'].max()
            self.files_info.loc[name, 'fraction_of_feedstock_fr'] = \
                df['fraction_of_feedstock_fr'].max()
            # total values
            self.files_info.loc[name, 'total_area'] = df['area'].sum()
            self.files_info.loc[name, 'total_conc_vial_mg_L'] = \
                df['conc_vial_mg_L'].sum()
            self.files_info.loc[name, 'total_fraction_of_sample_fr'] = \
                df['fraction_of_sample_fr'].sum()
            self.files_info.loc[name, 'total_fraction_of_feedstock_fr'] = \
                df['fraction_of_feedstock_fr'].sum()
            self.files_info.loc[name, 'compound_with_max_conc'] = \
                df[df['conc_vial_mg_L'] ==
                   self.files_info.loc[name, 'max_conc_vial_mg_L']].index[0]
        if save_to_excel:
            self.save_files_info()
        self.stats_to_files_info_added = True
        return self.files_info

    def create_samples_info(self):
        """ creates samples_info from files_info """
        if not self.files_info_loaded:
            self.load_files_info()
        _samples_info = self.files_info.reset_index(
            ).groupby('sample').agg(list)
        _samples_info.reset_index(inplace=True)
        _samples_info.set_index('sample', drop=True, inplace=True)
        self.samples_info = _samples_info
        self.samples_info_created = True
        print('Info: create_samples_info: samples_info created')
        return self.samples_info

    def create_samples_from_files(self, save_to_excel=True):
        """ create sample as ave and std of files with the same
        name (but different replicate (ex A_1 and A_2))
        """
        if not self.calibration_to_files_applied:
            self.apply_calibration_to_files()
        for samplename in self.samples_info.index:
            print('Sample: ', samplename)
            _files = []
            for filename in self.files_info.index[
                self.files_info['sample'] == samplename]:
                print('\tFile: ', filename)
                _files.append(self.files[filename])
            sample, sample_std = \
                self._create_sample_from_files(_files, samplename)
            self.samples[samplename] = sample
            self.samples_std[samplename] = sample_std
            if save_to_excel:
                self.save_sample(sample, sample_std, samplename)
        self.samples_created = True
        return self.samples, self.samples_std

    def _create_sample_from_files(self, files_in_sample, samplename):
        """ create a sample df and a sample_std df from
        files that are replicates of the same sample
        """

        all_ordered_columns = files_in_sample[0].columns.tolist()
        non_num_columns = ['iupac_name', 'compound_used_for_calibration']
        aligned_dfs = [df.align(files_in_sample[0], join='outer', axis=0)[0]
            for df in files_in_sample]  # Align indices
        # Keep non-numerical data separately and ensure no duplicates
        non_num_data = pd.concat([df[non_num_columns].drop_duplicates()
                                  for df in files_in_sample]).drop_duplicates()
        filled_dfs = [f.drop(columns=non_num_columns).fillna(0) for f in aligned_dfs]
        # Calculating the average and std for numerical data
        sample = pd.concat(filled_dfs).groupby(level=0).mean()
        sample_std = pd.concat(filled_dfs).groupby(level=0).std()
        # Merging non-numerical data with the numerical results
        sample = sample.merge(non_num_data, left_index=True,
                              right_index=True, how='left')
        sample_std = sample_std.merge(non_num_data, left_index=True,
                                      right_index=True, how='left')
        sample = sample.sort_values(by='retention_time')
        # Apply the same order to 'sample_std' using reindex
        sample_std = sample_std.reindex(sample.index)
        sample = sample[all_ordered_columns]
        sample_std = sample_std[all_ordered_columns]
        sample.index.name = samplename
        sample_std.index.name = samplename

        return sample, sample_std

    def add_stats_to_samples_info(self, save_to_excel=True):
        """ compute single sample statistics and
        """
        print('Info: add_stats_to_samples_info: started')
        if not self.samples_created:
            self.create_samples_from_files()
        for name, df in self.samples.items():
            self.samples_info.loc[name, 'max_height'] = df['height'].max()
            self.samples_info.loc[name, 'max_area'] = df['area'].max()
            #
            self.samples_info.loc[name, 'max_conc_vial_mg_L'] = \
                df['conc_vial_mg_L'].max()
            self.samples_info.loc[name, 'max_fraction_of_sample_fr'] = \
                df['fraction_of_sample_fr'].max()
            self.samples_info.loc[name, 'fraction_of_feedstock_fr'] = \
                df['fraction_of_feedstock_fr'].max()
            # total values
            self.samples_info.loc[name, 'total_area'] = df['area'].sum()
            self.samples_info.loc[name, 'total_conc_vial_mg_L'] = \
                df['conc_vial_mg_L'].sum()
            self.samples_info.loc[name, 'total_fraction_of_sample_fr'] = \
                df['fraction_of_sample_fr'].sum()
            self.samples_info.loc[name, 'total_fraction_of_feedstock_fr'] = \
                df['fraction_of_feedstock_fr'].sum()
            self.samples_info.loc[name, 'compound_with_max_conc'] = \
                df[df['conc_vial_mg_L'] ==
                   self.samples_info.loc[name, 'max_conc_vial_mg_L']].index[0]
        self.stats_to_samples_info_added = True
        if save_to_excel:
            self.save_samples_info()
        self.stats_to_samples_info_added = True
        return self.samples_info

    def create_param_report(self, param='conc_vial_mg_L', save_to_excel=True):
        """ create report with the results for each sample for each param
        """
        print('Info: create_param_report: ', param)
        if not self.samples_created:
            self.create_samples_from_files()
        _all_comps = self.compounds_properties['iupac_name'].tolist()
        if self.deriv_samples_present:
            _all_comps += self.deriv_compounds_properties['iupac_name'].tolist()
        rep = pd.DataFrame(index=list(set(_all_comps)),
            columns=self.samples_info.index, dtype='float')
        rep_std = pd.DataFrame(index=list(set(_all_comps)),
            columns=self.samples_info.index, dtype='float')
        rep.index.name, rep_std.index.name = param, param

        for comp in rep.index.tolist():  # add conc values
            for samplename in rep.columns.tolist():
                smp = self.samples[samplename].set_index('iupac_name')
                try:
                    ave = smp.loc[comp, param]
                except KeyError:
                    ave = 0
                smp_std = self.samples_std[samplename].set_index('iupac_name')
                try:
                    std = smp_std.loc[comp, param]
                except KeyError:
                    std = np.nan
                rep.loc[comp, samplename] = ave
                rep_std.loc[comp, samplename] = std

        rep = rep.sort_index(key=rep.max(1).get, ascending=False)
        rep = rep.loc[:, rep.any(axis=0)] # drop columns with only 0s
        rep = rep.loc[rep.any(axis=1), :] # drop rows with only 0s
        rep_std = rep_std.reindex(rep.index)
        self.reports[param] = rep
        self.reports_std[param] = rep_std
        self.list_of_created_param_reports.append(param)
        if save_to_excel:
            self.save_param_report(param=param)
        return rep, rep_std

    def create_param_aggrrep(self, param='conc_vial_mg_L', save_to_excel=True):
        """ create report with the results for each sample for each param
        but aggrefated by functional group
        """
        print('Info: create_param_aggrrep: ', param)
        if param not in self.list_of_created_param_reports:
            self.create_param_report(param)
        # fg = functional groups, mf = mass fraction
        samplenames = self.samples_info.index.tolist()
        _all_comps = self.reports[param].index.tolist()
        cols_with_fg_mf_labs = list(self.compounds_properties)
        if self.deriv_samples_present:
            for c in list(self.deriv_compounds_properties):
                if c not in cols_with_fg_mf_labs:
                    cols_with_fg_mf_labs.append(c)
        fg_mf_labs = [c for c in cols_with_fg_mf_labs if c.startswith('fg_mf_')]
        fg_labs = [c[6:] for c in fg_mf_labs]
        # create a df with iupac name index and fg_mf columns (underiv and deriv)
        comps_df = self.compounds_properties.set_index('iupac_name')
        if self.deriv_samples_present:
            deriv_comps_df = self.deriv_compounds_properties.set_index('iupac_name')
            all_comps_df = pd.concat([comps_df, deriv_comps_df])
        else:
            all_comps_df = comps_df
        all_comps_df = all_comps_df[~all_comps_df.index.duplicated(keep='first')]
        fg_mf_all = pd.DataFrame(index=_all_comps, columns=fg_mf_labs)
        for idx in fg_mf_all.index.tolist():
            fg_mf_all.loc[idx, fg_mf_labs] = all_comps_df.loc[idx, fg_mf_labs]
        # create the aggregated dataframes and compute aggregated results
        aggrrep = pd.DataFrame(columns=samplenames, index=fg_labs,
            dtype='float')
        aggrrep.index.name = param  # is the parameter
        aggrrep.fillna(0, inplace=True)
        aggrrep_std = pd.DataFrame(columns=samplenames, index=fg_labs,
                                   dtype='float')
        aggrrep_std.index.name = param  # is the parameter
        aggrrep_std.fillna(0, inplace=True)
        for col in samplenames:
            list_iupac = self.reports[param].index
            signal = self.reports[param].loc[:, col].values
            signal_std = self.reports_std[param].loc[:, col].values
            for fg, fg_mf in zip(fg_labs, fg_mf_labs):
                # each compound contributes to the cumulative sum of each
                # functional group for the based on the mass fraction it has
                # of that functional group (fg_mf act as weights)
                # if fg_mf in subrep: multiply signal for weigth and sum
                # to get aggregated
                weights = fg_mf_all.loc[list_iupac, fg_mf].astype(signal.dtype)

                aggrrep.loc[fg, col] = (signal*weights).sum()
                aggrrep_std.loc[fg, col] = (signal_std*weights).sum()
        aggrrep = aggrrep.loc[(aggrrep != 0).any(axis=1), :]  # drop rows with only 0
        aggrrep_std = aggrrep_std.reindex(aggrrep.index)
        aggrrep = aggrrep.sort_index(key=aggrrep[samplenames].max(1).get,
                            ascending=False)
        aggrrep_std = aggrrep_std.reindex(aggrrep.index)

        self.aggrreps[param] = aggrrep
        self.aggrreps_std[param] = aggrrep_std
        self.list_of_created_param_aggrreps.append(param)
        if save_to_excel:
            self.save_param_aggrrep(param=param)
        return aggrrep, aggrrep_std

    def save_files_info(self):
        """ save files_info as excel in the Files subfolder"""
        out_path = plib.Path(Project.out_path, 'files')
        out_path.mkdir(parents=True, exist_ok=True)
        self.files_info.to_excel(plib.Path(out_path, 'files_infos.xlsx'))
        print('Info: save_files_info: files_info saved')

    def save_file(self, file, filename):
        """ save files df as excel in the Files subfolder"""
        out_path = plib.Path(Project.out_path, 'files')
        out_path.mkdir(parents=True, exist_ok=True)
        file.to_excel(plib.Path(out_path, filename + '.xlsx'))
        print('Info: save_files: ', filename, ' saved')

    def save_samples_info(self):
        """ save files df as excel in the Files subfolder"""
        if not self.stats_to_samples_info_added:
            self.add_stats_to_samples_info()
        out_path = plib.Path(Project.out_path, 'samples')
        out_path.mkdir(parents=True, exist_ok=True)
        self.files_info.to_excel(plib.Path(out_path, 'samples_info.xlsx'))
        print('Info: save_samples_info: samples_info saved')

    def save_sample(self, sample, sample_std, samplename):
        """ save files df as excel in the Files subfolder"""
        out_path = plib.Path(Project.out_path, 'samples')
        out_path.mkdir(parents=True, exist_ok=True)
        sample.to_excel(plib.Path(out_path, samplename + '.xlsx'))
        sample_std.to_excel(plib.Path(out_path, samplename + '_std.xlsx'))
        print('Info: save_sample: ', samplename,'saved')

    def save_param_report(self, param='conc_inj_mg_L'):
        """ save param report to MultiReport folder
        """
        if param not in self.list_of_created_param_reports:
            self.create_param_report(param)
        name = 'Rep_' + param
        out_path = plib.Path(Project.out_path, 'reports')
        out_path.mkdir(parents=True, exist_ok=True)
        self.reports[param].to_excel(plib.Path(out_path, name + '.xlsx'))
        self.reports_std[param].to_excel(plib.Path(out_path, name + '_std.xlsx'))
        print('Info: save_param_report: Rep_' + param,'saved')

    def save_param_aggrrep(self, param='conc_inj_mg_L'):
        """ save param aggregated report to MultiReport folder """
        if param not in self.list_of_created_param_aggrreps:
            self.create_param_aggrrep(param)
        name = 'AggregRep_' + param
        out_path = plib.Path(Project.out_path, 'aggregated_reports')
        out_path.mkdir(parents=True, exist_ok=True)
        self.aggrreps[param].to_excel(plib.Path(out_path, name + '.xlsx'))
        self.aggrreps_std[param].to_excel(plib.Path(out_path, name + '_std.xlsx'))
        print('Info: save_param_aggrrep: AggregRep_' + param,'saved')

    def plot_ave_std(self, filename='plot',
        param='conc_vial_mg_L', aggr=False, min_y_thresh=None,
        only_samples_to_plot=None, rename_samples=None, reorder_samples=None,
        item_to_color_to_hatch=None,
        paper_col=.8, fig_hgt_mlt=1.5, xlab_rot=0, annotate_outliers=True,
        color_palette='deep',
        y_lab=None, y_lim=None, y_ticks=None,
        yt_sum=False, yt_lim=None, yt_lab=None, yt_ticks=None,
        yt_sum_label='total\n(right axis)',
        legend_location='best', legend_columns=1,
        legend_x_anchor=1, legend_y_anchor=1.02, legend_labelspacing=0.5,
        annotate_lttrs=False,
        note_plt=None):
        """
        Generates a bar plot from average and standard deviation values.
        legend_location can be None, outside, or all the best center left etc

        """
        # create folder where Plots are stored
        out_path = plib.Path(Project.out_path, 'plots')
        out_path.mkdir(parents=True, exist_ok=True)
        if not aggr:  # then use compounds reports
            df_ave = self.reports[param].T
            df_std = self.reports_std[param].T
        else:  # use aggregated reports
            df_ave = self.aggrreps[param].T
            df_std = self.aggrreps_std[param].T

        if only_samples_to_plot is not None:
            df_ave = df_ave.loc[only_samples_to_plot, :].copy()
            df_std = df_std.loc[only_samples_to_plot, :].copy()

        if rename_samples is not None:
            df_ave.index = rename_samples
            df_std.index = rename_samples

        if reorder_samples is not None:
            filtered_reorder_samples = [idx for idx in reorder_samples if idx in df_ave.index]
            df_ave = df_ave.reindex(filtered_reorder_samples)
            df_std = df_std.reindex(filtered_reorder_samples)

        if min_y_thresh is not None:
            df_ave = df_ave.loc[:, (df_ave > min_y_thresh).any(axis=0)].copy()
            df_std = df_std.loc[:, df_ave.columns].copy()

        if item_to_color_to_hatch is not None:  # specific color and hatches to each fg
            colors = [item_to_color_to_hatch.loc[item, 'clr'] for item in df_ave.columns]
            htchs = [item_to_color_to_hatch.loc[item, 'htch'] for item in df_ave.columns]
        else:  # no specific colors and hatches specified
            colors = sns.color_palette(color_palette, df_ave.shape[1])
            htchs = (None, '//', '...', '--', 'O', '\\\\', 'oo', '\\\\\\',
                    '/////', '.....', '//', '...', '--', 'O', '\\\\', 'oo',
                    '\\\\\\', '/////', '.....', '//', '...', '--', 'O', '\\\\',
                    'oo', '\\\\\\', '/////', '.....', '//', '...', '--', 'O',
                    '\\\\', 'oo', '\\\\\\', '/////', '.....')
        if yt_sum:
            plot_type = 1
        else:
            plot_type = 0

        fig, ax, axt, fig_par = figure_create(rows=1, cols=1, plot_type=plot_type,
            paper_col=paper_col, hgt_mltp=fig_hgt_mlt, font=Project.plot_font)
        if df_std.isna().all().all():  # means that no std is provided
            df_ave.plot(ax=ax[0], kind='bar', rot=xlab_rot, width=.9,
                        edgecolor='k', legend=False, yerr=df_std,
                        capsize=3, color=colors)
            bars = ax[0].patches  # needed to add patches to the bars
            n_different_hatches = int(len(bars)/df_ave.shape[0])
        else:  # no legend is represented but non-significant values are shaded
            mask = (df_ave.abs() > df_std.abs()) | df_std.isna()

            df_ave[mask].plot(ax=ax[0], kind='bar', rot=xlab_rot, width=.9,
                            edgecolor='k', legend=False, yerr=df_std[mask],
                            capsize=3, color=colors, label='_nolegend')
            df_ave[~mask].plot(ax=ax[0], kind='bar', rot=xlab_rot, width=.9,
                            legend=False, edgecolor='grey', color=colors,
                            alpha=.5, label='_nolegend')
            bars = ax[0].patches  # needed to add patches to the bars
            n_different_hatches = int(len(bars)/df_ave.shape[0]/2)
        if yt_sum:
            axt[0].scatter(df_ave.index, df_ave.sum(axis=1).values,
                        color='k', linestyle='None', edgecolor='k',
                        facecolor='grey', s=100, label=yt_sum_label, alpha=.5)
            if df_std is not None:
                axt[0].errorbar(df_ave.index, df_ave.sum(axis=1).values,
                                df_std.sum(axis=1).values, capsize=3,
                                linestyle='None', color='grey', ecolor='k')
        bar_htchs = []
        # get a list with the htchs
        for h in htchs[:n_different_hatches] + htchs[:n_different_hatches]:
            for n in range(df_ave.shape[0]):  # htcs repeated for samples
                bar_htchs.append(h)  # append based on samples number
        for bar, hatch in zip(bars, bar_htchs):  # assign htchs to each bar
            bar.set_hatch(hatch)
        ax[0].set(xlabel=None)
        if y_lab is None:
            y_lab = Project.param_to_axis_label[param]
        if yt_sum:
            legend_x_anchor += .14
            yt_lab = y_lab
        if xlab_rot != 0:
            ax[0].set_xticklabels(df_ave.index, rotation=xlab_rot, ha='right',
                rotation_mode='anchor')
        if legend_location is not None:
            hnd_ax, lab_ax = ax[0].get_legend_handles_labels()
            if df_std is not None:
                hnd_ax = hnd_ax[:len(hnd_ax)//2]
                lab_ax = lab_ax[:len(lab_ax)//2]
            if legend_labelspacing > 0.5:  # large legend spacing for molecules
                ax[0].plot(np.nan, np.nan, '-', color='None', label=' ')
                hhhh, aaaa = ax[0].get_legend_handles_labels()
                hnd_ax.append(hhhh[0])
                lab_ax.append(aaaa[0])
            if yt_sum:
                hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
            else:
                hnd_axt, lab_axt = [], []
            if legend_location == 'outside':  # legend goes outside of plot area
                ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt,
                    loc='upper left', ncol=legend_columns,
                    bbox_to_anchor=(legend_x_anchor, legend_y_anchor),
                    labelspacing=legend_labelspacing)
            else:  # legend is inside of plot area
                ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt,
                    loc=legend_location, ncol=legend_columns,
                    labelspacing=legend_labelspacing)
        # annotate ave+-std at the top of outliers bar (exceeding y_lim)
        if annotate_outliers and (y_lim is not None) and (df_std is not None):
            _annotate_outliers_in_plot(ax[0], df_ave, df_std, y_lim)
        if note_plt:
            ax[0].annotate(note_plt, ha='left', va='bottom',
                xycoords='axes fraction', xy=(0.005, .945+fig_hgt_mlt/100))
        figure_save(filename, out_path, fig, ax, axt, fig_par,
                y_lab=y_lab, yt_lab=yt_lab, y_lim=y_lim, yt_lim=yt_lim, legend=False,
                y_ticks=y_ticks, yt_ticks=yt_ticks, tight_layout=True,
                annotate_lttrs=annotate_lttrs, grid=Project.plot_grid)

if __name__ == '__main__':
    folder_path = plib.Path(r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\example\data")

    # class methods need to be called at the beginning to influence all instances
    Project.set_folder_path(folder_path)
    Project.set_plot_grid(False)
    Project.set_plot_font('Sans')  # ('Times New Roman')

    p = Project()
    # load files_info as provided by the user, if not given, create it
    # using the GC-MS .txt files in the folder
    files_info_0 = p.load_files_info()
    # load the provided calibrations as dict, store bool to know if are deriv
    calibrations, is_calibr_deriv = p.load_calibrations()
    c1, c2 = calibrations['CalDB'], calibrations['CalDBder']
    # load provided classificaiton codes and mass fractions for fun. groups
    class_code_frac = p.load_class_code_frac()
    # load all GCMS txt files as single files
    files0, is_files_deriv0 = p.load_all_files()
    f1, f2, f3 = files0['A_1'], files0['Ader_1'], files0['B_1']
    # create the list with all compounds in all samples
    list_of_all_compounds = p.create_list_of_all_compounds()
    # create the list with all derivatized compounds in all samples
    list_of_all_deriv_compounds = p.create_list_of_all_deriv_compounds()
    if 0: # set 0 to 1 if you want to recreate compounds properties databases
        compounds_properties = p.create_compounds_properties()
        deriv_compounds_properties = p.create_deriv_compounds_properties()
    else:  # otherwise load the available one (if unavailable it creates them)
        compounds_properties = p.load_compounds_properties()
        deriv_compounds_properties = p.load_deriv_compounds_properties()
    # apply the calibration to all files and store updated files as dict
    files, is_files_deriv = p.apply_calibration_to_files()
    f11, f22, f33 = files['A_1'], files['Ader_1'], files['B_1']
    # compute stats for each file in the files_info df
    files_info = p.add_stats_to_files_info()
    # create samples_info (ave and std) based on replicate data in files_info
    samples_info_0 = p.create_samples_info()
    # create samples and samples_std from files and store as dict
    samples, samples_std = p.create_samples_from_files()
    s1, s2, s3 = samples['A'], samples['Ader'], samples['B']
    sd1, sd2, sd3 = samples_std['A'], samples_std['Ader'], samples_std['B']
    # add stats to samples_info df
    samples_info = p.add_stats_to_samples_info()
    # create report (compounds based) for different parameters
    rep_conc, rep_conc_std = p.create_param_report(param='conc_vial_mg_L')
    rep_sample_fr, rep_sample_fr_std = p.create_param_report(param='fraction_of_sample_fr')
    # create aggreport (functionl group aggreageted based) for different parameters
    agg_conc, agg_conc_std = p.create_param_aggrrep(param='conc_vial_mg_L')
    agg_sample_fr, agg_sample_fr_std = p.create_param_aggrrep(param='fraction_of_sample_fr')
    # plot results bases on report
    #%%
    p.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0,
        legend_location='outside', only_samples_to_plot=['A', 'Ader'], #y_lim=[0, 5000]
             )
    # plot results bases on aggreport
    p.plot_ave_std(param='fraction_of_sample_fr', aggr=True, min_y_thresh=0.01,
        y_lim=[0, .5], color_palette='Set2')


