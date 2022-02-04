from tests import *

test_drawing_numbers_from_my_hist_distribution(means=[3, 5], bins_array=[1, 3, 5, 7], show=False, save=True)


@classmethod
def plot_1D_death_toll_max_x_mortality_series_betas(cls,
                                                    directory,
                                                    normalized,
                                                    show=True,
                                                    save=False):
    """
    PLots max death toll, one figure for one visibility.
        Death toll on y axis.
        Mortality on x axis.
        One line for one beta.

    :param directory: directory to averaged simulated data
    :type directory: str
    :param normalized: normalize death toll to 1 in order to see how quickly it saturates?
    :type normalized: bool
    :param show: show plot?
    :type show: bool
    :param save: save plot?
    :type save: bool
    """
    
    # hmm.. I'm sure what that is doing
    warnings.filterwarnings("error")
    fnames = all_fnames_from_dir(directory=directory)
    
    # get fixed simulation params from any (first) filename
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    # group fnames such: grouped[i][j][k] = fname(visibility_i, beta_j, mortality_k)
    grouped_fnames = group_fnames_standard_by_visibility_beta_mortality(directory=directory)
    
    # get range for ijk (explained above)
    num_visibilities, num_betas, num_mortalities = grouped_fnames.shape
    
    num_of_lines = num_betas
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
    # for each visibility create new figure
    for plot_id in range(num_visibilities):
        
        visibility = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])['visibility']
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # make title which includes normalization
        norm_info = ''
        if normalized:
            norm_info = '(normalized)'
        
        main_title = (f"Death toll max {norm_info} for "
                      f"visibility={float(visibility) * 100:.0f}%  "
                      f"(after {get_last_day(fname=grouped_fnames[plot_id][0][0]) + 1} days)\n")
        
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        
        # set x and y label which includes normalization
        ax.set_xlabel('mortality')
        if normalized:
            ax.set_ylabel(r'Death toll / max$_{\beta}$(Death toll)')
        else:
            ax.set_ylabel(r'Death toll')
        
        # prepare empty array for max death toll for each (beta, mortality) pair
        deaths = np.empty((num_betas, num_mortalities))
        # prepare dict for bata values ()
        betas = {}
        
        # run over betas
        for beta_id in range(num_betas):
            fname = grouped_fnames[plot_id][beta_id][0]
            variable_params = variable_params_from_fname(fname=fname)
            betas[beta_id] = variable_params[r'$\beta$']
            
            mortalities = {}
            # run over mortalities
            for mortality_id in range(num_mortalities):
                fname = grouped_fnames[plot_id][beta_id][mortality_id]
                variable_params = variable_params_from_fname(fname=fname)
                mortalities[mortality_id] = variable_params['mortality']
                
                df = pd.read_csv(fname)
                
                # fill deaths array by death toll for current beta and mortality
                deaths[beta_id][mortality_id] = np.max(df['Dead people'])
            
            # make label for one line (fixed beta)
            label = r'$\beta$='f"{betas[beta_id]}"
            
            # plot data (normalized or not)
            if normalized:
                # improve label if normalized
                label += '    'r"max$_{\beta}$(deaths)="f'{np.max(deaths[beta_id]):.0f}'
                
                # plot normalized data, but if for some reason that is impossible make normal plot
                try:
                    ax.plot(mortalities.values(), deaths[beta_id] / np.max(deaths[beta_id]), label=label,
                            color=colors[beta_id], marker='o', markersize=5)
                except RuntimeWarning:
                    print("Can't plot 1D_death_toll_max_x_mortality_series_betas probably np.max(deaths) equals 0.")
                    print("fname = ", fname)
                    ax.plot(mortalities.values(), np.zeros(len(mortalities)), label=label,
                            color=colors[beta_id], marker='o', markersize=5)
            else:
                ax.plot(mortalities.values(), deaths[beta_id], label=label, color=colors[beta_id],
                        marker='o', markersize=5)
        
        # set legend entries in same order as lines were plotted (which was ascending by beta values)
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        # set lower y limit to 0
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        
        main_title = main_title.replace('\n', '').replace(r'$\beta$', 'beta')
        cls.__show_and_save(fig=fig,
                            dir_to_data=directory,
                            plot_type='Death toll max, x mortality, beta series',
                            plot_name=" ".join(main_title.split()),
                            save=save,
                            show=show)
        