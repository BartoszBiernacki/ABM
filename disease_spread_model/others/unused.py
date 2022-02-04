class UnusedPlots(object):
    def __init__(self):
        pass

    @ classmethod
    def plot_fraction_of_susceptible(cls, fname):
        fixed_params = fixed_params_from_fname(fname=fname)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        main_title = r"Fraction of susceptible"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
    
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
    
        legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=None)
    
        ax.plot(df['Day'], df['Susceptible people'] / np.max(df['Susceptible people']), label=legend, color='blue')
        ax.plot(df['Day'], df['Susceptible cashiers'] / np.max(df['Replaced cashiers']), label=legend, color='red')
    
        ax.legend()
    
        plt.tight_layout()
        plt.show()
    
        def plot_1D_death_toll_dynamic_beta_sweep(cls,
                                                  directory: str,
                                                  normalized=False,
                                                  plot_real=True,
                                                  show=True,
                                                  save=False):
            """
            Plots averaged death toll from simulations.
            One figure for one mortality-visibility pair.
            One line for each beta.

            :param directory: directory to averaged simulated data
            :type directory: str
            :param normalized: normalize death toll to 1 in order to see how quickly it saturates?
            :type normalized: bool
            :param plot_real: also plot real data?
            :type plot_real: bool
            :param show: show plot?
            :type show: bool
            :param save: save plot?
            :type save: bool
            """
        
            fnames = all_fnames_from_dir(directory=directory)
            voivodeship = voivodeship_from_fname(fname=fnames[0])
        
            # get fixed simulation params from any (first) filename
            fixed_params = fixed_params_from_fname(fname=fnames[0])
        
            # group fnames such: grouped[i][j] = fname( (mortality, visibility)_i, beta_j )
            grouped_fnames = group_fnames_by_beta(directory=directory)
        
            # for each (mortality, visibility) pair create new figure
            for plot_id in range(grouped_fnames.shape[0]):
                fname = grouped_fnames[plot_id][0]
            
                mortality = variable_params_from_fname(fname=fname)['mortality']
                visibility = variable_params_from_fname(fname=fname)['visibility']
            
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
            
                # make title which includes voivodeship and normalization
                voivodeship_info = ''
                norm_info = ''
            
                if plot_real and voivodeship is not None:
                    voivodeship_info = f'for {voivodeship}'
                if normalized:
                    norm_info = ' (normalized)'
            
                main_title = (f"Death toll {voivodeship_info} {norm_info}\n"
                              f"mortality={float(mortality) * 100:.1f}%    "
                              f"visibility={float(visibility) * 100:.1f}%\n")
            
                title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
                ax.set_title(title)
                ax.set_xlabel('t, days')
            
                # prepare empty arrays for beta values and max death toll for each beta
                num_of_lines = grouped_fnames.shape[1]
                betas = np.empty(num_of_lines)
                deaths = np.empty_like(betas)
            
                cmap = plt.get_cmap('rainbow')
                colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
            
                # on created figure plot lines representing death toll for each beta
                for line_id in range(num_of_lines):
                
                    fname = grouped_fnames[plot_id][line_id]
                    df = pd.read_csv(fname)
                    variable_params = variable_params_from_fname(fname=fname)
                
                    betas[line_id] = variable_params['$\\beta$']
                
                    # make legend, but without ['mortality', 'visibility'] info because it already stands in fig title
                    ignored = ['mortality', 'visibility']
                    legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=ignored)
                
                    if normalized:
                        legend += ' ' * 4 + r'$\Sigma$'f' Deaths = {np.max(df["Dead people"]):.0f}'
                        ax.plot(df['Day'], df['Dead people'] / np.max(df['Dead people']), label=legend,
                                color=colors[line_id],
                                marker='o', markersize=3)
                        ax.set_ylabel(r'Death toll / max$_\beta$(Death toll)')
                    else:
                        ax.plot(df['Day'], df['Dead people'], label=legend, color=colors[line_id],
                                marker='o', markersize=3)
                        ax.set_ylabel(r'Death toll')
                
                    deaths[line_id] = np.max(df['Dead people'])
            
                # also plot real death toll if you want
                if plot_real and voivodeship is not None:
                    legend = 'Real data'
                    real_death_toll = RealData.get_real_death_toll()
                    y = np.array(real_death_toll.loc[voivodeship].to_list())
                
                    # find first day for which real death toll is grater than greatest simulated, to nicely truncate plot
                    last_real_day = np.argmax(y > max(deaths))
                    if last_real_day == 0:
                        last_real_day = len(y)
                
                    x = range(last_real_day)
                    y = y[:last_real_day]
                    ax.plot(x, y, label=legend, color='black', marker='o', markersize=3)
            
                # set legend entries in same order as lines were plotted (which was by ascending beta values)
                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles, labels)
            
                plt.tight_layout()
            
                cls.__show_and_save(fig=fig,
                                    dir_to_data=directory,
                                    plot_type='Death toll dynamic beta sweep',
                                    plot_name=" ".join(main_title.replace('\n', '').split()),
                                    save=save,
                                    show=show)
                
    @ classmethod
    def plot_2D_pandemic_time(cls, directory):
        fnames = all_fnames_from_dir(directory=directory)
        fixed_params = fixed_params_from_fname(fname=fnames[0])
    
        df = pd.read_csv(fnames[0])
        print(df.head().to_markdown())
    
        betas = np.empty(len(fnames))
        mortalities = np.empty_like(betas)
        days = np.empty_like(betas)
    
        for i, fname in enumerate(fnames):
            variable_params = variable_params_from_fname(fname=fname)
            df = pd.read_csv(fname)
        
            filt = df['Dead people'] / np.max(df['Dead people']) > 0.99
            days[i] = np.min(df[filt]['Day'])
        
            betas[i] = variable_params['$\\beta$']
            mortalities[i] = float(variable_params['mortality']) * 100
    
        unique_betas = list(set(betas))
        unique_mortalities = list(set(mortalities))
        unique_betas.sort()
        unique_mortalities.sort()
    
        days_matrix = np.empty((len(unique_mortalities), len(unique_betas)))
    
        for i in range(len(fnames)):
            beta = betas[i]
            mortality = mortalities[i]
        
            days_matrix[unique_mortalities.index(mortality)][unique_betas.index(beta)] = days[i]
    
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
    
        main_title = r"Pandemic duration in days as function of $\beta$ and mortality"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
    
        im = ax.pcolormesh(unique_betas, unique_mortalities, days_matrix, shading='nearest')
        ax.set_xticks(unique_betas[:])
        ax.set_yticks(unique_mortalities[:])
        ax.set_xlabel(r'$\beta$', fontsize=12)
        ax.set_ylabel('Mortality (in percent)', fontsize=12)
    
        fig.colorbar(im, cax=cax, orientation='vertical')
    
        for i in range(len(fnames)):
            ax.text(betas[i], mortalities[i], str(int(days[i])), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_1D_pandemic_duration_x_beta_series_visibilities(cls, directory, show=True, save=False):
        fnames = all_fnames_from_dir(directory=directory)
        fixed_params = fixed_params_from_fname(fname=fnames[0])
    
        grouped_fnames = group_fnames_standard_by_mortality_visibility_beta(directory=directory)
        num_mortalities, num_visibilities, num_betas = grouped_fnames.shape
    
        num_of_plots = num_mortalities
        num_of_lines = num_visibilities
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
        for plot_id in range(num_of_plots):
        
            mortality = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])['mortality']
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        
            main_title = "Pandemic duration"'\n' \
                         f"mortality={float(mortality) * 100:.1f}%"'\n'
            title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
            ax.set_title(title)
            ax.set_xlabel(r'Transmission parameter $\beta$')
            ax.set_ylabel('t, days')
        
            days = np.empty((num_visibilities, num_betas))
            visibilities = {}
            for visibility_id in range(num_visibilities):
                fname = grouped_fnames[plot_id][visibility_id][0]
                variable_params = variable_params_from_fname(fname=fname)
                visibilities[visibility_id] = variable_params['visibility']
            
                betas = {}
                for beta_id in range(num_betas):
                    fname = grouped_fnames[plot_id][visibility_id][beta_id]
                    variable_params = variable_params_from_fname(fname=fname)
                    betas[beta_id] = variable_params[r'$\beta$']
                
                    df = pd.read_csv(fname)
                    filt = df['Dead people'] / np.max(df['Dead people']) > 0.99
                    days[visibility_id][beta_id] = np.min(df[filt]['Day'])
            
                label = f'visibility = {float(visibilities[visibility_id]) * 100:.0f}%'
                ax.plot(betas.values(), days[visibility_id], label=label, color=colors[visibility_id],
                        marker='o', markersize=3)
        
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles),
                                          key=lambda t: float(t[0].replace('visibility = ', '').replace('%', ''))))
            ax.legend(handles, labels)
        
            plt.tight_layout()
        
            if save:
                plot_type = 'Pandemic duration, beta sweep, visibility series'
                save_dir = directory.replace('raw data/', 'plots/')
                save_dir += plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                name = ax.get_title()
                name = name.replace('\n', ' ')
                name = name.replace('    ', ' ')
                name = name.replace(r'$\beta$', 'beta')
                if name[-1:] == ' ':
                    name = name[:-1]
                plt.savefig(save_dir + name + '.pdf')
        
            if show:
                plt.show()
            plt.close(fig)

    @classmethod
    def plot_1D_tau_simplest_exp_fit(cls, directory, show=True, save=False):
        fnames = all_fnames_from_dir(directory=directory)
        fixed_params = fixed_params_from_fname(fname=fnames[0])
    
        for fname in fnames:
            variable_params = variable_params_from_fname(fname=fname)
            beta = variable_params[r'$\beta$']
            visibility = float(variable_params['visibility'])
            mortality = float(variable_params['mortality'])
            # Plot experimental data
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        
            main_title = "Exponential fit"'\n' \
                         r'$\beta$='f'{beta}    mortality={mortality * 100:.1f}%    visibility={visibility * 100:.0f}%''\n'
            title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
            ax.set_title(title)
            ax.set_xlabel('t, days', fontsize=12)
            ax.set_ylabel('Number of exposed', fontsize=12)
        
            df = pd.read_csv(fname)
            days = df['Day']
            exposed = df['Incubation people']
        
            A, tau = calc_tau(x_data=days, y_data=exposed)
        
            # Find peaks
            peak_pos_indexes = find_peaks(exposed)[0]
            peak_pos = days[peak_pos_indexes]
            peak_height = exposed[peak_pos_indexes]
        
            legend1 = 'experimental data'
            ax.plot(days, exposed, label=legend1, color="Green", linestyle='dashed', marker='o', markersize=5, zorder=0)
        
            legend2 = 'maxima'
            ax.scatter(peak_pos, peak_height, label=legend2, zorder=2)
        
            # Plot fitted function
            legend3 = r' $y =  Ae^{{-t/\tau}}$' '\n' r'$A={:.1f}$' '\n' r'$\tau = {:.1f}$'.format(A, tau)
            ax.plot(days, unknown_exp_func(days, A, tau), 'r', label=legend3, zorder=1)
        
            # Show legend and its entries in correct order
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[0], handles[2], handles[1]]
            labels = [labels[0], labels[2], labels[1]]
            ax.legend(handles, labels)
        
            plt.tight_layout()
        
            if save:
                plot_type = 'Simplest exp fits'
                save_dir = directory.replace('raw data/', 'plots/')
                save_dir += plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                name = ax.get_title()
                name = name.replace('\n', ' ')
                name = name.replace('    ', ' ')
                name = name.replace(r'$\beta$', 'beta')
                if name[-1:] == ' ':
                    name = name[:-1]
                plt.savefig(save_dir + name + '.pdf')
        
            if show:
                plt.show()
            plt.close(fig)

    @classmethod
    def plot_1D_tau_x_beta_series_visibilities(cls, directory, show=True, save=False):
        fnames = all_fnames_from_dir(directory=directory)
        fixed_params = fixed_params_from_fname(fname=fnames[0])
    
        grouped_fnames = group_fnames_standard_by_mortality_visibility_beta(directory=directory)
        num_mortalities, num_visibilities, num_betas = grouped_fnames.shape
    
        num_of_plots = num_mortalities
        num_of_lines = num_visibilities
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
        for plot_id in range(num_of_plots):
        
            mortality = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])['mortality']
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        
            main_title = 'Characteristic time 'r"$\tau$"'\n' \
                         f"mortality={float(mortality) * 100:.1f}%"'\n'
            title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
            ax.set_title(title)
            ax.set_xlabel(r'Transmission parameter $\beta$')
            ax.set_ylabel(r'$\tau$, days')
        
            taus = np.empty((num_visibilities, num_betas))
            visibilities = {}
            for visibility_id in range(num_visibilities):
                fname = grouped_fnames[plot_id][visibility_id][0]
                variable_params = variable_params_from_fname(fname=fname)
                visibilities[visibility_id] = variable_params['visibility']
            
                betas = {}
                for beta_id in range(num_betas):
                    fname = grouped_fnames[plot_id][visibility_id][beta_id]
                    variable_params = variable_params_from_fname(fname=fname)
                    betas[beta_id] = variable_params[r'$\beta$']
                
                    df = pd.read_csv(fname)
                    taus[visibility_id][beta_id] = calc_tau(y_data=df['Incubation people'], x_data=df['Day'])[1]
            
                label = f'visibility = {float(visibilities[visibility_id]) * 100:.0f}%'
                ax.plot(betas.values(), taus[visibility_id], label=label, color=colors[visibility_id],
                        marker='o', markersize=3)
        
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles),
                                          key=lambda t: float(t[0].replace('visibility = ', '').replace('%', ''))))
            ax.legend(handles, labels)
        
            plt.tight_layout()
        
            if save:
                plot_type = 'Tau, beta sweep, visibility series'
                save_dir = directory.replace('raw data/', 'plots/')
                save_dir += plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                name = ax.get_title()
                name = name.replace('\n', ' ')
                name = name.replace('    ', ' ')
                name = name.replace(r'$\beta$', 'beta')
                name = name.replace(r'$\tau$', 'tau')
                if name[-1:] == ' ':
                    name = name[:-1]
                plt.savefig(save_dir + name + '.pdf')
        
            if show:
                plt.show()
            plt.close(fig)
    
    @classmethod
    def plot_max_death_toll_prediction_x_visibility_series_betas(cls, directory, show=True, save=True):
        """
        Plot death toll prediction based on number of infected, mortality and visibility.
        """
        fnames = all_fnames_from_dir(directory=directory)
        fixed_params = fixed_params_from_fname(fname=fnames[0])
    
        grouped_fnames = group_fnames_standard_by_mortality_beta_visibility(directory=directory)
        num_mortalities, num_betas, num_visibilities = grouped_fnames.shape
    
        num_of_lines = num_betas
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
        for plot_id in range(num_mortalities):
        
            mortality = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])['mortality']
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            norm_info = ''
        
            main_title = "Death toll prediction by infected toll and mortality" + norm_info + '\n' \
                                                                                              f"mortality={float(mortality) * 100:.1f}%"'\n'
            title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
            ax.set_title(title)
            ax.set_xlabel('visibility')
            ax.set_ylabel(r'(Infected toll / all people) $\cdot$ visibility $\cdot$ mortality (in percent)')
        
            infected = np.empty((num_betas, num_visibilities))
            betas = {}
            for beta_id in range(num_betas):
                fname = grouped_fnames[plot_id][beta_id][0]
                variable_params = variable_params_from_fname(fname=fname)
                betas[beta_id] = variable_params[r'$\beta$']
            
                visibilities = {}
                for visibility_id in range(num_visibilities):
                    fname = grouped_fnames[plot_id][beta_id][visibility_id]
                    variable_params = variable_params_from_fname(fname=fname)
                    visibilities[visibility_id] = variable_params['visibility']
                
                    df = pd.read_csv(fname)
                    infected[beta_id][visibility_id] = np.max(df['Dead people']) + np.max(df['Recovery people'])
                    infected[beta_id][visibility_id] /= df['Susceptible people'][0]
                    infected[beta_id][visibility_id] *= 100
                    infected[beta_id][visibility_id] *= \
                        float(variable_params['visibility']) * float(variable_params['mortality'])
            
                label = r'$\beta$='f"{betas[beta_id]}"
                ax.plot(visibilities.values(), infected[beta_id], label=label, color=colors[beta_id],
                        marker='o', markersize=3)
        
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels)
        
            plt.tight_layout()
        
            if save:
                plot_type = 'Death toll prediction by infected toll and mortality'
                save_dir = directory.replace('raw data/', 'plots/')
                save_dir += plot_type + '/'
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                name = ax.get_title()
                name = name.replace('\n', ' ')
                name = name.replace('    ', ' ')
                name = name.replace(r'$\beta$', 'beta')
                if name[-1:] == ' ':
                    name = name[:-1]
                plt.savefig(save_dir + name + '.pdf')
        
            if show:
                plt.show()
            plt.close(fig)

    @classmethod
    def reproduce_plot_by_website(cls, filename="data_extracted_from_fig1.csv"):
        df = pd.read_csv(filename, header=None)
        x_img = np.array(df.iloc[:, [0]]).flatten()
        y_img = np.array(df.iloc[:, [1]]).flatten()
        print(x_img.size)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        ax.plot(x_img, y_img)
    
        x = [0, 4, 5, 9, 12, 13, 16, 17, 20, 21, 24, 25]
        y = [0, 56, 56, 13.5, 38.5, 38.5, 20.5, 19.2, 28.5, 29.4, 19.9, 18.3]
        ax.plot(x, y, color='red', linestyle='-')
        ax.scatter(x, y, color='red')
    
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, 51, 5)
        minor_ticks = np.arange(0, 51, 1)
    
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
    
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=1)
        ax.set(xlim=(-1, 26))
    
        # plt.show()
    
        x_int_indexes = np.array(np.linspace(0, len(x_img) - 1, int(max(x_img))), dtype=int)
    
        x_int = np.array(x_img[x_int_indexes], dtype=int)
        y_int = y_img[x_int_indexes]
    
        df = pd.DataFrame(data=np.array([x_int, y_int]))
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(df)
    
        ax.plot(x_int, y_int, color='green', linestyle='-')
        plt.show()
        print(np.argmax(y_img))
        print(x_img[np.argmax(y_img)])
    
        print(x_int_indexes)