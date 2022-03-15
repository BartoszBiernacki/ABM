class FindLastDayAnim(object):
    """
    Plot contains:
        - death toll
        - death toll completed
        - derivative completed
        - derivative smoothed up
        - derivative maxima and minima
        - vertical line showing last day of pandemic
        - moving vertical line to help see derivative - death toll relation

    Based on example from:
    https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
    """
    
    def __init__(self, start_days: dict, voivodeship: str, fps=50):
        
        self.start_days = start_days
        self.voivodeship = voivodeship
        
        self.days = Config.days_to_look_for_pandemic_end
        self.derivative_half_win_size = Config.death_toll_derivative_half_win_size
        self.derivative_smooth_out_win_size = Config.death_toll_derivative_smooth_out_win_size
        self.derivative_smooth_out_polyorder = Config.death_toll_derivative_smooth_out_savgol_polyorder
        
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax2 = self.ax.twinx()
        
        # Then setup FuncAnimation.
        self.ani = FuncAnimation(self.fig, self.update, interval=1000 / fps,
                                 init_func=self.setup_plot, frames=self.days,
                                 blit=True, repeat=True)
        
        self.axvline = None
    
    def setup_plot(self):
        """Initial drawing of plots."""
        
        # For some odd reason setup_plot is called twice which duplicates lines
        # and legend entries, so I just clear axes before doing anything.
        self.ax2.clear()
        self.ax.clear()
        
        self.ax.set_xlabel('t, days')
        self.ax.set_ylabel('Death toll')
        self.ax2.set_ylabel('Normalized derivative of death toll')
        
        self.ax.set_title(self.voivodeship)
        
        # get  last day of pandemic
        last_days = RealData.get_ending_days_for_voivodeships_based_on_death_toll_derivative(
            starting_days=self.start_days,
            days_to_search=self.days,
            derivative_half_win_size=self.derivative_half_win_size,
            derivative_smooth_out_win_size=self.derivative_smooth_out_win_size,
            derivative_smooth_out_polyorder=self.derivative_smooth_out_polyorder
        )
        start_day = self.start_days[self.voivodeship]
        last_day = last_days[self.voivodeship]
        
        print(f'start={start_day}, end={last_day}')
        
        # plot death toll segment
        death_toll = np.array((RealData.get_real_death_toll().loc[self.voivodeship])).astype(float)
        death_toll_segment = np.copy(death_toll[start_day: start_day + self.days])
        self.ax.plot(death_toll_segment, color='C0', zorder=1, label='death toll',
                     lw=4, alpha=0.7)
        
        # plot missing death toll segments approximated
        death_toll_completed = complete_missing_data(values=death_toll)
        missing_indices = get_indices_of_missing_data(data=death_toll)
        for i, indices in enumerate(missing_indices):
            
            idx_min = indices[0] - 1
            idx_max = indices[-1] + 2
            
            idx_min = max(0, idx_min - start_day)
            idx_max = min(self.days, idx_max - start_day)
            
            if idx_min < self.days:
                # plot with legend only first segment
                if i == 0:
                    self.ax.plot(range(idx_min, idx_max),
                                 death_toll_completed[start_day + idx_min: start_day + idx_max],
                                 color='black', lw=4, alpha=0.7, label='death toll interpolated')
                else:
                    self.ax.plot(range(idx_min, idx_max),
                                 death_toll_completed[start_day + idx_min: start_day + idx_max],
                                 color='black', lw=4, alpha=0.7)
        
        # # plot window derivative completed
        derivative_completed = window_derivative(
            y=death_toll_completed,
            half_win_size=self.derivative_half_win_size)
        derivative_completed_segment = np.copy(derivative_completed[start_day: start_day + self.days])
        derivative_completed_segment /= max(derivative_completed_segment)
        self.ax2.plot(derivative_completed_segment, color='C1', label='window derivative')
        
        # plot smooth out derivative completed
        yhat = savgol_filter(
            derivative_completed_segment,
            window_length=self.derivative_smooth_out_win_size,
            polyorder=self.derivative_smooth_out_polyorder
        )
        yhat /= max(yhat)
        self.ax2.plot(yhat, color='C2', label=f'derivative smoothed up', lw=4)
        
        # Plot peaks (minima and maxima) of smoothed up derivative.
        vec = np.copy(yhat)
        
        # Maxima
        x_peaks_max = argrelmax(data=vec, order=7)[0]
        self.ax2.scatter(x_peaks_max, vec[x_peaks_max], label='derivative maxima',
                         color='lime', s=140, zorder=100)
        # Minima
        x_peaks_min = argrelmin(data=vec, order=7)[0]
        self.ax2.scatter(x_peaks_min, vec[x_peaks_min], label='derivative minima',
                         color='darkgreen', s=140, zorder=100)
        
        # plot last day of initial pandemic phase
        self.axvline = self.ax2.axvline(last_day - start_day, color='red', lw=3, label='last day')
        
        # add legend (change y2 limits to avoid plots overlapping with legend)
        self.ax2.set_ylim(-0.05, 1.2)
        self.fig.legend(
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1),
            bbox_transform=self.ax.transAxes,
            fancybox=True,
            shadow=True)
        
        # plot moving axvline to easily show death toll - derivative relation
        self.axvline = self.ax.axvline(x=0, animated=True, color='black', lw=2)
        
        plt.tight_layout()
        
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.axvline,
    
    def update(self, i):
        """Update the vertical line."""
        self.axvline.set_xdata(i % self.days)
        
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.axvline,
    
    @staticmethod
    def make_animations(voivodeships: list,
                        fps=50,
                        start_days_by='deaths',
                        show=True,
                        save=False):
        if 'all' in voivodeships:
            voivodeships = RealData.get_voivodeships()
        
        if start_days_by == 'deaths':
            start_days = RealData.get_starting_days_for_voivodeships_based_on_district_deaths(
                percent_of_touched_counties=Config.percent_of_death_counties,
                ignore_healthy_counties=True)
        elif start_days_by == 'infections':
            start_days = RealData.get_starting_days_for_voivodeships_based_on_district_infections(
                percent_of_touched_counties=Config.percent_of_infected_counties)
        else:
            raise ValueError(f'start_days_by has to be "deaths" or "infections", but {start_days_by} was given')
        
        for voivodeship in voivodeships:
            a = FindLastDayAnim(
                start_days=start_days,
                voivodeship=voivodeship,
                fps=fps)
            
            if show:
                plt.show()
            
            if save:
                save_dir = f"{Config.ABM_dir}/RESULTS/plots/Finding last day of pandemic/"
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                fname = (f"{save_dir}{voivodeship} start_by{start_days_by} in "
                         f"{Config.percent_of_death_counties if start_days_by == 'deaths' else Config.percent_of_infected_counties}"
                         f" percent of counties.gif")
                a.ani.save(fname,
                           writer='pillow',
                           fps=fps)
                print(f'{voivodeship} {fps} saved')


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
    
        @classmethod
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
        
    @classmethod
    def get_days_and_deaths_by_beta_and_fixed_mortality_from_dir(cls, directory, const_mortality):
        fnames = all_fnames_from_dir(directory=directory)
    
        result = {}
        for i, fname in enumerate(fnames):
            df = pd.read_csv(fname)
            variable_params = variable_params_from_fname(fname=fname)
        
            mortality = float(variable_params['mortality'])
            if mortality == const_mortality:
                beta = variable_params['$\\beta$']
                result[beta] = df[['Day', 'Dead people']].copy()
    
        if not result:
            raise ValueError(f'Not found any data for const mortality = {const_mortality} in {directory}')
    
        return result

    @classmethod
    def get_days_and_deaths_by_mortality_and_fixed_beta_from_dir(cls, directory, const_beta):
        fnames = all_fnames_from_dir(directory=directory)
    
        result = {}
        for i, fname in enumerate(fnames):
            df = pd.read_csv(fname)
            variable_params = variable_params_from_fname(fname=fname)
        
            beta = float(variable_params['$\\beta$'])
            if beta == const_beta:
                mortality = variable_params['mortality']
                result[mortality] = df[['Day', 'Dead people']].copy()
    
        if not result:
            raise ValueError(f'Not found any data for const mortality = {const_beta} in {directory}')
    
        return result

    @classmethod
    def show_real_death_toll_for_voivodeship_shifted_by_hand(cls,
                                                             directory_to_data,
                                                             voivodeship,
                                                             starting_day=10,
                                                             last_day=100,
                                                             shift_simulated=True,
                                                             save=False,
                                                             show=True):
        """
        Plots real death toll for one voivodeship, but shifted in a way, that in day = starting_day
        it looks like pandemic started for good.

        Simulated data are also plotted to visually verify if parameters of simulation are correct.
        Simulated data are shifted, but in a way that in day=starting_day simulated death toll
        equal real shifted death toll.

        Intended to use with directory_to_data != None; to see similarity in real and
        simulated data.

        :param directory_to_data: folder which contain simulation result which will be compared to real data.
        :type directory_to_data: None or str
        :param voivodeship: name of voivodeship from which real data comes from.
        :type voivodeship: str
        :param starting_day: first day afer shift for which death toll is above threshold which was estimated by hand
        :type starting_day: int
        :param last_day: last day for which data will be plotted
        :type last_day: int
        :param shift_simulated: shift simulated death toll in the same way as real (same death toll threshold)?
        :type shift_simulated: bool
        :param save: save plot?
        :type save: bool
        :param show: show plot?
        :type show: bool
        """
    
        # get starting death toll for given voivodeship since from (by hand) death toll looks nicely
        starting_death = RealData.get_starting_deaths_by_hand()[voivodeship]
    
        # get real death toll for all voivodeships, shifted such in starting_day each voivodeship has starting deaths
        shifted_real_death_toll = \
            RealData.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(starting_day=starting_day,
                                                                                  minimum_deaths=starting_death)
        # get day number in which death toll = starting_death
        true_start_day = RealData.get_day_of_first_n_death(n=starting_death)
    
        # Plot data for given voivodeship *****************************************************************************
        # set figure, title and axis labels
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_title(f"Death toll for {voivodeship}, shifted in such a way, that in day {starting_day} "
                     f"death toll is not less than {starting_death}.\n")
        if directory_to_data is not None and shift_simulated:
            ax.set_title(ax.get_title() + 'Simulated death toll is shifted as well.')
        ax.set_xlabel(f't, days since first {starting_death} people died in given voivodeship')
        ax.set_ylabel(f'Death toll (since first {starting_death} people died in given voivodeship)')
    
        # set data
        x = shifted_real_death_toll.columns  # x = days of pandemic = [0, 1, ...]
        y = shifted_real_death_toll.loc[voivodeship]
    
        # make nice looking label
        left = voivodeship
        right = f"day {starting_day} = {true_start_day[voivodeship]}"
        label = '{:<20} {:>22}'.format(left, right)
    
        # plot shifted real data
        ax.plot(x[:last_day], y[:last_day], label=label, color='Black', linewidth=3)
        # ************************************************************************************************************
    
        # Plot data from simulations **********************************************************************************
        if directory_to_data:
            fnames = all_fnames_from_dir(directory=directory_to_data)
        
            num_of_lines = len(fnames)
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        
            for i, fname in enumerate(fnames):
                beta = float(variable_params_from_fname(fname=fname)[r'$\beta$'])
                mortality = float(variable_params_from_fname(fname=fname)['mortality'])
                visibility = float(variable_params_from_fname(fname=fname)['visibility'])
            
                df = pd.read_csv(fname)
            
                # decide if also shift simulated data
                common_day = 0
                if shift_simulated:
                    y = np.array(df['Dead people'])
                    common_day = np.where(y >= starting_death)[0][0]
                    y = y[common_day - starting_day:]
                    x = list(range(len(y)))
                else:
                    x = df['Day']
                    y = df['Dead people']
            
                # prepare label ingredients
                beta_info = r'$\beta$=' + f'{beta}'
                mortality_info = f'mortality={mortality * 100:.1f}%'
                visibility_info = f'visibility={visibility * 100:.0f}%'
                day_info = f"day {starting_day}=0"
                if shift_simulated:
                    day_info = f"day {starting_day}={common_day}"
            
                # create label for simulated data
                label = '{:<10} {:>15} {:>15} {:>10}'.format(beta_info,
                                                             mortality_info,
                                                             visibility_info,
                                                             day_info)
            
                # plot simulated data
                ax.plot(x[:last_day], y[:last_day], label=label, color=colors[i],
                        linewidth=1, linestyle='dashed')
            # ************************************************************************************************************
    
        # made legend and its entries nicely aligned
        ax.legend(prop={'family': 'monospace'}, loc='upper left')
        plt.tight_layout()
    
        cls.__show_and_save(fig=fig, dir_to_data=directory_to_data,
                            plot_type='Real shifted death toll up to threshold',
                            plot_name=f'starting_deaths={starting_death},   '
                                      f'starting_day={starting_day},   '
                                      f'last_day={last_day}',
                            save=save,
                            show=show)

    @classmethod
    def get_ending_days_by_derivative_of_smooth_death_toll(
            cls,
            starting_days: dict,
            days_to_search=200,
            death_toll_smooth_out_win_size=21,
            death_toll_smooth_out_polyorder=3,
            derivative_half_win_size=3,
    ):
        """
        Returns last day of first phase of pandemic.

        Last day is found as day between max delta peak_up - peak_down of
        death toll derivative.

        Algorithm:
            - get death toll and fill missing data in it
            - calculate window derivative of death toll
            - smooth up derivative
            - find peaks in derivative (up and down)
            - find two neighbouring peaks with max delta_y value
            - last day = avg(peak1_x peak2_x)

        :param starting_days: days considered as the beginning of pandemic
            for all voivodeships {voivodeship: start_day}
        :type starting_days: dict
        :param days_to_search: number of days since start_day where
            the last day will be searched for
        :type days_to_search: int
        :param derivative_half_win_size: half window size of death toll
            derivative e.g. val=3 --> 7 days.
            E.g. (2) data = [0, 10, 20, 30, 40, 50], val=1 -->
                derivative[2] = (30-10)/(idx(30) - idx(10)) = 20/(3-1) = 20/2 = 10
        :type derivative_half_win_size: int
        :param death_toll_smooth_out_win_size: window size used for smooth up
            death_toll in savgol_filter.
        :type death_toll_smooth_out_win_size: int
        :param death_toll_smooth_out_polyorder: polyorder used for smooth up
            death_toll in savgol_filter.
        :type death_toll_smooth_out_polyorder: int
        :return: dict {voivodeship: last_day_of_pandemic} for all voivodeships.
        :rtype: dict

        """
    
        # create empty result dict
        result = {}
    
        # get real death toll for all voivodeships (since 03.04.2019)
        death_tolls = RealData.get_real_death_toll()
    
        # fill gaps in real death toll
        for voivodeship in cls.get_voivodeships():
            death_tolls.loc[voivodeship] = complete_missing_data(values=death_tolls.loc[voivodeship])
    
        # smooth out death toll
        death_tolls_smooth = death_tolls.copy()
        for voivodeship in cls.get_voivodeships():
            death_tolls_smooth.loc[voivodeship] = savgol_filter(
                x=death_tolls.loc[voivodeship],
                window_length=death_toll_smooth_out_win_size,
                polyorder=death_toll_smooth_out_polyorder)
    
        # algorithm to find last day of pandemic for each voivodeship
        for voivodeship in cls.get_voivodeships():
            # get start day based on percent counties dead
            start_day = starting_days[voivodeship]
        
            # get derivative
            derivative = window_derivative(
                y=death_tolls_smooth.loc[voivodeship],
                half_win_size=derivative_half_win_size)
        
            # get just segment of this derivative which probably includes last day of
            # first phase of pandemic, it will be plotted
            derivative_segment = derivative[start_day: start_day + days_to_search]
        
            # normalize derivative to 1
            derivative_segment /= max(derivative_segment)
        
            # Find x pos of maxima
            x_peaks_max = argrelmax(data=derivative_segment, order=7)[0]
            # Find x pos of minima
            x_peaks_min = argrelmin(data=derivative_segment, order=7)[0]
        
            # make one sorted array of x coordinate of all found extremes
            x_extremes = np.sort(np.array([*x_peaks_max, *x_peaks_min]))
            # get indices of two neighbouring peaks with max delta val (from list of all peaks)
            idx1, idx2 = cls._get_neighbouring_indices_with_max_delta_value(data=derivative_segment[x_extremes])
            # get number of days where two neighbouring peaks, with max delta value, occurred
            # (since starting day)
            idx1, idx2 = x_extremes[[idx1, idx2]]
        
            # get day number where initial phase of death toll ends (since start day)
            idx_death_toll_change = int(np.average([idx1, idx2]))
        
            # add last day of first phase of pandemic (since 04.03.2019)
            # to resulting dictionary
            result[voivodeship] = idx_death_toll_change + start_day
    
        return result

    @classmethod
    def plot_last_day_finding_process(cls,
                                      voivodeships: list[str],
                                      start_days_by='deaths',
                                      percent_of_touched_counties=20,
                                      last_date='2020-07-01',
                                      death_toll_smooth_out_win_size=21,
                                      death_toll_smooth_out_polyorder=3,
                                      derivative_half_win_size=3,
                                      plot_redundant=False,
                                      show=True,
                                      save=False,
                                      ):
        """
        Plots crucial steps in finding last day of pandemic in voivodeships.

        Plots:
         - death tool
         - death tool smoothed
         - derivative of death toll
         - derivative of smoothed up death toll
         - smoothed up derivative of smoothed up death toll
         - slope of death toll
         - slope of smoothed up death toll
        """
    
        # get voivodeships
        if 'all' in voivodeships:
            voivodeships = RealData.get_voivodeships()
    
        # get start days dict
        start_days = RealData.starting_days(
            by=start_days_by,
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=False)
    
        # get real death toll for all voivodeships (since 03.04.2019)
        death_tolls = RealData.get_real_death_toll()
    
        # fill gaps in real death toll
        for voivodeship in RealData.get_voivodeships():
            death_tolls.loc[voivodeship] = complete_missing_data(values=death_tolls.loc[voivodeship])
    
        # smooth out death toll
        death_tolls_smooth = death_tolls.copy()
        for voivodeship in RealData.get_voivodeships():
            death_tolls_smooth.loc[voivodeship] = savgol_filter(
                x=death_tolls.loc[voivodeship],
                window_length=death_toll_smooth_out_win_size,
                polyorder=death_toll_smooth_out_polyorder)
    
        # get last day in which death pandemic last day will be looked for
        last_day_to_search = list(death_tolls.columns).index(last_date)
    
        # Make plots
        for voivodeship in voivodeships:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax2 = ax.twinx()
        
            # set title and axis labels
            date0 = (datetime.datetime(2020, 3, 4) +
                     datetime.timedelta(days=int(start_days[voivodeship])))
            ax.set_title(f'Woj. {voivodeship}, szukanie końca pierwszego etapu pandemii.\n'
                         f'Dzień 0 = {date0.strftime("%Y-%m-%d")}, na podstawie '
                         f'stwierdzonego '
                         f'{"zgonu" if start_days_by == "deaths" else "zachorowania"} w '
                         f'{percent_of_touched_counties}% powiatów.\n'
                         f'Ostatni dzień - {last_date}.')
            ax.set_xlabel('t, dni')
            ax.set_ylabel('Suma zgonów')
            ax2.set_ylabel('Przeskalowana pochodna lub nachylenie sumy zgonów')
        
            day0 = start_days[voivodeship]
            x_days = np.arange(-10, last_day_to_search - day0)
        
            # death toll
            death_toll = death_tolls.loc[voivodeship]
            ax.plot(x_days, death_toll[day0 - 10: last_day_to_search],
                    color='C0', label='suma zgonów', lw=4, alpha=0.7)
        
            # death toll smoothed up
            death_toll_smooth = death_tolls_smooth.loc[voivodeship]
            ax.plot(x_days, death_toll_smooth[day0 - 10: last_day_to_search],
                    color='C1', label='suma zgonów po wygładzeniu')
        
            if plot_redundant:
                # derivative
                derivative = window_derivative(
                    y=death_toll,
                    half_win_size=derivative_half_win_size)
                ax2.plot(derivative[day0 - 10: last_day_to_search],
                         color='C2', label='pochodna sumy zgonów', alpha=0.5)
            
                # smoothed up derivative
                derivative_smoothed_up = savgol_filter(
                    x=derivative,
                    window_length=death_toll_smooth_out_win_size,
                    polyorder=death_toll_smooth_out_polyorder)
                ax2.plot(derivative_smoothed_up[day0 - 10: last_day_to_search],
                         color='black', label='wygładzona pochodna sumy zgonów',
                         alpha=1, lw=10)
            
                # derivative of smooth death toll
                derivative_smooth = window_derivative(
                    y=death_toll_smooth,
                    half_win_size=derivative_half_win_size)
                ax2.plot(derivative_smooth[day0 - 10: last_day_to_search],
                         color='C3', lw=2,
                         label='pochodna wygładzonej sumy zgonów')
            
                # smoothed up derivative of smooth death toll
                derivative_smooth_smoothed_up = savgol_filter(
                    x=derivative_smooth,
                    window_length=death_toll_smooth_out_win_size,
                    polyorder=death_toll_smooth_out_polyorder)
                ax2.plot(derivative_smooth_smoothed_up[day0 - 10: last_day_to_search],
                         color='yellow', lw=4,
                         label='wygładzona pochodna wygładzonej sumy zgonów')
            
                # slope
                slope = slope_from_linear_fit(data=death_toll, half_win_size=3)
                ax2.plot(slope[day0 - 10: last_day_to_search],
                         color='C5', alpha=0.5,
                         label='nachylenie prostej dopasowanej do'
                               ' fragmentów sumy zgonów')
        
            # slope of smooth death toll
            slope_smooth = slope_from_linear_fit(
                data=death_toll_smooth, half_win_size=3)
        
            # normalize slope to 1
            if max(slope_smooth[day0 - 10: last_day_to_search]) > 0:
                slope_smooth /= max(slope_smooth[day0 - 10: last_day_to_search])
        
            # plot normalized slope
            ax2.plot(slope_smooth[day0 - 10: last_day_to_search],
                     color='green', lw=4,
                     label='nachylenie prostej dopasowanej do'
                           ' fragmentów wygładzonej sumy zgonów')
        
            # Plot peaks (minima and maxima) of slope of smoothed up death toll.
            vec = np.copy(slope_smooth[day0 - 10: last_day_to_search])
        
            # Maxima of slope
            x_peaks_max = argrelmax(data=vec, order=8)[0]
            ax2.scatter(x_peaks_max, vec[x_peaks_max],
                        color='lime', s=140, zorder=100,
                        label='maksima nachylenia sumy zgonów')
            # Minima of slope
            x_peaks_min = argrelmin(data=vec, order=8)[0]
            ax2.scatter(x_peaks_min, vec[x_peaks_min],
                        color='darkgreen', s=140, zorder=100,
                        label='minima nachylenia sumy zgonów')
        
            # last of of pandemic as day of peak closest to 2 moths with derivative > 0.5
            # get list of candidates for last day
            last_day_candidates = [x for x in x_peaks_max if vec[x] > 0.5]
            # if there are no candidates add day with largest peak
            if not last_day_candidates:
                try:
                    last_day_candidates.append(max(x_peaks_max, key=lambda x: vec[x]))
                except ValueError:
                    # if there are no peaks add 60
                    last_day_candidates.append(60)
        
            # choose find last day (nearest to 60) from candidates
            last_day = min(last_day_candidates, key=lambda x: abs(x - 60))
        
            # plot a line representing last day of pandemic
            ax.axvline(last_day, color='red')
        
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([0 - ymax * 0.1, ymax * 1.2])
        
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim([0 - ymax * 0.1, ymax * 1.2])
        
            fig.legend(
                loc="upper center",
                ncol=2,
                bbox_to_anchor=(0.5, 1),
                bbox_transform=ax.transAxes,
                fancybox=True,
                shadow=True)
        
            plt.tight_layout()
            cls.__show_and_save(fig=fig,
                                plot_type=f'Finding last day of pandemic up to {last_date}',
                                plot_name=(f'{voivodeship}, by {start_days_by} in {percent_of_touched_counties} '
                                           f'percent of counties'),
                                save=save,
                                show=show,
                                file_format='png')


