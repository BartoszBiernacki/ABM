from my_math_utils import *
from text_processing import *
import pandas as pd
import warnings


# plot_1D_death_toll_dynamic_beta_sweep(directory=directory_1D_v2, normalized=True, show=True, save=False)
# plot_1D_death_toll_dynamic_mortality_sweep(directory=directory_1D_v2, normalized=False, show=False, save=True)
# plot_1D_death_toll_dynamic_visibility_sweep(directory=directory_1D_v2, normalized=False, show=False, save=True)
# plot_1D_death_toll_max_visibility_sweep(directory=directory_1D_v2, show=False, save=True)
# plot_1D_death_toll_max_x_visibility_series_betas(directory=directory_1D_v2, normalized=True, show=False, save=True)
# plot_1D_death_toll_max_x_mortality_series_betas(directory=directory_1D_v2, normalized=False, show=False, save=True)
# plot_all_possible_death_toll_plots(directory=directory_1D_v2, normalized=True, show=True, save=False)

# 1D Death toll plots -----------------------------------------------------------------------------------------------
def plot_1D_death_toll_dynamic_beta_sweep(directory, normalized=False, show=True, save=False):
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    grouped_fnames = group_fnames_by_beta(directory=directory)
    for plot_id in range(grouped_fnames.shape[0]):
        fname = grouped_fnames[plot_id][0]
        
        mortality = variable_params_from_fname(fname=fname)['mortality']
        visibility = variable_params_from_fname(fname=fname)['visibility']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        norm_info = ''
        if normalized:
            norm_info = ' (normalized)'
        
        main_title = "Death toll" + norm_info + '\n'f"mortality={float(mortality) * 100:.1f}%"'    ' \
                                                f"visibility={float(visibility) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('t, days')
        
        lines = grouped_fnames.shape[1]
        
        betas = np.empty(lines)
        deaths = np.empty_like(betas)

        num_of_lines = len(betas)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        
        for line_id in range(lines):
            
            fname = grouped_fnames[plot_id][line_id]
            df = pd.read_csv(fname)
            variable_params = variable_params_from_fname(fname=fname)
            
            betas[line_id] = variable_params['$\\beta$']
            
            ignored = ['mortality', 'visibility']
            legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=ignored)
            if normalized:
                legend += ' ' * 4 + r'$\Sigma$'f' Deaths = {np.max(df["Dead people"]):.0f}'
                ax.plot(df['Day'], df['Dead people'] / np.max(df['Dead people']), label=legend, color=colors[line_id],
                        marker='o', markersize=3)
                ax.set_ylabel(r'Death toll / max$_\beta$(Death toll)')
            else:
                ax.plot(df['Day'], df['Dead people'], label=legend, color=colors[line_id],
                        marker='o', markersize=3)
                ax.set_ylabel(r'Death toll')
            
            deaths[line_id] = np.max(df['Dead people'])
        
        unique_betas = list(set(betas))
        unique_betas.sort()
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll dynamic beta sweep'
            save_dir = directory.replace('raw data/', 'plots/')
            save_dir += plot_type + '/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            name = ax.get_title()
            name = name.replace('\n', ' ')
            name = name.replace('    ', ' ')
            if name[-1:] == ' ':
                name = name[:-1]
            plt.savefig(save_dir + name + '.pdf')
        
        if show:
            plt.show()
        plt.close(fig)


def plot_1D_death_toll_dynamic_mortality_sweep(directory, normalized=False, show=True, save=False):
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    grouped_fnames = group_fnames_by_mortality(directory=directory)
    for plot_id in range(grouped_fnames.shape[0]):
        fname = grouped_fnames[plot_id][0]
        
        beta = variable_params_from_fname(fname=fname)[r'$\beta$']
        visibility = variable_params_from_fname(fname=fname)['visibility']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        norm_info = ''
        if normalized:
            norm_info = ' (normalized)'
        
        main_title = "Death toll" + norm_info + '\n'r"$\beta$="f"{float(beta):.3f}"'    ' \
                     f"visibility={float(visibility) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('t, days')
        
        lines = grouped_fnames.shape[1]
        
        mortalities = np.empty(lines)
        deaths = np.empty_like(mortalities)

        num_of_lines = len(mortalities)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        
        for line_id in range(lines):
            
            fname = grouped_fnames[plot_id][line_id]
            df = pd.read_csv(fname)
            variable_params = variable_params_from_fname(fname=fname)
            
            mortalities[line_id] = variable_params['mortality']
            
            ignored = [r'$\beta$', 'visibility']
            legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=ignored)
            if normalized:
                legend += ' ' * 4 + r'$\Sigma$'f' Deaths = {np.max(df["Dead people"]):.0f}'
                ax.plot(df['Day'], df['Dead people'] / np.max(df['Dead people']), label=legend, color=colors[line_id],
                        marker='o', markersize=3)
                ax.set_ylabel(r'Death toll / max$_\beta$(Death toll)')
            else:
                ax.plot(df['Day'], df['Dead people'], label=legend, color=colors[line_id],
                        marker='o', markersize=3)
                ax.set_ylabel(r'Death toll')
            
            deaths[line_id] = np.max(df['Dead people'])
        
        unique_mortalities = list(set(mortalities))
        unique_mortalities.sort()
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll dynamic mortality sweep'
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


def plot_1D_death_toll_dynamic_visibility_sweep(directory, normalized=False, show=True, save=False):
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    grouped_fnames = group_fnames_by_visibility(directory=directory)
    for plot_id in range(grouped_fnames.shape[0]):
        fname = grouped_fnames[plot_id][0]
        
        beta = variable_params_from_fname(fname=fname)[r'$\beta$']
        mortality = variable_params_from_fname(fname=fname)['mortality']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        norm_info = ''
        if normalized:
            norm_info = ' (normalized)'
        
        main_title = "Death toll" + norm_info + '\n'r"$\beta$="f"{float(beta):.3f}"'    ' \
                     f"mortality={float(mortality) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('t, days')
        
        lines = grouped_fnames.shape[1]
        
        visibilities = np.empty(lines)
        deaths = np.empty_like(visibilities)

        num_of_lines = len(visibilities)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        
        for line_id in range(lines):
            
            fname = grouped_fnames[plot_id][line_id]
            df = pd.read_csv(fname)
            variable_params = variable_params_from_fname(fname=fname)
            
            visibilities[line_id] = variable_params['visibility']
            
            ignored = [r'$\beta$', 'mortality']
            legend = get_legend_from_variable_params(variable_params=variable_params, ignored_params=ignored)
            if normalized:
                legend += ' ' * 4 + r'$\Sigma$'f' Deaths = {np.max(df["Dead people"]):.0f}'
                ax.plot(df['Day'], df['Dead people'] / np.max(df['Dead people']), label=legend, color=colors[line_id],
                        marker='o', markersize=3)
                ax.set_ylabel(r'Death toll / max$_\beta$(Death toll)')
            else:
                ax.plot(df['Day'], df['Dead people'], label=legend, color=colors[line_id], marker='o', markersize=3)
                ax.set_ylabel(r'Death toll')
            
            deaths[line_id] = np.max(df['Dead people'])
        
        unique_visibilities = list(set(visibilities))
        unique_visibilities.sort()
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll dynamic visibility sweep'
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


def plot_1D_death_toll_max_visibility_sweep(directory, show=True, save=False):
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    grouped_fnames = group_fnames_by_visibility(directory=directory)
    for plot_id in range(grouped_fnames.shape[0]):
        fname = grouped_fnames[plot_id][0]
        
        beta = variable_params_from_fname(fname=fname)[r'$\beta$']
        mortality = variable_params_from_fname(fname=fname)['mortality']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        main_title = "Death toll"'\n'r"$\beta$="f"{float(beta):.3f}"'    ' \
                     f"mortality={float(mortality) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('visibility')
        
        points = grouped_fnames.shape[1]
        
        visibilities = np.empty(points)
        deaths = np.empty_like(visibilities)
        for point_id in range(points):
            fname = grouped_fnames[plot_id][point_id]
            df = pd.read_csv(fname)
            variable_params = variable_params_from_fname(fname=fname)
            
            visibilities[point_id] = variable_params['visibility']
            deaths[point_id] = np.max(df['Dead people'])
        
        ax.scatter(visibilities, deaths)
        ax.set_ylabel(r'Death toll')
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll max visibility sweep'
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


def plot_1D_death_toll_max_x_visibility_series_betas(directory, normalized, show=True, save=False):
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
        if normalized:
            norm_info = ' (normalized)'
        
        main_title = "Death toll" + norm_info + '\n' \
                     f"mortality={float(mortality) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('visibility')
        if normalized:
            ax.set_ylabel(r'Death toll / max$_{\beta}$(Death toll)')
        else:
            ax.set_ylabel(r'Death toll')
        
        deaths = np.empty((num_betas, num_visibilities))
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
                deaths[beta_id][visibility_id] = np.max(df['Dead people'])
            
            label = r'$\beta$='f"{betas[beta_id]}"
            if normalized:
                label += '    'r"max$_{\beta}$(deaths)="f'{np.max(deaths[beta_id]):.0f}'
                ax.plot(visibilities.values(), deaths[beta_id] / np.max(deaths[beta_id]), label=label,
                        color=colors[beta_id], marker='o', markersize=3)
            else:
                ax.plot(visibilities.values(), deaths[beta_id], label=label, color=colors[beta_id],
                        marker='o', markersize=3)
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll max, visibility sweep, beta series'
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


def plot_1D_death_toll_max_x_mortality_series_betas(directory, normalized, show=True, save=False):
    warnings.filterwarnings("error")
    fnames = all_fnames_from_dir(directory=directory)
    fixed_params = fixed_params_from_fname(fname=fnames[0])
    
    grouped_fnames = group_fnames_standard_by_visibility_beta_mortality(directory=directory)
    num_visibilities, num_betas, num_mortalities = grouped_fnames.shape
    
    num_of_lines = num_betas
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
    for plot_id in range(num_visibilities):
        
        visibility = variable_params_from_fname(fname=grouped_fnames[plot_id][0][0])['visibility']
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        norm_info = ''
        if normalized:
            norm_info = ' (normalized)'
        
        main_title = "Death toll" + norm_info + '\n' \
                                                f"visibility={float(visibility) * 100:.0f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('mortality')
        if normalized:
            ax.set_ylabel(r'Death toll / max$_{\beta}$(Death toll)')
        else:
            ax.set_ylabel(r'Death toll')
        
        deaths = np.empty((num_betas, num_mortalities))
        betas = {}
        for beta_id in range(num_betas):
            fname = grouped_fnames[plot_id][beta_id][0]
            variable_params = variable_params_from_fname(fname=fname)
            betas[beta_id] = variable_params[r'$\beta$']
            
            mortalities = {}
            for mortality_id in range(num_mortalities):
                fname = grouped_fnames[plot_id][beta_id][mortality_id]
                variable_params = variable_params_from_fname(fname=fname)
                mortalities[mortality_id] = variable_params['mortality']
                
                df = pd.read_csv(fname)
                deaths[beta_id][mortality_id] = np.max(df['Dead people'])
            
            label = r'$\beta$='f"{betas[beta_id]}"
            if normalized:
                label += '    'r"max$_{\beta}$(deaths)="f'{np.max(deaths[beta_id]):.0f}'
                try:
                    ax.plot(mortalities.values(), deaths[beta_id] / np.max(deaths[beta_id]), label=label,
                            color=colors[beta_id], marker='o', markersize=5)
                except RuntimeWarning:
                    print("Can't plot 1D_death_toll_max_x_mortality_series_betas probably np.max(deaths) equals 0.")
                    print("Fname = ", fname)
                    ax.plot(mortalities.values(), np.zeros(len(mortalities)), label=label,
                            color=colors[beta_id], marker='o', markersize=5)
            else:
                ax.plot(mortalities.values(), deaths[beta_id], label=label, color=colors[beta_id],
                        marker='o', markersize=5)
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Death toll max, mortality sweep, beta series'
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


def plot_1D_infected_max_x_visibility_series_betas(directory, show=True, save=False):
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
        
        main_title = "Infected toll" + norm_info + '\n' \
                                                   f"mortality={float(mortality) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('visibility')
        ax.set_ylabel(r'Infected toll / all people (in percent)')
        
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
            
            label = r'$\beta$='f"{betas[beta_id]}"
            ax.plot(visibilities.values(), infected[beta_id], label=label, color=colors[beta_id],
                    marker='o', markersize=3)
        
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
        
        plt.tight_layout()
        
        if save:
            plot_type = 'Infected toll max, visibility sweep, beta series'
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


def plot_all_possible_death_toll_plots(directory, normalized=False, show=False, save=True):
    plot_1D_death_toll_dynamic_beta_sweep(directory=directory, normalized=normalized, show=show, save=save)
    plot_1D_death_toll_dynamic_mortality_sweep(directory=directory, normalized=normalized, show=show, save=save)
    plot_1D_death_toll_dynamic_visibility_sweep(directory=directory, normalized=normalized, show=show, save=save)
    
    plot_1D_death_toll_max_visibility_sweep(directory=directory, show=show, save=save)
    
    plot_1D_death_toll_max_x_visibility_series_betas(directory=directory, normalized=normalized, show=show, save=save)
    plot_1D_death_toll_max_x_mortality_series_betas(directory=directory, normalized=normalized, show=show, save=save)

    plot_1D_infected_max_x_visibility_series_betas(directory=directory, show=show, save=save)
# ------------------------------------------------------------------------------------------------------------------


# 1D tau and pandemic duration plots --------------------------------------------------------------------------------
def plot_1D_pandemic_duration_x_beta_series_visibilities(directory, show=True, save=False):
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


def plot_1D_tau_simplest_exp_fit(directory, show=True, save=False):
    import matplotlib
    matplotlib.use("Agg")
    
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


def plot_1D_tau_x_beta_series_visibilities(directory, show=True, save=False):
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


def plot_all_possible_tau_and_pandemic_duration_plots(directory, show=False, save=True):
    plot_1D_pandemic_duration_x_beta_series_visibilities(directory=directory, show=show, save=save)
    plot_1D_tau_simplest_exp_fit(directory=directory, show=show, save=save)
    plot_1D_tau_x_beta_series_visibilities(directory=directory, show=show, save=save)
# ------------------------------------------------------------------------------------------------------------------


# Plot all plots ----------------------------------------------------------------------------------------------------
def plot_all_possible_plots(directory):
    import matplotlib
    matplotlib.use("Agg")
    
    print("Plotting started ...")
    plot_all_possible_death_toll_plots(directory=directory, normalized=False, show=False, save=True)
    plot_all_possible_death_toll_plots(directory=directory, normalized=True, show=False, save=True)
    plot_all_possible_tau_and_pandemic_duration_plots(directory=directory, show=False, save=True)
    print("Plotting completed.")
# ------------------------------------------------------------------------------------------------------------------


# TODO Check if that works
def plot_tau_700_and_1000(directory_700, directory_1000):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    main_title = r"$\tau$"'\n'
    title = main_title
    ax.set_title(title)
    
    # FIRST PLOT----------------------------------------------------------------------------------------
    fnames = all_fnames_from_dir(directory=directory_700)
    betas = np.empty(len(fnames))
    mortalities = np.empty_like(betas)
    taus = np.empty_like(betas)
    
    for i, fname in enumerate(fnames):
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
        
        taus[i] = calc_tau(y_data=df['Incubation people'], x_data=df['Day'])
        
        betas[i] = variable_params['$\\beta$']
        mortalities[i] = variable_params['mortality']
    
    ax.scatter(betas, taus, color='black', label='N=700')
    # ***********************************************************************************************************
    
    # SECOND PLOT----------------------------------------------------------------------------------------
    fnames = all_fnames_from_dir(directory=directory_1000)
    betas = np.empty(len(fnames))
    mortalities = np.empty_like(betas)
    taus = np.empty_like(betas)
    
    for i, fname in enumerate(fnames):
        df = pd.read_csv(fname)
        variable_params = variable_params_from_fname(fname=fname)
        
        taus[i] = calc_tau(y_data=df['Incubation people'], x_data=df['Day'])
        
        betas[i] = variable_params['$\\beta$']
        mortalities[i] = variable_params['mortality']
    
    ax.scatter(betas, taus, color='blue', label='N=1000')
    # ***********************************************************************************************************
    
    ax.legend()
    ax.set_xlabel(r'Transmission parameter $\beta$')
    ax.set_ylabel(r'$\tau$ days')
    
    plt.tight_layout()
    plt.show()


def reproduce_plot_by_website(filename="data_extracted_from_fig1.csv"):
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