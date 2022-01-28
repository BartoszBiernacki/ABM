from my_math_utils import *
from text_processing import *
import pandas as pd
import warnings

from constants import *
from real_data import RealData


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


def plot_1D_death_toll_dynamic_real_and_beta_sweep(directory, real_death_toll, voivodeship,
                                                   normalized=False, show=True, save=False):
    """
    Plots entire real death toll for given voivodeship and simulated ones.
    
    :param directory: directory to averaged simulated data
    :type directory: string
    :param real_death_toll: dataframe containing death toll for all voivodeships
    :type real_death_toll: pd.DataFrame
    :param voivodeship: name of voivodeship
    :type voivodeship: string
    :param normalized: normalize death toll to 1?
    :type normalized: boolean
    :param show: show plot?
    :type show: boolean
    :param save: save plot?
    :type save: boolean
    """
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
            norm_info = ' (normalized) '

        main_title = "Death toll " + norm_info + 'for ' + voivodeship + \
                     '\n'f"mortality={float(mortality) * 100:.1f}%"'    ' \
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

        legend = 'Real data'
        x = list(range(len(real_death_toll.columns.to_list())))
        y = real_death_toll.loc[voivodeship].to_list()
        ax.plot(x, y, label=legend, color='black', marker='o', markersize=3)
        
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


def plot_stochastic_1D_death_toll_dynamic(dict_result: dict,
                                          avg_directory: str,
                                          voivodeship: str,
                                          show=False,
                                          save=True):
    """
    Designed to show stochasticity of simulations. Plots many death toll lines at the same Axes, each line was
    generated by simulation with exactly the same parameters.

    :param dict_result: dict in which:
        key is a tuple containing info about simulation run parameters
        value is dataframe containing model level DataCollector result for one simulation (not average)
    :type dict_result: dictionary
    :param avg_directory: directory to saved averaged model level DataCollector results. It's just for taking plot
        title name from filename (for reading simulation run parameters). Params are read from latest file in directory.
    :type avg_directory: str
    :param voivodeship: name of voivodeship that will be included in plot title.
    :type voivodeship: str
    :param save: Save plot?
    :type save: Boolean
    :param show: Show plot?
    :type show: Boolean
    
    If in run.py all params are fixed and iterations > 1 then desired not averaged data are stored in TMP_SAVE.
    """
    fnames = all_fnames_from_dir(directory=avg_directory)
    latest_file = max(fnames, key=os.path.getctime)
    variable_params = variable_params_from_fname(fname=latest_file)
    beta = variable_params['$\\beta$']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f"Death toll stochastic for {voivodeship} with " + r"$\beta$=" + f"{beta}")
    ax.set_xlabel('t, days')
    ax.set_ylabel(r'Death toll')
    
    for df in dict_result.values():
        ax.plot(df['Day'], df['Dead people'])
    plt.tight_layout()

    if save:
        plot_type = f"Death toll stochastic for {voivodeship}"
        save_dir = avg_directory.replace('raw data/', 'plots/')
        save_dir += plot_type + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f"Death toll stochastic for {voivodeship} with beta={beta}"
        plt.savefig(save_dir + name + '.pdf')

    if show:
        plt.show()
    plt.close(fig)

    plt.close(fig)


def plot_1D_recovered_dynamic_real_and_beta_sweep(directory, real_infected_toll, voivodeship,
                                                  normalized=False, show=True, save=False):
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
            norm_info = ' (normalized) '
        
        main_title = "Infected toll " + norm_info + 'for ' + voivodeship + \
                     '\n'f"mortality={float(mortality) * 100:.1f}%"'    ' \
                     f"visibility={float(visibility) * 100:.1f}%"'\n'
        title = main_title + get_ax_title_from_fixed_params(fixed_params=fixed_params)
        ax.set_title(title)
        ax.set_xlabel('t, days')
        
        lines = grouped_fnames.shape[1]
        
        betas = np.empty(lines)
        infected = np.empty_like(betas)
        
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

            incubation_people = df['Incubation people']
            prodromal_people = df['Prodromal people']
            illness_people = df['Illness people']
            recovery_people = df['Recovery people']
            dead_people = df['Dead people']
            
            infected_toll = incubation_people + prodromal_people + illness_people + recovery_people + dead_people
            
            ax.plot(df['Day'], infected_toll, label=legend, color=colors[line_id],
                    marker='o', markersize=3)
            ax.set_ylabel(r'Infected toll')
            
            infected[line_id] = np.max(infected_toll)
        
        unique_betas = list(set(betas))
        unique_betas.sort()
        
        legend = 'Real data'
        x = list(range(len(real_infected_toll.columns.to_list())))
        y = real_infected_toll.loc[voivodeship].to_list()
        ax.plot(x, y, label=legend, color='black', marker='o', markersize=3)
        
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


def show_real_disease_death_toll_normalized(voivodeships,
                                            last_day=None,
                                            show=True,
                                            save=False):
                                            
    """
    Function plots death toll since first day when data were collected up to last_day.
    Death toll for all voivodeships is normalized to one to show when pandemic started
    for each voivodeship.
    """
    real_data_obj = RealData(customers_in_household=3)
    real_death_toll_df = real_data_obj.get_real_death_toll()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    if last_day:
        ax.set_title(f"Death toll normalized. Data for {last_day} days since first case i.e. 04-03-2020.")
        ax.set_ylabel(f"Death toll / Day toll(day={last_day})")
    else:
        ax.set_title(f"Death toll normalized. Data for {len(real_death_toll_df.columns)} days since first case i.e. "
                     f"04-03-2020.")
        ax.set_ylabel(f"Death toll / Day toll(day={len(real_death_toll_df.columns)})")
    ax.set_xlabel("t, day since first day of collecting data i.e. 04-03-2020")

    cmap = plt.get_cmap('rainbow')
    if 'all' in voivodeships:
        num_of_lines = len(real_death_toll_df.index)
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    else:
        num_of_lines = len(voivodeships)
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]

    urbanization = get_urbanization()
    index = 0
    for voivodeship in urbanization.keys():
        x = range(len(real_death_toll_df.columns))
        y = real_death_toll_df.loc[voivodeship]
        
        if last_day:
            if np.max(y[: last_day]) > 0:
                y /= np.max(y[: last_day])
            if 'all' in voivodeships or voivodeship in voivodeships:
                ax.plot(x[: last_day], y[: last_day], label=voivodeship, color=colors[index])
                index += 1
        else:
            y /= np.max(y)
            if 'all' in voivodeships or voivodeship in voivodeships:
                ax.plot(x, y, label=voivodeship, color=colors[index])
                index += 1
    if last_day:
        ax.set_xlim(0, last_day+5)
        ax.set_ylim(0, 1.1)

    ax.legend()
    plt.tight_layout()

    if save:
        plot_type = 'Real death normalized, since 04-03-2020'
        save_dir = 'results/plots/'
        save_dir += plot_type + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = ax.get_title()
        name = name.replace('\n', ' ')
        name = name.replace('    ', ' ')
        if name[-1:] == ' ':
            name = name[:-1]
        plt.savefig(save_dir + name + '.pdf')
        print(save_dir + name + '.pdf')

    if show:
        plt.show()
    plt.close(fig)


def show_real_death_toll_shifted_to_match_death_toll_in_given_day(starting_day=10,
                                                                  day_in_which_colors_are_set=60,
                                                                  last_day=100,
                                                                  minimum_deaths=1,
                                                                  directory_to_data=None,
                                                                  shift_simulated=False,
                                                                  save=False,
                                                                  show=True):
    """
    Plots shifted death toll for all voivodeships in such manner that
    in day = starting_day in each voivodeship death toll is not less
    than minimum_deaths.
    
    If directory_to_data is given than also plots simulated death
    toll shifted in the same way.
    
    The idea is that if we look at death toll in voivodeships since
    fixed number of people died than death toll should looks similar.
    """
    
    real_data_obj = RealData(customers_in_household=3)
    shifted_real_death_toll = real_data_obj.get_shifted_real_death_toll_to_common_start(starting_day=starting_day,
                                                                                        minimum_deaths=minimum_deaths)

    true_start_day = real_data_obj.get_day_of_first_n_death(n=minimum_deaths)
    
    # df_indices_order[voivodeship] = num of voivodeship sorted shifted_real_death_toll
    # by death toll in day = 60. Used to determine colors of lines in the plot.
    df_indices_order = sort_df_indices_by_col(df=shifted_real_death_toll, column=day_in_which_colors_are_set)
    
    # dict df_indices_order where new_key = old_val and new_val = old_key (dict revered)
    inv_df_indices_order = {v: k for k, v in df_indices_order.items()}
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f"Death toll for every voivodeship, shifted in such a way, that in day {starting_day} "
                 f"death toll is not less than {minimum_deaths}.\n"
                 f"Mapping: (voivodeship, line color) was performed in day {day_in_which_colors_are_set} "
                 f"with respect to death toll in that day.")
    ax.set_xlabel(f't, days since first {minimum_deaths} people died in given voivodeship')
    ax.set_ylabel(f'Death toll (since first {minimum_deaths} people died in given voivodeship)')
    
    num_of_lines = len(shifted_real_death_toll.index)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
    
    for i in range(len(shifted_real_death_toll.index)):
        voivodeship = inv_df_indices_order[i]
        x = shifted_real_death_toll.columns  # x = days of pandemic = [0, 1, ...]
        y = shifted_real_death_toll.loc[voivodeship]

        left = voivodeship
        right = f"day {starting_day} = {true_start_day[voivodeship]}"
        label = '{:<20} {:>22}'.format(left, right)
        
        ax.plot(x[:last_day], y[:last_day],
                label=label,
                color=colors[i])
        
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

            common_day = 0
            if shift_simulated:
                y = np.array(df['Dead people'])
    
                common_day = np.where(y >= minimum_deaths)[0][0]
                y = y[common_day - starting_day:]
                x = list(range(len(y)))
            else:
                x = df['Day']
                y = df['Dead people']

            beta_info = r'$\beta$=' + f'{beta}'
            mortality_info = f'mortality={mortality * 100:.1f}%'
            visibility_info = f'visibility={visibility * 100:.0f}%'
            day_info = f"day {starting_day}=0"
            if shift_simulated:
                day_info = f"day {starting_day}={common_day}"

            label = '{:<10} {:>15} {:>15} {:>10}'.format(beta_info,
                                                         mortality_info,
                                                         visibility_info,
                                                         day_info)
            
            ax.plot(x[:last_day], y[:last_day], label=label, color=colors[i],
                    linewidth=3, linestyle='dashed')
         
    ax.legend(prop={'family': 'monospace'})
    plt.tight_layout()

    if save:
        plot_type = 'Real shifted death toll matching death toll in given day'
        save_dir = 'results/plots/' + plot_type + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f'minimum_deaths={minimum_deaths},   ' \
               f'starting_day={starting_day},   ' \
               f'day_in_which_colors_are_set={day_in_which_colors_are_set}   ' \
               f'last_day={last_day}'
        plt.savefig(save_dir + name + '.pdf')

    if show:
        plt.show()
    plt.close(fig)


def show_real_death_tol_shifted_by_hand(starting_day=10,
                                        day_in_which_colors_are_set=60,
                                        last_day=100,
                                        directory_to_data=None,
                                        shift_simulated=False,
                                        save=False,
                                        show=True):
    """
    Makes many death toll plots. On each plot there is death toll for group of similar
    voivodeships. Plots are shifted along X axis in such a way, that pandemic begins
    in starting_day.
    
    Similarity was defined by hand, by looking at death tolls of all voivodeships
    shifted such plots started with chosen value of death toll and looking for what
    initial value of death toll plot smoothly increases.
    
    If directory_to_data is given than shifted simulated data are also plotted.
    """
    
    real_data_obj = RealData(customers_in_household=3)
    # make dict: dict[starting_deaths] = list(voivodeship1, voivodeship2, ...) *************************************
    voivodeship_starting_deaths = RealData.get_starting_deaths_by_hand()
    unique_death_shifts = sorted(list(set(voivodeship_starting_deaths.values())))
    death_shifts = {}
    for death_shift in unique_death_shifts:
        death_shifts[death_shift] = []
        for voivodeship, val in voivodeship_starting_deaths.items():
            if val == death_shift:
                (death_shifts[death_shift]).append(voivodeship)
    # ***************************************************************************************************************
    
    for minimum_deaths, voivodeships in death_shifts.items():
        shifted_real_death_toll = \
            real_data_obj.get_shifted_real_death_toll_to_common_start(starting_day=starting_day,
                                                                      minimum_deaths=minimum_deaths)
    
        true_start_day = real_data_obj.get_day_of_first_n_death(n=minimum_deaths)
    
        # df_indices_order[voivodeship] = num of voivodeship sorted shifted_real_death_toll
        # by death toll in day = 60. Used to determine colors of lines in the plot.
        df_indices_order = sort_df_indices_by_col(df=shifted_real_death_toll, column=day_in_which_colors_are_set)
        death_toll_final_order = {}
        i = 0
        for voivodeship in df_indices_order:
            if voivodeship in voivodeships:
                death_toll_final_order[voivodeship] = i
                i += 1
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_title(f"Death toll for ,,similar'' voivodeships, shifted in such a way, that in day {starting_day} "
                     f"death toll is not less than {minimum_deaths}.\n"
                     f"Mapping: (voivodeship, line color) was performed in day {day_in_which_colors_are_set} "
                     f"with respect to death toll in that day.")
        ax.set_xlabel(f't, days since first {minimum_deaths} people died in given voivodeship')
        ax.set_ylabel(f'Death toll (since first {minimum_deaths} people died in given voivodeship)')
        
        num_of_lines = len(voivodeships)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, num_of_lines)]
        
        for voivodeship, index in death_toll_final_order.items():
            x = shifted_real_death_toll.columns  # x = days of pandemic = [0, 1, ...]
            y = shifted_real_death_toll.loc[voivodeship]
            
            left = voivodeship
            right = f"day {starting_day} = {true_start_day[voivodeship]}"
            label = '{:<20} {:>22}'.format(left, right)
            
            ax.plot(x[:last_day], y[:last_day],
                    label=label,
                    color=colors[index],
                    linewidth=3)
        
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
                
                common_day = 0
                if shift_simulated:
                    y = np.array(df['Dead people'])
                    try:
                        common_day = np.where(y >= minimum_deaths)[0][0]
                    except IndexError:
                        common_day = 0
                    y = y[common_day - starting_day:]
                    x = list(range(len(y)))
                else:
                    x = df['Day']
                    y = df['Dead people']
                
                beta_info = r'$\beta$=' + f'{beta}'
                mortality_info = f'mortality={mortality * 100:.1f}%'
                visibility_info = f'visibility={visibility * 100:.0f}%'
                day_info = f"day {starting_day}=0"
                if shift_simulated:
                    day_info = f"day {starting_day}={common_day}"
                
                label = '{:<10} {:>15} {:>15} {:>10}'.format(beta_info,
                                                             mortality_info,
                                                             visibility_info,
                                                             day_info)
                
                ax.plot(x[:last_day], y[:last_day], label=label, color=colors[i],
                        linewidth=1, linestyle='dashed')
        
        ax.legend(prop={'family': 'monospace'}, loc='upper left')
        plt.tight_layout()
        
        if save:
            plot_type = 'Real death toll for similar voivodeships, shifted by hand'
            save_dir = 'results/plots/' + plot_type + '/'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            name = f'minimum_deaths={minimum_deaths},   ' \
                   f'starting_day={starting_day},   ' \
                   f'day_in_which_colors_are_set={day_in_which_colors_are_set}   ' \
                   f'last_day={last_day}'
            plt.savefig(save_dir + name + '.pdf')
        
        if show:
            plt.show()
        plt.close(fig)


def show_real_death_toll_voivodeship_shifted_by_hand(directory_to_data,
                                                     voivodeship,
                                                     starting_day=10,
                                                     day_in_which_colors_are_set=60,
                                                     last_day=100,
                                                     shift_simulated=False,
                                                     save=False,
                                                     show=True):
    """
    Plots real death toll for one voivodeship, but shifted in a way, that in day = starting_day
    it looks like pandemic started for good.
    
    Simulation data are also plotted to visually verify if parameters of simulation are correct.
    Simulation data are shifted, but in a way that in day=starting_day simulated death toll
    equal real shifted death toll.
    
    Intended to use with directory_to_data != None to see similarity in real and
    simulated data.
    """
    real_data_obj = RealData(customers_in_household=3)
    voivodeship_starting_deaths = RealData.get_starting_deaths_by_hand()
    minimum_deaths = voivodeship_starting_deaths[voivodeship]

    shifted_real_death_toll = \
        real_data_obj.get_shifted_real_death_toll_to_common_start_by_num_of_deaths(starting_day=starting_day,
                                                                                   minimum_deaths=minimum_deaths)
    
    true_start_day = real_data_obj.get_day_of_first_n_death(n=minimum_deaths)
    
    # Plot data for given voivodeship *******************************************************************************
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f"Death toll for {voivodeship}, shifted in such a way, that in day {starting_day} "
                 f"death toll is not less than {minimum_deaths}.\n")
    ax.set_xlabel(f't, days since first {minimum_deaths} people died in given voivodeship')
    ax.set_ylabel(f'Death toll (since first {minimum_deaths} people died in given voivodeship)')

    x = shifted_real_death_toll.columns  # x = days of pandemic = [0, 1, ...]
    y = shifted_real_death_toll.loc[voivodeship]

    left = voivodeship
    right = f"day {starting_day} = {true_start_day[voivodeship]}"
    label = '{:<20} {:>22}'.format(left, right)

    ax.plot(x[:last_day], y[:last_day], label=label, color='Black', linewidth=3)
    # ****************************************************************************************************************
    
    # Plot data from simulations *************************************************************************************
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
            
            common_day = 0
            if shift_simulated:
                y = np.array(df['Dead people'])
                common_day = np.where(y >= minimum_deaths)[0][0]
                y = y[common_day - starting_day:]
                x = list(range(len(y)))
            else:
                x = df['Day']
                y = df['Dead people']
            
            beta_info = r'$\beta$=' + f'{beta}'
            mortality_info = f'mortality={mortality * 100:.1f}%'
            visibility_info = f'visibility={visibility * 100:.0f}%'
            day_info = f"day {starting_day}=0"
            if shift_simulated:
                day_info = f"day {starting_day}={common_day}"
            
            label = '{:<10} {:>15} {:>15} {:>10}'.format(beta_info,
                                                         mortality_info,
                                                         visibility_info,
                                                         day_info)
            
            ax.plot(x[:last_day], y[:last_day], label=label, color=colors[i],
                    linewidth=1, linestyle='dashed')
        # ************************************************************************************************************
    
    ax.legend(prop={'family': 'monospace'}, loc='upper left')
    plt.tight_layout()
    
    if save:
        plot_type = 'Real shifted initial death toll for similar voivodeships'
        save_dir = directory_to_data.replace('raw data', 'plots')
        save_dir += plot_type + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f'minimum_deaths={minimum_deaths},   ' \
               f'starting_day={starting_day},   ' \
               f'day_in_which_colors_are_set={day_in_which_colors_are_set}   ' \
               f'last_day={last_day}'
        plt.savefig(save_dir + name + '.pdf')
    
    if show:
        plt.show()
    plt.close(fig)


def plot_matched_real_death_toll_to_simulated(y1_simulated,
                                              y2_real,
                                              y2_start=None,
                                              y2_end=None,
                                              show=True):
    """
    Assumes y(x=0) = y[0], y(x=1) = y[1] and so on.
    Function moves y2 data along X axis to find out for what shift subset of y2 = y2[start: end]
    best matches subset of the same length of y1. Then plot y1, y2, y2_shifted.
    
    As it is a general function it returns plt.figure, so that it can be saved in desired folder
    function one level above.

    """
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(range(len(y1_simulated)), y1_simulated, label='simulated', color='blue')
    ax.plot(range(len(y2_real)), y2_real, label='real', color='orange')
    
    if y2_start and y2_end:
        ax.plot(np.arange(y2_start, y2_end), y2_real[y2_start: y2_end],
                color='red', linewidth=5)
        
        shift, error = find_best_x_shift_to_match_plots(y1_reference=y1_simulated,
                                                        y2=y2_real,
                                                        y2_start=y2_start,
                                                        y2_end=y2_end)
        if shift < 0:
            y2_new = [np.NaN] * (-shift) + y2_real.to_list()
        else:
            y2_new = y2_real[shift:]
        
        ax.plot(range(len(y2_new)), y2_new, label='real shifted', color='green')
        ax.plot(np.arange(y2_start - shift, y2_end - shift),
                y2_new[y2_start - shift: y2_end - shift],
                color='darkgreen',
                linewidth=5)
        
        ax.plot(np.arange(y2_start - shift, y2_end - shift),
                y1_simulated[y2_start - shift: y2_end - shift],
                color='darkblue',
                linewidth=5)
        
        ax.set_xlim(0, 265)
        ax.set_ylim(0, 7 * y2_real[y2_end])
    else:
        ax.set_xlim(0, y2_end)
        ax.set_ylim(0, 1.1 * y2_real[y2_end])
    
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
        
    return fig, ax


def plot_auto_fit_result(directory,
                         voivodeship,
                         percent_of_touched_counties,
                         days_to_fit,
                         ignore_healthy_counties=True,
                         show=True,
                         save=False):
    """
    Assumes y(x=0) = y[0], y(x=1) = y[1] and so on.
    Function moves y2 data along X axis to find out for what shift, subset of real death toll
    best matches subset of the same length of simulated death toll.
    Then plot: real death toll, simulated death toll, real death toll shifted.
    
    Simulated data is the latest file in directory.
    Start results from the percent_of_touched_counties.
    End = start + days_to_fit
    """
    real_data_obj = RealData(customers_in_household=3)
    real_death_toll = real_data_obj.get_real_death_toll()
    
    fnames = all_fnames_from_dir(directory=directory)
    latest_file = max(fnames, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    starting_day = real_data_obj.get_starting_days_for_voivodeships_based_on_district_deaths(
        percent_of_touched_counties=percent_of_touched_counties,
        ignore_healthy_counties=ignore_healthy_counties)
    
    print(f'Pandemic in {voivodeship} started in day {starting_day[voivodeship]} if '
          f'assumed that in {percent_of_touched_counties}% of counties must be at least one death. '
          f'Counties in no one ever died are {"ignored" if ignore_healthy_counties else "included"}.')

    fig, ax = plot_matched_real_death_toll_to_simulated(y1_simulated=df['Dead people'],
                                                        y2_real=real_death_toll.loc[voivodeship],
                                                        y2_start=starting_day[voivodeship],
                                                        y2_end=starting_day[voivodeship] + days_to_fit,
                                                        show=False)
    
    ax.set_title(f'Real death toll for {voivodeship} shifted to best match simulated data on '
                 f'{days_to_fit} days, \nsince in {percent_of_touched_counties} percent of counties '
                 f'first death was reported.')
    ax.set_xlabel('t, days')
    ax.set_ylabel('death toll')
    plt.tight_layout()

    if save:
        plot_type = 'Auto fit by percent_of_touched_counties'
        save_dir = directory.replace('raw data', 'plots')
        save_dir += plot_type + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f'percent_of_touched_counties={percent_of_touched_counties},   ' \
               f'ignore_healthy_counties={ignore_healthy_counties},   ' \
               f'days_to_fit={days_to_fit}'
        fig.savefig(save_dir + name + '.pdf')

    if show:
        plt.show()
    plt.close(fig)


def plot_pandemic_starting_days(percent_of_touched_counties,
                                normalize_by_population,
                                save=False,
                                show=True):
    """
    Plots first day of pandemic for all voivodeships since data were collected.
    ALso plots death toll in that day.

    First day results from the percent_of_touched_counties
    """
    
    real_data_obj = RealData(customers_in_household=3)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # PLot first (main) plot on ax ***********************************************************************************
    main_info = f"Day that is consider as starting day of pandemic for given voivodeship and death toll in that day.\n" \
                f" Day 0 is 04-03-2020."
    detail_info = f'Starting day is the first day on which at least one person died in approximately' \
                  f' {percent_of_touched_counties}% of counties.'
    ignore_healthy_info = f'Counties in which no one died (or data are missing) are ignored or included (see legend).'
    
    ax.set_title(main_info + '\n' + detail_info + '\n' + ignore_healthy_info)
    ax.set_xlabel(f'Voivodeship')
    ax.set_ylabel(f'Day number since 04-03-2020 which is considered to be the beginning of a pandemic', color='blue')
    
    starting_days_healthy_ignored = \
        real_data_obj.get_starting_days_for_voivodeships_based_on_district_deaths(
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=True)
    
    starting_days_healthy_included = \
        real_data_obj.get_starting_days_for_voivodeships_based_on_district_deaths(
            percent_of_touched_counties=percent_of_touched_counties,
            ignore_healthy_counties=False)
    
    voivodeships_ignored = starting_days_healthy_ignored.keys()
    
    p1 = ax.scatter(voivodeships_ignored,
                    [starting_days_healthy_ignored[voivodeship] for voivodeship in voivodeships_ignored],
                    color='blue',
                    alpha=0.8,
                    label='Starting day, counties without deaths are ignored.')
    
    ax.plot(voivodeships_ignored,
            [starting_days_healthy_ignored[voivodeship] for voivodeship in voivodeships_ignored],
            color='blue')
    
    p2 = ax.scatter(voivodeships_ignored,
                    [starting_days_healthy_included[voivodeship] for voivodeship in voivodeships_ignored],
                    color='purple',
                    alpha=0.4,
                    label='Starting day, counties without deaths are included.')
    
    ax.set_ylim([0, None])
    
    # rotate label of outer x axis
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    # *****************************************************************************************************************
    
    # Plot second plot on the other y axis ***************************************************************************
    starting_death_toll_ignored = real_data_obj.get_starting_death_toll_for_voivodeships_by_days(
        voivodeships_days=starting_days_healthy_ignored)
    
    starting_death_toll_included = real_data_obj.get_starting_death_toll_for_voivodeships_by_days(
        voivodeships_days=starting_days_healthy_included)

    ax2 = ax.twinx()
    
    if normalize_by_population:
        y_label2 = '(Death toll / population) ' r'$\cdot 10^5$'
        population = real_data_obj.get_real_general_data()['population']
        p3 = ax2.scatter(voivodeships_ignored,
                         [starting_death_toll_ignored[voivodeship] / population[voivodeship] * (10**5)
                          for voivodeship in voivodeships_ignored],
                         color='Red',
                         alpha=0.8,
                         label='Death toll, counties without deaths are ignored.')

        p4 = ax2.scatter(voivodeships_ignored,
                         [starting_death_toll_included[voivodeship] / population[voivodeship] * (10**5)
                          for voivodeship in voivodeships_ignored],
                         color='Orange',
                         alpha=0.4,
                         label='Death toll, counties without deaths are included.')
    else:
        y_label2 = 'Death toll (in given day)'
        p3 = ax2.scatter(voivodeships_ignored,
                         [starting_death_toll_ignored[voivodeship] for voivodeship in voivodeships_ignored],
                         color='Red',
                         alpha=0.8,
                         label='Death toll, counties without deaths are ignored.')
        
        p4 = ax2.scatter(voivodeships_ignored,
                         [starting_death_toll_included[voivodeship] for voivodeship in voivodeships_ignored],
                         color='Orange',
                         alpha=0.4,
                         label='Death toll, counties without deaths are included.')
        
    ax2.set_ylabel(y_label2, color='red')
    ax2.set_ylim([0, None])
    
    # ----------------------------------------------------------------------------------------------------------------
    # *****************************************************************************************************************
    # added plots for legend
    ps = [p1, p2, p3, p4]
    labs = [p.get_label() for p in ps]
    ax.legend(ps, labs)
    plt.tight_layout()
    
    if save:
        plot_type = 'Starting days for voivodeships based on district deaths'
        save_dir = 'results/plots/' + plot_type + '/'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        name = f'percent_of_touched_counties={percent_of_touched_counties},   ' \
               f'normalize_by_population={normalize_by_population}'
        plt.savefig(save_dir + name + '.pdf')
    
    if show:
        plt.show()
    plt.close(fig)


def plot_max_death_toll_prediction_x_visibility_series_betas(directory,
                                                             show=True,
                                                             save=False):
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
    