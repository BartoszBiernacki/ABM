import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmax
from disease_spread_model.data_processing.real_data import RealData
from disease_spread_model.config import StartingDayBy


def slope_from_linear_fit(data, half_win_size):
    from scipy.stats import linregress

    slope = np.zeros_like(data)

    for i in range(len(data)):
        # Avoid index error
        left = max(0, i - half_win_size)
        right = min(len(data) - 1, i + half_win_size)

        y = np.array(data[left: right + 1]).astype(float)
        x = np.arange(len(y))

        linefit = linregress(x, y)
        slope[i] = linefit.slope

    return slope


def plot_last_day_finding_process(
        death_toll_smooth_out_win_size=21,
        death_toll_smooth_out_polyorder=3,
        derivative_half_win_size=3,
):
    start_days = RealData.starting_days(
        by=StartingDayBy.INFECTIONS,
        percent_of_touched_counties=80,
        ignore_healthy_counties=False)

    voivodeship = 'mazowieckie'

    day0 = start_days[voivodeship]
    extra_days = 15

    death_toll = np.sign(np.arange(-30, 20))
    last_day_to_search = RealData.date_to_day('2020-07-01')
    death_toll = RealData.death_tolls().loc[voivodeship][:last_day_to_search]
    # death_toll = np.array([
    #     1, 2, 4, 5, 7, 8, 7, 6, 11, 14, 11, 15, 9, 19, 17, 13, 5, 6
    # ])

    death_toll_smooth = savgol_filter(
        x=death_toll,
        window_length=death_toll_smooth_out_win_size,
        polyorder=death_toll_smooth_out_polyorder)

    last_day_to_search = len(death_toll)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax2 = ax.twinx()

    DT_days = np.arange(-extra_days, last_day_to_search - day0)

    # plot death_toll toll
    ax.plot(DT_days, death_toll[day0 - extra_days: last_day_to_search],
            color='C0', label='suma zgonów', lw=4, alpha=0.7)

    ax.plot(DT_days, death_toll_smooth[day0 - extra_days: last_day_to_search],
            color='C1', label='suma zgonów po wygładzeniu')

    # slope of smooth death toll
    slope_smooth = slope_from_linear_fit(
        data=death_toll_smooth, half_win_size=derivative_half_win_size)

    # normalize slope to 1
    if max(slope_smooth[day0 - extra_days: last_day_to_search]) > 0:
        slope_smooth /= max(
            slope_smooth[day0 - extra_days: last_day_to_search])

    # plot normalized slope
    slope_days = np.arange(last_day_to_search - day0)
    ax2.plot(slope_days,
             slope_smooth[day0: last_day_to_search],
             color='green', lw=4,
             label='nachylenie prostej dopasowanej do'
                   ' fragmentów wygładzonej sumy zgonów')

    # Maxima of slope
    vec = np.copy(slope_smooth[day0: last_day_to_search])
    x_peaks_max = argrelmax(data=vec, order=8)[0]
    ax2.scatter(x_peaks_max, vec[x_peaks_max],
                color='lime', s=140, zorder=100,
                label='maksima nachylenia sumy zgonów')
    for day in x_peaks_max:
        ax.axvline(day+7, color='red', lw=5)

    last_day_candidates = [x for x in x_peaks_max if vec[x] > 0.5]
    if not last_day_candidates:
        try:
            last_day_candidates.append(
                max(x_peaks_max, key=lambda x: vec[x]))
        except ValueError:
            # if there are no peaks add 60
            last_day_candidates.append(60)

    end_days = RealData.ending_days_by_death_toll_slope(
        starting_days_by=StartingDayBy.INFECTIONS,
        percent_of_touched_counties=80,
        last_date='2020-07-01',
        death_toll_smooth_out_win_size=21,
        death_toll_smooth_out_polyorder=3,
    )

    ax.axvline(end_days[voivodeship] - day0, color='black', lw=3)
    ax.axvline(end_days[voivodeship] - day0 + 7, color='gray', lw=3)

    plt.show()

    end_days = RealData.ending_days_by_death_toll_slope(
        starting_days_by=StartingDayBy.INFECTIONS,
        percent_of_touched_counties=80,
        last_date='2020-07-01',
        death_toll_smooth_out_win_size=21,
        death_toll_smooth_out_polyorder=3,
    )
    print(end_days)
    for voivodeship, last_day in end_days.items():
        print(voivodeship, RealData.day_to_date(last_day))


plot_last_day_finding_process()