import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import pandas as pd
import scipy.fftpack as fft
import numpy as np
from bokeh.plotting import figure, output_file, Column, Row, save
from bokeh.models import ColumnDataSource, HoverTool, Legend, LinearColorMapper, LabelSet
from bokeh.palettes import Spectral, RdYlBu, Colorblind, Category10
import holoviews as hv
from bokeh.layouts import row, column
hv.Store.current_backend = 'bokeh'


def generate_dummy_data():
    """
    Generates dummy data in pivot format to be used for testing and demoing
    """
    df = pd.DataFrame(columns=['bacon', 'spam', 'beans', 'ham'],
                      index=pd.date_range(start='2015-01-01', end='2018-01-01'))
    t = np.array(range(len(df)))
    for c in df.columns:
        df.loc[:, c] = np.random.random(1) * np.sin(2 * np.pi * t / len(df)) + np.random.random(len(df))
    return df


def plot_cat(df_in, smooth_period=1, plot_std=False, title='', plot_original=True, linestyle='-', marker='+', ax=None, show_weekday_on=None):
    """
    Plots a df categorized
    :param df_in: the input df
    :param smooth_period: if needs to be smoothed by how muchh
    :param plot_std: whether to plot the STD of the smoothing period
    :param title: title
    :return:
    """
    df_in = df_in.interpolate(axis=0)
    if ax is None:
        ax = df_in.rolling(smooth_period).mean().plot(figsize=(14, 4), linestyle=linestyle, marker=marker)
    else:
        df_in.rolling(smooth_period).mean().plot(linestyle=linestyle, ax=ax, marker=marker)
    df_in = pd.DataFrame(df_in) if type(df_in) == pd.Series else df_in
    if show_weekday_on is not None:
        plot_weekdays(df_in, show_weekday_on)
    if plot_std:
        low = df_in.rolling(smooth_period).std() + df_in.rolling(smooth_period).mean()
        high = df_in.rolling(smooth_period).mean() - df_in.rolling(smooth_period).std()
        for c in df_in.columns:
            ax.fill_between(low[c].dropna().index, low[c].dropna().values, high[c].dropna().values, alpha=0.2)
    if smooth_period > 1 and plot_original:
        ax.set_prop_cycle(None)
        df_in.plot(ax=ax, alpha=0.7, marker='+', linestyle='--')
    ax.grid()
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), mode='expand', fontsize='x-small')


def plot_cat_fft(df_in, title='', show_max_labels=False):
    df_in = pd.DataFrame(df_in) if type(df_in) == pd.Series else df_in
    f_axis = np.linspace(-0.5, 0.5, len(df_in))

    def my_fft(x):
        return np.abs(fft.fftshift(fft.fft(x - np.mean(x))))
    df_in_fft = pd.DataFrame(index=f_axis, columns=df_in.columns)
    for c in df_in:
        df_in_fft[c] = my_fft(df_in[c])
        df_in_fft.plot(figsize=(14, 4))
        if show_max_labels:
            max_labels = df_in_fft[df_in_fft.index>0].sort_values(by=c, ascending=False).iloc[:5]
            for i in max_labels.index:
                plt.text(s=1/i, x=i, y=max_labels.loc[i])
    plt.title(title)
    plt.grid()
    plt.show()


def plot_weekdays(df, key):
    df['wd'] = [x.strftime('%A') for x in df.index]
    for d in df.index:
        plt.text(s=df.loc[d, 'wd'], x=d, y=df.loc[d, key])


def plot_pivot(df_in, x, y, cat, smooth_period=1, plot_std=False, title='', linestyle='-'):
    """
    Plots a df categorized unstacked, first pivots it
    :param df_in: the input df
    :param x: the x axis
    :param y: the y axis
    :param cat: the category (number of lines)
    :param smooth_period: if needs to be smoothed by how muchh
    :param plot_std: whether to plot the STD of the smoothing period
    :param title: title
    :return:
    """
    df = pd.pivot_table(df_in, index=x, columns=cat, values=y)
    plot_cat(df, smooth_period=smooth_period, plot_std=plot_std, title=title, linestyle=linestyle)


def plot_pivot_hist(df, x, y, label, separate_plots=True):
    """
    Plots a categorized histogram, first pivots it
    :param df:
    :param x: x axis
    :param y: y axis
    :param label: the category
    :param separate_plots: whether to plot all histograms on the same plot or different figures.
    :return:
    """
    dfp = pd.pivot_table(df, index=x, columns=label, values=y)
    if separate_plots:
        for c in dfp.columns:
            plt.figure(figsize=(12,4))
            dfp[c].plot(kind='hist', bins=40)
            plt.grid(alpha=0.5)
            plt.xlabel('number of events')
            plt.title(label + " = " + str(c))
    else:
        dfp.plot(kind='hist', bins=40, alpha=0.5, figsize=(12, 4))
        plt.grid(alpha=True)
    plt.show()


def plot_acf(x, n_lags, title='ACF plot'):
    """
    Plots the Autocorrelation function of a time series
    :param x: the time series
    :param n_lags: the number of lags to be shown (computed)
    :param title:
    :return:
    """
    acf_sm, qstat, pval = acf(x, nlags=n_lags, qstat=True)
    plt.figure(figsize=(12,6))
    plt.hlines(0.05, 0, n_lags, linestyle='--')
    plt.hlines(0, 0, n_lags)
    plt.hlines(-0.05, 0, n_lags, linestyle='--')
    plt.plot(acf_sm, '-+')
    plt.title(title)
    plt.grid()
    plt.show()


def plot_acf_df(df, n_lags, title='ACF plot'):
    """
    Plots the Autocorrelation function of a time series
    :param df: the input df organized as pivot_table
    :param n_lags: the number of lags to be shown (computed)
    :param title:
    :return:
    """
    df_acf = pd.DataFrame(columns=df.columns)
    for c in df.columns:
        df_acf[c] = acf(df[c].values, nlags=n_lags, qstat=False)
    df_acf.plot(figsize=(12, 6), style='-+')
    plt.hlines(0.05, 0, n_lags, linestyle='--')
    plt.hlines(0, 0, n_lags)
    plt.hlines(-0.05, 0, n_lags, linestyle='--')
    plt.title(title)
    plt.grid()
    plt.show()


def plot_delta(df, delta_cols, target_col, title='Delta plot'):
    f, ax = plt.subplots(3, 1, figsize=(14, 8))
    all_cols = delta_cols + [target_col]
    plot_cat(df[all_cols], ax=ax[0], title=title)
    df_delta = pd.DataFrame(index=df.index)
    for col in delta_cols:
        df_delta['delta_'+col] = df[col] - df[target_col]
    plot_cat(df_delta, ax=ax[1])
    df_delta = df_delta[df_delta.columns[df_delta.sum() != 0]]
    if not df_delta.empty:
        df_delta.plot(kind='kde', ax=ax[2])
        ax[2].grid()
        ax[2].legend(fontsize='x-small')


def plot_error (df, title='Error'):
    f, ax = plt.subplots(2, 1, figsize=(20, 10))
    plot_cat(df, ax=ax[0], title=title)
    df.plot(kind='kde', ax=ax[1])
    ax[1].grid()


def plot_imagesc(df, title=''):
    plt.figure(figsize=(10, 10))
    xticks = [pd.Timestamp(x).date() for x in df.index.values]
    plt.matshow(df.transpose().values)
    plt.colorbar()
    plt.xticks(range(len(xticks)), xticks, rotation=45)
    plt.title(title)


def bokeh_color(label, label_set, palette_num=10, offset=0):
    n = list(label_set).index(label)
    return Category10[palette_num][(n + offset) % len(Category10[palette_num])]


def plot_cat_bokeh(df, title='Sample Bokeh plot', x_type='event_date', color_offset=0, width=1200, height=300):
    # ensure the columns are strings
    df.columns = [str(x) for x in df.columns]
    df = df.loc[:, df.count() > 0]  # ensure no column is just nan
    if x_type == 'event_date':
        source = ColumnDataSource(df.reset_index().rename(columns={"index": "event_date"}))
        hover_tooltips = [('event_date', '@event_date{%F}')]+[(x, '@'+x+'{0.00}') for x in df.columns]
        hover = HoverTool(tooltips=hover_tooltips, formatters={'event_date': 'datetime'}, mode='mouse')
        p = figure(x_axis_type="datetime", title=title,
                   plot_width=width, plot_height=height,
                   toolbar_location='above')
        x_col = 'event_date'
    else:
        source = ColumnDataSource(df.reset_index())
        hover_tooltips = [('index', '@index')] + [(x, '@' + x + '{0.00}') for x in df.columns]
        hover = HoverTool(tooltips=hover_tooltips, formatters={'event_date': 'datetime'}, mode='mouse')
        p = figure(title=title,
                   plot_width=width, plot_height=height,
                   toolbar_location='above')
        x_col = 'index'
    p.add_tools(hover)
    p.grid.grid_line_alpha = 0.5
    plines = {}
    pmarks = {}
    for c in df.columns:
        plines[c] = p.line(x=x_col, y=c, source=source, color=bokeh_color(c, df.columns, offset=color_offset))
        pmarks[c] = p.scatter(x=x_col, y=c, source=source, marker='cross', color=bokeh_color(c, df.columns, offset=color_offset))
    legend = Legend(items=[(x, [plines[x], pmarks[x]]) for x in df.columns], location=(0, 0))
    legend.click_policy = "hide"
    p.add_layout(legend, 'right')
    p.background_fill_color = 'beige'
    p.background_fill_alpha = 0.2
    return p


def plot_hist_bokeh(df, title='Histogram', bins=100, color_offset=0):
    p = figure(title=title, plot_width=600, plot_height=500)
    p.grid.grid_line_alpha = 0.5
    p.legend.location = "bottom_left"
    df = df.loc[:, df.count() > 3]  # ensure at least 3 values to make a histogram
    df.rename(columns={v:str(v) for v in df.columns}, inplace=True)
    for c in df.columns:
        try:
            counts, vals = np.histogram(df[c].dropna(), bins=bins)
            vals = (vals[1:] + vals[:-1])/2
            width = (abs(vals[0]) + abs(vals[-1]))/len(vals)
            p.vbar(x=vals, top=counts, bottom=0, width=width, legend=c,
                   color=bokeh_color(c, df.columns, offset=color_offset),
                   alpha=0.5)
        except ValueError:
            pass
    p.legend.click_policy = "hide"
    p.background_fill_color = 'beige'
    p.background_fill_alpha = 0.2
    return p


def plot_bar_bokeh_from_dict(dict_df, title='Bar plot', plot_width=800, plot_height=300):
    p = [plot_bar_bokeh(dict_df[k], title=title + ' ' + k, plot_height=plot_height, plot_width=plot_width)
         for k in dict_df.keys()]
    return Row(*p)


def plot_bar_bokeh(df, title='Bar plot', plot_width=1000, plot_height=300):
    df = pd.DataFrame(df) if type(df) != pd.DataFrame else df
    top_y = df.columns[0]
    # for multiindex dfs, create a new index
    df['label'] = [str(x) for x in df.index]
    df['position'] = range(len(df))
    df[top_y] = df[top_y].abs()
    df.columns = [str(c) for c in df.columns]
    p = figure(title=title, plot_width=plot_width, plot_height=plot_height, y_range=[0, df[top_y].abs().max()*1.5])
    p.grid.grid_line_alpha = 0.5
    source = ColumnDataSource(df)
    hover = HoverTool(tooltips=[('label', '@label'), ('value', '@'+top_y)])
    #labels = LabelSet(x='position', y=top_y, text='label', level='glyph',
    #                  x_offset=0, y_offset=0, source=source, render_mode='canvas', angle=45,
    #                  text_align='left', text_font_size='6pt')
    p.vbar(x='position', top=top_y, width=0.9, line_color="white", source=source)
    #p.add_layout(labels)
    p.add_tools(hover)
    return p


def plot_boxplot_bokeh(df):
    foo = df.stack().reset_index()
    foo = foo[[foo.columns[1], 0]]
    foo.columns = ['group', 'error']
    table = hv.Table(foo)
    hvplot = hv.BoxWhisker(table.data, kdims=['group'])
    rend = hv.renderer('bokeh')
    fig = rend.get_plot(hvplot).state
    fig.plot_width = 600
    fig.plot_height = 500
    fig.xaxis.major_label_orientation = 45
    fig.grid.grid_line_alpha = 0.5
    fig.background_fill_color = 'beige'
    fig.background_fill_alpha = 0.2
    return fig


def plot_acf_bokeh(df, n_lags=50, title='ACF plot'):
    """
    Plots the Autocorrelation function of a time series
    :param df: the input df organized as pivot_table
    :param n_lags: the number of lags to be shown (computed)
    :param title:
    :return:
    """
    df_acf = pd.DataFrame(columns=df.columns)
    for c in df.columns:
        try:
            if len(df[c].dropna().values) > 5:
                df_acf[c] = acf(df[c].dropna().values, nlags=n_lags, qstat=False)
        except ValueError:
            pass
    fig = plot_cat_bokeh(df_acf, width=800, height=300, title=title, x_type='index')
    return fig


def plot_scatter_bokeh(df, x, y, c, s, max_size=30, title='Scatter plot'):
    df['attr_size'] = max_size*abs(df[s])/abs(df[s]).max() + 3  # minimum of 3 otherwise not seen
    mapper = LinearColorMapper(palette=Category10[10], low=df[c].min(), high=df[c].max())
    source = ColumnDataSource(df.reset_index())
    index_tt = [(str(v), "@"+str(v)) for v in df.index.names]
    val_tt = [(str(v), "@" + str(v) + '{0.00}') for v in [x, y, s, c]]
    hover = HoverTool(tooltips=index_tt + val_tt)
    p = figure(plot_width=600, plot_height=600, title=title, toolbar_location="right")
    p.add_tools(hover)
    p.circle(x, y, size='attr_size', fill_color={'field': c, 'transform': mapper},
             fill_alpha=0.5, line_alpha=1, line_color={'field': c, 'transform': mapper}, source=source)
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y
    p.background_fill_color = 'gray'
    p.background_fill_alpha = 0.05
    return p


def plot_scatter_bokeh_from_dict(df_dict, x, y, c, s, max_size=30, title='Scatter plots'):
    p_list = [plot_scatter_bokeh(df_dict[k], x=x+k, y=y+k, c=c, s=s, max_size=max_size, title=title + " {}".format(k))
              for k in df_dict.keys()]
    return p_list


def plot_scatter_bokeh_size_vars(df, x, y, c, size_vars, title="Scatter plot", save_to="unnamed_scatter_plot.html"):
    output_file(save_to, title=title)
    f = []
    for sv in size_vars:
        f.append(plot_scatter_bokeh(df, x, y, c, sv, title=title+" - sized by " + sv))
    p = Row(*f)
    save(p)


def generate_delta_report(df, delta_cols, target_col, title='Delta report', save_to='unnamed_delta_report.html'):
    p1 = plot_cat_bokeh(df[delta_cols+[target_col]], title=title)
    p1.title.text_font_size = '18pt'
    # deltas
    df_delta = pd.DataFrame(index=df.index)
    df_drel = pd.DataFrame(index=df.index)
    for col in delta_cols:
        df_delta['delta_' + col] = df[col] - df[target_col]
        df_drel['pct_delta_' + col] = 100*(df[col] - df[target_col])/df[target_col]
    p2 = plot_cat_bokeh(df_delta, title='Delta analysis vs ' + target_col, color_offset=0)
    p3 = plot_cat_bokeh(df_drel, title='Percent delta vs ' + target_col, color_offset=0)
    # histogram
    p4 = plot_hist_bokeh(df_delta, bins=100, color_offset=0)
    # box plot
    p5 = plot_boxplot_bokeh(df_delta)
    p6 = plot_acf_bokeh(df_delta)
    p = column(p1, p2, p3, Row(p4, p5), p6)
    if save_to is None:
        return p
    else:
        output_file(save_to, title=title)
        save(p)


def generate_error_report(df, title='Error report', x_type='event_date', save_to='unnamed_error_report.html'):
    # normal is already deltas
    p1 = plot_cat_bokeh(df.loc[:, df.count() > 0], title=title, x_type=x_type)
    p1.title.text_font_size = '18pt'
    # histogram
    p2 = plot_hist_bokeh(df, bins=100, color_offset=0)
    # box plot
    p3 = plot_boxplot_bokeh(df)
    p4 = plot_acf_bokeh(df)
    p = Column(p1, Row(p2, p3), p4)
    if save_to is None:
        return p
    else:
        output_file(save_to, title=title)
        save(p)

