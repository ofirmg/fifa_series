from typing import List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from tslearn.clustering import TimeSeriesKMeans

from utils import COLUMNS, FONT, FONT_FAMILY


def plot_diff_over_age(data: pd.DataFrame,
                       diff_attribute: str,
                       age_ranges=(18, 36),
                       yrange=(-10, 15),
                       chart_type: str = 'box',
                       color_col: str = None):
    """
    Util function to plot dataframe attribute over players age (yera of FIFA data)
    :param data: dataframe of all years players fifa ratings
    :param diff_attribute: the attribute to plot over the years (column of dataframe)
    :param age_ranges: ranges to use for plotting x axis (player age)
    :param yrange: ranges to use for plotting y axis (diff_attribute)
    :param chart_type: either 'box' or 'or else will be scatter plot
    :param color_col: name of data column to use to color the plots
    """
    clean_data = data[data['year_diff'].notna()]

    if chart_type == 'box':
        fig = px.box(clean_data,
                     x="age",
                     y=diff_attribute,
                     facet_row=color_col,
                     height=800,
                     title='Boxplots: Overall diff vs age')
    else:
        fig = px.scatter(clean_data,
                         x='age',
                         y=diff_attribute,
                         color=color_col,
                         labels='<i>Age</i>: %{x}' +
                                '<br><b>Mean diff %{y:.2f}</b>: <br>' +
                                '<b>#sample = %{text}</b>'
                         )
    fig.update_xaxes(range=age_ranges)
    fig.add_hline(y=0, name='zero', line_width=1, line_dash="dash", line_color='red')
    fig.update_yaxes(range=yrange)
    return fig


def career_clustering(data: pd.DataFrame,
                      n_clusters: int,
                      players_seasons_count: pd.DataFrame,
                      player_id2_name: dict,
                      max_overall_per_player: dict,
                      max_potential_per_player: dict,
                      metric: str = 'dtw',
                      min_age: int = 18,
                      max_age: int = 25,
                      max_iter: int = 50,
                      rating_attr='overall'):
    """
    Split players to clusters using dtw metric.
    :param data: dataframe of players ratings over years for clustering
    :param n_clusters: int - number of clusters to create
    :param players_seasons_count: holds the number of years each player has in the data,
    :param player_id2_name: mapping of player ids to their names
    :param max_overall_per_player: highest value of overall rating each player achieved in the data (for normalization)
    :param max_potential_per_player: highest value of potential rating each player achieved in the data
    :param metric: name of clustering distance metric for the clustering algo
    :param min_age: minimal age value to use
    :param max_age: max age value to use
    :param max_iter: max number of iteration to let clustering model (TimeSeriesKMeans)
    :param rating_attr: name of attribute used for clustering (over years)
    """
    k = max_age - min_age
    # 1. Data prep
    # All curves within age range
    players_from_min_age = set(data.loc[data['age'] == min_age, COLUMNS.PLAYER_ID].unique().tolist())
    print(f'Number of players with data at min_age({min_age}): {len(players_from_min_age)}')
    player_overkseasons = set(
        players_seasons_count.loc[players_seasons_count['num_seasons'] >= k, COLUMNS.PLAYER_ID].unique().tolist())
    print(f'Number of players with k ({k}) seasons data: {len(player_overkseasons)}')
    curves_population = players_from_min_age.intersection(player_overkseasons)
    print('Population for curves analysis:', len(curves_population))
    curves_data = data[data[COLUMNS.PLAYER_ID].isin(curves_population)]

    # 2. Pivot
    pivot_df = curves_data[[COLUMNS.PLAYER_ID, 'age', rating_attr]].drop_duplicates(subset=[COLUMNS.PLAYER_ID, 'age'])
    pivot_df = pivot_df[(pivot_df['age'] <= max_age) & (pivot_df['age'] >= min_age)]
    pivot_df = pd.pivot(pivot_df, index=[COLUMNS.PLAYER_ID], columns=['age'], values=[rating_attr])
    pivot_df.columns = [i for i in range(min_age, max_age + 1)]
    pivot_df = pivot_df[pivot_df.isna().sum(axis=1) == 0]
    print(f'> Final number of players for the analysis (after removing rows with any NA): {len(pivot_df)}')

    # Clustering with DTW
    clusterer = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, max_iter=max_iter, random_state=2603)
    clusterer.fit(pivot_df.values)
    centriods = pd.DataFrame([vec.flatten() for vec in clusterer.cluster_centers_],
                             columns=pivot_df.columns).T
    centriods.columns = [f'career_{i}' for i in range(len(centriods.columns))]
    pivot_df['cluster'] = list(clusterer.predict(pivot_df.values))
    pivot_df[COLUMNS.PLAYER_NAME] = pivot_df.index.copy().map(player_id2_name)
    pivot_df['max_overall'] = pivot_df.index.copy().map(max_overall_per_player)
    pivot_df['max_overall-potential_%'] = pivot_df.index.copy().map(max_potential_per_player)
    print('Samples within each cluster:')
    print(pivot_df['cluster'].value_counts())

    # Boxplots of max overall rating
    box_fig = go.Figure()
    box_fig.add_trace(go.Box(x=pivot_df['cluster'], y=pivot_df['max_overall'], name='Max overall'))
    box_fig.update_yaxes(title=f'Max value achieved', title_font_family=FONT)
    box_fig.update_layout(title_text='Cluster vs. max performance', font=dict(family=FONT_FAMILY, size=20))
    box_fig.update_xaxes(title=f'Cluster', title_font_family=FONT)
    box_fig.add_trace(
        go.Box(x=pivot_df['cluster'], y=pivot_df['max_overall-potential_%'], name='Max overall-potential %'))
    box_fig.show()

    centriods_fig = px.line(centriods)
    centriods_fig.update_yaxes(title=f'Overall rating', title_font_family=FONT)
    centriods_fig.update_layout(hovermode="x")
    centriods_fig.update_xaxes(title=f'Age', title_font_family=FONT)
    centriods_fig.update_layout(title_text=f"Player development by age", font=dict(family=FONT_FAMILY, size=20))

    centriods_fig.show()
    return pivot_df, centriods


def sankey_chart(G: nx.Graph,
                 value_column: str = 'weight',
                 title_suffix: str = '',
                 x_pos: List[float] = None,
                 add_age: bool = False,
                 fig_width: int = None,
                 fig_height: int = None,
                 nodes_color=None):
    """
    Function creates a sankey chart of players careers phases
    :param G: graph of players career phases
    :param value_column: the column that will be used for graph edges weights
    :param title_suffix: modification to title
    :param x_pos: optional list of x positions to use for each node (used when add_age is True)
    :param add_age: whether to add the age dimension (both location and printed text) to the graph or not
    :param fig_width: optional width value of produced chart
    :param fig_height: optional height value of produced chart
    :param nodes_color:
    """
    nodes_df = pd.DataFrame(G.nodes(data=True), columns=['node', 'attr'])
    edges_df = pd.DataFrame(G.edges(data=True))
    edges_df.columns = ['source', 'target', 'attr']
    edges_df = edges_df.merge(pd.DataFrame.from_dict(edges_df['attr'].to_dict(), orient='index'),
                              left_index=True,
                              right_index=True)
    nodes_lst = list(G.nodes)

    if add_age:
        link_dict = dict(
            source=[nodes_lst.index(n) for n in edges_df['source']],
            target=[nodes_lst.index(n) for n in edges_df['target']],
            value=[e / 10 for e in edges_df[value_column].tolist()],
            label=edges_df[value_column].tolist()
        )

        fig = go.Figure(data=[go.Sankey(valueformat=".0f", valuesuffix="TWh",
                                        node=dict(
                                            x=x_pos,
                                            pad=15,
                                            thickness=10,
                                            line=dict(color="black", width=0.25),
                                            label=nodes_df['node'].tolist(),
                                            color=nodes_color,
                                        ),
                                        link=link_dict)])
    else:
        fig = go.Figure(data=[go.Sankey(
            valueformat=".0f",
            valuesuffix="TWh",
            # Define nodes
            node=dict(
                pad=15,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=nodes_df['node'].tolist(),
            ),
            # Add links
            link=dict(
                source=[nodes_lst.index(n) for n in edges_df['source']],
                target=[nodes_lst.index(n) for n in edges_df['target']],
                value=edges_df[value_column].tolist(),
                label=edges_df[value_column].tolist()
            ))])

    fig.update_layout(title_text=f"Career phases flow chart{title_suffix}", font_size=10)
    if fig_width is not None:
        fig.update_layout(width=fig_width)
    if fig_height is not None:
        fig.update_layout(height=fig_height)
    return fig
