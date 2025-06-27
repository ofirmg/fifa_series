from typing import List, Dict
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

from stat_predict.static.config import ArtifactsPaths

PLAYERS_OF_INTEREST = [
    # Emerging Prodigies: Désiré Doué, Jamie Bynoe-Gittens, Francisco Conceição
    'João Pedro Gonçalves Neves',
    'Bradley Barcola',  # 264652
    'Désiré Doué',  # 271421
    'Conor Bradley',  # 264298
    'Victor Okoh Boniface',  # 247679
    'Jamie Bynoe-Gittens',  # 266032
    'Roméo Lavia',  # 263620
    'Milos Kerkez',  # 260908
    'Benjamin Šeško',  # 260592
    'Nicholas Williams Arthuer',  # 256516
    'Alejandro Garnacho Ferreyra',  # 268438
    'Sávio Moreira de Oliveira',  # 270409 Savinho
    'Alejandro Baena Rodríguez',  # 257279
    'Pablo Martín Páez Gavira',  # Gavi
    'Leny Yoro',

    # Rising/ proven Stars: Julián Álvarez, Cole Palmer, Jude Bellingham, Artem Dovbyk, Pedri
    'Pedro González López',  # 251854 Pedri
    'Erling Braut Haaland',  # 239085
    'Jude Victor William Bellingham',  # 252371
    'Julián Álvarez',  # 246191
    'Florian Richard Wirtz',  # 256630
    'Eduardo Celmi Camavinga',
    'Cole Palmer',  # 257534
    'Jeremie Agyekum Frimpong',  # 253149
    'Moisés Isaac Caicedo Corozo',  # 256079
    'Jamal Musiala',  # 256790
    'Artem Dovbyk',  # 242458
    'Patrik Schick',  # 234236
    'Declan Rice',  # 234378
    'Bukayo Saka',  # 246669
    'Ollie Watkins',  # 221697,
    'Alexis Mac Allister',  # 239837
    'Jules Koundé',  # 241486
    'Vítor Machado Ferreira',  # Vitinha
    'Alexander Isak',
    'Alexander Sørloth',  # 216549
    'Mohamed Salah Ghaly',  # 209331
    'Rafael Alexandre Conceição Leão',  # 241721
    'Federico Santiago Valverde Dipetta',  # 239053
    'Rodrygo Silva de Goes',  # 243812
    'Victor James Osimhen',  # 232293
    'Khvicha Kvaratskhelia',  # 247635
    'Ousmane Dembélé',  # 231443
    'Achraf Hakimi Mouh',  # 235212
    'Dominik Szoboszlai',  # 236772
    'Frenkie de Jong',  # 228702

    # Can go either way: Mason Mount, Ansu Fati, Giovanni Reyna
    'Anssumane Fati',  # 253004
    'Edward Nketiah',  # 236988
    'Gonçalo Matias Ramos',
    'Ikoma Loïs Openda',  # 243580
    'Jack Grealish',  # 206517
    'Conor Gallagher',  # 238216
    'Jadon Sancho',  # 233049 Jadon Malik Sancho
    'Federico Chiesa',  # 235805
    'Brahim Abdelkader Díaz',  # 231410
    'Ferran Torres García',  # 241461
    'Xavi Quentin Shay Simons',  # 245367
    'Marcus Lilian Thuram-Ulien',  # 228093
    'Marcus Rashford',  # 231677
    'Mason Mount',  # 233064
    'Ronald Federico Araújo da Silva',  # 253163
    'Benjamin White',  # 231936

    # Peak-Point Players: Casemiro, Neymar, Ciro Immobile
    'Carlos Henrique Venancio Casimiro',  # 200145
    'Thibaut Nicolas Marc Courtois',  # 192119
    'Christopher Nkunku',  # 232411
    'Riyad Mahrez',  # 204485
    'Kingsley Junior Coman',  # 213345
    'Neymar da Silva Santos Júnior',  # 190871
    'Antoine Griezmann',  # 194765
    'Raheem Sterling',  # 202652
    'Lorenzo Insigne',  # 198219
    'Ciro Immobile',  # 192387
    'Raphaël Varane',  # 201535

    # Late Bloomers: Einar Gyökeres, Dani Olmo (pre-Barcelona transfer)
    'Viktor Einar Gyökeres',  # 241651
    'Damián Emiliano Martínez',  # 202811
    'Daniel Olmo Carvajal',  # 244260
    'Andreas Bødtker Christensen',  # 213661
    'Serhou Yadaly Guirassy',  # 215441
    'Tijjani Reijnders',  # 240638
    'Exequiel Alejandro Palacios',  # 231521

    # Misc
    'Cody Mathès Gakpo',  # 242516
    'Trent Alexander-Arnold',  # 231281
    'Jérémy Doku',  # 246420
    'Nathan Aké',  # 208920
    'Matthijs de Ligt',  # 235243
    'Raphael Dias Belloli',  # 233419 Raphinia
]
SELECTED_PLAYERS_OF_INTEREST = [
    'Leny Yoro',
    'João Pedro Gonçalves Neves'
    'Nathan Aké',  # 208920
    'Jamal Musiala',
    'Pedro González López',  # 251854 Pedri
    'Jadon Sancho',  # 233049 Jadon Malik Sancho
    'Erling Braut Haaland',  # 239085
    'Mohamed Salah Ghaly',  # 209331
    'Cody Mathès Gakpo',  # 242516
    'Roméo Lavia',  # 263620
    'Marcus Rashford',  # 231677
    'Cole Palmer',  # 257534
    'Artem Dovbyk',  # 242458
    'Viktor Einar Gyökeres',  # 241651
    'Khvicha Kvaratskhelia',  # 247635
    'Victor Okoh Boniface',  # 247679
    'Benjamin Šeško',  # 260592
    'Ollie Watkins',  # 221697,
    'Serhou Yadaly Guirassy',  # 215441
    'Carlos Henrique Venancio Casimiro',  # 200145
    'Bukayo Saka',  # 246669
    'Rodrygo Silva de Goes',  # 243812
    'Jude Victor William Bellingham',  # 252371
    'Désiré Doué',  # 271421
    'Neymar da Silva Santos Júnior',  # 190871
    'Mason Mount',  # 233064
    'Moisés Isaac Caicedo Corozo',  # 256079
    'Julián Álvarez',  # 246191
    'Sávio Moreira de Oliveira',  # 270409 Savinho
    'Jamie Bynoe-Gittens',  # 266032
    'Alejandro Garnacho Ferreyra',  # 268438
    'Pedro González López',  # 251854
    'Florian Richard Wirtz',  # 256630
    'Giovanni Alejandro Reyna',
    'Francisco Fernandes Conceição',
    'Nicholas Williams Arthuer',  # 256516
    'Anssumane Fati',  # 253004
    'Bradley Barcola',
    'Kamaldeen Sulemana',
    'Bukayo Saka',
    'Eduardo Celmi Camavinga',
    'Alejandro Garnacho Ferreyra',  # 268438
    'Rasmus Winther Højlund'
]
SELECTED_PLAYERS_U21 = [
    'Alejandro Balde Martínez',  # 263578
    'Carney Chibueze Chukwuemeka',
    'Alberto Moleiro González',
    'Pablo Martín Páez Gavira',  # Gavi
    'Zeno Debast',
    'Alejandro Garnacho Ferreyra',  # 268438
    'Arda Güler',  # 264309
    'Mathys Tel',
    'Sávio Moreira de Oliveira',  # 270409 Savinho
    'António João Pereira de Albuquerque Tavares da Silva',
    'Cristian Andrey Mosquera Ibarguen',
    'Pedro González López',  # 251854
    'Jude Victor William Bellingham',  # 252371
    'Florian Richard Wirtz',  # 256630
    'Giovanni Alejandro Reyna',
    'Francisco Fernandes Conceição',
    'Nicholas Williams Arthuer',  # 256516
    'Xavi Quentin Shay Simons',  # 245367
    'Anssumane Fati',  # 253004
    'Bradley Barcola',
    'Yan Bueno Couto',
    'Riccardo Calafiori',
    'Jamal Musiala',
    'Alejandro Garnacho Ferreyra',  # 268438
    'Eduardo Celmi Camavinga',
    'Bukayo Saka',  # 246669
    'Lucas Chevalier',
    'Moisés Isaac Caicedo Corozo',  # 256079
    'Michael Olise',
    'Karim-David Adeyemi',
    'lyenoma Destiny Udogie',
    'Guillaume Restes',
    'Mason Greenwood',
    'Désiré Doué',  # 271421
    'Jamie Bynoe-Gittens'
]
FIFA24_FUTURE_STARS = [
    'Jérémy Doku',  # 246420
    'Arda Güler',  # 264309
    'Alejandro Garnacho Ferreyra',  # 268438
    'Victor Okoh Boniface',  # 247679
    'Alejandro Balde Martínez',  # 263578
    'Pablo Barrios Rivas',
    'Junior Castello Lukeba',
    'Yann Aurel Ludger Bisseck',
    'Harvey Elliott',
    'Youssoufa Moukoko',
    'Rasmus Winther Højlund',
    'Nicholas Williams Arthuer',  # 256516
    'Cole Palmer',  # 257534
    'Joshua Orobosa Zirkzee',
    'Rico Lewis'
]

PLAYERS_NAMES_CONVERTER = {
    'Benjamin Šeško': 'Šeško',
    'Mohamed Salah Ghaly': 'M.Salah',
    'Erling Braut Haaland': 'E.Haaland',
    'Eduardo Celmi Camavinga': 'Camavinga',
    'Anssumane Fati': 'Ansu Fati',
    'Serhou Yadaly Guirassy': 'Guirassy',
    'Jamal Musiala': 'J.Musiala',
    'Alejandro Balde Martínez': 'A.Balde',
    'Alberto Moleiro González': 'Alberto Moleiro',
    'Pablo Martín Páez Gavira': 'Gavi',
    'Carlos Henrique Venancio Casimiro': 'Casimiro',
    'Rodrygo Silva de Goes': 'Rodrygo',
    'Neymar da Silva Santos Júnior': 'Neymar',
    'Nicholas Williams Arthuer': 'Nico Williams',
    'Jude Victor William Bellingham': 'J.Bellingham',
    'Bradley Barcola': 'B.Barcola',
    'Alejandro Garnacho Ferreyra': 'A.Garnacho',
    'Moisés Isaac Caicedo Corozo': 'Moisés Caicedo',
    'Jamie Bynoe-Gittens': 'J.B-Gittens',
    'Khvicha Kvaratskhelia': 'Kvaratskhelia',
    'Marcus Rashford': 'M.Rashford',
    'Sávio Moreira de Oliveira': 'Savinho',
    'Pedro González López': 'Pedri'
}


def get_model_short_name(full_name: str) -> str:
    base_name = full_name.split('yback')[0]
    return base_name.split('-')[0][:-1].replace('_', ' ') + base_name[-1].replace('_', '') + 'years'


def get_recall_at_precisions(df: pd.DataFrame,
                             label_col: str = 'label',
                             pred_col: str = 'pred',
                             prob_col: str = 'prob',
                             precision_thresholds: List[float] = None) -> Dict:
    if precision_thresholds is None:
        precision_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    results = defaultdict(dict)
    for label_value in [0, 1]:
        # Filter to relevant rows — either labeled as class or predicted as class
        sub_df = df[(df[label_col] == label_value) | (df[pred_col] == label_value)].copy()

        # Define probability thresholds to test
        if label_value == 1:
            all_probs = sorted(set(sub_df.loc[sub_df[prob_col] >= 0.5, prob_col].round(4).tolist()), reverse=False)
        else:
            all_probs = sorted(set(sub_df.loc[sub_df[prob_col] < 0.5, prob_col].round(4).tolist()), reverse=True)

        for t in precision_thresholds:
            found = False
            for p in all_probs:
                # Make prediction based on current probability threshold
                preds = sub_df[prob_col].apply(lambda x: int(x >= p))

                # Predicted positives are defined as those with pred == 1
                tp = ((preds == label_value) & (sub_df[label_col] == label_value)).sum()
                fp = ((preds == label_value) & (sub_df[label_col] != label_value)).sum()
                total_actual = (sub_df[label_col] == label_value).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / total_actual if total_actual > 0 else 0

                if precision >= t:
                    results[label_value][int(100 * t)] = {'precision': precision, 'recall': recall}
                    found = True
                    break

            if not found:
                results[label_value][int(100 * t)] = {'precision': None, 'recall': None}

    return results


def plot_calibration_curve(probs,
                           y_true,
                           res: float = 0.01,
                           model_name: str = 'model',
                           output_dir: str = ArtifactsPaths.ARTIFACTS_DIR,
                           show_plots: bool = False,
                           export_plots: bool = False
                           ):
    """
    Plots a calibration curve using Plotly.

    Parameters:
        probs (array-like): Predicted probabilities.
        y_true (array-like): True binary labels (0 or 1).
        res (float): Resolution of probability bins (e.g., 0.01).
        model_name: name of model
        output_dir: directory to save the figure
    """
    # Create probability bins
    bins = np.arange(0, 1 + res, res)
    bin_centers = bins[:-1] + res / 2

    # Digitize probabilities into bins
    bin_indices = np.digitize(probs, bins) - 1  # Adjust to 0-based index
    bin_indices[bin_indices == len(bins) - 1] = len(bins) - 2  # Edge case

    # Compute true label proportions and counts per bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    true_counts = np.bincount(bin_indices, weights=y_true, minlength=len(bins) - 1)

    # Avoid division by zero
    valid_bins = bin_counts > 0
    prob_true = np.zeros_like(bin_counts, dtype=float)
    prob_true[valid_bins] = true_counts[valid_bins] / bin_counts[valid_bins]

    # Filter out empty bins
    scatter_x = bin_centers[valid_bins]
    scatter_y = prob_true[valid_bins]
    scatter_counts = bin_counts[valid_bins]

    # Create scatter plot with hover info
    scatter = go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='markers+lines',
        marker=dict(size=8, color='blue'),
        text=[f'Count: {c}' for c in scatter_counts],
        hoverinfo="text+x+y",
        name=f"Calibration Curve: {model_name}"
    )

    # Create histogram of probabilities
    hist = go.Bar(
        x=bin_centers,
        y=bin_counts,
        width=res * 0.9,
        marker=dict(color='gray', opacity=0.25),
        name="Probability Distribution",
        yaxis="y2",
    )

    line = go.Scatter(
        x=np.arange(0, 1.05, 0.05),
        y=np.arange(0, 1.05, 0.05),
        mode='markers+lines',
        name="perfect calibration",
        line=dict(color='dimgrey', width=4, dash='dash')
    )

    # Define layout
    layout = go.Layout(
        template="plotly_white",
        title="Calibration Curve",
        xaxis=dict(title="Predicted Probability"),
        yaxis=dict(title="True Label Proportion", range=[0, 1]),
        yaxis2=dict(title="Count", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.8, y=1.2),
        width=1600,
        height=1000
    )

    # Create figure
    fig = go.Figure(data=[scatter, hist, line], layout=layout)
    fig.update_layout(
        barmode="group",
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.)
    )
    if export_plots:
        short_model_name = get_model_short_name(model_name)
        fig.write_html(os.path.join(output_dir, f'{short_model_name}_calibration_curve.html'))
    if show_plots:
        fig.show()


def cls_metrics(preds: List[int],
                labels: List[int],
                probs: List[float],
                prefix: str = '',
                model_name: str = '',
                metrics: dict = None,
                conf_matrix: bool = False,
                prints: bool = True) -> Dict:
    if metrics is None:
        metrics = {}

    metrics[f'{prefix}count'] = len(preds)

    if conf_matrix:
        cm = confusion_matrix(labels, preds)
        cm = pd.DataFrame(cm, columns=['pred 0', 'pred 1'], index=['true 0', 'true 1'])
        cm_norm = np.round(confusion_matrix(labels, preds, normalize='true'), 2)
        cm_norm = pd.DataFrame(cm_norm, columns=['pred 0', 'pred 1'], index=['true 0', 'true 1'])
        if prints:
            print(f'\n{model_name} {prefix} confusion matrix\n{cm}')
            print(f'\n\n{model_name} {prefix} confusion matrix normed\n{cm_norm}\n')
        metrics[f'{prefix}confusion_matrix'] = cm
    else:
        metrics[f'{prefix}confusion_matrix'] = pd.Series(labels).value_counts()
        if prints:
            print('y Value counts')
            print(metrics[f'{prefix}confusion_matrix'])
            print('y Value counts - normed')
            print(pd.Series(labels).value_counts(normalize=True))

    try:
        metrics[f'{prefix}auc'] = round(roc_auc_score(labels, probs), 4)
    except ValueError:
        # one class appears in the predictions / labels
        pass

    cls_report = classification_report(labels, preds, zero_division=0, output_dict=True)
    for cls in ['0', '1']:
        if cls in cls_report:
            metrics[f'{prefix}class_{cls}_recall'] = cls_report[cls]['recall']
            metrics[f'{prefix}class_{cls}_precision'] = cls_report[cls]['precision']
            metrics[f'{prefix}class_{cls}_f1'] = cls_report[cls]['f1-score']
            metrics[f'{prefix}class_{cls}_support'] = cls_report[cls]['support']
        else:
            metrics[f'{prefix}class_{cls}_recall'] = None
            metrics[f'{prefix}class_{cls}_precision'] = None
            metrics[f'{prefix}class_{cls}_f1'] = None
            metrics[f'{prefix}class_{cls}_support'] = 0

    # Recall@Precision
    metrics['Recall@Precision'] = get_recall_at_precisions(pd.DataFrame(
        {'pred': preds, 'label': labels, 'prob': probs})
    )
    return metrics
