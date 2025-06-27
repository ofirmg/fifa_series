class COLUMNS:
    PLAYER_ID = 'sofifa_id'
    PLAYER_NAME = 'short_name'
    PLAYER_FULL_NAME = 'long_name'
    PLAYER_VALUE = 'value_eur'
    PLAYER_POSITION = 'club_position'
    POTENTIAL_RATIO = 'potential-overall-ratio'
    POTENTIAL_PERCENT = 'potential-overall-%'
    CAREER_PHASE = 'career_phase'
    GROW_RATE = 'grow_rate'
    PLAYER_RATING = 'overall'
    EUR_PER_RATING = 'euro_per_rating'
    EUR_PER_RATING_NORMED = 'euro_per_rating/500'


positions_mapper = {'CDM': 'CM',
                    'CM': 'CM',
                    'RCM': 'CM',
                    'LCM': 'CM',
                    'RDM': 'CM',
                    'LDM': 'CM',
                    'CAM': 'AM',
                    'RAM': 'AM',
                    'LAM': 'AM',
                   'LM': 'Winger',
                    'RM': 'Winger',
                    'RW': 'Winger',
                    'LW': 'Winger',
                    'CF': 'ST',
                    'ST': 'ST',
                    'RS': 'ST',
                    'LS': 'ST',
                    'LF': 'ST',
                    'RF': 'ST',
                    'RB': 'Full-back',
                    'RWB': 'Full-back',
                    'LB': 'Full-back',
                    'LWB': 'Full-back',
                    'LCB': 'CB',
                    'RCB': 'CB',
                    'CB': 'CB',
                    'GK': 'GK',
                   }

FONT_FAMILY = 'Lucida Grande'
ON_FIGURE_FONT_SIZE = 12
FONT = dict(family=FONT_FAMILY, size=18)
LABEL_FONT = dict(family=FONT_FAMILY, size=18)
TITLE_FONT = dict(family=FONT_FAMILY, size=20)