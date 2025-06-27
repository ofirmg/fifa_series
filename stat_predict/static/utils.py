class COLUMNS:
    POTENTIAL = 'potential'
    AGE = 'age'
    PLAYER_ID = 'player_id'  # 'sofifa_id'
    PLAYER_NAME = 'short_name'
    PLAYER_FULL_NAME = 'long_name'
    PLAYER_VALUE = 'value_eur'
    PLAYER_WAGE = 'wage_eur'
    PLAYER_POSITION = 'club_position'
    LEAGUE = 'league_name'
    LEAGUE_LEVEL = 'league_level'
    NATIONALITY = 'nationality_name'
    CLUB = 'club_name'
    POTENTIAL_PERCENT = 'potential-overall-%'
    CAREER_PHASE = 'career_phase'
    GROW_RATE = 'grow_rate'
    PLAYER_RATING = 'overall'
    YEAR = 'year'
    CONTRACT_EXPIRATION_DATE = 'club_contract_valid_until'
    JOIN_CLUB_DATE = 'club_joined_date'
    WORK_RATE = 'work_rate'


# vertical, horizontal
# vertical (gk, def, def-mid, mid, mid-attack, attack): -1.5, -1, -0.5, 0, 0.5, 1
# horizontal (left, mid, right): -1, 0, 1
positions_emb = {
    'CDM': (-0.5, 0),
    'CM': (0, 0),
    'RCM': (0, 0),
    'LCM': (0, 0),
    'RDM': (0, 0),
    'LDM': (0, 0),
    'CAM': (.5, 0),
    'RAM': (.5, 0),
    'LAM': (.5, 0),
    'LM': (0, -1),
    'RM': (0, 1),
    'RW': (.75, 1),
    'LW': (.75, -1),
    'CF': (1, 0),
    'ST': (1, 0),
    'RS': (1, 1),
    'LS': (1, -1),
    'LF': (1, -1),
    'RF': (1, 1),
    'RB': (-1, 1),
    'RWB': (-.5, 1),
    'LB': (-1, -1),
    'LWB': (-.5, -1),
    'LCB': (-1, 0),
    'RCB': (-1, 0),
    'CB': (-1, 0),
    'GK': (-1.5, 0),
}
SPECIFIC_POS_COLS = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm',
                     'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
                     'LWB Rating', 'RWB Rating']
REMOVE_COLS = ['player_url',
               'dob',
               'club_jersey_number',
               'nation_team_id',
               'nation_jersey_number',
               'real_face',
               'player_tags',
               'player_traits',
               'nation_flag_url',
               'nation_logo_url',
               'club_flag_url',
               'player_face_url',
               'club_logo_url',
               'Best Position',
               'years_left_in_contract',
               'work_rate_def',
               'work_rate_attack',
               'age_trans',
               '',
               'update_as_of'
               ]
work_rate_mapper = {'Medium': 2, 'Low': 1, 'High': 3}
IDENTIFIERS = [COLUMNS.PLAYER_ID, 'long_name', 'short_name', 'Player position', 'index']
METADATA_COLS = ['club_position',
                 'preferred_foot',
                 'body_type',
                 'nationality_id',
                 'nationality_name',
                 'club_loaned_from',
                 'player_positions',
                 'league_name',
                 'club_name',
                 'club_team_id']
SKILLS_PREFIXES = ['skill_', 'attacking_', 'movement_', 'power_', 'mentality_', 'defending_', 'goalkeeping_']
major_attrs = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'overall', 'potential']
body_types = ['Lean (170-185)',
              'Normal (170-185)',
              'Normal (170-)',
              'Stocky (170-)',
              'Lean (185+)',
              'Normal (185+)',
              'Stocky (170-185)',
              'Lean (170-)',
              'Stocky (185+)',
              'Unique']


# Defining time-series columns for simple time-series-flat representation change
class TS_COLS(COLUMNS):
    @classmethod
    def add_suffix(cls, suffix: str):
        # Loop over class attributes
        for name in dir(cls):
            # Get the actual attribute value
            value = getattr(cls, name)
            # Only apply suffix if it's a string and not a special attribute
            if isinstance(value, str) and not name.startswith('__'):
                setattr(cls, name, value + suffix)
        return cls


ts_cols_suffix = '_row_0'
TS_COLS.add_suffix(ts_cols_suffix)
