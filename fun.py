import os
import psycopg2
from pathlib import Path
from psycopg2.extras import DictCursor
import dedupe


MY_DIR = Path(__file__).parent.resolve()

SETTINGS_PATH = MY_DIR / 'learned_settings'

TRAINING_PATH = MY_DIR / 'training.json'

ZIPCODE = '11201'

DATABASE_URL = os.environ.get(
    'DATABASE_URL', 'postgres://nycdb:nycdb@localhost/nycdb')

FIELDS = [
    {'field': 'contactdescription', 'type': 'String', 'has missing': True},
    {'field': 'corporationname', 'type': 'String', 'has missing': True},
    {'field': 'title', 'type': 'String', 'has missing': True},
    {'field': 'firstname', 'type': 'String', 'has missing': True},
    {'field': 'middleinitial', 'type': 'String', 'has missing': True},
    {'field': 'lastname', 'type': 'String', 'has missing': True},
    {'field': 'businesshousenumber', 'type': 'String', 'has missing': True},
    {'field': 'businessstreetname', 'type': 'String'},
    {'field': 'businessapartment', 'type': 'String', 'has missing': True},
    {'field': 'businesscity', 'type': 'String', 'has missing': True},
    {'field': 'businessstate', 'type': 'String'},
    {'field': 'businesszip', 'type': 'Exact'}
]

PK_FIELD = 'registrationcontactid'

COLUMNS = [PK_FIELD, *[f['field'] for f in FIELDS]]


def to_dict(cur):
    d = {}
    for row in cur:
        pk = row[PK_FIELD]
        drow = {}
        for fieldcfg in FIELDS:
            field = fieldcfg['field']
            has_missing = fieldcfg.get('has missing', False)
            value = row[field]
            if value:
                value = value.lower().strip()
            if not has_missing and not value:
                raise AssertionError(field)
            drow[field] = value
        drow['businesszip'] = drow['businesszip'][:5]
        d[pk] = drow
    return d


def load_dict():
    print("Loading data from database...")
    with psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"select {', '.join(COLUMNS)} from hpd_contacts "
                f"where businesszip like '{ZIPCODE}%'")
            return to_dict(cur)


def label_and_train(d_dict):
    deduper = dedupe.Dedupe(FIELDS)

    print("Sampling data...")
    deduper.sample(d_dict, 15_000)

    if TRAINING_PATH.exists():
        with TRAINING_PATH.open('rb') as f:
            print(f"Loading trained examples from {TRAINING_PATH.name}.")
            deduper.readTraining(f)

    print("Starting active labeling...")
    dedupe.consoleLabel(deduper)

    deduper.train()

    with TRAINING_PATH.open('w') as tf:
        deduper.writeTraining(tf)
    
    with SETTINGS_PATH.open('wb') as sf:
        deduper.writeSettings(sf)


def main():
    d_dict = load_dict()
    if SETTINGS_PATH.exists():
        print(f"Reading settings from {SETTINGS_PATH.name}.")
        with SETTINGS_PATH.open('rb') as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        deduper = label_and_train(d_dict)


if __name__ == '__main__':
    main()
