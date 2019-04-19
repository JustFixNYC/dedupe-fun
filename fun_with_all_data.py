# This attempts to find duplicates just for one zip code.

# Much of this code is based on:
# https://github.com/dedupeio/dedupe-examples/blob/master/pgsql_big_dedupe_example/pgsql_big_dedupe_example.py

import os
import pickle
import psycopg2
from pathlib import Path
from psycopg2.extras import DictCursor
import numpy as np
import dedupe


MY_DIR = Path(__file__).parent.resolve()

SETTINGS_PATH = MY_DIR / 'all_data_learned_settings'

TRAINING_PATH = MY_DIR / 'all_data_training.json'

DUPES_PATH = MY_DIR / 'all_data_dupes.bin'

DATABASE_URL = os.environ.get(
    'DATABASE_URL', 'postgres://nycdb:nycdb@localhost/nycdb')

FIELDS = [
    {'field': 'contactdescription', 'type': 'String', 'has missing': True},
    {'field': 'corporationname', 'type': 'String', 'has missing': True},
    {'field': 'type', 'type': 'Categorical', 'categories': [
        'individualowner',
        'corporateowner',
        'agent',
        'headofficer',
        'officer',
        'shareholder',
        'sitemanager',
        'jointowner',
        'lessee'
    ]},
    {'field': 'title', 'type': 'String', 'has missing': True},
    {'field': 'firstname', 'type': 'String', 'has missing': True},
    {'field': 'middleinitial', 'type': 'String', 'has missing': True},
    {'field': 'lastname', 'type': 'String', 'has missing': True},
    {'field': 'businesshousenumber', 'type': 'ShortString', 'has missing': True},
    {'field': 'businessstreetname', 'type': 'String', 'has missing': True},
    {'field': 'businessapartment', 'type': 'String', 'has missing': True},
    {'field': 'businesscity', 'type': 'String', 'has missing': True},
    {'field': 'businessstate', 'type': 'String', 'has missing': True},
    {'field': 'businesszip', 'type': 'String', 'has missing': True}
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
            if fieldcfg['type'] == 'Categorical':
                if value not in fieldcfg['categories']:
                    raise AssertionError(value)
            if not has_missing and not value:
                raise AssertionError(field)
            drow[field] = value
        if drow.get('businesszip'):
            drow['businesszip'] = drow['businesszip'][:5]
        d[pk] = drow
    return d


def load_dict():
    print("Loading data from database...")
    with psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"select {', '.join(COLUMNS)} from hpd_contacts")
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


def get_the_freaking_minimum_index_and_score(scores):
    # Sometimes 'scores' is a tuple, sometimes it's an ndarray, this API is ridiculous
    minimum = scores[0]
    minimum_index = 0
    for i, score in enumerate(scores):
        if score < minimum:
            minimum = score
            minimum_index = i
    return minimum_index, minimum


def print_dict(d):
    dct = ' / '.join(
        filter(None, [d['contactdescription'], d['corporationname'], d['title']]))
    name = ' '.join(
        filter(None, [d['firstname'], d['middleinitial'], d['lastname']]))
    addr = ' '.join(
        filter(None, [
            d['businesshousenumber'], d['businessstreetname'], d['businessapartment'],
            d['businesscity'], d['businessstate'], d['businesszip']
        ]))
    print(f"  desc: {dct}")
    print(f"  name: {name}")
    print(f"  addr: {addr}")


def main():
    d_dict = load_dict()
    if SETTINGS_PATH.exists():
        print(f"Reading settings from {SETTINGS_PATH.name}.")
        with SETTINGS_PATH.open('rb') as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        deduper = label_and_train(d_dict)

    print("TODO: Do blocking stuff!")


if __name__ == '__main__':
    main()
