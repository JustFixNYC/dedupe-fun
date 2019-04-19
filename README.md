This is an experiment to see if [dedupe][] can help us
deduplicate HPD registration contacts.

## Quick start

In one terminal, start the database server:

```
docker-compose up
```

(Alternatively, you can use a local postgres instance.)

Then in another terminal, run:

```
python3 -m venv venv
source venv/bin/activate    # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
nycdb --download hpd_registrations
nycdb --load hpd_registrations
```

To find the duplicates in one zip code, run:

```
python fun_with_one_zipcode.py
```

[dedupe]: https://github.com/dedupeio/dedupe
