This is an experiment to see if [dedupe][] can help us
deduplicate landlord information.

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

[dedupe]: https://github.com/dedupeio/dedupe
