# Running in Docker

```bash
docker compose build          # ~6 min: installs deps, builds the UI, generates data
docker compose up             # serves http://localhost:8000
```

The image ships with the dataset already generated but **no trained model**,
because training is a 20-minute step that belongs to the user, not the build.
Train once into the mounted `artifacts/` volume:

```bash
docker compose run --rm predictops python run_experiments.py
docker compose run --rm predictops python evaluate.py
docker compose up
```

Both write into `./artifacts` on the host, so the results survive rebuilds and
can be inspected directly.

Tests:

```bash
docker compose run --rm predictops python -m pytest
```
