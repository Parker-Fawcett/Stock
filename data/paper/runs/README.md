# Prospective run manifests

Every `prospective-v2` prediction writes one immutable JSON manifest here.
The corresponding rows in `../log_v2.csv` store its SHA-256 hash. Grading
verifies that hash before using a prediction.

The older `../log.csv` has no reliable creation timestamps or manifests and
remains a separate historical series.
