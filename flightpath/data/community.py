"""Customer and employee text: reviews, forums, social - unstructured.

Sources (check each site's terms and API access):
  - Reddit r/delta, r/awardtravel (archives exist -> backtestable)
  - FlyerTalk Delta forums (upgrade clearance talk = premium demand signal)
  - App store reviews of the Fly Delta app
  - Employee reviews (Glassdoor-type) for labor-stress signal - optional

Raw posts -> extraction layer (OpenAI batch classification) ->
daily indices in extraction/text_signals.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import DataSource


class RedditSource(DataSource):
    name = "reddit"
    default_lag = timedelta(hours=1)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError


class ForumSource(DataSource):
    name = "flyertalk"
    default_lag = timedelta(hours=1)

    def fetch(self, start: date, end: date) -> None:
        raise NotImplementedError

    def parse(self) -> pd.DataFrame:
        raise NotImplementedError
