"""LLM information-extraction layer. Converts documents into structured Evidence.

Rule: the LLM never invents quantitative inputs. Numbers are only extracted when
stated verbatim in the source; qualitative evidence informs priors / uncertainty.
"""
