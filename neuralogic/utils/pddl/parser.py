import re
from typing import Any


def tokenize(text: str) -> list[str]:
    """Tokenize PDDL text into a list of strings and parentheses."""
    # Strip comments
    text = re.sub(r";.*", "", text)
    # Handle multi-line strings/names and parentheses
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(tokens: list[str]) -> Any:
    """Parse a list of tokens into a nested list structure (S-Expression)."""
    if not tokens:
        return None
    token = tokens.pop(0)
    if token == "(":
        lst = []
        while tokens and tokens[0] != ")":
            lst.append(parse_sexpr(tokens))
        if tokens:
            tokens.pop(0)  # Remove ')'
        return lst
    return token


class PDDLReader:
    """Utility class to read and parse PDDL files into S-Expressions."""

    @staticmethod
    def read(file_path: str) -> Any:
        """Reads a PDDL file and returns its S-Expression representation."""
        with open(file_path, "r") as f:
            content = f.read()
        tokens = tokenize(content)
        return parse_sexpr(tokens)

    @staticmethod
    def parse_string(content: str) -> Any:
        """Parses a PDDL string into its S-Expression representation."""
        tokens = tokenize(content)
        return parse_sexpr(tokens)
