import io
import re
import tokenize
from typing import List, Tuple

from .data.query_data import QueryData


class UnexpectedTokenException(Exception):
    def __init__(self, expected_token: str, actual_token: str) -> None:
        self.expected_token = expected_token
        self.actual_token = actual_token


def parse_insert_query(query_data: QueryData) -> QueryData:
    """Parses INSERT-queries and extracts the table, columns, values and column types if present
    Parameter query needs to be lowercase"""
    # TODO: '$' in table/column name doesn't work
    # INSERT INTO Cities (Name, Country) VALUES ("Paris", "France");
    # INSERT INTO Cities VALUES ("Paris", "France"), ("Berlin", "Germany");
    # INSERT INTO (Name, Country) VALUES ("Paris", "France");
    # INSERT INTO VALUES ("Paris", "France");
    tokens = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(query_data.query).readline)
    ]
    if tokens[0].upper() != "INSERT":
        raise UnexpectedTokenException("INSERT", tokens[0])
    if tokens[1].upper() != "INTO":
        raise UnexpectedTokenException("INTO", tokens[1])

    index = 2
    if tokens[index].upper() != "VALUES" and tokens[index] != "(":
        query_data.table, index = parse_table(tokens, index)
    if tokens[index] == "(":
        query_data.columns, index = parse_columns(tokens, index)
    if tokens[index].upper() != "VALUES":
        raise UnexpectedTokenException("VALUES", tokens[index])
    query_data.values, query_data.column_types, index = parse_values(tokens, index + 1)

    if index < len(tokens) and tokens[index] != ";" and "".join(tokens[index:]) != "":
        raise UnexpectedTokenException(";", tokens[index])

    # All tokens after the ';' are ignored

    return query_data


def parse_table(tokens: List[str], index: int) -> Tuple[str, int]:
    return (tokens[index], index + 1)


def parse_columns(tokens: List[str], index: int) -> Tuple[List[str], int]:
    if tokens[index] != "(":
        raise UnexpectedTokenException("(", tokens[index])

    columns = []
    columns.append(tokens[index + 1])
    index += 2

    while tokens[index] != ")":
        if tokens[index] == ",":
            columns.append(tokens[index + 1])
            index += 2
        else:
            columns[-1] = columns[-1] + tokens[index]
            index += 1

    return (columns, index + 1)


def parse_values(
    tokens: List[str], index: int
) -> Tuple[List[List[str]], List[str], int]:
    values = []
    row_values, column_types, index = parse_values_for_row(tokens, index)
    values.append(row_values)

    while tokens[index] == ",":
        row_values, index = parse_values_for_row(tokens, index + 1)
        values.append(row_values)

    return (values, column_types, index)


def parse_values_for_row(
    tokens: List[str], index: int
) -> Tuple[List[str], List[str], int]:
    if tokens[index] != "(":
        raise UnexpectedTokenException("(", tokens[index])

    values, column_types = [], []

    value, column_type, index = parse_single_value(tokens, index + 1)
    values.append(value)
    column_types.append(column_type)

    while tokens[index] == ",":
        value, column_type, index = parse_single_value(tokens, index + 1)
        values.append(value)
        column_types.append(column_type)

    if tokens[index] != ")":
        raise UnexpectedTokenException(")", tokens[index])

    return (values, column_types, index + 1)


def parse_single_value(tokens: List[str], index: int) -> Tuple[str, str, int]:
    # Replace call
    if tokens[index] == "replace":
        value, index = parse_replace_function(tokens, index + 1)
        return ("replace" + value, "VARCHAR(1023)", index)

    # Negative numbers
    if tokens[index] == "-":
        return ("-" + tokens[index + 1], get_column_type(tokens[index + 1]), index + 2)

    # Double ' or " is an escape sequence (e.g. 'O''Brian' -> O'Brian)
    if tokens[index][0] == "'" or tokens[index][0] == '"':
        double_escape_char = tokens[index][0]
        value = tokens[index]
        while tokens[index + 1][0] == double_escape_char:
            index += 1
            value += tokens[index]
        return (value, get_column_type(value), index + 1)

    # All other values
    return (tokens[index], get_column_type(tokens[index]), index + 1)


def parse_replace_function(tokens: List[str], index: int) -> Tuple[str, int]:
    if tokens[index] == "(":
        value, index = parse_replace_function_helper(tokens, index + 1, 1)
        return ("(" + value, index)
    else:
        raise UnexpectedTokenException("(", tokens[index])


def parse_replace_function_helper(
    tokens: List[str], index: int, open_parentheses: int
) -> Tuple[str, int]:
    if open_parentheses == 0:
        return ("", index)

    value, end_index = parse_replace_function_helper(
        tokens,
        index + 1,
        (
            open_parentheses + 1
            if tokens[index] == "("
            else open_parentheses - 1 if tokens[index] == ")" else open_parentheses
        ),
    )
    return (tokens[index] + value, end_index)


def get_column_type(value: str) -> str:
    """Computes the required column type for the given value"""
    if re.match(
        r"[\"\'][0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}[\"\']", value
    ):
        return "DATETIME"
    elif re.match(r"[\"\'][0-9]{4}-[0-9]{2}-[0-9]{2}[\"\']", value):
        return "DATE"
    elif value.startswith('"') or value.startswith("'") or value == "NULL":
        return "VARCHAR(1023)"
    elif "." in value:
        return "DOUBLE"
    else:
        return "BIGINT"
