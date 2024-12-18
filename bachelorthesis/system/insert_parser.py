import io
import logging
import re
import tokenize

from .data.insert_data import InsertData


class UnexpectedTokenException(Exception):
    def __init__(
        self,
        expected_token: str,
        actual_token: str,
        tokens: list[str],
        context: list[str],
    ) -> None:
        self.expected_token = expected_token
        self.actual_token = actual_token
        self.tokens = tokens
        self.context = context


def unexpected_token_encountered(
    tokens: list[str], expected_token: str, index: int
) -> None:
    raise UnexpectedTokenException(
        expected_token, tokens[index], tokens, tokens[index - 2 : index + 3]
    )


def parse_insert(insert_data: InsertData) -> InsertData:
    """Parses INSERTs and extracts the table, columns, values and column types if present"""
    # TODO: '$' in table/column name doesn't work
    # INSERT INTO Cities (Name, Country) VALUES ("Paris", "France");
    # INSERT INTO Cities VALUES ("Paris", "France"), ("Berlin", "Germany");
    # INSERT INTO (Name, Country) VALUES ("Paris", "France");
    # INSERT INTO VALUES ("Paris", "France");
    try:
        tokens = [
            token.string
            for token in tokenize.generate_tokens(
                io.StringIO(insert_data.insert).readline
            )
            if token.string.strip() != ""
        ]
    except Exception as e:
        logging.getLogger(__name__).error(
            "Could not parse insert: " + insert_data.insert
        )
        raise e

    if tokens[0].upper() != "INSERT":
        unexpected_token_encountered(tokens, "INSERT", 0)
    if tokens[1].upper() != "INTO":
        unexpected_token_encountered(tokens, "INTO", 1)

    index = 2
    if tokens[index].upper() != "VALUES" and tokens[index] != "(":
        insert_data.table, index = parse_table(tokens, index)
    if tokens[index] == "(":
        insert_data.columns, index = parse_columns(tokens, index)
    if tokens[index].upper() != "VALUES":
        unexpected_token_encountered(tokens, "VALUES", index)
    insert_data.values, insert_data.column_types, index = parse_values(
        tokens, index + 1
    )

    if index < len(tokens) and tokens[index] != ";" and "".join(tokens[index:]) != "":
        unexpected_token_encountered(tokens, ";", index)

    # All tokens after the ';' are ignored

    return insert_data


def parse_table(tokens: list[str], index: int) -> tuple[str, int]:
    return parse_identifier(tokens, index)


def parse_columns(tokens: list[str], index: int) -> tuple[list[str], int]:
    if tokens[index] != "(":
        unexpected_token_encountered(tokens, "(", index)

    columns = []
    column, index = parse_identifier(tokens, index + 1)
    columns.append(column)

    while tokens[index] != ")":
        column, index = parse_identifier(tokens, index + 1)
        columns.append(column)

    return (columns, index + 1)


def parse_identifier(tokens: list[str], index: int) -> tuple[str, int]:
    if tokens[index] == "`":
        identifier = tokens[index + 1]
        index += 2
        while tokens[index] != "`":
            identifier += tokens[index]
            index += 1
        return (identifier, index + 1)

    identifier = tokens[index]
    index += 1
    while tokens[index].upper() != "VALUES" and re.match(
        r"[A-Za-z0-9_$]+", tokens[index]
    ):
        identifier += tokens[index]
        index += 1
    return (identifier, index)


def parse_values(
    tokens: list[str], index: int
) -> tuple[list[list[str]], list[str], int]:
    values = []
    row_values, column_types, index = parse_values_for_row(tokens, index)
    values.append(row_values)

    while index < len(tokens) and tokens[index] == ",":
        row_values, index = parse_values_for_row(tokens, index + 1)
        values.append(row_values)

    return (values, column_types, index)


def parse_values_for_row(
    tokens: list[str], index: int
) -> tuple[list[str], list[str], int]:
    if tokens[index] != "(":
        unexpected_token_encountered(tokens, "(", index)

    values, column_types = [], []

    value, column_type, index = parse_single_value(tokens, index + 1)
    values.append(value)
    column_types.append(column_type)

    while tokens[index] == ",":
        value, column_type, index = parse_single_value(tokens, index + 1)
        values.append(value)
        column_types.append(column_type)

    if tokens[index] != ")":
        unexpected_token_encountered(tokens, ")", index)

    return (values, column_types, index + 1)


def parse_single_value(tokens: list[str], index: int) -> tuple[str, str, int]:
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


def parse_replace_function(tokens: list[str], index: int) -> tuple[str, int]:
    if tokens[index] == "(":
        value, index = parse_replace_function_helper(tokens, index + 1, 1)
        return ("(" + value, index)
    else:
        unexpected_token_encountered(tokens, "(", index)


def parse_replace_function_helper(
    tokens: list[str], index: int, open_parentheses: int
) -> tuple[str, int]:
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
