def parse_create_table(statement: str) -> tuple[str, list[str], list[str]]:
    """Get the name of the table and the column names and types from the statement"""
    # Get table name
    table_name_first_quote_index = statement.find("`")
    table_name_second_quote_index = statement.find(
        "`", table_name_first_quote_index + 1
    )
    table_name = statement[
        table_name_first_quote_index + 1 : table_name_second_quote_index
    ]

    # Get the column information needed
    attribute_start_index = statement.find("(")
    ignore_last_chars = 2 if statement.endswith(";") else 1
    attribute_data = [
        attribute.split()
        for attribute in statement[
            attribute_start_index + 1 : len(statement) - ignore_last_chars
        ].split(",\n")
    ]

    # Get the right data types and create the correct "CREATE TABLE" statement
    column_names = [
        attribute[0][1 : len(attribute[0]) - 1] for attribute in attribute_data
    ]
    column_types = [attribute[1] for attribute in attribute_data]

    return (table_name, column_names, column_types)
