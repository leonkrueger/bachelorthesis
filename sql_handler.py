class SQLHandler:
    def __init__(self, conn) -> None:
        self.conn = conn
        
    def execute_query(
        self, query: str, params: tuple[str, ...] = ()
    ) -> tuple[list[tuple[str | int | float, ...]], int]:
        # Executes the query with the given parameters on the database
        # Returns the result of the query and the value of the automatically incremented id
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        output = cursor.fetchall()
        auto_increment_id = cursor.lastrowid
        self.conn.commit()
        cursor.close()
        return output, auto_increment_id