import os
from typing import Any, Optional, Sequence
from collections import OrderedDict
from rich.table import Table
from rich.console import Console


def get_user_input(text: str, choices: Optional[Sequence[Any]]) -> str:
    from InquirerPy import inquirer
    import termios
    import sys

    termios.tcflush(sys.stdin, termios.TCIOFLUSH)

    if choices:
        return inquirer.select(text, choices=choices).execute()
    else:
        return inquirer.text(text).execute()


class LruCache(OrderedDict):
    def __init__(self, max_size: int = 100) -> None:
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key: Any, value: Any) -> None:
        if len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)


def output_table(
    rows: list[dict[str, Any]],
    markdown_file: Optional[str] = None,
    title: Optional[str] = None,
    column_names: Optional[list[str]] = None,
):
    """Outputs a table to the terminal and writes it as markdown to a file if markdown_file is provided."""
    if column_names is None:
        column_names = []
        for row in rows:
            for column_name in row.keys():
                if column_name not in column_names:
                    column_names.append(column_name)
        column_names.sort()

    table = Table(title=title)
    md_data = f"| {' | '.join(column_names)} |\n"
    md_data += "| " + " | ".join(["---"] * len(column_names)) + " |\n"

    for column_name in column_names:
        table.add_column(column_name)

    for row in rows:
        column_values = [row.get(column_name, "") for column_name in column_names]
        column_value_strs = [
            f"{column_value: .2f}" if type(column_value) == float else str(column_value)
            for column_value in column_values
        ]
        table.add_row(*column_value_strs)
        md_data += f"| {' | '.join(column_value_strs)} |\n"

    console = Console()
    console.print(table)

    if markdown_file is not None:
        dir = os.path.dirname(markdown_file)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(markdown_file, "w+") as f:
            f.write(md_data)
