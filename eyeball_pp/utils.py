from typing import Any, Optional, Sequence


def get_user_input(text: str, choices: Optional[Sequence[Any]]) -> str:
    from InquirerPy import inquirer
    import termios
    import sys

    termios.tcflush(sys.stdin, termios.TCIOFLUSH)

    if choices:
        return inquirer.select(text, choices=choices).execute()
    else:
        return inquirer.text(text).execute()
