from typing import Any, Optional, Sequence
from collections import OrderedDict


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
