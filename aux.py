from enum import Enum

class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    ORANGE = "\033[97m"
    RESET = "\033[0m"

def colored_print(s, color=Color.GREEN):
    print(f"{color.value}{s}{Color.RESET.value}")


def print_green(s):
    colored_print(s, Color.GREEN)

def print_yellow(s):
    colored_print(s, Color.YELLOW)


def print_cyan(s):
    colored_print(s, Color.CYAN)

def print_blue(s):
    colored_print(s, Color.BLUE)

def print_orange(s):
    colored_print(s, Color.ORANGE)

def print_red(s):
    colored_print(s, Color.RED)

def print_magenta(s):
    colored_print(s, Color.MAGENTA)


