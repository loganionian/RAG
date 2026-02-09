import re

# Updated regex pattern to allow trailing whitespace in page markers
_PAGE_MARKER_PATTERN = re.compile(r"^\[PAGE:(\d+)\]\s*$", re.MULTILINE)

# Rest of the existing code remains unchanged