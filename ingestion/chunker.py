import re

# Updated regex pattern to handle trailing whitespace
_PAGE_MARKER_PATTERN = re.compile(r"^\[PAGE:(\d+)\]\s*$", re.MULTILINE)

# Rest of the file remains unchanged