set -e

# Also makes sure directory is currently empty
[ ! -d "data" ]

# Expand the docs
ditto -V -x -k --sequesterRsrc --rsrc docs_dump.zip data

# Expand the discord
ditto -V -x -k --sequesterRsrc --rsrc discord_dump.zip data
