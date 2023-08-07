# Also makes sure directory is currently empty
[ -d "data" ] && ditto -V -x -k --sequesterRsrc --rsrc docs_dump.zip data
