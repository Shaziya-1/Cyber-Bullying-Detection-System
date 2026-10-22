import sqlite3
conn=sqlite3.connect('platform.db')
for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table';").fetchall():
  print(row[0])
