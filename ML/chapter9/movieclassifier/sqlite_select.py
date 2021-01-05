import sqlite3

conn = sqlite3.connect('reviews.sqlite')

c = conn.cursor()

c.execute("SELECT * FROM review_db WHERE date BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')")

result = c.fetchall()

conn.close()

print(result)