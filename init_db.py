import sqlite3

connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO users (username,password) VALUES (?,?)", ('ali4426623@gmail.com','admin'))
cur.execute("INSERT INTO users (username,password) VALUES (?,?)", ('admin','admin'))
cur.execute("INSERT INTO users (username,password) VALUES (?,?)", ('test@gmail.com','admin'))

connection.commit()
connection.close()