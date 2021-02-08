import sqlite3
con = sqlite3.connect('item.db')

c = con.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS items
             (date text, name text, buyqty real, buyprice real, sellqty real, sellprice real)''')

# Insert a row of data
date = "2020-02-06T06:12"
name = "Soldier Boots"
buyqty = 10
buyprice = 2000

# Larger example that inserts many records at a time
purchases = [('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
             ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
             ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
            ]

c.execute(f"INSERT INTO items VALUES (?,?,?,?)", (date, name, buyqty, buyprice))

# Save (commit) the changes
con.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
con.close()