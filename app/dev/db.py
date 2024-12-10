import sqlite3

# Function to create the database table
data_path = "./app/dev/data_form.db"
def create_table():
    conn = sqlite3.connect(data_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            jawa_text TEXT NOT NULL,
            indonesia_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            suggestion TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Function to insert data into the database
def insert_data(name, age, jawa_text, indonesia_text, rating, suggestion):
    conn = sqlite3.connect(data_path)
    c = conn.cursor()
    c.execute("INSERT INTO translations (name, age, jawa_text, indonesia_text, rating, suggestion) VALUES (?, ?, ?, ?, ?, ?)", 
              (name, age, jawa_text, indonesia_text, rating, suggestion))
    conn.commit()
    conn.close()
