from flask import Flask,render_template,Response,request, send_from_directory,session, send_file, jsonify, request,flash,redirect,url_for
import sqlite3
import webbrowser
app = Flask(__name__)
app.config['SECRET_KEY'] = '895623741'


database="Expenses.db"

def createtable():
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("create table if not exists register(id integer primary key autoincrement, name text,email text,password text,status text)")
    conn.commit()
    conn.close()
createtable()



@app.route('/')
def home():
    return render_template('register.html')


@app.route('/register', methods=["GET","POST"])
def register():
    if request.method=="POST":
        name=request.form['name']
        email=request.form['email']

        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute(" SELECT email FROM register WHERE email=?",(email,))
        registered=cursor.fetchall()
        if registered:
            return render_template('register.html', alert_message="Email Already Registered")
        else:
            cursor.execute("insert into register(name,email,password,status) values(?,?,?,?)",(name,email,password,0))
            conn.commit()
            return render_template('login.html', alert_message="Registered Succussfully")
    return render_template('register.html')



@app.route('/login', methods=["GET", "POST"])
def login():
    global data
    global email
    if request.method == "POST":        
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM register WHERE email=? AND password=?", (email, password))
        data = cursor.fetchone()


        if data is None:
            return render_template('register.html', alert_message="Email Not Registered or Check Password")
        else:
            session['email'] = email
            return render_template('dashboard.html')

    return render_template('login.html')



    
if __name__ == "__main__":
    app.run()

