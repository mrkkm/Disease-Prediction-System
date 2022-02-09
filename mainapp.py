from flask import Flask, render_template, request, redirect, url_for
import disease_prediction
from flask_mysqldb import MySQL

app = Flask(__name__)

app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "records"

mysql = MySQL(app)


@app.route('/symptom', methods=['POST', 'GET'])
def symptom():
    name = request.form.get('Name')
    s1 = request.form.get('Symptom1')
    s2 = request.form.get('Symptom2')
    s3 = request.form.get('Symptom3')
    s4 = request.form.get('Symptom4')
    s5 = request.form.get('Symptom5')
    selected_symptoms = [s1, s2, s3, s4, s5]
    disease_dt = disease_prediction.decisiontree(selected_symptoms)
    disease_rf = disease_prediction.randomforest(selected_symptoms)
    if request.method == "POST":
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO dpsrecords (name,symptom1,symptom2,symptom3,symptom4,symptom5,decisiontree,randomforest) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (name, s1, s2, s3, s4, s5, disease_dt, disease_rf))
        mysql.connection.commit()
        cur.close()
    return render_template('disease_prediction.html', sym=[s1, s2, s3, s4, s5], disease_dt=disease_dt, disease_rf=disease_rf)


@app.route('/records', methods=['POST', 'GET'])
def records():
    if request.method == "GET":
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM dpsrecords")
        recordsdetail = cur.fetchall()
        mysql.connection.commit()
        cur.close()
        return render_template('records.html', records=recordsdetail)


if __name__ == '__main__':
    app.run(debug=True)
