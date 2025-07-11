import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib


app = Flask(__name__)

nimgs = 50

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('face.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/result')
def result():
    names, rolls, times, l = extract_attendance()
    return render_template('result.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# @app.route('/start', methods=['GET'])
# def start():
    names, rolls, times, l = extract_attendance()

    if 'model.pkl' not in os.listdir('static'):
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('result.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'model.pkl' not in os.listdir('static'):
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Save the current frame for display in the browser
            cv2.imwrite('static/attendance_frame.jpg', frame)

        # Break after a single detection (or you can add your own stopping criteria)
        break

    cap.release()

    # Render the image in the result template
    names, rolls, times, l = extract_attendance()
    return render_template('result.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, image_path='static/attendance_frame.jpg')

#not working
# @app.route('/add', methods=['GET', 'POST'])
# def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        # Save the current frame for debugging/display
        cv2.imwrite('static/add_user_frame.jpg', frame)

        # Optionally break early for testing
        if i == nimgs:
            break

    cap.release()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, image_path='static/add_user_frame.jpg')

    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    # Ensure unique roll numbers
    existing_users = os.listdir('static/faces')
    for user in existing_users:
        _, roll = user.split('_')
        if newuserid == roll:
            return render_template('index.html', mess="Roll number already exists. Please use a unique roll number.")

    # Check for duplicate face
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    faces = extract_faces(frame)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        new_face = cv2.resize(frame[y:y+h, x:x+w], (200, 200))
        new_face_encoding = face_recognition.face_encodings(new_face)[0]
        
        # Load existing face encodings
        known_face_encodings = []
        for user_folder in os.listdir('static/faces'):
            user_folder_path = f'static/faces/{user_folder}'
            if os.path.isdir(user_folder_path):
                for img_name in os.listdir(user_folder_path):
                    img_path = os.path.join(user_folder_path, img_name)
                    image = cv2.imread(img_path)
                    face_encoding = face_recognition.face_encodings(image)
                    if face_encoding:
                        known_face_encodings.append(face_encoding[0])

        # Compare with existing faces
        matches = face_recognition.compare_faces(known_face_encodings, new_face_encoding)
        if any(matches):
            return render_template('index.html', mess="Face already registered. Registration aborted.")

    # Create folder and save images for training
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    while i < nimgs:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if j % 5 == 0:
                img_name = f'{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{img_name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1

        cv2.imwrite('static/current_frame.jpg', frame)
        print(f"Saved frame {i} to static/current_frame.jpg")

        if cv2.waitKey(1) == 27:  # Break on ESC key press
            break
    cap.release()
    cv2.destroyAllWindows()

    # Train the model after adding new user
    print('Training Model...')
    train_model()

    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    # Ensure unique roll numbers
    existing_users = os.listdir('static/faces')
    for user in existing_users:
        _, roll = user.split('_')
        if newuserid == roll:
            return render_template('index.html', mess="Roll number already exists. Please use a unique roll number.")

    # Check for duplicate face
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    faces = extract_faces(frame)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        new_face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).ravel()
        
        # Load existing model
        if 'model.pkl' in os.listdir('static'):
            model = joblib.load('static/model.pkl')
            existing_user = model.predict([new_face])
            if existing_user:
                return render_template('index.html', mess=f"Face already registered as {existing_user[0]}. Registration aborted.")

    # Create folder and save images for training
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    while i < nimgs:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            # Draw rectangle on face (optional)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if j % 5 == 0:
                img_name = f'{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{img_name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        # Save current frame to static directory for review (optional)
        cv2.imwrite('static/current_frame.jpg', frame)

        # Skip `imshow()`; you can serve 'static/current_frame.jpg' on the front end for live preview
        print(f"Saved frame {i} to static/current_frame.jpg")

        if cv2.waitKey(1) == 27:  # Break on ESC key press
            break
    cap.release()
    cv2.destroyAllWindows()

    # Train the model after adding new user
    print('Training Model...')
    train_model()

    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    # Ensure unique roll numbers
    existing_users = os.listdir('static/faces')
    for user in existing_users:
        _, roll = user.split('_')
        if newuserid == roll:
            return render_template('index.html', mess="Roll number already exists. Please use a unique roll number.")

    # Check for duplicate face
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    faces = extract_faces(frame)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        new_face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).ravel()
        
        # Load existing model
        if 'model.pkl' in os.listdir('static'):
            model = joblib.load('static/model.pkl')
            existing_user = model.predict([new_face])
            if existing_user:
                return render_template('index.html', mess=f"Face already registered as {existing_user[0]}. Registration aborted.")

    # Create folder and save images for training
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    while i < nimgs:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if j % 5 == 0:
                img_name = f'{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{img_name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # Train the model after adding new user
    print('Training Model...')
    train_model()

    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)
