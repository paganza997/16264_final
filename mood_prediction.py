from get_song_info import *
from model import *
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from sklearn.pipeline import Pipeline


top=tk.Tk()
top.geometry('500x400')
top.title('Song Classification Spotify API')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

inputtxt = tk.Text(top,height=2,width=25,)

inputtxt.pack()

def classify(id_song):
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=modelo,epochs=300,
                                                                             batch_size=200,verbose=0))])
    pip.fit(X2,encoded_y)
    preds = get_songs_features(id_song)
    preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T
    results = pip.predict(preds_features)
    mood = np.array(target['mood'][target['encode']==int(results)])
    name_song = preds[0][0]
    artist = preds[0][2]
    print("{0} by {1} is a {2} song".format(name_song,artist,mood[0].upper()))
    label.configure(foreground='#011638', text=mood[0]) 

def text_Input():
    try:
        inp = inputtxt.get(1.0, "end-1c")
        info = get_songs_features(inp)
        name_song = info[0][0]
        artist = info[0][2]
        lbl.config(text = "Artist: "+artist+"   Song:"+name_song)
        classify(inp)
    except:
        pass

upload=Button(top,text="Classify Song",command=text_Input, padx=10,pady=5)

upload.configure(background='#364156', foreground='black', font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Song Mood Classification",pady=20, font=('arial',20,'bold'))


heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
lbl = tk.Label(top, text = "")
lbl.pack()
top.mainloop()