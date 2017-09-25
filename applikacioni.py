import cv2
import wx
import numpy as np

class FaceDetectionApp(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Face Detection Application', size=(300, 300))
        panel = wx.Panel(self, -1)

        bmp1 = wx.Bitmap("joyful.png")
        button1 = wx.BitmapButton(panel, -1, bitmap=bmp1, size=(100,100), pos=(20,20) )
        self.Bind(wx.EVT_BUTTON, self.FaceDetector, button1)

        bmp2 = wx.Bitmap("eye.png")
        button2 = wx.BitmapButton(panel, -1, bitmap=bmp2, size=(100,100), pos=(160,20) )
        self.Bind(wx.EVT_BUTTON, self.EyeDetector, button2)

        bmp3 = wx.Bitmap("nose.png")
        button3 = wx.BitmapButton(panel, -1, bitmap=bmp3, size=(100,100), pos=(20,140) )
        self.Bind(wx.EVT_BUTTON, self.NoseDetector, button3)

        bmp4 = wx.Bitmap("mouth.png")
        button4 = wx.BitmapButton(panel, -1, bitmap=bmp4, size=(100,100), pos=(160,140) )
        self.Bind(wx.EVT_BUTTON, self.MouthDetector, button4)

    def FaceDetector(self, event):
        face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

        if face_cascade.empty():
        	raise IOError('Unable to load the face cascade classifier xml file')

        cap = cv2.VideoCapture(0)
        scaling_factor = 0.5

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in face_rects:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

            cv2.imshow('Face Detector', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def EyeDetector(self, event):
        face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')

        if face_cascade.empty():
        	raise IOError('Unable to load the face cascade classifier xml file')

        if eye_cascade.empty():
        	raise IOError('Unable to load the eye cascade classifier xml file')

        cap = cv2.VideoCapture(0)
        ds_factor = 0.5

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (x_eye,y_eye,w_eye,h_eye) in eyes:
                    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
                    radius = int(0.3 * (w_eye + h_eye))
                    color = (0, 255, 0)
                    thickness = 3
                    cv2.circle(roi_color, center, radius, color, thickness)

            cv2.imshow('Eye Detector', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


    def NoseDetector(self, event):
        nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')

        if nose_cascade.empty():
        	raise IOError('Unable to load the nose cascade classifier xml file')

        cap = cv2.VideoCapture(0)
        ds_factor = 0.5

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in nose_rects:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                break

            cv2.imshow('Nose Detector', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def MouthDetector(self, event):
        mouth_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_mouth.xml')

        if mouth_cascade.empty():
        	raise IOError('Unable to load the mouth cascade classifier xml file')

        cap = cv2.VideoCapture(0)
        ds_factor = 0.5

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
            for (x,y,w,h) in mouth_rects:
                y = int(y - 0.15*h)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                break

            cv2.imshow('Mouth Detector', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = wx.PySimpleApp()
    FaceDetectionApp().Show()
    app.MainLoop()
