import threading
import logging

import cv2 as cv
import numpy as np
import pyaudio
import mediapipe as mp
import tensorflow as tf
#from tf.keras.models import load_model

# https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
# https://realpython.com/traditional-face-detection-python/
# https://realpython.com/face-detection-in-python-using-a-webcam/

class Teremin:
    def __init__(self):
        # INFORMAÇÕES
        print(f"TEREMIN VIRTUAL")
        print(f"por Ewerton Bramos")
        print(f"(pressione \"q\" para sair)")

        # INICIALIZANDO LOGGING
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

        # ADICIONANDO HAAR-LIKE CLASSIFIER
        classifierXML = 'haarcascade_frontalface_alt.xml'
        self.faceCascade:cv.CascadeClassifier = cv.CascadeClassifier(classifierXML)
        #faceCascade = cv.CascadeClassifier('haarcascade_smile.xml')

        # FREQUÊNCIAS DE LYNMAN
        freqLynman = (
            586.67, 495.05, 469.33, 458.33, 452.48, 449.18, 447.27, 445.76,
            444.44, 443.85, 443.29, 442.96, 441.76, 441.59, 441.44, 441.16
        )

        # INICIALIZANDO PYAUDIO
        self.pyAudio:pyaudio.PyAudio = pyaudio.PyAudio()
        self.volume:float = 0.5  # range [0.0, 1.0]
        self.fs:int = 44100  # sampling rate, Hz, must be integer
        self.duracao:float = 0.05  # in seconds, may be float7
        self.startFreq:float = 320.0
        self.quantSamples:int = 400

        # for paFloat32 sample values must be in range [-1.0, 1.0]
        self.stream = self.pyAudio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.fs,
                        output=True)

        self.capturaVideo:cv.VideoCapture = cv.VideoCapture(0)

        # INICIALIZANDO HAND GESTURE
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        '''model = tf.keras.models.load_model('hand_gesture_recognition_code/mp_hand_gesture')
        #model = tf.keras.layers.TFSMLayer(
        #    'hand-gesture-recognition-code/mp_hand_gesture/',
        #    )#call_endpoint='serving_default')
        # Load class names
        f = open('gesture.names', 'r')
        classNames = f.read().split('\n')
        f.close()
        print(classNames)'''

        self.sair:bool = False
        self.main()

    def gerarSom(self, f:float, d:float) -> bytes:
        # generate samples, note conversion to float32 array
        samples = (np.sin(
            2 * np.pi * np.arange(self.fs * d) * f / self.fs)
            ).astype(np.float32)

        # per @yahweh comment explicitly convert to bytes sequence
        bytesSaida = (self.volume * samples).tobytes()

        return bytesSaida

    def tocarSom(self, proximidade:int):
        freq = self.startFreq + proximidade
        sample = self.gerarSom(freq, self.duracao)
        logging.info(f"Proximidade: {proximidade} | {freq}Hz")
        
        if self.stream.get_write_available():
            self.stream.write(sample)

    '''def reconhecerMao(self, frame, hands, mpHands, mpDraw, classNames, model):
        framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result = hands.process(framergb)
        className = ''
        # post process the result
        if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, 
        mpHands.HAND_CONNECTIONS)
                    
        # RECONHECER GESTOS
        prediction = model.predict([landmarks])
        print(prediction)
        classID = np.argmax(prediction)
        className = classNames[classID]
        # show the prediction on the frame
        cv.putText(frame, className, (10, 50), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv.LINE_AA)'''

    def main(self):
        while self.sair == False:
            # Capturando frame do vídeo da webcam e espelhando
            ret, frame = self.capturaVideo.read()
            #frameX, frameY, frameC = frame.shape
            #frame = cv.flip(frame,1)

            # Carregando imagem em escala de cinza para detecção de Haar-Like
            # feature usando o algoritmo de Viola-Jones
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            rectCor = (255, 0, 255)
            if len(faces):
                x = faces[0][0]
                y = faces[0][1]
                w = faces[0][2]
                h = faces[0][3]
                
                # Desenhando retângulos em volta dos rostos encontrados
                cv.rectangle(frame, (x, y), (x+w, y+h), rectCor, 2)

                normalizado = max(0,min(w - 100,self.quantSamples - 1))
                self.tocarSom(normalizado)
                #t = threading.Thread(target=self.tocarSom, args=(normalizado,))
                #t.start()

                cv.putText(frame,
                           f"{self.startFreq + normalizado}Hz",
                           (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           rectCor,
                           1,
                           cv.LINE_AA)
            else:
                t = threading.Thread(target=self.tocarSom, args=(0,))
                t.start()

            # Display the resulting frame
            cv.imshow('Teremin', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                self.sair = True

    def __del__(self):
        print(f"FINALIZANDO!")
        self.stream.stop_stream()
        self.stream.close()
        self.pyAudio.terminate()
        self.capturaVideo.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    T = Teremin() 
