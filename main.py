import cv2 as cv
import numpy as np
import threading
import logging
import pyaudio

# https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
# https://realpython.com/traditional-face-detection-python/
# https://realpython.com/face-detection-in-python-using-a-webcam/

class Teremin:
    def __init__(self):
        print(f"TEREMIN VIRTUAL")
        print(f"por Ewerton Bramos")
        print(f"(pressione \"q\" para sair)")

        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

        classifierXML = 'haarcascade_frontalface_alt.xml'
        self.faceCascade:cv.CascadeClassifier = cv.CascadeClassifier(classifierXML)
        #faceCascade = cv.CascadeClassifier('haarcascade_smile.xml')

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
        self.stream.write(sample)

    def main(self):
        while self.sair == False:
            # Carregando imagem em escala de cinza para detecção de Haar-Like
            # feature usando o algoritmo de Viola-Jones
            ret, frame = self.capturaVideo.read()
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
                t = threading.Thread(target=self.tocarSom, args=(normalizado,))
                t.start()

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
