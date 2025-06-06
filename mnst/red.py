import cv2
import tensorflow as tf
import numpy as np

model= tf.keras.models.load_model('mnist_models (1).keras')# # se agrega el modelo de la red neuronal al codigo

cap = cv2.VideoCapture(0)# Abre la cámara
while True:
    ret, frame = cap.read() #frame es para atrapar el fotograma exactamente donde debe de ser, y el ret es para decir si en realidad la lectura si de pudo hacer
    if ret == False:#Si el ret es False se detiene el bucle
        break
    image_prediction= np.zeros(shape=(250,200, 3)) #Convierte la imagen en negro y la vuelve en una matriz de 0
   

    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #vuelve el foto grama en escala de grises
    _, binary=cv2.threshold(frame_gray, 127,255,cv2.THRESH_BINARY_INV) #los pixeles los vuelve a una escala de 0 a 255 ( 0 el mas negro y 255 el más blanco)
    contours, _= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Primero busca los bordos blancos de la imagen y los simpifica para que no sean muy pesados
    contours=sorted(contours, key=cv2.contourArea, reverse=True)[:5]# Agarra los 5 contornos más grandes que aparezcan en la imagen y los ordena de mayor a menor
    cv2.drawContours(frame,contours, -1, (255, 0 ,0), 3) # Se le indica que a todos estos contornos de las listas les ponga una linea azul

    for cnt in contours: #este bucle va a recorrer todos los elementos de la lista contours
        if cv2.contourArea(cnt) > 3000: # Esta condicion solo se cumple si la imagen tiene más de 3000 pixeles,y el contour va encontrar los bordos de esa area
            x, y, w, h = cv2.boundingRect(cnt) #ademas de eso descarta los contornos muy pequeños
            #^ se sacan los puntos iniciales. h= largo  w=ancho x= la 
            if w / h < 1: #esta es la que extrae el digito de la imagen
                height, width = binary.shape #Guarda la imagen de blanco y negro en hfilas y wcolumnas

                iniarr_x = max(y - h // 10, 0)# se recortan las margenes de la imagen para que no se salga de los bordos
                iniaba_y = min(y + h + h // 10, height)

                crop_height = iniaba_y - iniarr_x #calcula el area total con todo lo recortado
                w_portion = (crop_height - w) // 2 #busca que el alto y el ancho coincidan para ser un cuadrado

                x_ini = max(x - w_portion, 0) #termina de arreglar todas las cordenadas y ver que no se salgan del borde
                x_fin = min(x + w + w_portion, width)

                if iniaba_y > iniarr_x  and x_fin > x_ini: # este if se utiliza para ver si las cordenadas de inicio están primero que las del fin
                    crop_image = binary[iniarr_x :iniaba_y, x_ini:x_fin] 

                    if (crop_image is not None and # Verifica que la imagen recortada no sea nula
                        crop_image.size > 0 and # Verifica que la imagen recortada tenga un tamaño mayor a 0
                        crop_image.shape[0] > 0 and # Verifica que la imagen recortada tenga una altura mayor a 0
                        crop_image.shape[1] > 0):# Verifica que la imagen recortada tenga un ancho mayor a 0

                        
                        crop_resize_image = cv2.resize(crop_image, (28, 28))# Redimensiona la imagen recortada a 28x28 píxeles (formato requerido por el modelo MNIST)
                        input_image = crop_resize_image.astype("float32") / 255.0# Convierte la imagen a tipo float y normaliza los valores entre 0 y 1                 
                        input_image = input_image.reshape(1, 28 * 28) # Reestructura la imagen a un vector de 784 elementos (1 fila, 28*28 columnas)

                        
                        prediction = model.predict(input_image)# Realiza la predicción con el modelo cargado
                        
                        predicted_class = prediction.argmax(axis=1)[0]# Obtiene la clase con mayor probabilidad (el dígito predicho)

                        
                        print(prediction)# Imprime en consola el vector de probabilidades
                        print(predicted_class)#imprime la clase predicha

                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1) #Dibuja un rectángulo azul alrededor del contorno detectado en el frame original
                        cv2.rectangle(frame, (x_ini, iniarr_x ), (x_fin, iniaba_y), (0, 255, 0), 1)# Dibuja un rectángulo verde alrededor del área recortada

                        # Escribe el porcentaje de confianza de la predicción en la imagen negra de resultados
                        cv2.putText(image_prediction,
                                    f"Prediction({prediction[0][predicted_class]*100: .1f}%):",
                                    (5, 20),
                                    1,
                                    1.2,
                                    (0, 255, 255),
                                    1)
                        # Escribe el dígito predicho en grande en la imagen negra de resultados
                        cv2.putText(image_prediction,
                                    str(predicted_class),
                                    (5, 240), #posicion de x and y
                                    1,#Tipo de letra
                                    20,#Tamaño de la letra
                                    (9, 9, 255 ), #Color, en este caso amarillo
                                    3)#Grosor de la letra

                    
                    cv2.imshow("crop_image", crop_image)# Muestra la imagen recortada 
                    cv2.imshow("crop_resize_image", crop_resize_image)#la imagen redimensionada en ventanas separadas
                else:
                    # si la imagen recortada es invalida, muestra un mensaje de error
                    print("Imagen recortada vacía o inválida. No se puede redimensionar.")


    
    cv2.imshow("Frame", frame)# Muestra el frame original
    cv2.imshow("image_prediction",image_prediction)# Muestra la imagen de predicción
    cv2.imshow("binary",binary)#Muestra la imagen binaria en ventanas

    # si se le hunde la tecla ESC se cierra el blucle
    key= cv2.waitKey(1) & 0xFF
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()# se destruyen todas las ventanas abiertas
