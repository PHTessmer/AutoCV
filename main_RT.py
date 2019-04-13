import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
import time
from threading import Thread
from queue import Queue

C_LEFT_SLOPE = 0
C_RIGHT_SLOPE = 0
C_LEFT = [0, 0, 0]
C_RIGHT = [0, 0, 0]

road = np.zeros((720, 1280, 3))
started = 0

def reset_globals():
    """
    Limpa as variáveis globais
    """
    print("RESETANDO VARIAVEIS GLOBAIS")
    global C_LEFT_SLOPE
    global C_RIGHT_SLOPE
    global C_LEFT
    global C_RIGHT
    C_LEFT_SLOPE = 0
    C_RIGHT_SLOPE = 0
    C_LEFT = [0, 0, 0]
    C_RIGHT = [0, 0, 0]

def draw_lines(img, lines, color=[0, 255, 0], thickness=14):
    """
    Função que desenha as linhas com a cor espessura determinadas.
    Ela itera pelas linhas de Hough, filtra e designa como linha esquerda ou direita.
    """
    global C_LEFT_SLOPE
    global C_RIGHT_SLOPE
    global C_LEFT
    global C_RIGHT

    # DECLARA VARIAVEIS
    c_weight = 0.9

    right_ys = []
    right_xs = []
    right_slopes = []

    left_ys = []
    left_xs = []
    left_slopes = []

    midpoint = img.shape[1] / 2
    bottom_of_image = img.shape[0]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope, yint = np.polyfit((x1, x2), (y1, y2), 1)
            # Filtra as linhas pela inclinação e posição x
            if .35 < np.absolute(slope) <= .85:
                if slope > 0 and x1 > midpoint and x2 > midpoint:
                    right_ys.append(y1)
                    right_ys.append(y2)
                    right_xs.append(x1)
                    right_xs.append(x2)
                    right_slopes.append(slope)
                elif slope < 0 and x1 < midpoint and x2 < midpoint:
                    left_ys.append(y1)
                    left_ys.append(y2)
                    left_xs.append(x1)
                    left_xs.append(x2)
                    left_slopes.append(slope)

    # DESENHA A LINHA DIREITA DA FAIXA
    if right_ys:
        right_index = right_ys.index(min(right_ys))
        right_x1 = right_xs[right_index]
        right_y1 = right_ys[right_index]
        right_slope = np.median(right_slopes)
        if C_RIGHT_SLOPE != 0:
            right_slope = right_slope + (C_RIGHT_SLOPE - right_slope) * c_weight

        right_x2 = int(right_x1 + (bottom_of_image - right_y1) / right_slope)

        if C_RIGHT_SLOPE != 0:
            right_x1 = int(right_x1 + (C_RIGHT[0] - right_x1) * c_weight)
            right_y1 = int(right_y1 + (C_RIGHT[1] - right_y1) * c_weight)
            right_x2 = int(right_x2 + (C_RIGHT[2] - right_x2) * c_weight)

        C_RIGHT_SLOPE = right_slope
        C_RIGHT = [right_x1, right_y1, right_x2]

        cv2.line(img, (right_x1, right_y1), (right_x2, bottom_of_image), color, thickness)

    # DESENHA A LINHA ESQUERDA DA FAIXA
    if left_ys:
        left_index = left_ys.index(min(left_ys))
        left_x1 = left_xs[left_index]
        left_y1 = left_ys[left_index]
        left_slope = np.median(left_slopes)
        if C_LEFT_SLOPE != 0:
            left_slope = left_slope + (C_LEFT_SLOPE - left_slope) * c_weight

        left_x2 = int(left_x1 + (bottom_of_image - left_y1) / left_slope)

        if C_LEFT_SLOPE != 0:
            left_x1 = int(left_x1 + (C_LEFT[0] - left_x1) * c_weight)
            left_y1 = int(left_y1 + (C_LEFT[1] - left_y1) * c_weight)
            left_x2 = int(left_x2 + (C_LEFT[2] - left_x2) * c_weight)

        C_LEFT_SLOPE = left_slope
        C_LEFT = [left_x1, left_y1, left_x2]

        cv2.line(img, (left_x1, left_y1), (left_x2, bottom_of_image), color, thickness)

def process_image(image):
    """
    Aplica Canny e Hough nas imagens para detectar as bordas da faixa e desenhar
    as linhas sobre a imagem original.
    """
    global started
    # PARAMETROS
    imshape = image.shape
    kernel_size = 3
    sigma_x = 0
    low_canny_threshold = 25
    high_canny_threshold = low_canny_threshold * 3
    vertices = np.array([[(0,imshape[0]), (9*imshape[1]/20, 11*imshape[0]/18), (11*imshape[1]/20, 11*imshape[0]/18), (imshape[1],imshape[0])]], dtype=np.int32)
    ignore_mask_color = 255
    rho = 1
    theta = np.pi/180
    hough_threshold = 10
    min_line_len = 30
    max_line_gap = 60
    α = 0.8
    β = 1.
    λ = 0.

    # GRAYSCALE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # GAUSSIAN BLUR
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x)

    # CANNY EDGES
    edges = cv2.Canny(blur, low_canny_threshold, high_canny_threshold)

    # REGION MASK
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(edges, mask)

    # HOUGH TRANSFORM
    lines = cv2.HoughLinesP(masked, rho, theta, hough_threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    hough_image = np.zeros((*masked.shape, 3), dtype=np.uint8)
    draw_lines(hough_image, lines)

    # WEIGHTED IMAGE
    processed = cv2.addWeighted(image, α, hough_image, β, λ)
    
    started = 1

    return processed

"""
# MOSTRA IMAGEM TESTE
#image = mpimg.imread("images/img1.jpg")
#plt.imshow(process_image(image))
#plt.show()

# SALVA IMAGENS PROCESSADAS EM ./final_images
imageNames = os.listdir('imagens/')
for name in imageNames:
    reset_globals()
    image = mpimg.imread("images/{}".format(name))
    plt.imsave("images_det/final_{}".format(name), process_image(image))
"""

if __name__ == '__main__':
    
    frames_counts = 1
    cap=cv2.VideoCapture('test_videos/v_HD.mp4')  
    class MyThread(Thread):

        def __init__(self, q):
            Thread.__init__(self)
            self.q = q

        def run(self):
            while(1):
                if (not self.q.empty()):
                    image = self.q.get()
                    process_image(image)

    q = Queue()
    q.queue.clear()
    thd1 = MyThread(q)
    thd1.setDaemon(True)
    thd1.start()

    while (True):  
        start=time.time()
        ret,frame=cap.read()

            # Detect as faixas a cada 5 frames
        if frames_counts % 5 == 0:
            q.put(frame)

            # Adiciona a detecção das faixas na imagem original
        if started:
            frame = process_image(frame)
        cv2.imshow("RealTime_lane_detection",frame)  
        if cv2.waitKey(1)&0xFF==ord('q'):  
            break  
        frames_counts+=1
        cv2.waitKey(12)
        finish=time.time()
        print ('FPS:  ' + str(int(1/(finish-start))))

    cap.release()  
    cv2.destroyAllWindows() 