import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

C_LEFT_SLOPE = 0
C_RIGHT_SLOPE = 0
C_LEFT = [0, 0, 0]
C_RIGHT = [0, 0, 0]

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

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([0, 150,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

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
    # PARAMETROS
    imshape = image.shape
    kernel_size = 3
    sigma_x = 0
    low_canny_threshold = 25
    high_canny_threshold = low_canny_threshold * 3
    vertices = np.array([[(0,imshape[0]), (9*imshape[1]/20, 11*imshape[0]/18), (11*imshape[1]/20, 11*imshape[0]/18), (imshape[1],imshape[0])]], dtype=np.int32)
    #vertices = np.array([[(0,imshape[0]), (7*imshape[1]/20, 8*imshape[0]/18), (13*imshape[1]/20, 8*imshape[0]/18), (imshape[1],imshape[0])]], dtype=np.int32)
    #vertices = np.array([[(0,3*imshape[0]/4), (7*imshape[1]/20, 8*imshape[0]/18), (13*imshape[1]/20, 8*imshape[0]/18), (imshape[1],3*imshape[0]/4)]], dtype=np.int32)
    ignore_mask_color = 255
    rho = 1
    theta = np.pi/180
    hough_threshold = 10
    min_line_len = 30
    max_line_gap = 60
    α = 0.8
    β = 1.
    λ = 0.

    #SELECT WHITE/YELLOW
    white_yellow = select_white_yellow(image)
    #plt.imshow(white_yellow)
    #cv2.imshow('white/yellow', white_yellow)
    #cv2.waitKey(25)

    # GRAYSCALE
    gray = cv2.cvtColor(white_yellow, cv2.COLOR_BGR2GRAY)

    # GAUSSIAN BLUR
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x)

    # CANNY EDGES
    edges = cv2.Canny(blur, low_canny_threshold, high_canny_threshold)
    #plt.imshow(edges)
    #cv2.imshow('canny', edges)
    #cv2.waitKey(25)
    
    # REGION MASK
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(edges, mask)
    #plt.imshow(masked)
    #cv2.imshow('masked', masked)
    #cv2.waitKey(25)

    # HOUGH TRANSFORM
    lines = cv2.HoughLinesP(masked, rho, theta, hough_threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    hough_image = np.zeros((*masked.shape, 3), dtype=np.uint8)
    draw_lines(hough_image, lines)
    #plt.imshow(hough_image)

    # WEIGHTED IMAGE
    processed = cv2.addWeighted(image, α, hough_image, β, λ)
    #plt.imshow(processed)
    #cv2.imshow('lines', processed)
    #cv2.waitKey(25)
    
    return processed

# CRIA O VIDEO COM AS LINHAS DESENHADAS SOBRE AS FAIXAS
reset_globals()
out1 = 'outhls_vHD.mp4'
clip1 = VideoFileClip(os.path.join('test_videos', 'v_HD.mp4'))
proc_clip = clip1.fl_image(process_image)
proc_clip.write_videofile(os.path.join('output_videos', out1), audio=False)
