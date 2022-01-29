import time
from points import initialize_points, update_points, w, h
from pygame.locals import *
import pygame


def drawPoints(points, screen):
    for p in points:
        pygame.draw.circle(screen, 'black', (p.x, p.y), 3)


def main():
    points = initialize_points()
    frame_rate = 60
    is_mouse_down = False
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption('Point Packing')

    while 1:
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN:
                is_mouse_down = True
            if event.type == MOUSEBUTTONUP:
                is_mouse_down = False
            if event.type == QUIT:
                return
        screen.fill('white')
        drawPoints(points, screen)
        update_points(points, pygame.mouse.get_pos(), is_mouse_down)
        pygame.display.update()
        pygame.time.delay(1000 // frame_rate)


main()
