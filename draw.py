from points import initialize_points, update_points, setting
from pygame.locals import *
import pygame


def draw_points(points, screen):
    for p in points:
        pygame.draw.circle(screen, 'black', (p.x, p.y), setting.point_radius)


def main():
    points = initialize_points()
    frame_rate = 60
    pygame.init()
    screen = pygame.display.set_mode((setting.w, setting.h))
    pygame.display.set_caption('Point Packing')
    print(pygame.key.key_code("r"))
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                points = initialize_points()
        is_mouse_down = pygame.mouse.get_pressed()[0]
        screen.fill('white')
        draw_points(points, screen)
        update_points(points, pygame.mouse.get_pos(), is_mouse_down)
        pygame.display.update()
        pygame.time.delay(1000 // frame_rate)


main()
