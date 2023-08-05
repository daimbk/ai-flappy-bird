import random  # For generating random numbers
import sys  # We will use sys.exit to exit the program
import pygame
from pygame.locals import *  # Basic pygame imports
import neat

# Global Variables for the game
FPS = 32
SCREENWIDTH = 289
SCREENHEIGHT = 511
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
GROUNDY = SCREENHEIGHT * 0.8
GAME_SPRITES = {}
GAME_SOUNDS = {}
PLAYER = 'gallery/sprites/bird.png'
BACKGROUND = 'gallery/sprites/background.png'
PIPE = 'gallery/sprites/pipe.png'

#NEAT options
generation = 0 #note that the first generation of the birds is 0 because index starts from zero. XD
max_gen = 50 #the maximum number of generation to run
prob_threshold_to_jump = 0.8 #the probability threshold to activate the bird to jump
failed_punishment = 10 #the amount of fitness decrease after collision

def welcomeScreen():
    """
    Shows welcome images on the screen
    """

    playerx = int(SCREENWIDTH/5)
    playery = int((SCREENHEIGHT - GAME_SPRITES['player'].get_height())/2)
    messagex = int((SCREENWIDTH - GAME_SPRITES['message'].get_width())/2)
    messagey = int(SCREENHEIGHT*0.13)
    basex = 0
    while True:
        for event in pygame.event.get():
            # if user clicks on cross button, close the game
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

            # If the user presses space or up key, start the game for them
            elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                return
            else:
                SCREEN.blit(GAME_SPRITES['background'], (0, 0))
                SCREEN.blit(GAME_SPRITES['player'], (playerx, playery))
                SCREEN.blit(GAME_SPRITES['message'], (messagex, messagey))
                SCREEN.blit(GAME_SPRITES['base'], (basex, GROUNDY))
                pygame.display.update()
                FPSCLOCK.tick(FPS)

#define a function to get the input index of the pipes
def get_index(lower_pipes, upper_pipes, birds):
    # get the bird's x position (assuming all birds have the same x position)
    bird_x = birds[0].x

    # calculate the x distance between birds and each lower pipe
    lower_list_distance = [lower_pipe['x'] + GAME_SPRITES['pipe'][1].get_width() - bird_x for lower_pipe in lower_pipes]

    # calculate the x distance between birds and each upper pipe
    upper_list_distance = [upper_pipe['x'] + GAME_SPRITES['pipe'][0].get_width() - bird_x for upper_pipe in upper_pipes]

    # combine the distances from both lists
    combined_distances = lower_list_distance + upper_list_distance

    # get the index of the pipe that has the minimum non-negative distance (the closest pipe in front of the bird)
    index = combined_distances.index(min(i for i in combined_distances if i >= 0))

    return index


def mainGame(config, genomes):
    global generation, SCREEN #use the global variable gen and SCREEN
    screen = SCREEN
    generation += 1 #update the generation

    score = 0
    playerx = int(SCREENWIDTH/5)
    playery = int(SCREENWIDTH/2)
    basex = 0

    models_list = [] #create an empty list to store all the training neural networks
    genomes_list = [] #create an empty list to store all the training genomes
    birds_list = [] #create an empty list to store all the training birds

    # Create 2 pipes for blitting on the screen
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # my List of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH+200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH+200+(SCREENWIDTH/2), 'y': newPipe2[0]['y']},
    ]
    # my List of lower pipes
    lowerPipes = [
        {'x': SCREENWIDTH+200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH+200+(SCREENWIDTH/2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    playerVelY = -9
    playerMaxVelY = 10
    playerMinVelY = -8
    playerAccY = 1

    playerFlapAccv = -8  # velocity while flapping
    playerFlapped = False  # It is true only when the bird is flapping

    for genome_id, genome in genomes: #for each genome
        birds_list.append(playerx, playery) #create a bird and append the bird in the list
        genome.fitness = 0 #start with fitness of 0
        genomes_list.append(genome) #append the genome in the list
        model = neat.nn.FeedForwardNetwork.create(genome, config) #set up the neural network for each genome using the configuration we set
        models_list.append(model) #append the neural network in the list
        

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            '''if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery > 0:
                    playerVelY = playerFlapAccv
                    playerFlapped = True
                    GAME_SOUNDS['wing'].play()'''
        #stop the game when the score exceed the maximum score
        #break the loop and restart when no bird left
        if score >= max_score or len(birds_list) == 0:
            run = False
            break

        '''# This function will return true if the player is crashed
        crashTest = isCollide(playerx, playery, upperPipes, lowerPipes)
        if crashTest:
            return'''
        
        pipe_input_index = get_index(lowerPipes, upperPipes, birds_list) #get the input index of the pipes list
        passed_pipes = [] #create an empty list to hold all the passed pipes

        # both lowerPipes and upperPipes lists have the same length
        for i in range(len(lowerPipes)):
            lower_pipe = lowerPipes[i]
            upper_pipe = upperPipes[i]

            # Move the pipes
            lower_pipe['x'] -= pipeVelX
            upper_pipe['x'] -= pipeVelX

            # Check if the bird has passed either the lower or the upper pipe
            if lower_pipe['x'] + GAME_SPRITES['pipe'][1].get_width() < birds_list[0].x:
                passed_pipes.append(lower_pipe)
            elif upper_pipe['x'] + GAME_SPRITES['pipe'][0].get_width() < birds_list[0].x:
                passed_pipes.append(upper_pipe)


        # check for score
        score = len(passed_pipes) #calculate the score of the game, which equals to the number of pipes the bird passed
        
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY

        if playerFlapped:
            playerFlapped = False
        playerHeight = GAME_SPRITES['player'].get_height()
        playery = playery + min(playerVelY, GROUNDY - playery - playerHeight)

        # move pipes to the left
        for upperPipe, lowerPipe in zip(upperPipes, lowerPipes):
            upperPipe['x'] += pipeVelX
            lowerPipe['x'] += pipeVelX

        # Add a new pipe when the first is about to cross the leftmost part of the screen
        if 0 < upperPipes[0]['x'] < 5:
            newpipe = getRandomPipe()
            upperPipes.append(newpipe[0])
            lowerPipes.append(newpipe[1])

        # if the pipe is out of the screen, remove it
        if upperPipes[0]['x'] < -GAME_SPRITES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # Lets blit our sprites now
        SCREEN.blit(GAME_SPRITES['background'], (0, 0))
        for upperPipe, lowerPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(GAME_SPRITES['pipe'][0],
                        (upperPipe['x'], upperPipe['y']))
            SCREEN.blit(GAME_SPRITES['pipe'][1],
                        (lowerPipe['x'], lowerPipe['y']))

        SCREEN.blit(GAME_SPRITES['base'], (basex, GROUNDY))
        SCREEN.blit(GAME_SPRITES['player'], (playerx, playery))
        myDigits = [int(x) for x in list(str(score))]
        width = 0
        for digit in myDigits:
            width += GAME_SPRITES['numbers'][digit].get_width()
        Xoffset = (SCREENWIDTH - width)/2

        for digit in myDigits:
            SCREEN.blit(GAME_SPRITES['numbers'][digit],
                        (Xoffset, SCREENHEIGHT*0.12))
            Xoffset += GAME_SPRITES['numbers'][digit].get_width()
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def isCollide(playerx, playery, upperPipes, lowerPipes):
    if playery > GROUNDY - 25 or playery < 0:
        GAME_SOUNDS['hit'].play()
        return True

    for pipe in upperPipes:
        pipeHeight = GAME_SPRITES['pipe'][0].get_height()
        if (playery < pipeHeight + pipe['y'] and abs(playerx - pipe['x']) < GAME_SPRITES['pipe'][0].get_width()):
            GAME_SOUNDS['hit'].play()
            return True

    for pipe in lowerPipes:
        if (playery + GAME_SPRITES['player'].get_height() > pipe['y']) and abs(playerx - pipe['x']) < GAME_SPRITES['pipe'][0].get_width():
            GAME_SOUNDS['hit'].play()
            return True

    return False


def getRandomPipe():
    """
    Generate positions of two pipes(one bottom straight and one top rotated ) for blitting on the screen
    """
    pipeHeight = GAME_SPRITES['pipe'][0].get_height()
    offset = SCREENHEIGHT/3
    y2 = offset + random.randrange(0, int(SCREENHEIGHT -
                                   GAME_SPRITES['base'].get_height() - 1.2 * offset))
    pipeX = SCREENWIDTH + 10
    y1 = pipeHeight - y2 + offset
    pipe = [
        {'x': pipeX, 'y': -y1},  # upper Pipe
        {'x': pipeX, 'y': y2}  # lower Pipe
    ]
    return pipe

#define a function to run NEAT algorithm to play flappy bird
def run_NEAT(config_file):

    #use NEAT algorithm to build a neural network based on the pre-set configuration
    #Create a neat.config.Config object from the configuration file
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation,
                                config_file)
    
    #Create a neat.population.Population object using the Config object created above
    neat_pop = neat.population.Population(config)
    
    #show the summary statistics of the learning progress
    neat_pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    neat_pop.add_reporter(stats)
    
    #Call the run method on the Population object, giving it your fitness function and (optionally) the maximum number of generations you want NEAT to run
    neat_pop.run(mainGame, max_gen)
    
    #get the most fit genome genome as our winner with the statistics.best_genome() function
    winner = stats.best_genome()
    
    #visualize the results
    node_names = {-1:'delta_x', -2: 'delta_y_top', -3:'delta_y_bottom', 0:'Jump or Not'}
    draw_net(config, winner, True, node_names = node_names)
    plot_stats(stats, ylog = False, view = True)
    plot_species(stats, view = True)
    
    #show the final statistics
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    # This will be the main point from where our game will start
    pygame.init()  # Initialize all pygame's modules
    FPSCLOCK = pygame.time.Clock()
    pygame.display.set_caption('Flappy Bird')
    GAME_SPRITES['numbers'] = (
        pygame.image.load('gallery/sprites/0.png').convert_alpha(),
        pygame.image.load('gallery/sprites/1.png').convert_alpha(),
        pygame.image.load('gallery/sprites/2.png').convert_alpha(),
        pygame.image.load('gallery/sprites/3.png').convert_alpha(),
        pygame.image.load('gallery/sprites/4.png').convert_alpha(),
        pygame.image.load('gallery/sprites/5.png').convert_alpha(),
        pygame.image.load('gallery/sprites/6.png').convert_alpha(),
        pygame.image.load('gallery/sprites/7.png').convert_alpha(),
        pygame.image.load('gallery/sprites/8.png').convert_alpha(),
        pygame.image.load('gallery/sprites/9.png').convert_alpha(),
    )

    GAME_SPRITES['message'] = pygame.image.load(
        'gallery/sprites/message.png').convert_alpha()
    GAME_SPRITES['base'] = pygame.image.load(
        'gallery/sprites/base.png').convert_alpha()
    GAME_SPRITES['pipe'] = (pygame.transform.rotate(pygame.image.load(PIPE).convert_alpha(), 180),
                            pygame.image.load(PIPE).convert_alpha()
                            )

    # Game sounds
    GAME_SOUNDS['die'] = pygame.mixer.Sound('gallery/audio/die.wav')
    GAME_SOUNDS['hit'] = pygame.mixer.Sound('gallery/audio/hit.wav')
    GAME_SOUNDS['point'] = pygame.mixer.Sound('gallery/audio/point.wav')
    GAME_SOUNDS['swoosh'] = pygame.mixer.Sound('gallery/audio/swoosh.wav')
    GAME_SOUNDS['wing'] = pygame.mixer.Sound('gallery/audio/wing.wav')

    GAME_SPRITES['background'] = pygame.image.load(BACKGROUND).convert()
    GAME_SPRITES['player'] = pygame.image.load(PLAYER).convert_alpha()

    while True:
        welcomeScreen()  # Shows welcome screen to the user until he presses a button
        mainGame()  # This is the main game function

