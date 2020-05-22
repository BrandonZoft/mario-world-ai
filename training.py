import retro
import numpy as np
import cv2
import neat
import pickle
import csv
import warnings
import sys

# C:\Users\brand\anaconda3\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes
# C:\Users\brand\anaconda3\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes
# max xpos on end level = 2427

with open('stats.csv', 'w+', newline='') as csvfile:
    filewrite = csv.writer(csvfile, delimiter=',',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewrite.writerow(['"generation"', '"id"', '"xpos"'])

warnings.filterwarnings("ignore", category=RuntimeWarning)

env = retro.make('SuperMarioWorld-Snes')

imgarray = []

resume = False

if len(sys.argv) == 2:
    resume = True
    restore_file = 'neat-checkpoint-{}'.format(sys.argv[1])

generation = 1
id_count = 1
globalcounter = 0


def eval_genomes(genomes, config):
    global generation
    global id_count
    global globalcounter
    for genome_id, genome in genomes:
        ob = env.reset()
        action = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        # cv2.namedWindow("Machine Learning", cv2.WINDOW_NORMAL)

        while not done:
            env.render()
            frame += 1
            # scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            # scaledimg = cv2.resize(scaledimg, (iny, inx))

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            # ob = np.reshape(ob, (inx, iny))
            # cv2.imshow('main', scaledimg)
            # cv2.waitKey(1)

            for x in ob:
                for y in x:
                    imgarray.append(y)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)

            imgarray.clear()

            xpos = info['x']

            if xpos > xpos_max:
                fitness_current = xpos
                xpos_max = xpos

            if xpos_max >= 2427:
                globalcounter += 1
                done = True

            if globalcounter >= 10:
                fitness_current += 10000

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 200:
                done = True
                with open('stats.csv', 'a+', newline='') as csvfile:
                    filewrite = csv.writer(
                        csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewrite.writerow([generation, id_count, xpos])
                print("Posicion Maxima: ", xpos_max)
                # print('id:', id_count, 'gen:', generation)
                id_count += 1

                if id_count == 61:
                    generation += 1
                    id_count = 1

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

if resume == True:
    try:
        p = neat.Checkpointer.restore_checkpoint(restore_file)
    except:
        print("ERROR: checkpoint doesn't exist")
        exit()

else:
    p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
