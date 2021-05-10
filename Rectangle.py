

from shapely.geometry import Polygon
import random
from random import randint, uniform
import copy
from math import sqrt
import numpy as np
import numpy as np
from scipy import *
from math import *
from matplotlib.pyplot import *
from functools import *
import pyclipper
import sys
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Polygon as PltPolygon
from copy import deepcopy
import statistics

def draw_polygon(polygon, ax=None, color="green"):
    """Draw a polygon over an axis (not shown directly)

    @param polygon: Whether a polygon or a path
    """
    if ax is None:
        ax = plt.gca()

    p = PltPolygon(polygon, closed=False, color=color, alpha=0.3, lw=0)
    ax.add_patch(p)
    ax.axis('equal')
    return ax

def draw_polygons(polygons, colors=None, verbose=False):
    """Draw polygons and print the figure

    @param polygons: SimpyPolygon or list(SympyPolygon)
    @param colors: colors to match to the polygons
    """
    ax = plt.gca()
    if colors is None:
        colors = ["grey"]*len(polygons)
    if len(colors) > 0 and len(colors) < len(polygons):
        colors = colors + [colors[-1]] * (len(polygons) - len(colors))
    colors.reverse()

    colors = deepcopy(colors)
    for polygon in polygons:
        color = colors.pop()
        if verbose:
            print(color, "Polygon : ", polygon)
        ax = draw_polygon(polygon, ax, color=color)
    return ax

def column(matrix, i):
    return [row[i] for row in matrix]

class Point(object):
    """docstring for Point."""

    def __init__(self, x,y):
        super(Point, self).__init__()
        self.x = x
        self.y = y

    def print_Point(self):
        print("[",self.x,";",self.y,"]")

    def distance(self ,other):
        return sqrt( pow(self.x - other.x,2) + pow(self.y - other.y,2) )


class Rectangle(object):
    """docstring for Rectangle."""

    def __init__(self, x1,y1,x2,y2,l, trial = 0):
        super(Rectangle, self).__init__()
        self.p1 = Point(x1,y1)
        self.p2 = Point(x2,y2)
        self.l= l
        self.list = [x1,y1,x2,y2,l]
        self.trial=trial

    def print_Rect(self):
        print("_________________________")
        self.p1.print_Point()
        self.p2.print_Point()
        print("l : ",self.l)
        print("_________________________")

    def update(self):
        x1 =self.list[0]
        y1 =self.list[1]
        x2 =self.list[2]
        y2 =self.list[3]
        l =self.list[4]
        self.p1 = Point(x1,y1)
        self.p2 = Point(x2,y2)
        self.l= l

    def to4points(self):
        b2=self.p2.y
        a2=self.p1.y
        b1=self.p2.x
        a1=self.p1.x
        l=self.l
        #print(a1,a2,b1,b2,l)
        n = sqrt((b2 - a2)**2 + (b1 - a1)**2)

        l = -l/n

        d1 = l*(a2 - b2) + a1
        d2 = l*(b1 - a1) + a2
        c1 = l*(a2 - b2) + b1
        c2 = l*(b1 - a1) + b2
        return(np.array([[(a1), (a2)], [(b1), (b2)], [(c1), (c2)], [(d1), (d2)]]))



    def eval (self):
        return self.p1.distance(self.p2) * self.l


    def toList(self):
        return [self.p1.x,self.p1.y,self.p2.x,self.p2.y,self.l]

    def move(self,speed):
        self.p1.x = self.p1.x + speed[0]
        self.p1.y = self.p1.y + speed[1]
        self.p2.x = self.p2.x + speed[2]
        self.p2.y = self.p2.y + speed[3]
        self.l = self.l +speed[4]
        return(self)

def toRectangle(l , trial = 0):
    #print(l)
    return Rectangle(l[0],l[1],l[2],l[3],l[4], trial)



class Probleme(object):
    """docstring for Probleme."""
    def __init__(self, poly, param):
        super(Probleme, self).__init__()
        self.poly = poly
        self.infy = min(column(poly,1))
        self.supy = max(column(poly,1))
        self.infx = min(column(poly,0))
        self.supx = max(column(poly,0))
        self.param= param

    def evaluate(self,sol):
        #print("#sol ",sol)
        poly_res = Polygon(sol.to4points())
        if Polygon(self.poly).contains(poly_res):
            return sol.eval()
        else :
            #print("out")
            return(-1)

    def initOne(self):
        l=random.uniform(0,max(self.supx-self.infx,self.supy-self.infy))
        p1=Point(random.uniform(self.infx,self.supx),random.uniform(self.infy,self.supy))
        p2=Point(random.uniform(self.infx,self.supx),random.uniform(self.infy,self.supy))
        food=Rectangle(p1.x,p1.y,p2.x,p2.y,l)
        while not Polygon(self.poly).contains(Polygon(food.to4points())):
            l=random.uniform(0,max(self.supx-self.infx,self.supy-self.infy))
            p1=Point(random.uniform(self.infx,self.supx),random.uniform(self.infy,self.supy))
            p2=Point(random.uniform(self.infx,self.supx),random.uniform(self.infy,self.supy))
            food=Rectangle(p1.x,p1.y,p2.x,p2.y,l)
        return food

    def initSeveral(self):
        food_sources=[]
        while len(food_sources) < self.param['population_size']:
            food = self.initOne()
            food_sources.append(food)
        return(food_sources)

    def condition_arret(self, i):
        if (i%100)==0 :
            buffer = ("iter : "+ str(i)+" / "+str(self.param['max_num_iteration']))
            print (buffer, end="\r")
        return i>self.param['max_num_iteration']

    #ABC
    def generate_neighbour(self, food_rank, food_sources):
        #Generation
        voisin_valide = False
        while not voisin_valide:

            random_partner = randint(0,self.param['population_size']-1)
            while random_partner == food_rank:
                random_partner = randint(0,self.param['population_size']-1)
            random_variable = randint(0,self.param['dimension']-1)

            old_food=food_sources[food_rank].toList()
            #print(old_food)
            new_food = deepcopy(old_food)

            new_variable = old_food[random_variable] \
                + uniform(-1,1) * (old_food[random_variable] - food_sources[random_partner].toList()[random_variable])
            new_food [random_variable] = new_variable
            new_food = toRectangle(new_food)

            poly_new_food = Polygon(new_food.to4points())
            if Polygon(self.poly).contains(poly_new_food):
                voisin_valide = True

            old_food=food_sources[food_rank]

        #Evaluation
        if self.evaluate(new_food)>self.evaluate(old_food):
            return new_food
        else:
            old_food.trial += 1
            return old_food

    def compute_proba_foods(self, food_sources):
        eval_all_food_neg = [-food.eval() for food in food_sources]#the minus is here to transpose to a minimization probleme
        fitness_all_food=[]
        for eval in eval_all_food_neg:
            if eval>= 0:
                fit = 1/(eval+1)
            else:
                fit = 1 + abs(eval)
            fitness_all_food.append(fit)
        fit_total = np.sum(fitness_all_food)
        proba_all_food = [fit/fit_total for fit in fitness_all_food]
        return proba_all_food

    def search_Best(self ,old_best, food_sources):
        for food in food_sources:
            if food.eval()>old_best.eval():
                old_best = deepcopy(food)
        return old_best

    def scout_phases(self, food_sources):
        limit_value = self.param['max_cycle_on_food_source']
        all_trials = [food.trial for food in food_sources]
        max_trial = max(all_trials)
        if max_trial<limit_value:
            return food_sources
        else:
            selected = []
            not_selected = []
            for food in food_sources:
                if food.trial == max_trial:
                    selected.append(food)
                else:
                    not_selected.append(food)
            random_rank = randint(0,len(selected)-1)
            selected[random_rank] = self.initOne()
            return selected + not_selected

    def optimize_ABC(self):
        if self.param['max_cycle_on_food_source']==None:
            self.param['max_cycle_on_food_source'] = self.param['population_size']*self.param['dimension']
        iter=0
        food_sources = self.initSeveral()
        best_solution = Rectangle(0,0,0,0,0)
        while not self.condition_arret(iter):

            #Employed Bee Phase
            for food_rank in range(self.param['population_size']-1):
                food_sources[food_rank] = self.generate_neighbour(food_rank, food_sources)

            #calcul the vect of probabilities for Onlooker Bee Phase
            proba_all_food = self.compute_proba_foods(food_sources)
            #Onlooker Bee Phase
            for onlooker in range(self.param['population_size']-1):
                found_food = False
                food_rank=0
                while not found_food:
                    r = uniform(0,1)
                    if r <=proba_all_food[food_rank]:
                        food_sources[food_rank] = self.generate_neighbour(food_rank, food_sources)
                        found_food  = True
                    food_rank += 1
                    if food_rank>self.param['population_size']-1:
                        food_rank = 0

            best_solution = self.search_Best(best_solution, food_sources)
            #Scout Phase
            food_sources = self.scout_phases(food_sources)
            iter+=1
        #print("Best sol :",best_solution)
        print("Done ! ")
        return(best_solution)

    #PSO
    class Particle_PSO(object):
        """docstring for Particle_PSO."""
        def __init__(self, pos,param):
            self.pos = pos
            self.best_perso = pos
            self.speed = [0]*param['dimension']

        def update_best(self):
            if self.pos.eval()>self.best_perso.eval():
                self.best_perso = self.pos
            return(self)

        def update_speed(self,param,best_global):
            psi = param['Psi']
            cMax = param['Cmax']
            pos = self.pos.toList()
            pos_Best = self.best_perso.toList()
            for rank in range(param['dimension']):
                self.speed[rank]= psi*self.speed[rank] \
                    +cMax*uniform(0,1)*(pos_Best[rank]-pos[rank]) \
                    +cMax*uniform(0,1)*(best_global.toList()[rank]-pos[rank])
            return(self)

        def update_position(self):
            self.pos=self.pos.move(self.speed)
            return self

    def init_PSO(self):
        return [self.Particle_PSO(self.initOne(),self.param) for i in range(self.param['population_size'])]

    def search_Best_Pso(self,best,swarm):
        for particle in swarm:
            if self.evaluate(particle.pos)>self.evaluate(best):
                best=particle.pos
        return best

    def optimize_PSO(self):
        iter = 0
        swarm = self.init_PSO()
        best_solution = self.search_Best_Pso(Rectangle(0,0,0,1,1),swarm)
        while not self.condition_arret(iter):
            #update best position
            swarm=[particle.update_best() for particle in swarm]
            #compute new speed
            swarm=[particle.update_speed(self.param,best_solution) for particle in swarm]
            #update new position
            for rank in range(len(swarm)-1):
                new_particle = swarm[rank].update_position()
                if self.evaluate(new_particle.pos)>=0:
                    swarm[rank]=new_particle
                    #print("changed")
                else:
                    #print("stopped")
                    swarm[rank].speed = [0]*self.param['dimension']
            #update best global solution
            best_solution=self.search_Best_Pso(best_solution,swarm)
            iter += 1
        print("Done ! ")
        return(best_solution)



def draw_from_Rect(rect,poly):
    res= rect.to4points().tolist()
    print("Coordinates = ",res)
    print("Res = ",rect.eval())
    draw_polygons([res,poly], colors = ["blue", "green"])

def opti_ABC_and_Draw(poly,param):
    model = Probleme(poly,param)
    rect=model.optimize_ABC()
    draw_from_Rect(rect,poly)

algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':20,\
                   'dimension':5,\
                   'max_cycle_on_food_source':None,\
                   'Psi': 0.7,\
                   'Cmax':1.47} #If max_cycle_on_food_source is None then is set as population_size*dimension
#'max_iteration_without_improv':10,\
#population_size is the number of employed bees and food sources

poly1 = [[50,150],[200,50],[350,150],[350,300],[250,300],[200,250],[150,350],[100,250],[100,200]]
poly2 = [[10,10],[10,400],[400,400],[400,10]]
poly3 = [[10,10],[10,300],[250,300],[350,130],[200,10]]
poly4 = [[50,50],[50,400],[220,310],[220,170],[330,170],[330,480],[450,480],[450,50]]


model = Probleme(poly2,algorithm_param)

rect=model.optimize_ABC().to4points()
print(rect)
rect=model.optimize_PSO().to4points()
print(rect)
