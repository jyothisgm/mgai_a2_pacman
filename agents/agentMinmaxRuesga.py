# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import math
import time
import numpy as np
import util
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, copy
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSPacman', second = 'MCTSPacman'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex, mode="attack"), eval(second)(secondIndex, mode="defense")]

##########
# Agents #
##########

class Node:
    def __init__(self, state, agentIndex, index,  parent=None, action=None,  agent= None, mode="attack", myTurn=True):
        self.state = state
        self.agentIndex = agentIndex
        self.index = index
        self.agent = agent
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.untriedActions = self.state.getLegalActions(index)
        self.mode = mode
        self.myTurn=myTurn
            
    def isLeaf(self):
        return len(self.children) == 0
    
    def isTerminal(self):
        return self.state.isOver()
    
    def isFullyExpanded(self):
        return len(self.untriedActions) == 0
    
    def bestChild(self):
        minVisits = np.inf
        bestChild = None
        
        for child in self.children:
            if child.visits < minVisits:
                minVisits = child.visits
                bestChild = child
        
        return bestChild
    
    def expand(self):
        action = np.random.choice(self.untriedActions)
        self.untriedActions.remove(action)
        nextState = self.state.generateSuccessor(self.index, action)
        child = Node(nextState, self.agentIndex, (self.index+1) % nextState.getNumAgents(), parent=self, action=action, agent=self.agent, mode=self.mode, myTurn=(not self.myTurn))
        self.children.append(child)

        return child
    
    def treePolicy (self):
        node = self
        while not node.isTerminal():
            if not node.isFullyExpanded(): return node.expand()
            else: node = node.bestChild()
        return node
    
    def backup(self):
        self.visits += 1
        
        if self.parent is not None:
            self.parent.backup()
    
    def rewardFunction(self, state, initScore):
        features = self.getFeatures(state, initScore)
        weights = self.getWeights()

        return features * weights
    
    def getFeatures(self, state, initScore):
        features = util.Counter()

        # Get the position and carrying status of the agent
        agent_pos = state.getAgentPosition(self.agentIndex)

        # Get lists of items to use
        enemies = [state.getAgentState(opp) for opp in self.agent.getOpponents(state)]
        invaders = [opp for opp in enemies if opp.isPacman and opp.getPosition() != None]
        defenders = [opp for opp in enemies if (not opp.isPacman) and opp.getPosition() != None]
        food = self.agent.getFood(state).asList()
        oppFood = self.agent.getFoodYouAreDefending(state).asList()

        
        # Calculate and add features
        features['score'] = self.agent.getScore(state) - initScore
        features['carry'] = state.getAgentState(self.agentIndex).numCarrying
        features['oppCarry'] = sum([opp.numCarrying for opp in enemies])
        
        if len(food) > 0:
            features['distFood'] = min([self.agent.getMazeDistance(agent_pos, food_pos) for food_pos in food])

        if len(oppFood) > 0:
            features['distOppFood'] = min([self.agent.getMazeDistance(agent_pos, food_pos) for food_pos in oppFood])

        if len(invaders) > 0:
            features['distInvaders'] = min([self.agent.getMazeDistance(agent_pos, opp.getPosition()) for opp in invaders])

        if len(defenders) > 0:
            features['distDefenders'] = min([self.agent.getMazeDistance(agent_pos, opp.getPosition()) for opp in defenders])

        return features
    
    def getWeights(self):
        if self.mode == "attack":
            return {'score': 1000.0, 'distFood': -1.0, 'distOppFood': 0.0, 'distInvaders': -0.5, 'distDefenders': 0.5, 'carry': 10.0, 'oppCarry': 0.0}
        
        elif self.mode == "defense":
            return {'score': 10.0, 'distFood': 0.0, 'distOppFood': -1.0, 'distInvaders': -10.0, 'distDefenders': 0.0, 'carry': 0.0, 'oppCarry': -10.0}

    def printRewards(self, initScore):
        print(" ")
        print("Agent Index: " + str(self.agentIndex) + " | Agent position: " + str(self.state.getAgentPosition(self.agentIndex)) + " | visits: " + str(self.visits) + " | Carrying: " + str(self.state.getAgentState(self.agentIndex).numCarrying))
        print("---- ACTIONS: REWARDS / VISITS / AVERAGE ----")
        self.printTree("- ", initScore)

    def printTree(self, string, initScore):
        for child in self.children:
            rewards = child.evaluateTree(initScore)
            print(string + child.action + ": " + str(rewards) + " / " + str(child.visits) + " / " + str(rewards/self.visits))
            #child.printTree("   " + string)

    def evaluateTree(self, initScore):
        if self.isFullyExpanded():
            if self.myTurn: return max([child.evaluateTree(initScore) for child in self.children])
            else: return min([child.evaluateTree(initScore) for child in self.children])
        else: return self.rewardFunction(self.state, initScore)

    def minmaxAction(self, initScore):
        maxScore = float("-inf")
        bestChild = None
        for child in self.children:
            score = child.evaluateTree(initScore)
            if score > maxScore:
                bestChild = child
                maxScore = score

        return bestChild.action

class MCTSPacman(CaptureAgent):
    def __init__(self, index, mode="attack", timeForComputing=0.5):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing
        self.serchTime = 0.1*timeForComputing
        self.mode = mode

    def registerInitialState(self, state):
        CaptureAgent.registerInitialState(self, state)
        self.start = state.getAgentPosition(self.index)
        self.prevAction = None
        
    def chooseAction(self, initState):
        # Set the start state
        state = initState.deepCopy()

        # Create the root node
        rootNode = Node(state, self.index, self.index, action=self.prevAction, agent=self, mode=self.mode)
        initScore = copy.deepcopy(self.getScore(state))

        # Start the MCTS algorithm
        startTime = time.time()
        while time.time() - startTime < self.serchTime:
            # Select - Tree Policy
            node = rootNode.treePolicy()

            # Backup
            node.backup()

        rootNode.printRewards(initScore)
        action = rootNode.minmaxAction(initScore)

        # Update the search time
        self.serchTime += (self.timeForComputing - time.time() + startTime)*0.01
        print("Search time: " + str(self.serchTime))
        print("Total time: " + str(time.time() - startTime))
        print("")

        return action