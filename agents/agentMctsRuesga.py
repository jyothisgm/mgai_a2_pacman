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
    def __init__(self, state, agentIndex, index,  parent=None, action=None, depth=0, max_depth=np.inf, agent= None, mode="attack", myTurn=True, initScore=0):
        self.state = state
        self.agentIndex = agentIndex
        self.index = index
        self.agent = agent
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.totalReward = 0
        self.untriedActions = self.state.getLegalActions(index)
        self.depth=depth
        self.max_depth = max_depth
        self.mode = mode
        self.myTurn=myTurn
        self.initScore=initScore
            
    def isLeaf(self):
        return len(self.children) == 0
    
    def isTerminal(self):
        return self.state.isOver()
    
    def isFullyExpanded(self):
        return len(self.untriedActions) == 0
    
    def bestChild(self, c=10000.0):
        bestScore = float('-inf')
        bestChild = None
        
        for child in self.children:
            exploitation = child.totalReward / child.visits
            exploration =  (2*math.log(self.visits) / child.visits) ** 0.5
            score = exploitation + c * exploration
            
            if score > bestScore:
                bestScore = score
                bestChild = child
        
        return bestChild
    
    def expand(self):
        action = np.random.choice(self.untriedActions)
        self.untriedActions.remove(action) #np.delete(self.untriedActions, np.where(self.untriedActions == action))
        nextState = self.state.generateSuccessor(self.index, action)
        child = Node(nextState, self.agentIndex, (self.index+1) % nextState.getNumAgents(), parent=self, action=action, depth=self.depth+1, max_depth=self.max_depth, agent=self.agent, mode=self.mode, myTurn=(not self.myTurn), initScore=self.initScore)
        self.children.append(child)

        return child
    
    def defaultPolicy(self):
        state = self.state.deepCopy()
        agentIndex = self.index
        i = self.depth
        while (not state.isOver()) and i < self.max_depth:
            actions = state.getLegalActions(agentIndex)
            action = np.random.choice(actions)
            state = state.generateSuccessor(agentIndex, action)
            agentIndex = (agentIndex + 1) % state.getNumAgents()
            i+=1
        
        return self.rewardFunction(state)
    
    def treePolicy (self):
        node = self
        while not node.isTerminal():
            if not node.isFullyExpanded():
                return node.expand()
            else:
                node = node.bestChild()
        return node
    
    def backup(self, reward):
        self.visits += 1
        self.totalReward += reward
        
        if self.parent is not None:
            self.parent.backup(reward)
    
    def rewardFunction(self, state):
        features = self.getFeatures(state)
        weights = self.getWeights()

        return features * weights
    
    def getFeatures(self, state):
        features = util.Counter()
        enemies = [state.getAgentState(opp) for opp in self.agent.getOpponents(state)]
        invaders = [opp for opp in enemies if opp.isPacman and opp.getPosition() != None]
        defenders = [opp for opp in enemies if (not opp.isPacman) and opp.getPosition() != None]
        food = self.agent.getFood(state).asList()
        oppFood = self.agent.getFoodYouAreDefending(state).asList()


        # Get the position and carrying status of the agent
        agent_pos = state.getAgentPosition(self.agentIndex)
        features['carry'] = state.getAgentState(self.agentIndex).numCarrying
        features['score'] = self.agent.getScore(state) - self.initScore
        if not self.parent is None:
            if not self.parent.action is None:
                if self.action == Directions.REVERSE[self.parent.action] and self.agentIndex == self.index: features['reverse'] = 1
        if self.action == Directions.STOP and self.agentIndex == self.index: features['stop'] = 1
        
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
            return {'score': 100.0, 'distFood': 0.0, 'distOppFood': -1.0, 'distInvaders': -10.0, 'distDefenders': 0.0, 'carry': 0.0, 'oppCarry': -50.0}

    def printRewards(self):
        print(" ")
        print("Agent Index: " + str(self.agentIndex) + " | Agent position: " + str(self.state.getAgentPosition(self.agentIndex)) + " | visits: " + str(self.visits) + " | Carrying: " + str(self.state.getAgentState(self.agentIndex).numCarrying))
        print("---- ACTIONS: REWARDS / VISITS / AVERAGE ----")
        self.printTree("- ")

    def printTree(self, string):
        for child in self.children:
            print(string + child.action + ": " + str(child.totalReward) + " / " + str(child.visits) + " / " + str(child.totalReward/child.visits))
            #child.printTree("   " + string)

class MCTSPacman(CaptureAgent):
    def __init__(self, index, mode="attack", max_depth=40, timeForComputing=0.9):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing
        self.max_depth = max_depth
        self.mode = mode

    def registerInitialState(self, state):
        CaptureAgent.registerInitialState(self, state)
        self.start = state.getAgentPosition(self.index)
        self.prevAction = None
        
    def chooseAction(self, initState):
        # Set the start state
        state = initState.deepCopy()
        
        # Create the root node
        rootNode = Node(state, self.index, self.index, max_depth=self.max_depth, action=self.prevAction, agent=self, mode=self.mode, initScore=self.getScore(state))
        
        # Start the MCTS algorithm
        startTime = time.time()
        while time.time() - startTime < self.timeForComputing:
            # Select - Tree Policy
            node = rootNode.treePolicy()

            reward = node.defaultPolicy()

            # Backup
            node.backup(reward)
        
        rootNode.printRewards()
        print("Total time: " + str(time.time() - startTime))
        print("")
        # Return the best action
        return rootNode.bestChild(c=0.0).action