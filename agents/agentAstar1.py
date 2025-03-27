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

from captureAgents import CaptureAgent
import random
import util
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAstarAgent', second = 'DefensiveAstarAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

from captureAgents import CaptureAgent
from game import Directions
from util import manhattanDistance
import heapq

class AStarAgent(CaptureAgent):
    def registerInitialState(self, gameState, parent=None, action=None):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.parent = parent
        self.action = action

    def getFeatures(self, state, initScore):
        features = util.Counter()
        enemies = [state.getAgentState(opp) for opp in self.getOpponents(state)]
        invaders = [opp for opp in enemies if opp.isPacman and opp.getPosition() != None]
        defenders = [opp for opp in enemies if (not opp.isPacman) and opp.getPosition() != None]
        food = self.getFood(state).asList()
        oppFood = self.getFoodYouAreDefending(state).asList()


        # Get the position and carrying status of the agent
        agent_pos = state.getAgentPosition(self.index)
        features['carry'] = state.getAgentState(self.index).numCarrying
        features['score'] = self.getScore(state) - initScore
        if not self.parent is None:
            if not self.parent.action is None:
                if self.action == Directions.REVERSE[self.parent.action] and self.index == self.index: features['reverse'] = 1
        if self.action == Directions.STOP and self.index == self.index: features['stop'] = 1

        # Get the number of opponents scared and caught
        features['oppCarry'] = sum([opp.numCarrying for opp in enemies])

        if len(food) > 0:
            features['distFood'] = min([self.getMazeDistance(agent_pos, food_pos) for food_pos in food])

        if len(oppFood) > 0:
            features['distOppFood'] = min([self.getMazeDistance(agent_pos, food_pos) for food_pos in oppFood])

        if len(invaders) > 0:
            features['distInvaders'] = min([self.getMazeDistance(agent_pos, opp.getPosition()) for opp in invaders])

        if len(defenders) > 0:
            features['distDefenders'] = min([self.getMazeDistance(agent_pos, opp.getPosition()) for opp in defenders])

        return features
    
    def rewardFunction(self, state, initScore):
        features = self.getFeatures(state, initScore)
        weights = self.getWeights()

        return features * weights
    
    def getSuccessors(self, state):
        successors = []
        for action in state.getLegalActions(self.index):
            successor = state.generateSuccessor(self.index, action)
            cost = self.rewardFunction(successor, self.getScore(successor))
            successors.append((successor, action, cost))
        return successors

    def aStarSearch(self, gameState):
        startState = gameState
        startNode = (self.getStateId(startState), startState.getLegalActions(self.index), 0)
        visited = set()
        queue = []
        heapq.heappush(queue, (0, startNode))
        actions_list = []
        while queue:
            _, currentNode = heapq.heappop(queue)
            _, actions, cost = currentNode
            currentState = self.getStateFromId()
            if currentState.isOver():
                return actions
            if currentState not in visited:
                visited.add(currentState)
                successors = self.getSuccessors(currentState)
                for nextState, action, stepCost in successors:
                    actions_list.append(action)
                    nextCost = cost + stepCost
                    nextStateId = self.getStateId(nextState)
                    nextNode = (nextStateId, actions + [action], nextCost)
                    priority = nextCost + self.rewardFunction(nextState, self.getScore(nextState))
                    heapq.heappush(queue, (priority, nextNode))
            return max(queue)

    def getStateId(self, state):
        agentState = state.getAgentState(self.index)
        pos = agentState.getPosition()
        direction = agentState.getDirection()
        carrying = agentState.numCarrying
        return (pos, direction, carrying)

    def getStateFromId(self):
        agentState = self.getCurrentObservation().getAgentState(self.index)
        state = self.getCurrentObservation().deepCopy()
        state.data.agentStates[self.index] = agentState
        return state

    def oppositeAction(self, action):
        if action == "North": return "South"
        elif action == "South": return "North"
        elif action == "East": return "West"
        elif action == "West": return "East"


class OffensiveAstarAgent(AStarAgent):
    def __init__(self, index, timeForComputing=0.2):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing

    def getWeights(self):
        return {'score': 1000.0, 'distFood': -1.0, 'distOppFood': 0.0, 'distInvaders': -0.2, 'distDefenders': 0.8, 
                'carry': 10.0, 'oppCarry': 0.0, 'stop': -2.0, 'reverse': -2.0}
    
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actionPath = self.aStarSearch(gameState)
        action = actionPath[1][0][1]
        if action in gameState.getLegalActions(self.index):
            return action
        else:
            # return Directions.STOP
            return self.oppositeAction(action)


class DefensiveAstarAgent(AStarAgent):
    def __init__(self, index, timeForComputing=0.2):
        CaptureAgent.__init__(self, index)
        self.timeForComputing = timeForComputing

    def getWeights(self):
        return {'score': 0.0, 'distFood': 0.0, 'distOppFood': -1.0, 'distInvaders': -10.0, 'distDefenders': 0.0, 
        'carry': 0.0, 'oppCarry': -10.0, 'stop': -2.0, 'reverse': -1.0}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actionPath = self.aStarSearch(gameState)
        action = actionPath[1][0][1]
        if action in gameState.getLegalActions(self.index):
            return action
        else:
            # return Directions.STOP
            return self.oppositeAction(action)