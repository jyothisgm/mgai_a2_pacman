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

from game import Directions, Agent, Actions
from captureAgents import CaptureAgent, AgentFactory
import random, util, distanceCalculator 
from util import nearestPoint

import math

######################
# Parameters of MCTS #
######################

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR = 1/math.sqrt(2.0)
NUM_SIM = 10    # number of simulation times, at lease 1
# REWARD_DISCOUNT=0.8
REWARD_DISCOUNT = 0   # if don't want to use roll out, just turn reward discount into 0
SIM_LEVEL = 1   # level of tree expanding
LEVEL = 0       # level of simulation tree (not used)


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
              first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##############
# MCTSAgents #
##############

class MCTSAgent(CaptureAgent):
  """
  A Monte-Carlo Tree Search base Agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # # Once attacker capture food, then it go back to home
    # agentState = gameState.getAgentState(self.index)

    # if agentState.numCarrying > 1:
    #   bestDist = 9999
    #   for action in actions:
    #     successor = self.getSuccessor(gameState, action)
    #     pos2 = successor.getAgentPosition(self.index)
    #     dist = self.getMazeDistance(self.start,pos2)
    #     if dist < bestDist:
    #       bestAction = action
    #       bestDist = dist
    #   return bestAction

    # # Once total food left <= 2, then they go back to home
    # foodLeft = len(self.getFood(gameState).asList())

    # if foodLeft <= 2:
    #   bestDist = 9999
    #   for action in actions:
    #     successor = self.getSuccessor(gameState, action)
    #     pos2 = successor.getAgentPosition(self.index)
    #     dist = self.getMazeDistance(self.start,pos2)
    #     if dist < bestDist:
    #       bestAction = action
    #       bestDist = dist
    #   return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.evaluateAttackParameters(gameState, action)
    weights = self.getCostOfAttackParameter(gameState, action)
    return features * weights

  def evaluateAttackParameters(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getCostOfAttackParameter(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

######################
# OffensiveReflexAgent #
######################

class OffensiveReflexAgent(MCTSAgent):
    """
    A Monte Carlo Tree Search-based offensive agent that seeks food efficiently.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        self.numSims = NUM_SIM
        # Maximum number of simulations per turn
        self.sturns = SIM_LEVEL
        # Depth of the MCTS tree
        self.levels = LEVEL
        # Initial value of the agent's state
        self.svalue = 0
        # List to store the current agent's move history
        self.smoves = []
        self.gameState = gameState
        # The current tree node representing the agent's state
        self.current_node = Node(MState(self.gameState, self.index, self.svalue, self.smoves, self.sturns, agent_type='OffensiveReflexAgent'))
        # Starting position of the agent
        self.start = self.current_node.mstate.gameState.getAgentState(self.index).getPosition()

    def getOffAction(self, gameState):
        """
        Prevent CaptureAgent from always using the overridden chooseAction.
        """
        self.observationHistory.append(gameState)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos != nearestPoint(myPos):
            # If the agent is between grid positions, keep moving in the current direction.
            return gameState.getLegalActions(self.index)[0]
        else:
            return self.chooseOffAction(gameState)

    def chooseOffAction(self, gameState):
        """
        Selects an action using Monte Carlo Tree Search.
        """
        return self.runSimulation(self.current_node, gameState, self.index, self.levels, self.numSims)

    def runSimulation(self, current_node, gameState, index, numSims=5):
        """
        Runs MCTS to find the best action.
        """
        self.gameState = gameState
        self.index = index
        self.current_node = current_node

        value = 0
        self.current_node.resetNode()
        self.current_node.mstate.resetMState(self.gameState, index, value)
        
        self.index = index
        self.current_node.children = []

        # l = levels
        child_node = UCTSEARCH(numSims, self.current_node, self.index)

        return child_node.mstate.fromMove

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluateAttackParameters(self, gameState, action):
        """
        Compute attack strategy features for weighted evaluation in evaluate().
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        position = successor.getAgentState(self.index).getPosition()
        previousGameState = self.getPreviousObservation()
        
        features['successorScore'] = self.getScore(successor)
        features['offence'] = 1 if successor.getAgentState(self.index).isPacman else 0

        # # food number
        # features['foodNum'] = len(self.getFood(successor).asList())

        # # Compute features related to food
        # features['foodDistance'] = self.getNearestFoodDistance(successor, position)

        # Compute features related to retreat
        features['RetreatScore'] = self.getRetreatScore(successor, position)
        
        # foodList = self.getFood(successor).asList()
        # localFoodCount = sum(1 for food in foodList if self.getMazeDistance(position, food) <= 10)  # Count food in 5-step range

    
        # Compute food score: balance food density and nearest food distance
        features['foodScore'] = self.getNearestFoodDistance(successor, position)

        # Compute features related to ghosts
        features['distanceToGhost'] = self.getGhostThreat(successor, position, features['successorScore'])

        # Number of nearby ghosts
        features['numNearbyGhosts'] = self.getNearbyGhostCount(successor, position, radius=3)

        # Distance to nearest power capsule
        features['capsuleDistance'] = self.getNearestCapsuleDistance(successor, position)
        # features['pathDiversion'] = self.getPathDiversion(gameState)
        
        # Food consumption efficiency
        if previousGameState:
            prevFood = len(self.getFood(previousGameState).asList())
            features['foodEaten'] = prevFood - features['foodNum']  # Food reduction count, incentivizes eating food
        if action == Directions.STOP:
            features['stopPenalty'] = 1  # Penalize stopping
        
        currentDirection = gameState.getAgentState(self.index).configuration.direction
        reverseDirection = Directions.REVERSE[currentDirection]
        if successor.getAgentState(self.index).numCarrying > 0:
            features['foodScore'] *= 0.1
        if action == reverseDirection:
            features['reversePenalty'] = 1
        
        if not hasattr(self, 'recentPositions'):
            self.recentPositions = []
        
        self.recentPositions.append(position)

        if len(self.recentPositions) > 4:
            self.recentPositions.pop(0) 

        if len(self.recentPositions) == 4:
            A, B, C, D = self.recentPositions
            # Check for clockwise cycle (A → B → C → D → A) or counterclockwise cycle (A → D → C → B → A)
            if (A == C and B == D) or (A == D and B == C):
                features['cyclePenalty'] = 100
        return features

    ###########
    # Capsule #
    ###########
    def getNearestCapsuleDistance(self, successor, position):
        """Returns the distance to the nearest power capsule, if available."""
        capsuleList = self.getCapsules(successor)
        return min([self.getMazeDistance(position, cap) for cap in capsuleList]) if capsuleList else float('inf')
    
    ##############
    # GhostCount #
    ##############    
    def getNearbyGhostCount(self, successor, position, radius=3):
        """
        Compute the number of ghosts within the given `radius`.
        """
        ghostPositions = [
            successor.getAgentState(enemy).getPosition()
            for enemy in self.getOpponents(successor)
            if not successor.getAgentState(enemy).isPacman and successor.getAgentState(enemy).getPosition() is not None
        ]

        return sum(1 for ghostPos in ghostPositions if self.getMazeDistance(position, ghostPos) <= radius)
    
    ################
    # RetreatScore #
    ################      
    def getRetreatScore(self, successor, position):
        """
        Computes a retreat score to ensure the agent prioritizes returning directly to its own half.
        Higher values indicate a greater need to retreat.
        """
        # Determine the boundary of the agent's own half of the map
        mapWidth = successor.data.layout.width
        borderX = (mapWidth // 2) - 1 if self.red else (mapWidth // 2)
        borderPositions = [(borderX, y) for y in range(successor.data.layout.height) if not successor.hasWall(borderX, y)]
        
        # Compute the shortest path to the boundary
        borderDistance = min(self.getMazeDistance(position, border) for border in borderPositions)

        # Calculate the nearest ghost distance
        ghostPositions = [
            successor.getAgentState(enemy).getPosition()
            for enemy in self.getOpponents(successor)
            if not successor.getAgentState(enemy).isPacman and successor.getAgentState(enemy).getPosition() is not None
        ]
        minGhostDistance = min((self.getMazeDistance(position, ghost) for ghost in ghostPositions), default=float('inf'))
        
        # Calculate the urgency of retreat (boundary distance, ghost threat, food carried)
        ghostThreatFactor = 20 / (minGhostDistance + 1)  # The closer the ghost, the higher the threat
        carriedFood = successor.getAgentState(self.index).numCarrying
        foodWeight = carriedFood ** 1.5  # The more food carried, the stronger the retreat urge
        retreatScore = (100 / (borderDistance + 1)) + ghostThreatFactor + foodWeight  # Higher retreatScore means higher urgency to retreat
        return retreatScore
    
    #######################
    # NearestFoodDistance #
    #######################     
    def getNearestFoodDistance(self, successor, position):
        """
        Compute the maze distance from the current position to the nearest food, 
        and boost the score when the food is close to encourage Pacman to move toward it.
        """
        foodList = self.getFood(successor).asList()  # Get list of food positions
        if not foodList:
            return 0  # Return 0 if there is no food

        # Calculate the distance to each food item and find the minimum distance
        minFoodDistance = min(self.getMazeDistance(position, food) for food in foodList)

        # **Encourage moving toward food if it is close** (foodScore should be stronger when the food is nearby)
        if minFoodDistance <= 3:  # If the food is within 3 steps
            successorScore = self.getScore(successor)
            minFoodDistance -= successorScore  # Encourage closer food by reducing distance with score

        return minFoodDistance
    
    ###############
    # GhostThreat #
    ###############
    def getGhostThreat(self, successor, position, successorScore):
        """Calculates the threat posed by enemy ghosts."""
        disToGhost = []
        for enemyPos in self.getOpponents(successor):
            enemy = successor.getAgentState(enemyPos)
            if not enemy.isPacman and enemy.getPosition() is not None:
                ghostPos = enemy.getPosition()
                disToGhost.append(self.getMazeDistance(position, ghostPos))
        
        if disToGhost:
            minDisToGhost = min(disToGhost)
            # threat score
            return minDisToGhost + successorScore if minDisToGhost < 5 else 0
        return 0
    


    
    def getCostOfAttackParameter(self, gameState, action):
        """
        Compute the weights for the attack strategy to yield a more reasonable evaluation score, 
        while enhancing ghost avoidance, making the agent favor safe offensive paths over pure retreat.
        Also integrates the capsule distance to enhance survival and retreat strategy.
        """
        successor = self.getSuccessor(gameState, action)

        # **Initialize weights**
        weights = {
            'successorScore': 2000,  
            'offence': 100,  
            'foodScore': 5000,  
            'distanceToGhost': 0,  
            'RetreatScore': 0,  
            'foodEaten': 500,
            'numNearbyGhosts': 0,
            'stopPenalty': -1000,  # Penalize stopping
            'reversePenalty': -1000,
            'cyclePenalty': -5000,
            'capsuleDistance': -200  # Initialize the capsule distance weight
        }

        # **Retrieve current state information**
        agent_state = successor.getAgentState(self.index)
        carrying_food = agent_state.numCarrying
        position = agent_state.getPosition()
        
        # **Compute minimum ghost distance**
        ghostPositions = [
            successor.getAgentState(enemy).getPosition()
            for enemy in self.getOpponents(successor)
            if not successor.getAgentState(enemy).isPacman and successor.getAgentState(enemy).getPosition() is not None
        ]
        ghostDistances = [self.getMazeDistance(position, ghost) for ghost in ghostPositions]
        minGhostDistance = min(ghostDistances) if ghostDistances else float('inf')

        # **Compute the number of nearby ghosts**
        numNearbyGhosts = self.getNearbyGhostCount(successor, position, radius=5)

        # **Track consecutive steps without eating food**
        if not hasattr(self, 'stepsWithoutFood'):
            self.stepsWithoutFood = 0

        previousGameState = self.getPreviousObservation()
        if previousGameState:
            prevFood = len(self.getFood(previousGameState).asList())
            currFood = len(self.getFood(successor).asList())
            foodEaten = prevFood - currFood  
            if foodEaten > 0:
                self.stepsWithoutFood = 0  # Reset if food is eaten
            else:
                self.stepsWithoutFood += 1  # Increment if no food was eaten

        # **Boost offense if no food has been eaten for 10 steps**
        if self.stepsWithoutFood >= 10:
            weights['offence'] += 500
            weights['foodScore'] *= 2  # Prioritize food even more

        # **Dynamically adjust ghost avoidance strategy**
        if numNearbyGhosts >= 2 and minGhostDistance <= 3:
            # **When multiple ghosts are nearby, prioritize retreat**
            weights['distanceToGhost'] = 1000
            weights['RetreatScore'] += 500
        elif numNearbyGhosts == 1 and minGhostDistance <= 3:
            # **With a single ghost, still consider offense**
            weights['distanceToGhost'] = 400
            weights['foodScore'] = -300  
        
        # # **Optimize offensive behavior**
        if minGhostDistance > 5:
             weights['foodScore'] = -400 
        # elif minGhostDistance <= 3:
        #     weights['foodScore'] = -250  

        if carrying_food >= 1:
            weights['offence'] = 500  # Maintain a smaller offensive weight even when carrying food
            weights['foodScore'] *= 0.5  # Reduce the weight of foodScore to prioritize retreat, but still consider food
            weights['RetreatScore'] = weights['foodEaten']  # Increase retreat priority when food is carried

        # If ghosts are near, prioritize capsules more to aid in retreat
        if minGhostDistance <= 3:
            weights['capsuleDistance'] *= -2  # Stronger preference for capsules when retreating

        # If there are no ghosts nearby, we can ignore the capsules and focus more on food collection
        if minGhostDistance > 5:
            weights['capsuleDistance'] *= -0.5  # Weaken capsule preference when no ghost threat

        return weights
    
#####################
# MCTS DefensiveReflexAgent #
#####################

class DefensiveReflexAgent(MCTSAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.previousFood = []
        self.counter = 0
        self.patrolPoints = []

#######  Monte Carlo Tree Search Simulation
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

    self.numSims = NUM_SIM
    self.sturns = SIM_LEVEL
    self.levels = LEVEL
    self.svalue = 0 
    self.smoves = []
    self.gameState = gameState
    self.current_node = Node( MState(self.gameState, self.index, self.svalue, self.smoves, self.sturns) )
    self.start = self.current_node.mstate.gameState.getAgentState(self.index).getPosition()

  def getDefAction(self, gameState):
    """
    Prevent CaptureAgent always use the overriden chooseAction from DefensiveReflexAgent
    """
    self.observationHistory.append(gameState)

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    if myPos != nearestPoint(myPos):
      # We're halfway from one position to the next
      return gameState.getLegalActions(self.index)[0]
    else:
      return self.chooseDefAction(gameState)

  def chooseDefAction(self, gameState):

    """
    Picks among the actions with the highest Q(s,a).
    """
    return self.runSimulation(self.current_node, gameState, self.index, self.levels, self.numSims)

  def runSimulation(self, current_node, gameState, index, numSims=5):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    self.gameState = gameState
    self.index = index  
    self.current_node = current_node

    value = 0
    self.current_node.resetNode()
    self.current_node.mstate.resetMState(self.gameState, index, value)

    self.index = index

    self.current_node.children = []

    # l = LEVEL
    child_node = UCTSEARCH(numSims, self.current_node, self.index)

    return child_node.mstate.fromMove


  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor


########  End of Monte Carlo Tree Search Simulation

  def getIsRed(self):
    if self.index%2 == 0:
      return True
    else:
      return False

  def evaluateAttackParameters(self, gameState, action):

        self.gameState = gameState

        features = util.Counter()
        successor = self.getSuccessor(self.gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0
        # Adds the sonar signal
        pos = successor.getAgentPosition(self.index)
        n = successor.getNumAgents()
        distances = []
    
        dists = []

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # Invader: Enemy in the vision
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            poss = [a.getPosition() for a in invaders]
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
     
            if dists != []:
                features['invaderDistance'] = min(dists)
        
            else:
                features['invaderDistance'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        startPos = gameState.getInitialAgentPosition(self.index)

        if len(invaders) > 0:
            features['distToHome'] = 0
        else:
            features['distToHome'] = distanceCalculator.Distancer(gameState.data.layout).getDistance(startPos, myPos)

        if self.getIsRed():
            centralX = (gameState.data.layout.width - 2)/2
        else:
            centralX = ((gameState.data.layout.width - 2)/2) + 1

        centralY = (gameState.data.layout.height)/2
        centralPos = (centralX, centralY)

        if len(invaders) > 0:
            features['distToCentral'] = 0
        else:
            features['distToCentral'] = distanceCalculator.Distancer(gameState.data.layout).getDistance(centralPos, myPos)
    
        features['distToHome'] = 0
        features['distToCentral'] = 0
        
        
        position = successor.getAgentState(self.index).getPosition()
        if not hasattr(self, 'recentPositions'):
            self.recentPositions = []
        
        self.recentPositions.append(position)

        if len(self.recentPositions) > 4:
            self.recentPositions.pop(0) 

        if len(self.recentPositions) == 4:
            A, B, C, D = self.recentPositions
            # Check for clockwise cycle (A → B → C → D → A) or counterclockwise cycle (A → D → C → B → A)
            if (A == C and B == D) or (A == D and B == C):
                features['cyclePenalty'] = 100
        return features


  def getCostOfAttackParameter(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -200, 'cyclePenalty': -5000}




class MState():
    NUM_TURNS = 5
    GOAL = 0

    def __init__(self, gameState, index, value=0, move=None, fromMove=None, turn=NUM_TURNS, agent_type='DefensiveReflexAgent'):  
        self.gameState = gameState
        self.index = index
        self.value = value
        self.turn = turn
        self.move = move
        self.fromMove = fromMove
        self.agent_type = agent_type  # Add agent_type to differentiate between Offensive and Defensive agents

    def setMState(self, gameState, index):
        self.gameState = gameState
        self.index = index

    def resetMState(self, gameState, index, value):
        self.gameState = gameState
        self.index = index
        self.value = value

    def deepCopy(self):
        mstate = MState(self, self.gameState, self.index)
        mstate.gameState = self.gameState
        mstate.index = self.index
        mstate.value = self.value
        mstate.turn = self.turn
        mstate.move = self.move
        mstate.fromMove = self.fromMove
        mstate.agent_type = self.agent_type  # Copy the agent_type as well
        return mstate

    def getFromMove(self):
        return self.fromMove
  
    def getMove(self):
        return self.move

    def next_mstate(self, oNode=None):
        actions = self.gameState.getLegalActions(self.index)
        # actions.remove(Directions.STOP)
        # actions.remove(Directions.REVERSE)
        if oNode is None:
            otherNode = []
        else:
            otherNode = oNode
            if otherNode != []:
                for n in otherNode:
                    actions.remove(n.mstate.getFromMove())  
        if self.agent_type == 'DefensiveReflexAgent':
            da = DefensiveReflexAgent(self.index)
        else:
            da = OffensiveReflexAgent(self.index)

        da.registerInitialState(self.gameState.deepCopy())
        
        nextmove = random.choice([x for x in actions])
        nextValue = da.evaluate(self.gameState, nextmove)
        self.move = nextmove

        nextGameState = self.gameState.generateSuccessor(self.index, nextmove)
        nextMState = MState(nextGameState, self.index, nextValue, None, nextmove, self.turn-1, self.agent_type)
        
        return nextMState

    
    def terminal(self):
        return self.turn == 0
  
    def reward(self):
        return self.value
  
    def __repr__(self):
        return f"Value: {self.value}; Move: {self.move}; Agent Type: {self.agent_type}"
  


class Node():
    def __init__(self, mstate, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.mstate = mstate
        self.children = []
        self.parent = parent  

    def add_child(self, child_mstate):
        child = Node(child_mstate, self)
        self.children.append(child)

    def getState(self):
        return self.mstate.deepCopy()

    def resetNode(self):
        self.visits = 0
        self.reward = 0.0
        self.children = []

    def setParentNode(self, node):
        self.parent = node

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        # Get legal actions excluding STOP
        actions = self.mstate.gameState.getLegalActions(self.mstate.index)
        actions.remove(Directions.STOP)
        availableActions = len(actions)

        # If the number of children equals the number of available actions, return True
        if len(self.children) == availableActions:
            return True
        return False

    def __repr__(self):
        return f"Node; children: {len(self.children)}; visits: {self.visits}; reward: {self.reward:.6f}"


def UCTSEARCH(budget, root, index):
    for _ in range(budget):   
      front = TREEPOLICY(root, index)
      reward = DEFAULTPOLICY(front.mstate)
      BACKUP(root, front, reward)
    return BESTCHILD(root, 0, index)

def TREEPOLICY(node, index):
    """
    Selects and expands nodes in the Monte Carlo Tree Search (MCTS) process.
    The function explores the tree based on the UCT algorithm.
    """
    # If the node is a list, we handle multiple nodes in parallel
    if isinstance(node, list):  
        while not node[0].getState().terminal():
            if not node[0].fully_expanded():  
                return EXPAND(node[0])  # Expand the node if it's not fully expanded
            else:
                node[0] = BESTCHILD(node[0], SCALAR, index)  # Select the best child if expanded
        return node
    
    # If the node is a single node, we continue the process with this node
    else:
        while not node.getState().terminal():  # Continue until terminal state is reached
            if not node.fully_expanded():  
                return EXPAND(node)  # Expand the node if it's not fully expanded
            else:
                node = BESTCHILD(node, SCALAR, index)  # Select the best child if expanded
        return node

def EXPAND(node):
    # Create a list of the current node's children
    children = [child for child in node.children]

    # Generate the next state based on the current state and the children that have already been tried
    new_mstate = node.mstate.next_mstate(children)

    # Add the new state as a child of the current node
    node.add_child(new_mstate)

    # Set the parent of the newly added child node
    node.children[-1].setParentNode(node)

    # Return the newly added child node
    return node.children[-1]

def BESTCHILD(node, scalar, index):
    best_score = float('-inf')  # Start with a very low best score
    best_children = []  # List to store best child nodes

    # Iterate over each child node of the current node
    for child in node.children:
        # Exploitation: reward per visit
        exploit = child.reward / child.visits

        # Exploration: sqrt(log(visits) / visits), promoting less explored nodes
        explore = math.sqrt(math.log(2 * node.visits) / float(child.visits))

        # Total score combines both exploitation and exploration
        score = exploit + scalar * explore

        # If score is equal to the best score, add the child to the list of best children
        if score == best_score:
            best_children.append(child)

        # If this child's score is better than the best score, update the best score and best children
        if score > best_score:
            best_children = [child]  # Reset best children list with this one child
            best_score = score  # Update best score

    # Randomly choose one of the best children (if multiple have the same score)
    return random.choice(best_children) if best_children else None

def DEFAULTPOLICY(mstate):
  while mstate.terminal() == False:
    mstate = mstate.next_mstate()
  return mstate.reward()

def BACKUP(root,node,reward):
  while node != None:
    node.visits += 1
    node.reward += reward*(REWARD_DISCOUNT**SIM_LEVEL) # discounted reward after several turns of simulation
    node.reward += node.mstate.reward() # add the root reward and the last reward together

    node = node.parent
  return 0

