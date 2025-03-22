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
NUM_SIM = 20    # number of simulation times, at lease 1
# REWARD_DISCOUNT=0.8
# REWARD_DISCOUNT = 0   # if don't want to use roll out, just turn reward discount into 0
SIM_LEVEL = 15   # level of tree expanding
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
        self.recentPositions = []  # Tracks the last few positions to detect cycles
        self.visitedPositions = set()  # Stores recently visited positions to avoid local loops
        self.escapeMode = False  # Flag indicating if the agent is in escape mode
        self.lastEscapeDirection = None  # Stores the last escape direction to prevent circling back

        self.distancer.getMazeDistances()
        self.boundary = []
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
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        pos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        features['foods'] = -len(foodList)

        # Get opponents' positions and scared status
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        
        # Separate normal (dangerous) ghosts and scared (edible) ghosts
        normalGhosts = [o.getPosition() for o in opponents if not o.isPacman and o.getPosition() is not None and o.scaredTimer == 0]
        scaredGhosts = [o.getPosition() for o in opponents if not o.isPacman and o.getPosition() is not None and o.scaredTimer > 0]

        # Distance to the closest normal ghost (avoidance)
        features['disToNormalGhost'] = min([self.getMazeDistance(pos, g) for g in normalGhosts], default=100)

        # Distance to the closest scared ghost (chasing)
        features['disToScaredGhost'] = min([self.getMazeDistance(pos, g) for g in scaredGhosts], default=100)

        # Distance to home (to return food safely)
        features['gohome'] = min([self.getMazeDistance(pos, b) for b in self.boundary], default=1000)

        # Avoid reversing direction
        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            features['reverse'] = 10  

        # Avoid stopping
        if action == Directions.STOP:
            features['stop'] = 1  

        # Distance to the closest food
        if foodList:
            features['distanceToFood'] = min([self.getMazeDistance(pos, food) for food in foodList])

        # Distance to the closest capsule
        capsules = list(set(gameState.data.capsules) - set(self.getCapsulesYouAreDefending(gameState)))
        features['distanceToCapsule'] = min([self.getMazeDistance(pos, c) for c in capsules], default=0)

        # **Escape mode (avoid only normal ghosts, not scared ones)**
        if features['disToNormalGhost'] <= 3:
            self.escapeMode = True
            self.lastEscapeDirection = action  
        elif features['disToNormalGhost'] > 5:
            self.escapeMode = False  

        if self.escapeMode and action == self.lastEscapeDirection:
            features['avoidSameEscape'] = 30  

        # **Cycle detection (Avoid looping behavior)**
        self.recentPositions.append(pos)
        if len(self.recentPositions) > 6:
            self.recentPositions.pop(0)

        # **Detect 4-step loops (A → B → C → D → A)**
        if len(self.recentPositions) == 4:
            A, B, C, D = self.recentPositions
            if (A == C and B == D) or (A == D and B == C):
                features['cyclePenalty'] = 100  

        # **Detect 3-step loops (A → B → A)**
        if len(self.recentPositions) == 3:
            A, B, C = self.recentPositions
            if A == C:
                features['shortCyclePenalty'] = 50  

        # **Encourage exploration (Avoid visiting the same spot repeatedly)**
        if pos in self.visitedPositions:
            features['explorePenalty'] = 30  
        else:
            self.visitedPositions.add(pos)  

        return features

    def getCostOfAttackParameter(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        # Separate normal (dangerous) ghosts and scared (edible) ghosts
        normalGhosts = [o.getPosition() for o in opponents if not o.isPacman and o.getPosition() is not None and o.scaredTimer == 0]
        scaredGhosts = [o.getPosition() for o in opponents if not o.isPacman and o.getPosition() is not None and o.scaredTimer > 0]

        # Distance to the closest normal ghost
        disToNormalGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index), g) for g in normalGhosts], default=100)
        
        # Distance to the closest scared ghost
        disToScaredGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index), g) for g in scaredGhosts], default=100)

        foodList = self.getFood(gameState).asList()
        foodOfCarry = gameState.getAgentState(self.index).numCarrying
        isScared = any(o.scaredTimer > 10 for o in opponents)

        # **Escape mode (avoiding normal ghosts)**
        if self.escapeMode:
            return {
                'foods': 0,
                'distanceToFood': 0,
                'disToNormalGhost': 200,  # Strongly avoid normal ghosts
                'disToScaredGhost': 0,  # Ignore scared ghosts when escaping
                'gohome': -200,
                'reverse': -10,
                'stop': -200,
                'distanceToCapsule': -100,
                'cyclePenalty': -200,
                'shortCyclePenalty': -100,
                'explorePenalty': -50,
                'avoidSameEscape': -50 
            }

        # **If carrying enough food, prioritize returning home**
        if foodOfCarry >= max(3, len(foodList) // 2) and not isScared:
            return {
                'foods': 0,
                'distanceToFood': 0,
                'disToNormalGhost': 8000,
                'disToScaredGhost': 0,
                'gohome': -12000,
                'reverse': -10,
                'stop': -150,
                'distanceToCapsule': -30,
                'cyclePenalty': -100,  
                'shortCyclePenalty': -50,
                'explorePenalty': -40,
            }

        # **Retreat if a normal ghost is too close**
        if disToNormalGhost <= 3:
            return {
                'foods': 0,
                'distanceToFood': 0,
                'disToNormalGhost': 100,
                'disToScaredGhost': 0,
                'gohome': -100,
                'reverse': -10,
                'stop': -150,
                'distanceToCapsule': -100,
                'cyclePenalty': -100,
                'shortCyclePenalty': -50,
                'explorePenalty': -40,
            }

        # **Chase scared ghosts if available**
        if disToScaredGhost <= 6:
            return {
                'foods': 50,
                'distanceToFood': -9,
                'disToNormalGhost': 0,  # Ignore normal ghosts when ghosts are scared
                'disToScaredGhost': -30,  # Encourage moving toward scared ghosts
                'gohome': -5,
                'reverse': -3,
                'stop': -50,
                'distanceToCapsule': -20,
                'cyclePenalty': -50,
                'shortCyclePenalty': -20,
                'explorePenalty': -30,
            }

        # **If there is still plenty of food, keep eating**
        if len(foodList) > 2:
            return {
                'foods': 100,
                'distanceToFood': -9,
                'disToNormalGhost': 14,
                'disToScaredGhost': 0,
                'gohome': -8,
                'reverse': -5,
                'stop': -100,
                'distanceToCapsule': -15,
                'cyclePenalty': -100,
                'shortCyclePenalty': -50,
                'explorePenalty': -40,
            }

        # **If only a little food is left, prioritize going home**
        return {
            'foods': 0,
            'distanceToFood': 0,
            'disToNormalGhost': 14,
            'disToScaredGhost': 0,
            'gohome': -9000,
            'reverse': -10,
            'stop': -150,
            'distanceToCapsule': 0,
            'cyclePenalty': -100,
            'shortCyclePenalty': -50,
            'explorePenalty': -40,
        }
    
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


#######  Monte Carlo Tree Search Simulation
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

    if self.red:
        self.middle = (gameState.data.layout.width - 2) // 2
    else:
        self.middle = (gameState.data.layout.width - 2) // 2 + 1
        
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

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Determine if the agent is on defense
        features['onDefense'] = 0 if myState.isPacman else 1

        # Compute the defensive position (fallback position if all food is eaten)
        foodCenter = self.getCenterPointOfDefensiveFood(gameState)
        if gameState.hasWall(*foodCenter):
            foodCenter = self.nearPosInGrid(gameState, foodCenter)
        features['distToFoodCenter'] = self.getMazeDistance(myPos, foodCenter)

        # Identify invaders
        invaders = [a for a in [successor.getAgentState(i) for i in self.getOpponents(successor)] if a.isPacman and a.getPosition()]
        features['numInvaders'] = len(invaders)

        # Select the highest-priority invader to chase
        if invaders:
            self.defenseMode = False  # Switch to chase mode
            # Select the closest invader
            closestInvader = min(invaders, key=lambda inv: self.getMazeDistance(myPos, inv.getPosition()))
            self.targetInvader = closestInvader.getPosition()
            features['invaderDistance'] = self.getMazeDistance(myPos, self.targetInvader)

            # If the invader is very close, increase defensive pressure
            if features['invaderDistance'] < 2:
                features['inDangerousZone'] = 1  # Enemy is too close
        else:
            self.defenseMode = True  # No invaders, return to defensive mode
            self.targetInvader = None

        # Fallback strategy when scared
        if successor.getAgentState(self.index).scaredTimer > 0:
            features['fallback'] = features['invaderDistance'] * 2  # Reduce chase weight if scared

        # Avoid meaningless reverses & stopping
        if action == Directions.STOP:
            features['stop'] = 1
        if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
            features['reverse'] = 1

        return features


  def getCostOfAttackParameter(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        invaders = [a for a in [successor.getAgentState(i) for i in self.getOpponents(successor)] if a.isPacman and a.getPosition()]
        scaredTime = successor.getAgentState(self.index).scaredTimer

        # If scared, avoid approaching invaders
        if scaredTime > 0:
            return {
                'numInvaders': -1000,
                'onDefense': 100,
                'invaderDistance': -5,  # Lower chase weight
                'stop': -100,
                'reverse': -2,
                'distToFoodCenter': 0,
                'inDangerousZone': -10000,
                'fallback': -200  # Prioritize retreating
            }

        # If multiple invaders are present, prioritize food protection
        if len(invaders) > 1:
            return {
                'numInvaders': -2000,  # Stronger weight, must defend
                'onDefense': 200,
                'invaderDistance': -15,
                'stop': -100,
                'reverse': -5,
                'distToFoodCenter': -1,
                'inDangerousZone': -20000,
                'fallback': 0
            }

        # Normal defensive behavior
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -12,  # Slightly reduce focus on enemy distance
            'stop': -100,
            'reverse': -3,  # Allow some reversing
            'distToFoodCenter': -2,  # Adjust defensive position
            'inDangerousZone': -15000,
            'fallback': 0
        }
  def getCenterPointOfDefensiveFood(self, gameState):
        """
        Compute the central point of the remaining food as the default defensive position.
        """
        homeFoods = self.getFoodYouAreDefending(gameState).asList()
        if not homeFoods:
            return self.middle, gameState.data.layout.height // 2

        # If food is widely spread, pick the one closest to the boundary
        minBoundaryFood = min(homeFoods, key=lambda food: abs(food[0] - self.middle))
        return minBoundaryFood

    # Find the nearest valid position without walls
  def nearPosInGrid(self, gameState, pos):
        """
        Find a nearby position that is not blocked by walls.
        """
        neighbors = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), 
                     (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]
        validPositions = [p for p in neighbors if self.inGrid(p, gameState) and not gameState.hasWall(p[0], p[1])]
        return random.choice(validPositions) if validPositions else pos

  def inGrid(self, pos, gameState):
        """
        Ensure that the position is within the valid map boundaries.
        """
        return 1 <= pos[0] < gameState.data.layout.width - 1 and \
               1 <= pos[1] < gameState.data.layout.height - 1



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
        actions = [a for a in actions if a != Directions.STOP]
        # actions.remove(Directions.STOP)
        # actions.remove(Directions.REVERSE)
        if oNode:
            for n in oNode:
                if n.mstate.getFromMove() in actions:
                    actions.remove(n.mstate.getFromMove())  
        

        # Choose the appropriate agent type based on the state
        if self.agent_type == 'DefensiveReflexAgent':
            da = DefensiveReflexAgent(self.index)
        else:
            da = OffensiveReflexAgent(self.index)

        da.registerInitialState(self.gameState.deepCopy())
        
        nextmove = random.choice([x for x in actions if x != Directions.STOP])
        nextValue = da.evaluate(self.gameState, nextmove)
        self.move = nextmove

        nextGameState = self.gameState.generateSuccessor(self.index, nextmove)
        nextMState = MState(nextGameState, self.index, nextValue, None, nextmove, self.turn-1)
        
        return nextMState
    
    def terminal(self):
        return self.turn == 0
  
    def reward(self):
        opponents = self.gameState.getOpponents(self.gameState.getAgentState(self.index))
        scared_ghosts = [
            self.gameState.getAgentState(i) 
            for i in opponents 
            if not self.gameState.getAgentState(i).isPacman 
            and self.gameState.getAgentState(i).scaredTimer > 0
        ]
        if scared_ghosts:
            return self.value + 100 
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
        exploit = child.reward / child.visits
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
    rollout_depth = 0
    while not mstate.terminal() and rollout_depth < SIM_LEVEL:
        mstate = mstate.next_mstate()
        rollout_depth += 1
    return mstate.reward()

def BACKUP(node, reward):
    while node:
        node.update(reward)
        node = node.parent

