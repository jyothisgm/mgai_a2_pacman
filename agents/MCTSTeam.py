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
import random,util,time,distanceCalculator 
import game
from util import nearestPoint
from util import pause
from capture import noisyDistance
import math
import hashlib
import logging
import argparse

######################
# Parameters of MCTS #
######################

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/math.sqrt(2.0)
NUM_SIM=1    # number of simulation times, at lease 1
# REWARD_DISCOUNT=0.8
REWARD_DISCOUNT=0   # if don't want to use roll out, just turn reward discount into 0
SIM_LEVEL=1   # level of tree expanding
LEVEL=0       # level of simulation tree (not used)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

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

    # Once attacker capture food, then it go back to home
    agentState = gameState.getAgentState(self.index)

    if agentState.numCarrying > 1:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # Once total food left <= 2, then they go back to home
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

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
# Normal OffensiveReflexAgent #
######################

class OffensiveReflexAgent(MCTSAgent):
    """
    A Monte Carlo Tree Search-based offensive agent that seeks food efficiently.
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        self.numSims = NUM_SIM
        self.sturns = SIM_LEVEL
        self.levels = LEVEL
        self.svalue = 0
        self.smoves = []
        self.gameState = gameState
        self.current_node = Node(MState(self.gameState, self.index, self.svalue, self.smoves, self.sturns))
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

    def runSimulation(self, current_node, gameState, index, levels=3, numSims=5):
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

        l = LEVEL
        child_node = UCTSEARCH(numSims / (l + 1), self.current_node, self.index)

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
        self.gameState = gameState
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        # Compute distance to the nearest food
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to the nearest ghost
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if (not a.isPacman) and a.getPosition() is not None]

        if len(ghosts) > 0:
            myPos = successor.getAgentPosition(self.index)
            positions = [agent.getPosition() for agent in ghosts]
            self.target = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            self.distanceToGhost = self.getMazeDistance(myPos, self.target)
            if self.distanceToGhost < 10:
                features['distanceToGhost'] = self.distanceToGhost
        else:
            features['distanceToGhost'] = 0

        return features

    def getCostOfAttackParameter(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': 10}

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

    self.numSims=NUM_SIM
    self.sturns=SIM_LEVEL
    self.levels=LEVEL
    self.svalue=0 
    self.smoves=[]
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

  def runSimulation(self, current_node, gameState, index, levels=3, numSims=5):
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

    l = LEVEL
    child_node = UCTSEARCH(numSims/(l+1), self.current_node, self.index)

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
        # print 'min(dists)', min(dists)
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

    return features


  def getCostOfAttackParameter(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class MState():
  NUM_TURNS = 5
  GOAL = 0

  def __init__(self, gameState, index, value=0, move = None, fromMove = None, turn=NUM_TURNS):  
    self.gameState=gameState
    self.index=index
    self.value=value
    self.turn=turn
    self.move=move
    self.fromMove=fromMove

  def setMState(self, gameState, index):
    self.gameState=gameState
    self.index=index

  def resetMState(self, gameState, index, value):
    self.gameState=gameState
    self.index=index
    self.value=value

  def deepCopy(self):
    mstate = MState( self, self.gameState, self.index )
    mstate.gameState = self.gameState
    mstate.index = self.index
    mstate.value = self.value
    mstate.turn = self.turn
    mstate.move = self.move
    mstate.fromMove = self.fromMove
    return mstate

  def getFromMove(self):
    return self.fromMove
  
  def getMove(self):
    return self.move

  # def next_mstate(self, oMState = None):
  def next_mstate(self, oNode = None):

    actions = self.gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)

    if oNode == None:
      otherNode = []
    else:
      otherNode = oNode
      if otherNode != []:
        for n in otherNode:
          actions.remove(n.mstate.getFromMove())  

    da = DefensiveReflexAgent(self.index, 0.1)
    da.registerInitialState(self.gameState.deepCopy())
    
    nextValues = [da.evaluate(self.gameState, a) for a in actions]

    nextValue = max(nextValues)
    bestActions = [a for a, v in zip(actions, nextValues) if v == nextValue]

    nextmove = random.choice([x for x in bestActions])
    self.move = nextmove

    nextGameState = self.gameState.generateSuccessor(self.index, nextmove)
    nextMState = MState(nextGameState, self.index, nextValue, None, nextmove, self.turn-1)
    
    return nextMState
    
  def terminal(self):
    if self.turn == 0:
      return True
    return False
  def reward(self):
    r = self.value
    return r
  def __repr__(self):
    s="Value: %d; Move: %s"%(self.value,self.move)
    return s
  


class Node():
  def __init__(self, mstate, parent=None):
    self.visits=1
    self.reward=0.0 
    self.mstate=mstate
    self.children=[]
    self.parent=parent  
  def add_child(self,child_mstate):
    child=Node(child_mstate,self)
    self.children.append(child)
  def getState(self):
    return self.mstate.deepCopy()
  def resetNode(self):
    self.visits=0
    self.reward=0.0 
    self.children=[]
  def setParentNode(self, node):
    self.parent = node
  def update(self,reward):
    self.reward+=reward
    self.visits+=1
  def fully_expanded(self):
    # if len(self.children)==self.mstate.num_moves:
    actions = self.mstate.gameState.getLegalActions(self.mstate.index)
    actions.remove(Directions.STOP)
    availableActions = len(actions)
    #print 'availableActions', availableActions, actions
    #print 'len(self.children)', len(self.children), self.children
    if len(self.children) == availableActions:   # If it's fully expanded
      #printChildrenPosition(self.children)
      return True
    return False
  def __repr__(self):
    s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
    return s


def UCTSEARCH(budget,root,index):
  #print 'UCTSEARCH'
  ## print 'location', root.mstate.gameState.getAgentPosition(index)
  for iter in range(budget):
    if iter%10000==9999:
      logger.info("simulation: %d"%iter)
      logger.info(root)
    ## print 'root', root, 'type(root)', type(root)
    front=TREEPOLICY(root,index)
    reward=DEFAULTPOLICY(front.mstate)
    BACKUP(root,front,reward)
  ## print 'root location', root.mstate.gameState.getAgentPosition(index)
  ## print 'c location', BESTCHILD(root,0,index).mstate.gameState.getAgentPosition(index)
  return BESTCHILD(root,0,index)

def TREEPOLICY(node,index):
  #print 'TREEPOLICY'
  ## print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^root before if', node, 'type(root)', type(node)

  if type(node) is list:  
    #print 'node', node
    while node.getState.terminal()==False:
      ## print 'terminal()==False'
      if node[0].fully_expanded()==False:  
        ## print 'fully_expanded()==False'
        return EXPAND(node[0])
      else:
        node[0]=BESTCHILD(node[0],SCALAR,index)
    return node
  else:
    while node.getState().terminal()==False:
      ## print 'terminal()==False'
      if node.fully_expanded()==False:  
        ## print 'fully_expanded()==False'
        return EXPAND(node)
      else:
        #print 'Fully Expanded'
        node=BESTCHILD(node,SCALAR,index)
    return node

def EXPAND(node):
  tried_children = [c for c in node.children]

  new_mstate = node.mstate.next_mstate(tried_children)

  tried_children_mstate = []

  if tried_children != []:
    for t in tried_children:
      tried_children_mstate.append(t.mstate)
  node.add_child(new_mstate)
  node.children[-1].setParentNode(node)


  return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar,index):
  bestscore = -9999999999
  bestchildren = []
  for c in node.children:
    #print 'c location', c.mstate.gameState.getAgentPosition(index)
    exploit = c.reward / c.visits
    explore = math.sqrt(math.log(2*node.visits) / float(c.visits))  
    score = exploit + scalar*explore
    if score == bestscore:
      bestchildren.append(c)  # more than one best child
    if score>bestscore:       # best child
      bestchildren=[c]
      bestscore=score
  
  if len(bestchildren)==0:
    logger.warn("OOPS: no best child found, probably fatal")

  if bestchildren!=[]:
    return random.choice(bestchildren)
  else:
    return bestchildren

def DEFAULTPOLICY(mstate):
  #print 'DEFAULTPOLICY'
  while mstate.terminal()==False:
    ## print 'mstate.terminal()==False'
    mstate=mstate.next_mstate()
  return mstate.reward()

def BACKUP(root,node,reward):
  while node!=None:
    node.visits+=1
    node.reward+=reward*(REWARD_DISCOUNT**SIM_LEVEL) # discounted reward after several turns of simulation
    node.reward+=node.mstate.reward() # add the root reward and the last reward together

    node=node.parent
  return 0

