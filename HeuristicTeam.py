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
import random, util as util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    
    actions = gameState.getLegalActions(self.index)
    # You should change this in your own agent.
    return random.choice(actions)
  
  def getSuccessor(self, gameState, action):
        # Calculate the new state after performing the action
        successor = gameState.generateSuccessor(self.index, action)
        # Position of the new state
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # If the position is not the nearest grid point, move one more step forward
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

  def evaluate(self, gameState, action): 
        features = self.evaluateAttackParameters(gameState, action)
        weights = self.getCostOfAttackParameter(gameState, action)
        return features * weights
 
  def evaluateAttackParameters(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        # Calculate the features when attacking and store them in the 'features' variable
        features['successorScore'] = self.getScore(successor)
        return features
    
  def getCostOfAttackParameter(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveReflexAgent(DummyAgent):
    '''
    Inheriting properties of Base Class
    '''
    def __init__(self, index):
        CaptureAgent.__init__(self, index)   
        # Current coordinates     
        self.presentCoordinates = (-5, -5)
        # Initial target position
        self.initialTarget = []


    
    def evaluateAttackParameters(self, gameState, action):
        """
        Evaluates the given action for offensive strategy by considering multiple factors.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        position = successor.getAgentState(self.index).getPosition()
        previousGameState = self.getPreviousObservation()

        features['successorScore'] = self.getScore(successor)
        features['offence'] = 1 if successor.getAgentState(self.index).isPacman else 0
        
        # Compute food-related features
        features['foodDistance'] = self.getNearestFoodDistance(successor, position)
        features['capsuleDistance'] = self.getNearestCapsuleDistance(successor, position)
        features['RetreatScore'] = self.getRetreatScore(successor, position)

        # Compute ghost-related features
        features['distanceToGhost'] = self.getGhostThreat(successor, position, features['successorScore'])
        
        return features
    
    def getNearestCapsuleDistance(self, successor, position):
        """Returns the distance to the nearest power capsule, if available."""
        capsuleList = self.getCapsules(successor)
        return min([self.getMazeDistance(position, cap) for cap in capsuleList]) if capsuleList else float('inf')
    
    def getRetreatScore(self, successor, position):
        """
        Computes a retreat score to determine whether the agent should prioritize retreating.
        Higher values indicate a greater need to retreat.
        """
        # Determine the team's boundary position
        mapWidth = successor.data.layout.width
        borderX = (mapWidth // 2) - 1 if self.red else (mapWidth // 2)
        borderPositions = [(borderX, y) for y in range(successor.data.layout.height) if not successor.hasWall(borderX, y)]
        
        # Distance to the closest border point
        borderDistance = min(self.getMazeDistance(position, border) for border in borderPositions)

        # Get enemy ghost positions
        ghostPositions = [
            successor.getAgentState(enemy).getPosition()
            for enemy in self.getOpponents(successor)
            if not successor.getAgentState(enemy).isPacman and successor.getAgentState(enemy).getPosition() is not None
        ]

        # Calculate the minimum ghost distance
        minGhostDistance = min((self.getMazeDistance(position, ghost) for ghost in ghostPositions), default=float('inf'))

        # Encourage retreat if ghosts are near (动态威胁计算)
        ghostThreatFactor = 10 if minGhostDistance <= 2 else (5 if minGhostDistance <= 5 else 0)

        # 考虑食物携带量，指数级增长
        carriedFood = successor.getAgentState(self.index).numCarrying
        foodWeight = carriedFood * 1

        # 计算最终回家分数（分数越高，回家意愿越强）
        retreatScore = (5 / (borderDistance + 1)) + ghostThreatFactor + foodWeight  

        return retreatScore
    
    def getNearestFoodDistance(self, successor, position):
        """
        Calculates the best food distance considering multiple factors:
        - Food proximity
        - Ghost threat
        - Nearby food clusters (to encourage efficient eating)
        - Distance to retreat (border)
        """
        foodList = self.getFood(successor).asList()
        if not foodList:
            return 0  # No food available

        # Get enemy ghost positions
        ghostPositions = [
            successor.getAgentState(enemy).getPosition()
            for enemy in self.getOpponents(successor)
            if not successor.getAgentState(enemy).isPacman and successor.getAgentState(enemy).getPosition() is not None
        ]

        # Determine the team's boundary position
        mapWidth = successor.data.layout.width
        borderX = (mapWidth // 2) - 1 if self.red else (mapWidth // 2)
        borderPositions = [(borderX, y) for y in range(successor.data.layout.height) if not successor.hasWall(borderX, y)]

        # Evaluate all food positions and select the best one
        bestScore = float('inf')
        for food in foodList:
            foodDistance = self.getMazeDistance(position, food)
            borderDistance = self.getMazeDistance(position, food) + min(self.getMazeDistance(food, border) for border in borderPositions)

            # Count nearby food items within a 2-tile radius
            nearbyFoodCount = sum(1 for other in foodList if self.getMazeDistance(food, other) <= 2)

            # Compute ghost threat dynamically
            ghostThreatLevel = min((self.getMazeDistance(food, ghost) for ghost in ghostPositions), default=float('inf'))
            ghostPenalty = 50 / (ghostThreatLevel + 1)  # 动态计算鬼的影响，避免固定惩罚

            # 如果已经在敌方半场，不考虑食物密集度
            if self.isInEnemyTerritory(successor):
                nearbyFoodCount = 0  

            # 调整优先级评分
            priorityScore = (
                foodDistance
                - (nearbyFoodCount * 1.5)  # 适当考虑密集度
                + (borderDistance * 0.3)   # 回家路线更重要
                + ghostPenalty             # 避免鬼，但不过度惩罚
            )

            bestScore = min(bestScore, priorityScore)

        return bestScore

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
            return minDisToGhost + successorScore if minDisToGhost < 5 else 0
        return 0
    
    def getCostOfAttackParameter(self, gameState, action):
        """
        计算攻击参数的权重，决定 Pacman 在不同情况下的行为优先级。
        """
        successor = self.getSuccessor(gameState, action)

        # 默认权重
        weights = {
            'offence': 0,  
            'successorScore': 202,  
            'foodDistance': -8,  
            'distanceToGhost': 215,  
            'capsuleDistance' : 10
        }

        # 获取鬼魂信息
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        scaredGhosts = [g for g in ghosts if g.scaredTimer > 0]

        # 如果有害怕的鬼魂，则降低鬼魂威胁权重
        if scaredGhosts:
            weights['distanceToGhost'] = 0  

        # 计算撤退需求
        agent_state = successor.getAgentState(self.index)
        carrying_food = agent_state.numCarrying
        agent_position = agent_state.getPosition()

        if carrying_food >= 5:
            weights['RetreatScore'] = 3010  # 高优先级撤退

        return weights

    
    def getOpponentPositions(self, gameState):
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]


    
    def isInEnemyTerritory(self, gameState):
        my_pos = gameState.getAgentPosition(self.index)
        return my_pos[0] > gameState.data.layout.width // 2  
    
        
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

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
                # action that gets home fastest/shortest
            return bestAction

        finalAction = random.choices(
        population=[random.choice(bestActions), random.choice(actions)],
        weights=[0.8, 0.2]
    )[0]

        return finalAction
    
    def registerInitialState(self, gameState):
        self.currentFoodSize = 9999999
        
        CaptureAgent.registerInitialState(self, gameState)
        # Initial position of the agent
        self.initPosition = gameState.getAgentState(self.index).getPosition()
        # Initial attack target position
        self.initialAttackCoordinates(gameState)

    def initialAttackCoordinates(self, gameState):
        layoutInfo = []
        # Central X coordinate of the map
        x = (gameState.data.layout.width - 2) // 2
        # If not red (i.e., blue), shift the central X coordinate one step to the right
        if not self.red:
            x += 1
        
        y = (gameState.data.layout.height - 2) // 2
        
        # Store map information: the central X coordinate and all walkable Y coordinates.
        layoutInfo.extend((gameState.data.layout.width, gameState.data.layout.height, x, y))
        
        # Store all walkable Y coordinates at the central X coordinate
        # Iterate over Y coordinates from 1 to height-1 to avoid checking boundary walls
        self.initialTarget = []
        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.initialTarget.append((layoutInfo[2], i))
        
        # Number of walkable central column coordinates; always choose the one closest to the middle of the map as the initial attack target
        noTargets = len(self.initialTarget)
        if (noTargets % 2 == 0):
            noTargets = (noTargets // 2)
            self.initialTarget = [self.initialTarget[noTargets]]
        else:
            noTargets = (noTargets - 1) // 2
            self.initialTarget = [self.initialTarget[noTargets]]




class DefensiveReflexAgent(DummyAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.previousFood = []
        self.counter = 0
        self.patrolPoints = []

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.setPatrolPoint(gameState)

    def setPatrolPoint(self, gameState):
        '''Dynamic patrol point selection based on the center of the maze.'''
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x += 1

        self.patrolPoints = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(x, i):
                self.patrolPoints.append((x, i))

        # Keep middle patrol points
        if len(self.patrolPoints) > 2:
            self.patrolPoints = self.patrolPoints[1:-1]

    def heuristicEvaluation(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        myPos = successor.getAgentPosition(self.index)
        features = util.Counter()
        
        # Defensive state check
        myState = successor.getAgentState(self.index)
        features['onDefense'] = 1 if not myState.isPacman else 0

        # Calculate distance to visible invaders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        if invaders:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # Penalize stopping or reversing
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Move towards target if no invaders are detected
        if not invaders and self.target:
            features['targetDistance'] = self.getMazeDistance(myPos, self.target)

        weights = {
            'numInvaders': -1000, 
            'onDefense': 100, 
            'invaderDistance': -10, 
            'stop': -100, 
            'reverse': -2, 
            'targetDistance': -1
        }
        return features * weights

    def chooseAction(self, gameState):
        position = gameState.getAgentPosition(self.index)
        
        # Reset target if we have reached it
        if position == self.target:
            self.target = None

        # Detect invaders
        invaders = [gameState.getAgentState(i).getPosition() for i in self.getOpponents(gameState) 
                    if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() is not None]
        
        if invaders:
            # Prioritize invaders closer to the food or important areas
            self.target = min(invaders, key=lambda x: self.getMazeDistance(position, x))
        else:
            # Handle stolen food
            if self.previousFood and len(self.getFoodYouAreDefending(gameState).asList()) < len(self.previousFood):
                stolenFood = set(self.previousFood) - set(self.getFoodYouAreDefending(gameState).asList())
                if stolenFood:
                    self.target = stolenFood.pop()

        # Update previous food state
        self.previousFood = self.getFoodYouAreDefending(gameState).asList()

        # If no invader or stolen food, patrol
        if self.target is None:
            self.target = random.choice(self.patrolPoints)

        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP) if Directions.STOP in actions else None

        # Select the best action based on the heuristic

        bestAction = max(actions, key=lambda a: self.heuristicEvaluation(gameState, a))
        return bestAction