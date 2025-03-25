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
import random, time, util
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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

################
# Parent Agent #
################
class DummyAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        ######################
        ### setup boundary ###
        ######################
        if self.red:
            self.middle = (gameState.data.layout.width - 2) // 2
        else:
            self.middle = (gameState.data.layout.width - 2) // 2 + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.middle, i):
                self.boundary.append((self.middle, i))

    def chooseAction(self, gameState):
        gameStateCopy = gameState.deepCopy()
        actions = gameStateCopy.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        # values = []
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()

        return features

    def getWeights(self, gameState, action):
        return {'foods': 100, 'distanceToFood': -1, 'disToOpponent': 0}


class OffensiveReflexAgent(DummyAgent):
    def __init__(self, index):
        super().__init__(index)
        self.recentPositions = []  # Track last few positions to detect cycles
        self.visitedPositions = set()  # Track visited positions to encourage exploration
        self.escapeMode = False  # Whether the agent is in escape mode
        self.lastEscapeDirection = None  # Track last escape direction

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        pos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        features['foods'] = -len(foodList)

        # Get opponents' positions and scared status
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleGhosts = [(o.getPosition(), o.scaredTimer) for o in opponents if not o.isPacman and o.getPosition() is not None and o.scaredTimer == 0]

        # Only consider normal ghosts (dangerous ones)
        normalGhosts = [g[0] for g in visibleGhosts if g[1] == 0]  # Only track ghosts that are NOT scared

        # Distance to the closest normal ghost (to escape)
        if normalGhosts:
            minNormalGhostDist = min([self.getMazeDistance(pos, g) for g in normalGhosts])
            features['disToOpponent'] = minNormalGhostDist
        else:
            minNormalGhostDist = 100  # No ghost around

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

        # **Escape mode (avoid normal ghosts)**
        if minNormalGhostDist <= 3:
            self.escapeMode = True
            self.lastEscapeDirection = action  
        elif minNormalGhostDist > 5:
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

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visibleGhosts = [o.getPosition() for o in opponents if not o.isPacman and o.getPosition() is not None]
        minGhostDist = min([self.getMazeDistance(successor.getAgentPosition(self.index), g) for g in visibleGhosts], default=100)
        foodList = self.getFood(gameState).asList()
        foodOfCarry = gameState.getAgentState(self.index).numCarrying
        isScared = any(o.scaredTimer > 10 for o in opponents)

        if self.escapeMode:
            return {
                'foods': 0,
                'distanceToFood': 0,
                'disToOpponent': 200,
                'gohome': -200,
                'reverse': -10,
                'stop': -200,
                'distanceToCapsule': -100,
                'cyclePenalty': -200,
                'shortCyclePenalty': -100,
                'explorePenalty': -50,
                'avoidSameEscape': -50 
            }

        if foodOfCarry >= max(3, len(foodList) // 2) and not isScared:
            return {
                'foods': 0,
                'distanceToFood': 0,
                'disToOpponent': 8000,
                'gohome': -12000,
                'reverse': -10,
                'stop': -150,
                'distanceToCapsule': -30,
                'cyclePenalty': -100,  
                'shortCyclePenalty': -50,
                'explorePenalty': -40,
            }

        # retreat
        if minGhostDist <= 3:
            return {
                'foods': 0,
                'distanceToFood': 0,
                'disToOpponent': 100,
                'gohome': -100,
                'reverse': -10,
                'stop': -150,
                'distanceToCapsule': -100,
                'cyclePenalty': -100,
                'shortCyclePenalty': -50,
                'explorePenalty': -40,
            }

        # eat food
        if len(foodList) > 2:
            return {
                'foods': 100,
                'distanceToFood': -9,
                'disToOpponent': 14,
                'gohome': -8,
                'reverse': -5,
                'stop': -100,
                'distanceToCapsule': -15,
                'cyclePenalty': -100,
                'shortCyclePenalty': -50,
                'explorePenalty': -40,
            }

        # Only a little food left → Go home first
        return {
            'foods': 0,
            'distanceToFood': 0,
            'disToOpponent': 14,
            'gohome': -9000,
            'reverse': -10,
            'stop': -150,
            'distanceToCapsule': 0,
            'cyclePenalty': -100,
            'shortCyclePenalty': -50,
            'explorePenalty': -40,
        }
    
    
class DefensiveReflexAgent(DummyAgent):
    def __init__(self, index):
        super().__init__(index)
        self.defenseMode = True  # Default mode is defense
        self.targetInvader = None  # Tracks the current invader being chased
        self.lastDefensePosition = None  # Stores the last defensive position

    def getFeatures(self, gameState, action):
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

    def getWeights(self, gameState, action):
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