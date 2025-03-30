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
            first='OffensiveReflexAgent', second='DefenderAgent'):
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


class DefenderAgent(CaptureAgent):
    def __init__(self, index):
        # Initialize defender agent
        CaptureAgent.__init__(self, index)
        self.targetPosition = None  # Current target position
        self.previousFoodState = []  # Previous food state to track changes
        self.patrolCounter = 0  # Counter for patrol behavior
        self.attack = False
        self.chase = False
        self.foodTargetPosition = None

    def calculateBorderCrossingPoints(self, gameState):
        '''
        Look for entry points along the middle of the board
        '''
        mapDimensions = []
        # Calculate the x-coordinate of the border
        borderX = (gameState.data.layout.width - 2) // 2
        if not self.red:
            borderX += 1  # Adjust for blue team
        mapHeight = (gameState.data.layout.height - 2) // 2
        # Store map dimensions and border position
        mapDimensions.extend((gameState.data.layout.width, gameState.data.layout.height, borderX, mapHeight))

        self.borderCrossingPoint = []

        # Find all points along the border that aren't walls
        for yCoord in range(1, mapDimensions[1] - 1):
            if not gameState.hasWall(mapDimensions[2], yCoord):
                self.borderCrossingPoint.append((mapDimensions[2], yCoord))

    def registerInitialState(self, gameState):
        # Initialize game state data
        CaptureAgent.registerInitialState(self, gameState)
        self.determinePatrolPoints(gameState)  # Determine patrol points

    def determinePatrolPoints(self, gameState):
        '''
        Look for center of the maze for patrolling
        '''
        # Calculate border position
        borderX = (gameState.data.layout.width - 2) // 2
        if not self.red:
            borderX += 1  # Adjust for blue team
        self.patrolPositions = []
        # Find all positions along border that aren't walls
        for yCoord in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(borderX, yCoord):
                self.patrolPositions.append((borderX, yCoord))

        # Focus on patrolling near the closest food point near border
        criticalDefensePoints = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
        minDefenseDistance = 999999
        patrolPositions = []
        for each_point in criticalDefensePoints:
            for each_patrol in self.patrolPositions:
                distance = self.getMazeDistance(each_point, each_patrol)
                if minDefenseDistance > distance:
                    minDefenseDistance = distance
                    patrolPositions = [each_patrol]
                elif minDefenseDistance == distance:
                    patrolPositions.append(each_patrol)
        if len(patrolPositions):
            self.patrolPositions = patrolPositions

    def getDefensiveMovementOptions(self, gameState):
        # Get valid defensive moves
        validMoves = []
        allMoves = gameState.getLegalActions(self.index)

        # Identify reverse direction
        oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        allMoves.remove(Directions.STOP)  # Don't consider stopping

        # Remove reverse direction from options if possible
        for moveIdx in range(0, len(allMoves)-1):
            if oppositeDirection == allMoves[moveIdx]:
                allMoves.remove(oppositeDirection)

        # Find moves that keep us in our territory
        if not self.attack or not self.chase:
            for moveIdx in range(len(allMoves)):
                currentMove = allMoves[moveIdx]
                nextState = gameState.generateSuccessor(self.index, currentMove)
                if gameState.getAgentState(self.index).isPacman or not nextState.getAgentState(self.index).isPacman:
                    validMoves.append(currentMove)
        else:
            validMoves = allMoves

        # If no valid moves or after several moves, allow reversing
        if len(validMoves) == 0:
            self.patrolCounter = 0
        else:
            self.patrolCounter = self.patrolCounter + 1
        if self.patrolCounter > 4 or self.patrolCounter == 0:
            validMoves.append(oppositeDirection)

        return validMoves

    def chooseAction(self, gameState):
        legal_actions = self.getDefensiveMovementOptions(gameState)

        action_scores = self.evaluate(gameState,legal_actions)
        maxValue = max(action_scores.values())
        bestActions = [action for action, score in action_scores.items() if score == maxValue]
        return random.choice(bestActions)


    def evaluate(self, gameState, legal_actions):
        features_dict = self.getFeatures(gameState, legal_actions)
        weights = self.getWeights()
        action_scores = {}

        for action, feature_values in features_dict.items():
            score = sum(feature_values[f] * weights.get(f, 0) for f in feature_values)
            action_scores[action] = score

        return action_scores

    def getWeights(self):
        return {
        "invShortDefendFoodDist": 50,
        "scaredTimer": 100,
        "generalDefendFood": 60
    }

    def getFeatures(self, gameState, possibleMoves):
        # Main decision method for defender
        currentPosition = gameState.getAgentPosition(self.index)
        # Reset target if reached
        if currentPosition == self.targetPosition:
            self.targetPosition = None
        intruders = []
        closestIntruderPosition = []
        minimumDistance = 9999999

        # Look for enemy intruders in our territory
        enemyIndices = self.getOpponents(gameState)
        enemyIdx = 0
        foodLocations = self.getFood(gameState).asList()
        attack = []
        nearestFood = min([self.getMazeDistance(currentPosition, food) for food in foodLocations]) if foodLocations else 9999
        while enemyIdx != len(enemyIndices):
            enemyIndex = enemyIndices[enemyIdx]
            enemyAgent = gameState.getAgentState(enemyIndex)
            if enemyAgent.isPacman and enemyAgent.getPosition() != None:
                enemyPosition = enemyAgent.getPosition()
                intruders.append(enemyPosition)
            else:
                # Check if attack is possible
                if enemyAgent.getPosition() and self.getMazeDistance(enemyAgent.getPosition(), currentPosition) > nearestFood * 2 + 2 or enemyAgent.scaredTimer > 5:
                    attack.append(True)
                else:
                    attack.append(False)
                    self.attack = False
                    self.targetPosition = None
            enemyIdx = enemyIdx + 1
        # If intruders found, target the closest one
        if len(intruders) > 0:
            self.attack = False
            self.chase = True
            for intruderPosition in intruders:
                distance = self.getMazeDistance(intruderPosition, currentPosition)
                if distance < minimumDistance:
                    minimumDistance = distance
                    closestIntruderPosition.append(intruderPosition)
            self.targetPosition = closestIntruderPosition[-1]
        # Attack if possible
        else:
            if not len(attack) or all(attack):
                foodDistances = [self.getMazeDistance(currentPosition, food) for food in foodLocations]
                if foodDistances:
                    self.targetPosition = foodLocations[foodDistances.index(min(foodDistances))]
                    self.attack = True
            elif len(self.previousFoodState) > 0:
                if len(self.getFoodYouAreDefending(gameState).asList()) < len(self.previousFoodState):
                    missingFood = set(self.previousFoodState) - set(self.getFoodYouAreDefending(gameState).asList())
                    self.targetPosition = missingFood.pop()

        # Update food state for next comparison
        self.previousFoodState = self.getFoodYouAreDefending(gameState).asList()
        self.determinePatrolPoints(gameState)

        # If no target, Patrol
        if self.targetPosition == None:
            self.targetPosition = random.choice(self.patrolPositions)

        # Get valid defensive moves
        moveOptions = []
        distanceValues = []

        moveIdx = 0
        #  distance to target from  each move
        while moveIdx < len(possibleMoves):
            currentMove = possibleMoves[moveIdx]
            nextState = gameState.generateSuccessor(self.index, currentMove)
            newPosition = nextState.getAgentPosition(self.index)
            moveOptions.append(currentMove)
            distanceValues.append(self.getMazeDistance(newPosition, self.targetPosition))
            moveIdx = moveIdx + 1
        # Choose move that minimizes distance to target
        shortestDistance = min(distanceValues)
        feature_value_per_action = {}

        for move, distance in zip(moveOptions, distanceValues):
            if gameState.getAgentState(self.index).scaredTimer > 1 and shortestDistance < 2:
                if distance != shortestDistance:
                    feature_value_per_action.setdefault(move, {})['scaredTimer'] = 40
            else:
                if distance == shortestDistance:
                    feature_value_per_action.setdefault(move, {})['invShortDefendFood'] = 20

        if not feature_value_per_action:
            for action in possibleMoves:
                feature_value_per_action.setdefault(move, {})['generalDefendFood'] = 10

        return feature_value_per_action
