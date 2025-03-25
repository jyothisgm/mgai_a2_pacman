from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math

#####################
## Team Pac-Champs ##
#####################
def createTeam(firstIndex, secondIndex, isRed,
            first = 'AttackerAgent', second = 'DefenderAgent'):
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
    # Creates and returns a team with two agents evaluated from the provided strings
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class MCTSNode:
    def __init__(self, gameState, parent=None, action=None, agentIndex=None):
        self.gameState = gameState
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.agentIndex = agentIndex
        self.untriedActions = self.getLegalMovesRestrictingOpposite(gameState)
        if Directions.STOP in self.untriedActions:
            self.untriedActions.remove(Directions.STOP)
        self.action = action

    def getLegalMovesRestrictingOpposite(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.agentIndex)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping
        oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.agentIndex).configuration.direction]
        if len(legalMoves) == 1:
            return legalMoves  # Only one option
        else:
            # Avoid going back if possible
            if oppositeDirection in legalMoves:
                legalMoves.remove(oppositeDirection)
            return legalMoves  # Choose randomly among remaining legal moves

    def uct_select_child(self):
        C = 0.8  # Try tuning this based on reward scale
        def uct_score(child):
            average_value = child.value / (child.visits + 1e-4)
            exploration = C * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4))
            return average_value + exploration

        return max(self.children, key=uct_score)

    def print_tree(self, indent=0):
        indent_str = " " * indent
        avg_value = self.value / self.visits if self.visits > 0 else 0
        print(f"{indent_str}- Action: {self.action}, Visits: {self.visits}, AvgValue: {avg_value:.2f}")
        for child in self.children:
            child.print_tree(indent + 4)

##########
# Agents #
##########
class BaseStrategyAgent(CaptureAgent):
    '''
    Methods inherited from the baselineTeam.py
    '''
    def getSuccessor(self, gameState, action):
        # Generate the successor state after taking an action
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()
        # Check if we need another action to reach a precise position
        if position != nearestPoint(position):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action): 
        # Evaluate an action by calculating features and multiplying by weights
        featureMap = self.calculateFeatures(gameState, action)
        weightMap = self.getWeights(gameState, action)
        return featureMap * weightMap / 10.0


    def calculateFeatures(self, gameState, action):
        # Feature Extraction
        featureMap = util.Counter()
        successor = self.getSuccessor(gameState, action)
        featureMap['scoreChange'] = self.getScore(successor)
        return featureMap

    def getWeights(self, gameState, action):
        # Default weights
        return {'scoreChange': 1.0}

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

        borderCrossingPoint = []

        # Find all points along the border that aren't walls
        for yCoord in range(1, mapDimensions[1] - 1):
            if not gameState.hasWall(mapDimensions[2], yCoord):
                borderCrossingPoint.append((mapDimensions[2], yCoord))
        return borderCrossingPoint

    def getGhosts(self, gameState):
        enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        currentPosition = gameState.getAgentState(self.index).getPosition()
        ghostDistances = [self.getMazeDistance(currentPosition, a.getPosition()) for a in ghosts]

        return ghosts, ghostDistances

    def getIntruders(self, gameState):
        # Return positions of all enemies
        enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        intruders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        currentPosition = gameState.getAgentState(self.index).getPosition()
        intruderDistances = [self.getMazeDistance(currentPosition, a.getPosition()) for a in intruders]

        return intruders, intruderDistances
    
    def getLegalMovesRestrictingOpposite(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.index)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping
        oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if len(legalMoves) == 1:
            return legalMoves  # Only one option
        else:
            # Avoid going back if possible
            if oppositeDirection in legalMoves:
                legalMoves.remove(oppositeDirection)
            return legalMoves  # Choose randomly among remaining legal moves


    def findLongestDistanceInMap(self, gameState):
        from itertools import combinations

        all_positions = []
        width = gameState.data.layout.width
        height = gameState.data.layout.height

        # 1. Collect all legal (non-wall) positions
        for x in range(width):
            for y in range(height):
                if not gameState.hasWall(x, y):
                    all_positions.append((x, y))

        # 2. Check all unique position pairs
        max_distance = 0
        for pos1, pos2 in combinations(all_positions, 2):
            dist = self.getMazeDistance(pos1, pos2)
            if dist > max_distance:
                max_distance = dist
        return max_distance


class AttackerAgent(BaseStrategyAgent):
    '''
    Inheriting properties of Base Class
    '''
    def __init__(self, index):
        # Initialize the agent with default values
        CaptureAgent.__init__(self, index)
        self.stuckCounter = 0  # Counter to track if agent is stuck
        self.aggressiveMode = False  # Flag for aggressive behavior
        self.returnHome = False  # Flag to return home after collecting food
        self.borderCrossingPoint = []  # Points to cross into enemy territory
        self.isStuck = False  # Flag to indicate if agent is stuck
        self.remainingPowerPellets = 0  # Count of remaining power pellets
        self.currentScore = 0

        self.longestDistanceInMap = 999999999
        self.numberOfSimulations = 80
        self.mcDepth = 20

    def registerInitialState(self, gameState):
        # Initialize game state data
        self.foodRemaining = 9999999  # High initial value for food remaining

        CaptureAgent.registerInitialState(self, gameState)
        # Store home position
        self.homeBase = gameState.getAgentState(self.index).getPosition()
        self.longestDistanceInMap = self.findLongestDistanceInMap(gameState)
        # Calculate border crossing points
        self.borderCrossingPoint = self.calculateBorderCrossingPoints(gameState)


    def calculateFeatures(self, gameState, action):
        # Create feature counter for evaluating actions
        featureMap = util.Counter()
        if action != Directions.STOP:
            successor = self.getSuccessor(gameState, action) 
        else:
            successor = gameState
        foodLocations = self.getFood(successor).asList()

        # Penalty for Each Step
        featureMap['step'] = -1

        # Incentive for being a PAcman
        if successor.getAgentState(self.index).isPacman:
            featureMap['isAttacker'] = 1
        else:
            featureMap['isAttacker'] = 0

        # Incentive to capture food
        if foodLocations:
            foodCapture =  self.foodRemaining - len(foodLocations)
            featureMap['foodCapture'] = min(foodCapture, 1)

        # Helps in return Home
        scoreDiff = self.getScore(successor) - self.currentScore
        featureMap['scoreChange'] = scoreDiff

        featureMap['capsuleCapture'] = 0
        if self.remainingPowerPellets > len(self.getCapsules(successor)):
            featureMap['capsuleCapture'] = 1

        return featureMap

    def getWeights(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        # Adjust weights based on current mode
        weights = {
            'step': 0.1,
            'foodCapture': 0.5,
            'capsuleCapture': 3,
            'isAttacker': 0.5,
            'scoreChange': 3,
        }
        if self.aggressiveMode:
            # Weights for aggressive mode
            weights['isAttacker'] = 1

        return weights


    def runMCTS(self, rootState, numSimulations=80, maxDepth=10):
        rootNode = MCTSNode(rootState, agentIndex=self.index)

        for _ in range(numSimulations):
            node = rootNode
            state = rootState.deepCopy()

            # SELECTION
            while node.untriedActions == [] and node.children:
                node = node.uct_select_child()
                state = state.generateSuccessor(self.index, node.action)

            # EXPANSION
            if node.untriedActions:
                action = random.choice(node.untriedActions)
                node.untriedActions.remove(action)
                nextState = state.generateSuccessor(self.index, action)
                childNode = MCTSNode(nextState, parent=node, action=action, agentIndex=self.index)
                node.children.append(childNode)
                node = childNode
                state = nextState

            # SIMULATION
            totalReward = 0
            depth = 0
            last_position = state.getAgentState(self.index).getPosition()
            visited_positions = set(last_position)
            while depth < maxDepth:
                legalActions = self.getLegalMovesRestrictingOpposite(state)
                if not legalActions:
                    break
                action = random.choice(legalActions)
                currentStateIsPacman = state.getAgentState(self.index).isPacman
                state = state.generateSuccessor(self.index, action)

                new_position = state.getAgentState(self.index).getPosition()
                totalReward += self.evaluate(state, Directions.STOP)

                if currentStateIsPacman and not state.getAgentState(self.index).isPacman and (self.getScore(state) - self.currentScore):
                    totalReward += .2 * (depth)
                    break
                if state.getAgentState(self.index).getPosition() == self.homeBase:
                    totalReward -= 10
                    break
                if new_position in visited_positions:
                    totalReward -= 2  # You can tune this penalty
                visited_positions.add(state)
                
                depth += 1
            totalReward = totalReward / (depth + 1)
            # BACKPROPAGATION
            while node is not None:
                node.visits += 1
                node.value += totalReward
                node = node.parent

        # print("\n==== MCTS Tree ====")
        # rootNode.print_tree()
        # print("===================\n")
        bestChild = max(rootNode.children, key=lambda c: c.visits)
        # print(bestChild.action)
        return bestChild.action

    def chooseAction(self, gameState):
        currentPosition = gameState.getAgentState(self.index).getPosition()
        self.remainingPowerPellets = len(self.getCapsules(gameState))
        self.currentScore = self.getScore(gameState)

        if self.foodRemaining == len(self.getFood(gameState).asList()):
            self.stuckCounter += 1
        else:
            self.stuckCounter = 0
            self.foodRemaining = len(self.getFood(gameState).asList())

        if self.homeBase == currentPosition:
            self.stuckCounter = 0  # Reset counter if agent has been captured

        if self.stuckCounter > 20:
            self.aggressiveMode = True
        else:
            self.aggressiveMode = False

        # Run MCTS instead of plain simulation
        bestMove = self.runMCTS(gameState, numSimulations=self.numberOfSimulations, maxDepth=self.mcDepth)

        # print(f"CurrentPos: {currentPosition} BestMove: {bestMove}")
        return bestMove


class DefenderAgent(BaseStrategyAgent):
    def __init__(self, index):
        # Initialize defender agent
        CaptureAgent.__init__(self, index)
        self.targetPosition = None  # Current target position
        self.previousFoodState = []  # Previous food state to track changes
        self.patrolCounter = 0  # Counter for patrol behavior
        self.attack = False
        self.chase = False
        self.foodTargetPosition = None

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
        nearestFood = min([self.getMazeDistance(currentPosition, food) for food in foodLocations])
        while enemyIdx != len(enemyIndices):
            enemyIndex = enemyIndices[enemyIdx]
            enemyAgent = gameState.getAgentState(enemyIndex)
            if enemyAgent.isPacman and enemyAgent.getPosition() != None:
                enemyPosition = enemyAgent.getPosition()
                intruders.append(enemyPosition)
            else:
                if self.getMazeDistance(enemyAgent.getPosition(), currentPosition) > nearestFood * 2 + 2 or enemyAgent.scaredTimer > 5:
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
        # If food was eaten, target the eaten food position
        else:
            if not len(attack) or all(attack):
                foodDistances = [self.getMazeDistance(currentPosition, food) for food in foodLocations]
                self.targetPosition = foodLocations[foodDistances.index(min(foodDistances))]
                self.attack = True
            elif len(self.previousFoodState) > 0:
                if len(self.getFoodYouAreDefending(gameState).asList()) < len(self.previousFoodState):
                    missingFood = set(self.previousFoodState) - set(self.getFoodYouAreDefending(gameState).asList())
                    self.targetPosition = missingFood.pop()


        # Update food state for next comparison
        self.previousFoodState = self.getFoodYouAreDefending(gameState).asList()
        self.determinePatrolPoints(gameState)
        
        # If no target, choose based on food/capsule count
        if self.targetPosition == None:
            if len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
                # Few food left, protect remaining food and capsules
                criticalDefensePoints = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
                self.targetPosition = random.choice(criticalDefensePoints)
            else:
                # Normal patrol behavior
                self.targetPosition = random.choice(self.patrolPositions)
        
        # Get valid defensive moves
        possibleMoves = self.getDefensiveMovementOptions(gameState)
        moveOptions = []
        distanceValues = []

        moveIdx = 0
        # Evaluate each move by distance to target     
        while moveIdx < len(possibleMoves):
            currentMove = possibleMoves[moveIdx]
            nextState = gameState.generateSuccessor(self.index, currentMove)
            newPosition = nextState.getAgentPosition(self.index)
            moveOptions.append(currentMove)
            distanceValues.append(self.getMazeDistance(newPosition, self.targetPosition))
            moveIdx = moveIdx + 1
        # Choose move that minimizes distance to target
        shortestDistance = min(distanceValues)
        if gameState.getAgentState(self.index).scaredTimer > 1 and shortestDistance < 2:
            bestMoves = [move for move, distance in zip(moveOptions, distanceValues) if distance != shortestDistance]
        else:
            bestMoves = [move for move, distance in zip(moveOptions, distanceValues) if distance == shortestDistance]
        if not bestMoves:
            bestMove = random.choice(possibleMoves)
        else:
            bestMove = random.choice(bestMoves)
        return bestMove
