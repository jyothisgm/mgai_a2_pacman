from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math


### Penalty for worst case scenerio for enemy movement

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
        self.action = action

    def getLegalMovesRestrictingOpposite(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.agentIndex)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping
        oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.agentIndex).configuration.direction]
        if len(legalMoves) == 1:
            return legalMoves  # Only one option
        else:
            # Avoid going back if possible except for parent
            if not self.parent and oppositeDirection in legalMoves:
                legalMoves.remove(oppositeDirection)
            return legalMoves  # Choose randomly among remaining legal moves

    def uct_select_child(self):
        C = 1.4  # Try tuning this based on reward scale
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
    def registerInitialState(self, gameState):
        # Initialize game state data
        CaptureAgent.registerInitialState(self, gameState)
        self.discountRate = 0.9
        self.homeBase = gameState.getAgentState(self.index).getPosition()
        self.attack = False
        self.chase = False
        self.targets = None
        self.maxDistance = self.findLongestDistanceInMap(gameState)
        self.borderCrossingPoint = self.calculateBorderCrossingPoints(gameState)
        self.foodRemaining = len(self.getFood(gameState).asList())

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
        return featureMap * weightMap

    def calculateFeatures(self, gameState, action):
        # Feature Extraction
        featureMap = util.Counter()
        successor = self.getSuccessor(gameState, action)
        featureMap['scoreChange'] = self.getScore(successor)
        return featureMap

    def getWeights(self, gameState, action):
        # Default weights
        return {'scoreChange': 1.0}

    def getGhosts(self, gameState):
        enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        currentPosition = gameState.getAgentState(self.index).getPosition()
        ghostDistances = [self.getMazeDistance(currentPosition, a.getPosition()) for a in ghosts]

        return ghosts, ghostDistances

    def getIntruders(self, gameState):
        # Return positions of all enemies
        enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        intruders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        intrudersDist = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a) for a in intruders]

        return intruders, intrudersDist
    
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
        point_a = point_b = None
        for pos1, pos2 in combinations(all_positions, 2):
            dist = self.getMazeDistance(pos1, pos2)
            if dist > max_distance:
                max_distance = dist
                point_a, point_b = pos1, pos2
        # print(f"Longest distance is {max_distance} between {point_a} and {point_b}")
        return max_distance

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
    
    def getLegalMovesRestrictingOpposite(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.index)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping
        oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if len(legalMoves) == 1:
            return legalMoves  # Only one option
        else:
            # Avoid going back if possible
            # if oppositeDirection in legalMoves:
            #     legalMoves.remove(oppositeDirection)
            return legalMoves  # Choose randomly among remaining legal moves

class AttackerAgent(BaseStrategyAgent):
    '''
    Inheriting properties of Base Class
    '''
    def __init__(self, index):
        # Initialize the agent with default values
        CaptureAgent.__init__(self, index)
        self.stuckCounter = 0  # Counter to track if agent is stuck
        self.aggressiveMode = False  # Flag for aggressive behavior
        self.remainingPowerPellets = 0  # Count of remaining power pellets
        self.currentScore = 0
        self.numberOfSimulations = 50
        self.mcDepth = 20

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
                # action = max(legalActions, key=lambda a: self.evaluate(state, a))
                action = random.choice(legalActions)
                currentStateIsPacman = state.getAgentState(self.index).isPacman
                state = state.generateSuccessor(self.index, action)

                new_position = state.getAgentState(self.index).getPosition()
                totalReward += self.evaluate(state, Directions.STOP)

                if currentStateIsPacman and not state.getAgentState(self.index).isPacman and (self.getScore(state) - self.currentScore):
                    totalReward += .2 * depth
                    break
                if state.getAgentState(self.index).getPosition() == self.homeBase:
                    totalReward -= .7 * (maxDepth - depth)
                    break
                if new_position in visited_positions:
                    totalReward -= 1  # You can tune this penalty
                visited_positions.add(new_position)
                
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
        self.remainingPowerPellets = 0  # Count of remaining power pellets
        self.currentScore = 0
        self.numberOfSimulations = 20
        self.mcDepth = 5

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
        return patrolPositions

    def chooseAction(self, gameState):
        # Main decision method for defender
        currentPosition = gameState.getAgentPosition(self.index)
        self.patrolPositions = self.determinePatrolPoints(gameState)
        
        self.remainingPowerPellets = len(self.getCapsules(gameState))
        self.currentScore = self.getScore(gameState)

        ghosts, ghostDist = self.getGhosts(gameState)
        foodLocations = self.getFood(gameState).asList()
        self.foodRemaining = len(foodLocations)
        intruders, intrudersDist = self.getIntruders(gameState)
        self.targets = None
        if len(intruders):
            # self.mcDepth = min([self.getMazeDistance(currentPosition, a) for a in intruders]) + 1
            self.chase = True
            self.attack = False
            # self.targets = intruders
        elif (len(ghostDist) and min([self.getMazeDistance(currentPosition, a) for a in foodLocations]) <  min(ghostDist) * 2 + 1) or ghosts[ghostDist.index(min(ghostDist))].scaredTimer > 5:
            self.chase = False
            self.attack = True
            self.targets = self.patrolPositions
        else:
            self.chase = False
            self.attack = False
            self.targets = self.patrolPositions
        
        bestMove = self.runMCTS(gameState, numSimulations=self.numberOfSimulations, maxDepth=self.mcDepth)
        return bestMove

    def calculateFeatures(self, gameState, action):
        # Create feature counter for evaluating actions
        featureMap = util.Counter()
        if action != Directions.STOP:
            successor = self.getSuccessor(gameState, action) 
        else:
            successor = gameState
        intruders, intruderDist = self.getIntruders(successor)
        currentPosition = successor.getAgentPosition(self.index)
        foodLocations = self.getFood(successor).asList()
        minFoodDist = [self.getMazeDistance(currentPosition, a) for a in foodLocations]
        capsuleDist = [self.getMazeDistance(currentPosition, a) for a in self.getCapsulesYouAreDefending(gameState)]
        # Penalty for Each Step
        featureMap['step'] = -1
        if self.chase:
            if intruders:
                featureMap['intruder'] = ((self.maxDistance - min(intruderDist))/self.maxDistance)**5
                if currentPosition in self.getCapsulesYouAreDefending(gameState):
                    featureMap['saveCapsule'] = 1
            if successor.getAgentState(self.index).isPacman:
                featureMap['defend'] = -2
            else:
                featureMap['defend'] = 1
        # Incentive to capture food
        elif self.attack and foodLocations:
            foodCapture =  self.foodRemaining - len(foodLocations)
            featureMap['foodCapture'] = min(foodCapture, 1)
            if not foodCapture:
                featureMap['distanceToFood'] = ((self.maxDistance - min(minFoodDist))/self.maxDistance)**5

            # Helps in return Home
            scoreDiff = self.getScore(successor) - self.currentScore
            featureMap['scoreChange'] = scoreDiff

            featureMap['capsuleCapture'] = 0
            if self.remainingPowerPellets > len(self.getCapsules(successor)):
                featureMap['capsuleCapture'] = 1
            featureMap['defend'] = 0
        else:
            featureMap['defend'] = ((self.maxDistance - min([self.getMazeDistance(currentPosition, a) for a in self.patrolPositions])) / self.maxDistance)**5
            if successor.getAgentState(self.index).isPacman:
                featureMap['defend'] -= - 0.1
        return featureMap

    def getWeights(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        # Adjust weights based on current mode
        weights = {
            'step': 1,
            'foodCapture': 1,
            'capsuleCapture': 1,
            'scoreChange': 3,
            'intruder': 15,
            'saveCapsule': 10,
            'defend': 2,
            'distanceToFood': 2
        }

        return weights

    def runMCTS(self, rootState, numSimulations=80, maxDepth=10):
        rootNode = MCTSNode(rootState, agentIndex=self.index)
        for i in range(numSimulations):
            node = rootNode
            state = rootState.deepCopy()
            
            totalReward = 0
            depth = 0

            # SELECTION
            while node.untriedActions == [] and node.children:
                node = node.uct_select_child()
                state = state.generateSuccessor(self.index, node.action)

            simulate = True
            # EXPANSION
            if node.untriedActions:
                action = random.choice(node.untriedActions)
                node.untriedActions.remove(action)
                nextState = state.generateSuccessor(self.index, action)
                if state.getAgentState(self.index).getPosition() == self.homeBase:
                    totalReward = self.evaluate(state, Directions.STOP)
                    totalReward -= 20
                    simulate = False
                if self.targets and state.getAgentState(self.index).getPosition() in self.targets:
                    totalReward += 15
                    simulate = False
                if simulate:
                    currentStateIsPacman = state.getAgentState(self.index).isPacman
                    childNode = MCTSNode(nextState, parent=node, action=action, agentIndex=self.index)
                    node.children.append(childNode)
                    node = childNode
                    state = nextState

            # SIMULATION
            last_position = state.getAgentState(self.index).getPosition()
            visited_positions = set(last_position)
            while simulate and depth < maxDepth:
                legalActions = self.getLegalMovesRestrictingOpposite(state)
                if not legalActions:
                    break
                if self.chase:
                    print([(each, self.evaluate(state, each)) for each in legalActions])
                action = max(legalActions, key=lambda a: self.evaluate(state, a))
                # action = random.choice(legalActions)
                currentStateIsPacman = state.getAgentState(self.index).isPacman
                state = state.generateSuccessor(self.index, action)

                new_position = state.getAgentState(self.index).getPosition()
                depth += 1
                reward = (self.discountRate)**depth * self.evaluate(state, Directions.STOP)
                totalReward += reward

                if currentStateIsPacman and not state.getAgentState(self.index).isPacman and (self.getScore(state) - self.currentScore):
                    totalReward += 15 * (self.discountRate)**(depth) * self.getScore(state) - self.currentScore
                    break
                if state.getAgentState(self.index).getPosition() == self.homeBase:
                    totalReward -=  20 * (self.discountRate)**(depth)
                    break
                if self.targets and state.getAgentState(self.index).getPosition() in self.targets:
                    totalReward += 10 * (self.discountRate)**(depth)
                    break
                if not self.chase and new_position in visited_positions:
                    totalReward -= .3  # You can tune this penalty

            totalReward = totalReward / max(depth, 1)

            # BACKPROPAGATION
            counter = 0
            while node is not None:
                node.visits += 1
                node.value += (self.discountRate ** counter) * totalReward
                node = node.parent
                counter += 1
        
        if rootNode.children:
            bestChild = max(rootNode.children, key=lambda c: c.value/c.visits)
        else:
            return random.choice(rootState.getLegalActions(self.index))
        
        if self.chase:
            print([(c.action, c.value/c.visits) for c in rootNode.children])
        return bestChild.action
