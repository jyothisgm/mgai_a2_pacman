from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint
import math

EXPLORE_RATE = math.sqrt(2.0)
NUM_SIM = 500
REWARD_DISCOUNT = 0.8
DEPTH = 10
MAX_TIME = 0.5 # 70ms
EPSILON = 0.02

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
        self.untriedActions = self.getLegalMoves(gameState)
        self.action = action

    def getLegalMoves(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.agentIndex)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping\
        return legalMoves  # Choose randomly among remaining legal moves

    def uctSelectChild(self):
        C = EXPLORE_RATE  # Try tuning this based on reward scale
        def uctScore(child):
            averageValue = child.value / (child.visits + 1e-4)
            exploration = C * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4))
            return averageValue + exploration

        return max(self.children, key=uctScore)

    def printTree(self, indent=0):
        indentStr = " " * indent
        avgValue = self.value / self.visits if self.visits > 0 else 0
        print(f"{indentStr}- Action: {self.action or None}, Visits: {self.visits}, AvgValue: {avgValue:.2f}")
        for child in self.children:
            child.printTree(indent + 4)

##########
# Agents #
##########
class BaseStrategyAgent(CaptureAgent):
    '''
    Methods inherited from the baselineTeam.py
    '''
    def __init__(self, index):
        # Initialize the agent with default values
        CaptureAgent.__init__(self, index)
        self.discountRate = REWARD_DISCOUNT
        self.homeBase = []
        self.maxTime = MAX_TIME
        self.epsilon = EPSILON
        self.numberOfSimulations = NUM_SIM
        self.mcDepth = DEPTH

    def registerInitialState(self, gameState):
        # Initialize game state data
        self.foodRemaining = 9999999  # High initial value for food remaining

        CaptureAgent.registerInitialState(self, gameState)
        # Store home position
        self.homeBase.append(gameState.getAgentState(self.index).getPosition())
        # Calculate border crossing points
        self.borderCrossingPoint = self.calculateBorderCrossingPoints(gameState)
        self.maxDistance = self.findLongestDistanceInMap(gameState)

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
        intruders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        currentPosition = gameState.getAgentState(self.index).getPosition()
        intruderDistances = [self.getMazeDistance(currentPosition, a) for a in intruders]

        return intruders, intruderDistances

    def getLegalMoves(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.index)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping
        return legalMoves  # Choose randomly among remaining legaQl moves

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

        allPositions = []
        width = gameState.data.layout.width
        height = gameState.data.layout.height

        # 1. Collect all legal (non-wall) positions
        for x in range(width):
            for y in range(height):
                if not gameState.hasWall(x, y):
                    allPositions.append((x, y))

        # 2. Check all unique position pairs
        maxDistance = 0
        for pos1, pos2 in combinations(allPositions, 2):
            dist = self.getMazeDistance(pos1, pos2)
            if dist > maxDistance:
                maxDistance = dist
        return maxDistance

    def updateGhostsTowardPacman(self, gameState, temperature=1.0):
        """
        Ghosts move probabilistically toward our Pacman based on maze distance.
        Higher temperature = more randomness, lower = more greedy.
        """
        pacmanPos = gameState.getAgentState(self.index).getPosition()

        for enemyIndex in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemyIndex)

            if enemyState.isPacman or enemyState.getPosition() is None:
                continue  # Skip attackers and invisible ghosts

            ghostPos = enemyState.getPosition()
            if self.getMazeDistance(ghostPos, pacmanPos) > 6:
                continue
            legalActions = gameState.getLegalActions(enemyIndex)
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)

            # Compute maze distances for each possible move
            distances = []
            actions = []
            for action in legalActions:
                successor = gameState.generateSuccessor(enemyIndex, action)
                newPos = successor.getAgentState(enemyIndex).getPosition()
                dist = self.getMazeDistance(newPos, pacmanPos)
                distances.append(dist)
                actions.append(action)

            # Convert distances to a probability distribution (lower distance = higher prob)
            # We invert and scale using a temperature parameter
            expWeights = [math.exp(-d / temperature) for d in distances]
            total = sum(expWeights)
            probs = [w / total for w in expWeights]

            # Sample based on weighted probability
            chosenAction = random.choices(actions, weights=probs, k=1)[0]
            gameState = gameState.generateSuccessor(enemyIndex, chosenAction)

        return gameState

    def canBeCapturedInNSteps(self, gameState, n):
        """
        Returns True if any ghost can reach the agent within n steps.
        """
        currentState = gameState.getAgentState(self.index)
        currentPos = currentState.getPosition()

        # If you cannot be captured (i.e., you're not scared and not a Pacman), return False
        if not currentState.isPacman and currentState.scaredTimer == 0:
            return False

        enemyIndices = self.getOpponents(gameState)

        for enemyIndex in enemyIndices:
            enemyState = gameState.getAgentState(enemyIndex)
            enemyPos = enemyState.getPosition()

            # Enemy must be in a capturing state: not Pacman and not scared
            if enemyState.isPacman or enemyState.scaredTimer > 0:
                continue

            # Check if the enemy can reach you in n steps
            dist = self.getMazeDistance(enemyPos, currentPos)
            if dist <= n:
                return random.choices([True, False], weights=[1, dist - 1])[0]

        return False


class AttackerAgent(BaseStrategyAgent):
    '''
    Inheriting properties of Base Class
    '''
    def __init__(self, index):
        # Initialize the agent with default values
        BaseStrategyAgent.__init__(self, index)
        self.stuckCounter = 1  # Counter to track if agent is stuck
        self.aggressiveMode = False  # Flag for aggressive behavior
        self.returnHome = False  # Flag to return home after collecting food
        self.borderCrossingPoint = []  # Points to cross into enemy territory
        self.isStuck = False  # Flag to indicate if agent is stuck
        self.remainingPowerPellets = 0  # Count of remaining power pellets
        self.currentScore = 0
        self.discardedFoodPosition = []

    def calculateFeatures(self, gameState, action):
        # Create feature counter for evaluating actions
        featureMap = util.Counter()
        if action != Directions.STOP:
            successor = self.getSuccessor(gameState, action)
        else:
            successor = gameState
        foodLocations = self.getFood(successor).asList()
        foodCapture = successor.getAgentState(self.index).numCarrying
        if not foodCapture:
            foodLocations = [item for item in foodLocations if item not in self.discardedFoodPosition]

        # Penalty for Each Step
        featureMap['step'] = -1

        # Incentive for being a Pacman
        if successor.getAgentState(self.index).isPacman:
            featureMap['isAttacker'] = 1
        else:
            featureMap['isAttacker'] = 0

        currentPosition = successor.getAgentPosition(self.index)
        foodDists = [self.getMazeDistance(currentPosition, a) for a in foodLocations]
        borderDists = [self.getMazeDistance(currentPosition, a) for a in self.borderCrossingPoint]

        # Incentive to capture food
        featureMap['foodCapture'] = min(foodCapture, 2)
        # totalFoodDists = 0
        # for eachFood in foodDists:
        # foodDists += ((self.maxDistance - min(foodDists))/self.maxDistance)**5

        if len(foodDists):
            featureMap['distanceToFood'] = ((self.maxDistance - min(foodDists))/self.maxDistance)**5
        totalBorderDists = 0
        for eachBorder in borderDists:
            totalBorderDists += ((self.maxDistance - eachBorder)/self.maxDistance)**5

        featureMap['scoreChange'] = ((self.maxDistance - min(borderDists))/self.maxDistance)**5 * foodCapture

        featureMap['capsuleCapture'] = 0
        if self.remainingPowerPellets > len(self.getCapsules(successor)):
            featureMap['capsuleCapture'] = 1

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        intruders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        if successor.getAgentState(self.index).scaredTimer:
            featureMap['intruder'] = -1
        if successor.getAgentState(self.index).getPosition() in intruders:
            featureMap['intruder'] = 1
        if intruders and successor.getAgentState(self.index).getPosition() in self.getCapsulesYouAreDefending(gameState):
            featureMap['intruder'] = 2
        return featureMap

    def getWeights(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        # Adjust weights based on current mode
        weights = {
            'step': 0.5,
            'foodCapture': 10,
            'capsuleCapture': 25,
            'isAttacker': 15,
            'scoreChange': 10,
            'intruder': 4,
            'distanceToFood': 7
        }
        if self.aggressiveMode:
            # Weights for aggressive mode
            weights['isAttacker'] = 25
        ghosts, ghostDist = self.getGhosts(gameState)
        if not ghosts or ghosts[ghostDist.index(min(ghostDist))].scaredTimer > 2:
            weights['isAttacker'] = 40
            weights['scoreChange'] = 3
            weights['foodCapture'] = 30

        return weights

    def chooseAction(self, gameState):
        currentPosition = gameState.getAgentState(self.index).getPosition()
        self.remainingPowerPellets = len(self.getCapsules(gameState))
        self.currentScore = self.getScore(gameState)
        self.attack = True
        allFood = self.getFood(gameState).asList()
        foodDist = [self.getMazeDistance(currentPosition, a) for a in allFood]

        # Aggressive Mode
        if self.homeBase in currentPosition or self.foodRemaining != len(allFood):
            self.stuckCounter = 1
            self.foodRemaining = len(allFood)
            self.aggressiveMode = False
            self.discardedFoodPosition = []
        else:
            self.stuckCounter += 1
            if self.stuckCounter > 20:
                self.aggressiveMode = True
            if self.stuckCounter % 5 == 0:
                self.discardedFoodPosition.append(allFood[foodDist.index(min(foodDist))])
                if len(self.discardedFoodPosition) == self.foodRemaining:
                    self.discardedFoodPosition = []

        # Run MCTS instead of plain simulation
        bestMove = self.runMCTS(gameState, numSimulations=self.numberOfSimulations, maxDepth=self.mcDepth)
        return bestMove

    def runMCTS(self, rootState, numSimulations=10, maxDepth=10):
        rootNode = MCTSNode(rootState, agentIndex=self.index)
        rootIsPacman = rootState.getAgentState(self.index).isPacman

        startTime = time.time()
        simulations = 0
        while time.time() - startTime < self.maxTime and simulations < numSimulations:
            node = rootNode
            state = rootState.deepCopy()
            depthCounter = 0

            # SELECTION
            while node.untriedActions == [] and node.children:
                node = node.uctSelectChild()
                state = state.generateSuccessor(self.index, node.action)
                depthCounter +=1

            # EXPANSION
            if node.untriedActions:
                action = random.choice(node.untriedActions)
                node.untriedActions.remove(action)
                nextState = state.generateSuccessor(self.index, action)
                # state = self.updateGhostsTowardPacman(state)

                # Check if the new state is the end condition
                statePos = state.getAgentState(self.index).getPosition()
                if (self.remainingPowerPellets > len(self.getCapsules(state)) or
                        statePos in self.homeBase or
                        self.getScore(state) - self.currentScore):
                    node.untriedActions = []
                else:
                    childNode = MCTSNode(nextState, parent=node, action=action, agentIndex=self.index)
                    node.children.append(childNode)
                    node = childNode
                    state = nextState

            # SIMULATION
            totalReward = 0
            depth = 0
            lastPosition = state.getAgentState(self.index).getPosition()
            visitedPositions = set(lastPosition)
            while depth < maxDepth:
                legalActions = self.getLegalMoves(state)
                if not legalActions:
                    break
                if random.random() < self.epsilon:
                    action = random.choice(legalActions)
                else:
                    action = max(legalActions, key=lambda a: self.evaluate(state, a))
                action = random.choice(legalActions)
                state = state.generateSuccessor(self.index, action)
                # state = self.updateGhostsTowardPacman(state)
                newPosition = state.getAgentState(self.index).getPosition()
                depth += 1
                depthCounter +=1

                # Get reward for new state
                reward = (self.discountRate)**depth * self.evaluate(state, Directions.STOP)
                totalReward += reward

                # Penalty for Visiting the same Position
                if newPosition in visitedPositions:
                    totalReward -= (self.discountRate)**depth * 1  # You can tune this penalty

                ''' Check if End condition has reached '''
                # Capsule captured
                if self.remainingPowerPellets > len(self.getCapsules(state)):
                    totalReward += 20 * (self.discountRate)**(depth)
                    break

                # Score increased
                if self.getScore(state) - self.currentScore:
                    totalReward += 10 * (self.discountRate)**(depth) * (self.getScore(state) - self.currentScore)
                    break

                # Reset to home base
                if newPosition in self.homeBase or (rootIsPacman and self.canBeCapturedInNSteps(state, depthCounter+1)):
                    totalReward -= 55 * (self.discountRate)**(depth)
                    break

            totalReward = totalReward / max(depth, 1)

            # BACKPROPAGATION
            counter = 0
            while node is not None:
                node.visits += 1
                node.value += (self.discountRate ** counter) * totalReward
                node = node.parent
                counter += 1
            simulations += 1
        if rootNode.children:
            bestChild = max(rootNode.children, key=lambda c: c.value/c.visits)
        else:
            return random.choice(rootState.getLegalActions(self.index))
        return bestChild.action


class DefenderAgent(BaseStrategyAgent):
    def __init__(self, index):
        # Initialize defender agent
        BaseStrategyAgent.__init__(self, index)
        self.remainingPowerPellets = 0  # Count of remaining power pellets
        self.currentScore = 0

    def determinePatrolPoints(self, gameState):
        '''
        Look for center of the maze for patrolling
        '''
        # Calculate border position
        borderX = (gameState.data.layout.width - 2) // 2
        if not self.red:
            borderX += 2  # Adjust for blue team
        else:
            borderX -= 1
        self.patrolPositions = []
        # Find all positions along border that aren't walls
        for yCoord in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(borderX, yCoord):
                self.patrolPositions.append((borderX, yCoord))

        # Focus on patrolling near the closest food point near border
        criticalDefensePoints = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
        minDefenseDistance = 999999
        patrolPositions = []
        for eachPoint in criticalDefensePoints:
            for eachPatrol in self.patrolPositions:
                distance = self.getMazeDistance(eachPoint, eachPatrol)
                if minDefenseDistance > distance:
                    minDefenseDistance = distance
                    patrolPositions = [eachPatrol]
                elif minDefenseDistance == distance:
                    patrolPositions.append(eachPatrol)
        return patrolPositions

    def chooseAction(self, gameState):
        # Main decision method for defender
        currentPosition = gameState.getAgentPosition(self.index)
        self.patrolPositions = self.determinePatrolPoints(gameState)

        self.remainingPowerPellets = len(self.getCapsules(gameState))
        self.currentScore = self.getScore(gameState)

        ghosts, ghostDist = self.getGhosts(gameState)
        foodLocations = self.getFood(gameState).asList()
        minFoodDist = min([self.getMazeDistance(currentPosition, a) for a in foodLocations]) if foodLocations else 99999
        self.foodRemaining = len(foodLocations)
        intruders, intrudersDist = self.getIntruders(gameState)
        self.targets = None
        if len(intruders) and gameState.getAgentState(self.index).scaredTimer < 5:
            self.chase = True
            self.attack = False
            self.targets = None
        elif not len(ghostDist) or minFoodDist * 2 <  min(ghostDist) or ghosts[ghostDist.index(min(ghostDist))].scaredTimer > 5:
            self.chase = False
            self.attack = True
            self.targets = None
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
        minBorderDist = min([self.getMazeDistance(currentPosition, a) for a in self.borderCrossingPoint])

        # Penalty for Each Step
        featureMap['step'] = -1

        # Attack Intruders
        if self.chase:
            featureMap['intruder'] = 1
            if intruders:
                featureMap['intruder'] = ((self.maxDistance - min(intruderDist))/self.maxDistance)**5
                # if capsuleDist and min(intruderDist) > min(capsuleDist) and min(intruderDist)/2 < min(capsuleDist):
                #     featureMap['saveCapsule'] = ((self.maxDistance - min(capsuleDist))/self.maxDistance)**5
            if successor.getAgentState(self.index).isPacman:
                featureMap['defend'] = -1
            else:
                featureMap['defend'] = 1
        # Incentive to capture food
        elif self.attack and foodLocations:
            foodCapture = successor.getAgentState(self.index).numCarrying
            featureMap['foodCapture'] = min(foodCapture, 1)
            if not foodCapture and minFoodDist:
                featureMap['distanceToFood'] = ((self.maxDistance - min(minFoodDist))/self.maxDistance)**5
            else:
                featureMap['distanceToFood'] = 1
            # Helps in return Home
            scoreDiff = self.getScore(successor) - self.currentScore
            featureMap['scoreChange'] = scoreDiff
            featureMap['returnHome'] = ((self.maxDistance - minBorderDist)/self.maxDistance)**5 * (foodCapture + scoreDiff)

            featureMap['capsuleCapture'] = 0
            # capsuleAttackDist = [self.getMazeDistance(currentPosition, a) for a in self.getCapsules(successor)]
            # if capsuleAttackDist:
            #     featureMap['capsuleCapture'] = ((self.maxDistance - min(capsuleAttackDist))/self.maxDistance)**5
            featureMap['defend'] = 0
        else:
            if successor.getAgentState(self.index).isPacman:
                featureMap['defend'] = 1 - ((self.maxDistance - minBorderDist) / self.maxDistance)**5
            else:
                patrolDist = min([self.getMazeDistance(currentPosition, a) for a in self.patrolPositions])
                featureMap['defend'] = ((self.maxDistance - patrolDist) / self.maxDistance)**5
        return featureMap

    def getWeights(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        # Adjust weights based on current mode
        weights = {
            'step': 0.1,
            'foodCapture': 8,
            'capsuleCapture': 20,
            'scoreChange': 5,
            'intruder': 30,
            'saveCapsule': 20,
            'defend': 5,
            'distanceToFood': 5,
            'returnHome': 10
        }

        return weights

    def runMCTS(self, rootState, numSimulations=80, maxDepth=10):
        rootNode = MCTSNode(rootState, agentIndex=self.index)
        rootIntruder, _ = self.getIntruders(rootState)
        rootIsPacman = rootState.getAgentState(self.index).isPacman

        startTime = time.time()
        simulations = 0
        while time.time() - startTime < self.maxTime and simulations < numSimulations:
            node = rootNode
            state = rootState.deepCopy()
            depthCounter = 0

            # SELECTION
            while node.untriedActions == [] and node.children:
                node = node.uctSelectChild()
                state = state.generateSuccessor(self.index, node.action)
                depthCounter +=1

            # EXPANSION
            if node.untriedActions:
                action = random.choice(node.untriedActions)
                node.untriedActions.remove(action)
                nextState = state.generateSuccessor(self.index, action)

                # Check if the new state is the end condition
                statePos = state.getAgentState(self.index).getPosition()
                capsuleAttackDist = [self.getMazeDistance(statePos, a) for a in self.getCapsules(state)]
                intruders, _ = self.getIntruders(state)
                if (self.remainingPowerPellets > len(capsuleAttackDist) or 
                        statePos in self.homeBase or 
                        (self.targets and statePos in self.targets) or 
                        len(rootIntruder) > len(intruders)):
                    node.untriedActions = []
                else:
                    childNode = MCTSNode(nextState, parent=node, action=action, agentIndex=self.index)
                    node.children.append(childNode)
                    node = childNode
                    state = nextState

            # SIMULATION
            totalReward = 0
            depth = 0
            lastPosition = state.getAgentState(self.index).getPosition()
            visitedPositions = set(lastPosition)
            while depth < maxDepth:
                legalActions = self.getLegalMoves(state)
                if not legalActions:
                    break
                if random.random() < self.epsilon:
                    action = random.choice(legalActions)
                else:
                    action = max(legalActions, key=lambda a: self.evaluate(state, a))
                state = state.generateSuccessor(self.index, action)
                newPosition = state.getAgentState(self.index).getPosition()
                depthCounter +=1
                depth += 1

                # Get reward for new state
                reward = (self.discountRate)**depth * self.evaluate(state, Directions.STOP)
                totalReward += reward

                # Penalty for Visiting the same Position
                if not self.chase and newPosition in visitedPositions:
                    totalReward -=  (self.discountRate)**depth * 1  # You can tune this penalty

                ''' Check if End condition has reached '''
                capsuleAttackDist = [self.getMazeDistance(newPosition, a) for a in self.getCapsules(state)]
                # Capsule captured
                if self.remainingPowerPellets > len(capsuleAttackDist):
                    totalReward += 20 * (self.discountRate)**(depth)
                    break
                
                # Score increased
                if self.getScore(state) - self.currentScore:
                    totalReward += 40 * (self.discountRate)**(depth) * (self.getScore(state) - self.currentScore)
                    break
                
                # Reset to home base
                if newPosition in self.homeBase or (rootIsPacman and self.canBeCapturedInNSteps(state, depthCounter+1)):
                    totalReward -= 30 * (self.discountRate)**(depth)
                    break
                
                # Reached a target
                if self.targets and newPosition in self.targets:
                    totalReward += 40 * (self.discountRate)**(depth)
                    break
                
                # Intruder Killed
                intruders, _ = self.getIntruders(state)
                if len(rootIntruder) > len(intruders):
                    totalReward += 40 * (self.discountRate)**(depth)
                    break

            totalReward = totalReward / max(depth, 1)

            # Backpropagation with Discount
            counter = 0
            while node is not None:
                node.visits += 1
                node.value += (self.discountRate ** counter) * totalReward
                node = node.parent
                counter += 1
            simulations += 1

        if rootNode.children:
            bestChild = max(rootNode.children, key=lambda c: c.value/c.visits)
        else:
            return random.choice(rootState.getLegalActions(self.index))
        return bestChild.action