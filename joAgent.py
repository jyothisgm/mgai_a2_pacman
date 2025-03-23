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

        self.borderCrossingPoint = []

        # Find all points along the border that aren't walls
        for yCoord in range(1, mapDimensions[1] - 1):
            if not gameState.hasWall(mapDimensions[2], yCoord):
                self.borderCrossingPoint.append((mapDimensions[2], yCoord))

        # crossingPointCount = len(self.borderCrossingPoint)
        # # Choose the middle point to focus on
        # if (crossingPointCount % 2 == 0):
        #     midPoint = (crossingPointCount // 2) 
        #     self.borderCrossingPoint = [self.borderCrossingPoint[midPoint]]
        # else:
        #     midPoint = (crossingPointCount - 1) // 2
        #     self.borderCrossingPoint = [self.borderCrossingPoint[midPoint]] 
        print("Border Crossing Point: ", self.borderCrossingPoint)  # Debug output


class AttackerAgent(BaseStrategyAgent):
    '''
    Inheriting properties of Base Class
    '''
    def __init__(self, index):
        # Initialize the agent with default values
        CaptureAgent.__init__(self, index)        
        self.currentPosition = (-5, -5)  # Initial invalid position
        self.stuckCounter = 0  # Counter to track if agent is stuck
        self.aggressiveMode = False  # Flag for aggressive behavior
        self.previousFoodState = []  # Previous food state to track changes
        self.currentFoodState = []  # Current food state
        self.returnHome = False  # Flag to return home after collecting food
        self.powerPelletActive = False  # Flag for power pellet activation
        self.currentTarget = None  # Current target position
        self.foodConsumed = 0  # Count of food consumed
        self.borderCrossingPoint = []  # Points to cross into enemy territory
        self.isStuck = False  # Flag to indicate if agent is stuck
        self.remainingPowerPellets = 0  # Count of remaining power pellets
        self.previousPowerPellets = 0  # Previous power pellet count

    def registerInitialState(self, gameState):
        # Initialize game state data
        self.foodRemaining = 9999999  # High initial value for food remaining

        CaptureAgent.registerInitialState(self, gameState)
        self.homeBase = gameState.getAgentState(self.index).getPosition()  # Store home position
        self.calculateBorderCrossingPoints(gameState)  # Calculate border crossing points

    def calculateFeatures(self, gameState, action):
        # Create feature counter for evaluating actions
        featureMap = util.Counter()
        successor = self.getSuccessor(gameState, action) 
        nextPosition = successor.getAgentState(self.index).getPosition() 
        foodLocations = self.getFood(successor).asList()
        
        # Helps in return Home 
        featureMap['scoreChange'] = self.getScore(successor)  # Score change after action

        # Check if agent will be in attack mode (Pacman)
        if successor.getAgentState(self.index).isPacman:
            featureMap['isAttacker'] = 1
        else:
            featureMap['isAttacker'] = 0

        # Find distance to closest food
        if foodLocations: 
            featureMap['distanceToFood'] = min([self.getMazeDistance(nextPosition, food) for food in foodLocations])

        if len(self.getCapsules(gameState)):
            featureMap['distanceToCapsule'] = min([self.getMazeDistance(nextPosition, capsule) for capsule in self.getCapsules(gameState)])
        enemyTeam = []
        distancesToGhosts = []
        enemyTeam = self.getOpponents(successor)

        # Calculate distances to enemy ghosts
        for enemyIdx in range(len(enemyTeam)):
            enemyAgentIdx = enemyTeam[enemyIdx]
            enemyAgent = successor.getAgentState(enemyAgentIdx)
            if not enemyAgent.isPacman and enemyAgent.getPosition() != None:
                ghostPosition = enemyAgent.getPosition()
                distancesToGhosts.append(self.getMazeDistance(nextPosition, ghostPosition))

        # If ghosts are nearby, factor in distance to closest ghost
        if len(distancesToGhosts) > 0:
            closestGhostDistance = min(distancesToGhosts)
            if closestGhostDistance < 5:  # Only relevant if ghost is close
                featureMap['ghostProximity'] = closestGhostDistance + featureMap['scoreChange']
                if closestGhostDistance < 2:
                    featureMap['ghostProximity'] = 2 * closestGhostDistance
            else:
                featureMap['ghostProximity'] = 0

        return featureMap
    
    def getWeights(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        # Adjust weights based on current mode
        if self.aggressiveMode:
            # Weights for aggressive mode
            weights = {
                    'isAttacker': 20,
                    'scoreChange': 202,
                    'distanceToFood': -8,
                    'distanceToCapsule': -210,
                    'ghostProximity': 210}
        else:
            # Normal mode weights
            weights = {
                    'isAttacker': 10,
                    'scoreChange': 202,
                    'distanceToFood': -8,
                    'distanceToCapsule': -210,
                    'ghostProximity': 210}
        
        successor = self.getSuccessor(gameState, action) 
        
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        # If ghosts are scared, don't avoid them
        if len(ghosts) > 0:
            min_dist = 999999
            for each in ghosts:
                ghost_dist = self.getMazeDistance(self.currentPosition, each.getPosition())
                if min_dist > ghost_dist:
                    nearest_ghost = each
                    min_dist = ghost_dist
            if nearest_ghost.scaredTimer > 1:
                weights['ghostProximity'] = 0
        
        return weights

    def getEnemyPositions(self, gameState):
        # Return positions of all enemies
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def getRandomLegalMove(self, gameState):
        # Get a random legal move, avoiding STOP and preferably not reversing
        legalMoves = gameState.getLegalActions(self.index)
        legalMoves.remove(Directions.STOP)  # Don't consider stopping
        oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if len(legalMoves) == 1:
            return legalMoves[0]  # Only one option
        else:
            # Avoid going back if possible
            if oppositeDirection in legalMoves:
                legalMoves.remove(oppositeDirection)
            return random.choice(legalMoves)  # Choose randomly among remaining legal moves

    def simulateMonteCarlo(self, gameState, depth):
        # Monte Carlo simulation of random moves to evaluate positions
        simulationState = gameState.deepCopy()
        value = 0
        direction_change = 0
        while depth > 0:
            next_move = self.getRandomLegalMove(simulationState)
            oppositeDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer == 0]
            # if ghosts and min([self.getMazeDistance(self.currentPosition, a.getPosition()) for a in ghosts]) < 5 and next_move == oppositeDirection:
            #     direction_change += 1
            simulationState = simulationState.generateSuccessor(self.index, next_move)
            # simulationState
            # current_sim_pos = simulationState.getAgentState(self.index).getPosition()
            depth -= 1
            
            value += self.evaluate(simulationState, Directions.STOP)
        value -= abs(direction_change * 0.05 * value)
        return value  # Evaluate final state

    def findBestPathToTarget(self, legalMoves, gameState, moveOptions, distancesToTarget):
        # Find best move to reach the target point
        shortestPathLength = 9999999999  # Very large initial value
        for moveIdx in range(0, len(legalMoves)):    
            move = legalMoves[moveIdx]
            nextState = gameState.generateSuccessor(self.index, move)
            nextPosition = nextState.getAgentPosition(self.index)
            # Calculate distance to border crossing point
            distance = self.getMazeDistance(nextPosition, self.borderCrossingPoint[0])
            distancesToTarget.append(distance)
            if(distance < shortestPathLength):
                shortestPathLength = distance

        # Find all moves that lead to shortest path
        bestMoves = [a for a, distance in zip(legalMoves, distancesToTarget) if distance == shortestPathLength]
        bestMove = random.choice(bestMoves)  # Randomly choose among best moves
        return bestMove
        
    def chooseAction(self, gameState):
        # Main decision method for selecting next action
        self.currentPosition = gameState.getAgentState(self.index).getPosition()
        self.currentFoodState = self.getFood(gameState).asList()
    
        # Check if agent is at home base or border crossing point
        if self.currentPosition == self.homeBase:
            self.isStuck = True  # Needs to move toward border
        if self.currentPosition == self.borderCrossingPoint[0]:
            self.isStuck = False  # At border, can proceed normally


        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        intruders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        intruderDistances = [self.getMazeDistance(self.currentPosition, a.getPosition()) for a in intruders]
        breakif = False
        if not gameState.getAgentState(self.index).isPacman and intruders:
            nearestIntruder = intruders[intruderDistances.index(min(intruderDistances))]
            target = nearestIntruder.getPosition()
            defCapsules = self.getCapsulesYouAreDefending(gameState)
            if defCapsules:
                capsuleDist = [self.getMazeDistance(self.currentPosition, c) for c in defCapsules]
                nearestCapsule = defCapsules[capsuleDist.index(min(capsuleDist))]
                if min(capsuleDist) < min(intruderDistances):
                    target = nearestCapsule
                    if min(capsuleDist) <= 2:
                        target = nearestIntruder.getPosition()
                else:
                    breakif = True
            elif min([self.getMazeDistance(self.currentPosition, food) for food in self.currentFoodState]) < min(intruderDistances):
                breakif = True
            if not breakif:
                legalMoves = gameState.getLegalActions(self.index)
                legalMoves.remove(Directions.STOP)
                moveIdx = 0
                moveOptions = []
                distancesToTarget = []
                while moveIdx != len(legalMoves):
                    currentMove = legalMoves[moveIdx]
                    newPosition = (gameState.generateSuccessor(self.index, currentMove)).getAgentPosition(self.index)
                    moveOptions.append(currentMove)
                    distancesToTarget.append(self.getMazeDistance(newPosition, target))
                    moveIdx += 1
                
                # Choose move that minimizes distance to target
                shortestDistance = min(distancesToTarget)
                bestMoves = [move for move, distance in zip(moveOptions, distancesToTarget) if distance == shortestDistance]
                bestMove = random.choice(bestMoves)
                return bestMove
        
        # If stuck at home, find path to border crossing
        if self.isStuck:
            legalMoves = gameState.getLegalActions(self.index)
            legalMoves.remove(Directions.STOP)
            moveOptions = []
            distancesToTarget = []
            bestMove = self.findBestPathToTarget(legalMoves, gameState, moveOptions, distancesToTarget)
            return bestMove

        if not self.isStuck:
            # Update food and power pellet states
            self.remainingPowerPellets = len(self.getCapsules(gameState))
            previousPowerPelletCount = self.previousPowerPellets
            previousFoodCount = len(self.previousFoodState)

            # Check if food was eaten and update return status           
            if len(self.currentFoodState) < len(self.previousFoodState):
                self.returnHome = True
            self.previousFoodState = self.currentFoodState
            self.previousPowerPellets = self.remainingPowerPellets

            # Reset return flag if back in home territory
            if not gameState.getAgentState(self.index).isPacman:
                self.returnHome = False

            # Check if agent is stuck in a pattern           
            availableFoodList = self.getFood(gameState).asList()
            availableFoodCount = len(availableFoodList)

            if availableFoodCount == self.foodRemaining:
                self.stuckCounter = self.stuckCounter + 1  # Increment stuck counter if food hasn't changed
            else:
                self.foodRemaining = availableFoodCount
                self.stuckCounter = 0  # Reset counter if food has changed
            if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
                self.stuckCounter = 0  # Reset counter if agent has been captured
            if self.stuckCounter > 20:
                self.aggressiveMode = True  # Switch to aggressive mode if stuck too long
            else:
                self.aggressiveMode = False
            
            possibleMoves = gameState.getLegalActions(self.index)
            possibleMoves.remove(Directions.STOP)

            # Calculate distance to closest enemy ghost      
            distanceToNearestEnemy = 999999
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 5]
            if len(ghosts) > 0:
                distanceToNearestEnemy = min([self.getMazeDistance(self.currentPosition, a.getPosition()) for a in ghosts])
            
            '''
            Power pellet strategy:
            -> If there is a power pellet available then powerPelletActive is True.
            -> If enemy Distance is less than 5 then powerPelletActive is False.
            -> If pacman scored a food then return to home powerPelletActive is False.
            '''
            # Update power pellet status
            if self.remainingPowerPellets < previousPowerPelletCount:
                self.powerPelletActive = True  # Just ate a power pellet
                self.foodConsumed = 0
            if distanceToNearestEnemy <= 5:
                self.powerPelletActive = False  # Ghost too close, be cautious
            if (len(self.currentFoodState) < len(self.previousFoodState)):
                self.powerPelletActive = False  # Prioritize returning with food

            if self.powerPelletActive:
                # Reset food counter if back in home territory
                if not gameState.getAgentState(self.index).isPacman:
                    self.foodConsumed = 0

                shortestDistanceToTarget = 999999

                # Increment food counter if food was eaten
                if len(self.currentFoodState) < previousFoodCount:
                    self.foodConsumed += 1

                # Set target based on food consumption and availability
                if len(self.currentFoodState) == 0 or self.foodConsumed >= 7:
                    self.currentTarget = self.homeBase  # Return home if enough food collected or none left
                else:
                    # Find closest food as target
                    for foodPosition in self.currentFoodState:
                        distance = self.getMazeDistance(self.currentPosition, foodPosition)
                        if distance < shortestDistanceToTarget:
                            shortestDistanceToTarget = distance
                            self.currentTarget = foodPosition

                legalMoves = gameState.getLegalActions(self.index)
                legalMoves.remove(Directions.STOP)
                moveOptions = []
                distancesToTarget = []
                
                # Evaluate each move by distance to target
                moveIdx = 0
                while moveIdx != len(legalMoves):
                    currentMove = legalMoves[moveIdx]
                    newPosition = (gameState.generateSuccessor(self.index, currentMove)).getAgentPosition(self.index)
                    moveOptions.append(currentMove)
                    distancesToTarget.append(self.getMazeDistance(newPosition, self.currentTarget))
                    moveIdx += 1
                
                # Choose move that minimizes distance to target
                shortestDistance = min(distancesToTarget)
                bestMoves = [move for move, distance in zip(moveOptions, distancesToTarget) if distance == shortestDistance]
                bestMove = random.choice(bestMoves)
                return bestMove
            else:
                # Not in power pellet mode - use Monte Carlo evaluation
                self.foodConsumed = 0
                moveEvaluations = []
                for move in possibleMoves:
                    nextState = gameState.generateSuccessor(self.index, move)
                    simulationValue = 0
                    # Run multiple simulations for each possible move
                    
                    for _ in range(20):
                        simulationValue += self.simulateMonteCarlo(nextState, 20)
                    moveEvaluations.append(simulationValue)

                # Choose move with highest evaluation score
                bestEvaluation = max(moveEvaluations)
                bestMoves = [move for move, value in zip(possibleMoves, moveEvaluations) if value == bestEvaluation]
                bestMove = random.choice(bestMoves)
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

        # Remove positions at the edges to focus on central patrol area
        # for i in range(len(self.patrolPositions)):
        #     if len(self.patrolPositions) > 2:
        #         self.patrolPositions.remove(self.patrolPositions[0])
        #         self.patrolPositions.remove(self.patrolPositions[-1])
        #     else:
        #         break

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
                self.chase = True
            else:
                if self.getMazeDistance(enemyAgent.getPosition(), currentPosition) > nearestFood * 2 + 2 or enemyAgent.scaredTimer > 4:
                    attack.append(True)
                else:
                    attack.append(False)
                    self.attack = False
            enemyIdx = enemyIdx + 1
        # If intruders found, target the closest one
        if len(intruders) > 0:
            self.attack = False
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
