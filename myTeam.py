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
import random, time, util as util
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
        # Counter (not eat food)
        self.counter = 0
        # Attack mode
        self.attack = False
        # Record of last food state
        self.lastFood = []
        # Record of current food state
        self.presentFoodList = []
        # Whether the agent should return to its base
        self.shouldReturn = False
        # Whether the agent has eaten a capsule
        self.capsulePower = False
        # Target mode
        self.targetMode = None
        # How many food items the agent has eaten
        self.eatenFood = 0
        # Initial target position
        self.initialTarget = []
        # Whether the agent has stopped
        self.hasStopped = 0
        # Number of capsules left
        self.capsuleLeft = 0
        # Number of capsules left in the previous round
        self.prevCapsuleLeft = 0

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

    
    def evaluateAttackParameters(self, gameState, action):
        # Store various evaluation features for this action
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        position = successor.getAgentState(self.index).getPosition()
        # List of remaining food
        foodList = self.getFood(successor).asList()
        # New score after performing the action
        features['successorScore'] = self.getScore(successor)
        
        # Check if the agent is still in attack mode
        if successor.getAgentState(self.index).isPacman:
            features['offence'] = 1
        else:
            features['offence'] = 0

        # Calculate the distance to the nearest food, take the minimum distance
        if foodList:
            features['foodDistance'] = min([self.getMazeDistance(position, food) for food in foodList])

        opponentsList = []
        # Distance to all enemy ghosts
        disToGhost = []
        # List of opponent agent indexes
        opponentsList = self.getOpponents(successor)

        for i in range(len(opponentsList)):
            enemyPos = opponentsList[i]
            enemy = successor.getAgentState(enemyPos)
            if not enemy.isPacman and enemy.getPosition() is not None:
                ghostPos = enemy.getPosition()
                disToGhost.append(self.getMazeDistance(position, ghostPos))

        if len(disToGhost) > 0:
            # Distance to the nearest ghost
            minDisToGhost = min(disToGhost)
            if minDisToGhost < 5:
                features['distanceToGhost'] = minDisToGhost + features['successorScore']
            else:
                features['distanceToGhost'] = 0
        return features
    
    def getCostOfAttackParameter(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        # not eat for a long time
        if self.attack:
            # have to return
            if self.shouldReturn is True:
                return {'offence' :3010,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
            else:
                return {'offence' :0,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
        else:
            successor = self.getSuccessor(gameState, action) 
            weightGhost = 210
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            if len(invaders) > 0:
                if invaders[-1].scaredTimer > 0:
                    weightGhost = 0
                    
            return {'offence' :0,
                    'successorScore': 202,
                    'foodDistance': -8,
                    'distancesToGhost' :weightGhost}

    def getOpponentPositions(self, gameState):
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    # **** action ****
    def bestPossibleAction(self ,mcsc):
        ab = mcsc.getLegalActions(self.index)
        # avoid stagnate
        ab.remove(Directions.STOP)

        if len(ab) == 1:
            # only one action available
            return ab[0]
        else:
            reverseDir = Directions.REVERSE[mcsc.getAgentState(self.index).configuration.direction]
            # not moving reversely
            if reverseDir in ab:
                ab.remove(reverseDir)
            return random.choice(ab)

    # ********* modify
    def monteCarloSimulation(self ,gameState ,depth):
        ss = gameState.deepCopy()
        while depth > 0:
            ss = ss.generateSuccessor(self.index ,self.bestPossibleAction(ss))
            depth -= 1
        return self.evaluate(ss ,Directions.STOP)

    def getBestAction(self,legalActions,gameState,possibleActions,distanceToTarget):
        shortestDistance = 9999999999
        for i in range (0,len(legalActions)):    
            action = legalActions[i]
            nextState = gameState.generateSuccessor(self.index, action)
            nextPosition = nextState.getAgentPosition(self.index)
            distance = self.getMazeDistance(nextPosition, self.initialTarget[0])
            distanceToTarget.append(distance)
            if(distance<shortestDistance):
                shortestDistance = distance

        bestActionsList = [a for a, distance in zip(legalActions, distanceToTarget) if distance == shortestDistance]
        bestAction = random.choice(bestActionsList)
        return bestAction
        
    def chooseAction(self, gameState):
        self.presentCoordinates = gameState.getAgentState(self.index).getPosition()
    
        if self.presentCoordinates == self.initPosition:
            self.hasStopped = 1
        if self.presentCoordinates == self.initialTarget[0]:
            self.hasStopped = 0

        # find next possible best move 
        if self.hasStopped == 1:
            legalActions = gameState.getLegalActions(self.index)
            legalActions.remove(Directions.STOP)
            possibleActions = []
            distanceToTarget = []
            
            bestAction=self.getBestAction(legalActions,gameState,possibleActions,distanceToTarget)
            
            return bestAction

        if self.hasStopped==0:
            self.presentFoodList = self.getFood(gameState).asList()
            self.capsuleLeft = len(self.getCapsules(gameState))
            realLastCapsuleLen = self.prevCapsuleLeft
            realLastFoodLen = len(self.lastFood)

            # Set returned = 1 when pacman has secured some food and should to return back home           
            if len(self.presentFoodList) < len(self.lastFood):
                self.shouldReturn = True
            self.lastFood = self.presentFoodList
            self.prevCapsuleLeft = self.capsuleLeft

           
            if not gameState.getAgentState(self.index).isPacman:
                self.shouldReturn = False

            # checks the attack situation           
            remainingFoodList = self.getFood(gameState).asList()
            remainingFoodSize = len(remainingFoodList)
    
        
            if remainingFoodSize == self.currentFoodSize:
                self.counter = self.counter + 1
            else:
                self.currentFoodSize = remainingFoodSize
                self.counter = 0
            if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
                self.counter = 0
            if self.counter > 20:
                self.attack = True
            else:
                self.attack = False
            
            
            actionsBase = gameState.getLegalActions(self.index)
            actionsBase.remove(Directions.STOP)

            # distance to closest enemy        
            distanceToEnemy = 999999
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer == 0]
            if len(invaders) > 0:
                distanceToEnemy = min([self.getMazeDistance(self.presentCoordinates, a.getPosition()) for a in invaders])
            
            '''
            Capsule eating:
            -> If there is capsule available then capsulePower is True.
            -> If enemy Distance is less than 5 then capsulePower is False.
            -> If pacman scored a food then return to home capsulePower is False.
            '''
            if self.capsuleLeft < realLastCapsuleLen:
                self.capsulePower = True
                self.eatenFood = 0
            if distanceToEnemy <= 5:
                self.capsulePower = False
            if (len(self.presentFoodList) < len (self.lastFood)):
                self.capsulePower = False

        
            if self.capsulePower:
                if not gameState.getAgentState(self.index).isPacman:
                    self.eatenFood = 0

                modeMinDistance = 999999

                if len(self.presentFoodList) < realLastFoodLen:
                    self.eatenFood += 1

                if len(self.presentFoodList )==0 or self.eatenFood >= 5:
                    self.targetMode = self.initPosition
        
                else:
                    for food in self.presentFoodList:
                        distance = self.getMazeDistance(self.presentCoordinates ,food)
                        if distance < modeMinDistance:
                            modeMinDistance = distance
                            self.targetMode = food

                legalActions = gameState.getLegalActions(self.index)
                legalActions.remove(Directions.STOP)
                possibleActions = []
                distanceToTarget = []
                
                k=0
                while k!=len(legalActions):
                    a = legalActions[k]
                    newpos = (gameState.generateSuccessor(self.index, a)).getAgentPosition(self.index)
                    possibleActions.append(a)
                    distanceToTarget.append(self.getMazeDistance(newpos, self.targetMode))
                    k+=1
                
                minDis = min(distanceToTarget)
                bestActions = [a for a, dis in zip(possibleActions, distanceToTarget) if dis== minDis]
                bestAction = random.choice(bestActions)
                return bestAction
            else:
               
                self.eatenFood = 0
                distanceToTarget = []
                for a in actionsBase:
                    nextState = gameState.generateSuccessor(self.index, a)
                    value = 0
                    for i in range(1, 24):
                        value += self.monteCarloSimulation(nextState ,20)
                    distanceToTarget.append(value)

                best = max(distanceToTarget)
                bestActions = [a for a, v in zip(actionsBase, distanceToTarget) if v == best]
                bestAction = random.choice(bestActions)
            return bestAction


class DefensiveReflexAgent(DummyAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Current defensive target (could be an invader's position or food).
        self.target = None
        # Previous round's food list (used to detect stolen food).
        self.previousFood = []
        self.counter = 0

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.setPatrolPoint(gameState)

    def setPatrolPoint(self, gameState):
        '''
        Look for the center of the maze to patrol.
        '''
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x += 1

        self.patrolPoints = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(x, i):
                self.patrolPoints.append((x, i))

        for i in range(len(self.patrolPoints)):
            # If there are too many patrol points, remove the first and last ones to keep the middle ones
            if len(self.patrolPoints) > 2:
                self.patrolPoints.remove(self.patrolPoints[0])
                self.patrolPoints.remove(self.patrolPoints[-1])
            else:
                break
    

    def getNextDefensiveMove(self, gameState):
        agentActions = []
        actions = gameState.getLegalActions(self.index)
        
        rev_dir = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        actions.remove(Directions.STOP)
        # Remove "STOP" action and "reverse" direction
        for i in range(0, len(actions)-1):
            if rev_dir == actions[i]:
                actions.remove(rev_dir)

        # Filter out actions that would turn the agent into a Pacman (eating beans)
        for i in range(len(actions)):
            a = actions[i]
            new_state = gameState.generateSuccessor(self.index, a)
            if not new_state.getAgentState(self.index).isPacman:
                agentActions.append(a)

        # If the agent has been defending for 4 consecutive rounds, add reverse direction to change its movement
        if len(agentActions) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
        if self.counter > 4 or self.counter == 0:
            agentActions.append(rev_dir)

        return agentActions

    def chooseAction(self, gameState):
        
        position = gameState.getAgentPosition(self.index)
        # If the agent reaches the target, clear the target and look for a new defensive task
        if position == self.target:
            self.target = None
        invaders = []
        nearestInvader = []
        minDistance = float("inf")

        # Look for enemy positions in our home
        opponentsPositions = self.getOpponents(gameState)
        i = 0
        while i != len(opponentsPositions):
            opponentPos = opponentsPositions[i]
            opponent = gameState.getAgentState(opponentPos)
            if opponent.isPacman and opponent.getPosition() is not None:
                opponentPos = opponent.getPosition()
                invaders.append(opponentPos)
            i = i + 1

        # Find the closest enemy Pacman and set it as the target
        if len(invaders) > 0:
            for oppPosition in invaders:
                dist = self.getMazeDistance(oppPosition, position)
                if dist < minDistance:
                    minDistance = dist
                    nearestInvader.append(oppPosition)
            self.target = nearestInvader[-1]

        # If no invaders are found, check if any food has been stolen: if the food count has decreased, it means food was stolen, and the agent should go to the stolen food location.
        else:
            if len(self.previousFood) > 0:
                if len(self.getFoodYouAreDefending(gameState).asList()) < len(self.previousFood):
                    stolenFood = set(self.previousFood) - set(self.getFoodYouAreDefending(gameState).asList())
                    self.target = stolenFood.pop()

        self.previousFood = self.getFoodYouAreDefending(gameState).asList()
        
        if self.target is None:
            # If the number of food ≤ 4, prioritize defending remaining food and capsules
            if len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
                highPriorityFood = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
                self.target = random.choice(highPriorityFood)
            else:
                self.target = random.choice(self.patrolPoints)

        # Get possible defensive moves
        candAct = self.getNextDefensiveMove(gameState)
        awsomeMoves = []
        fvalues = []

        i = 0
        # Find the best move       
        while i < len(candAct):
            a = candAct[i]
            nextState = gameState.generateSuccessor(self.index, a)
            newpos = nextState.getAgentPosition(self.index)
            awsomeMoves.append(a)
            fvalues.append(self.getMazeDistance(newpos, self.target))
            i = i + 1

        best = min(fvalues)
        bestActions = [a for a, v in zip(awsomeMoves, fvalues) if v == best]
        bestAction = random.choice(bestActions)
        return bestAction


