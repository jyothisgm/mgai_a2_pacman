from captureAgents import CaptureAgent
from util import nearestPoint, Counter
from game import Directions
from collections import deque



def createTeam(firstIndex, secondIndex, isRed,
                first = 'OffensiveAgent', second = 'DefensiveAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# ============================ #
#  Base Agent
# ============================ #
class BaseAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.food_collected = 0

        self.home = gameState.getAgentPosition(self.index) # Store initial position
        self.midline = [pos for pos in gameState.getWalls().asList(False) if pos[0] == gameState.data.layout.width // 2]
        self.safe_zones = self.getSafeZones(gameState)
        self.offensiveBuffer = deque(maxlen=4)
        self.ignoredFood = set()
        self.foodChoices = []

    def chooseAction(self, gameState):
        self.updateFoodCollected(gameState)
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        action_dict = {}
        for a in actions:
            action_dict[a] = self.evaluate(gameState, a)
        
        best_action = max(action_dict, key=action_dict.get)
        return best_action
    
    def updateFoodCollected(self, gameState):
        """ Updates the food count based on what has been collected. """
        # Check if Pac-Man is back at spawn (indicating death)
        if not gameState.getAgentState(self.index).isPacman:
            self.food_collected = 0  # Reset food count upon death

        previousGameState = self.getPreviousObservation()
        if previousGameState:
            previousFoodList = self.getFood(previousGameState).asList()
            currentFoodList = self.getFood(gameState).asList()
            collected = len(previousFoodList) - len(currentFoodList)
            if collected > 0:
                self.food_collected += collected
                self.ignoredFood = set()

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
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}
    
    def getSafeZones(self, gameState):
        """ Returns a list of safe positions (friendly territory border). """
        layoutWidth = gameState.data.layout.width
        layoutHeight = gameState.data.layout.height
        mid_x = layoutWidth // 2 - (1 if self.red else 0)  # Midpoint of the map (adjust for red team)
        
        safe_zones = [
            (mid_x, y) for y in range(layoutHeight)
            if not gameState.hasWall(mid_x, y)
        ]
        return safe_zones

    def findNearestSafeZone(self, gameState):
        """ Finds the closest safe zone for retreating. """
        current_position = gameState.getAgentPosition(self.index)
        if not current_position:
            return None

        # Compute the closest safe zone using Manhattan distance
        safe_zone = min(self.safe_zones, key=lambda pos: self.getMazeDistance(current_position, pos))
        return safe_zone
    

class OffensiveAgent(BaseAgent):
    def chooseAction(self, gameState):
        self.updateFoodCollected(gameState)
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        scores = []
        for a in actions:
            scores.append(self.evaluate(gameState, a))
        
        print(actions)
        print(scores)
        best_action = actions[scores.index(max(scores))]
        print(best_action)
        myPos = gameState.getAgentState(self.index).getPosition()
        
        foodList = self.getFood(gameState).asList()
        foodList = list(set(foodList) - self.ignoredFood)
        foodDistances = [self.getMazeDistance(myPos, food) for food in foodList]
        foodPos = foodList[foodDistances.index(min(foodDistances))]
        if (myPos, foodPos) in self.offensiveBuffer:
            self.ignoredFood.add(foodPos)
        
        print(self.ignoredFood)
        print(gameState)

        self.offensiveBuffer.append((myPos, foodPos))
        
        return best_action
    
    def getFeatures(self, gameState, action):
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()   
        features['successorScore'] = -len(foodList)#self.getScore(successor)
        

        # Compute distance to the nearest food
        foodList = list(set(foodList) - self.ignoredFood)
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
        else:
            myPos = successor.getAgentState(self.index).getPosition()
            self.ignoredFood = set()
            foodList = self.getFood(successor).asList()
        
        foodDistances = [self.getMazeDistance(myPos, food) for food in foodList]
        foodPos = foodList[foodDistances.index(min(foodDistances))]
        print(self.offensiveBuffer)
        if (myPos, foodPos) in self.offensiveBuffer:
            features['distanceToFood'] = 9999
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        capsuleList = self.getCapsules(successor)
        capsuleDistances = [self.getMazeDistance(myPos, capsule) for capsule in capsuleList]
        if capsuleDistances:
            distanceToCapsule = min(capsuleDistances)
        else:
            distanceToCapsule = 0
        features['distanceToCapsule'] = distanceToCapsule
        features['ghostDistance'] = 0
        features['numGhosts'] = len(ghosts)
        features['safeZone'] = 0
        features['distanceToFood'] = self.getMazeDistance(myPos, foodPos)
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            features['ghostDistance'] = min(dists)
            if (not features['distanceToCapsule'] or features['distanceToCapsule'] > min(dists)) and min(dists) < 7:
                features['ghostDistance'] = 10 * min(dists) + 2
                if min(dists) < 4:
                    features['ghostDistance'] = 20 * min(dists) + 3
                    features['distanceToFood'] = 999
                        
                if self.food_collected:
                    features['safeZone'] = self.getMazeDistance(myPos, self.findNearestSafeZone(gameState))
            else:
                features['distanceToFood'] *= 10

        print(features)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1, "ghostDistance": - 1, "safeZone": -1.5, "distanceToCapsule": -1}

class DefensiveAgent(BaseAgent):
    def getFeatures(self, gameState, action):
        features = Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0
        
        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
