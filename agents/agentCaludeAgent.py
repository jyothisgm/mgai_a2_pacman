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
        self.home = gameState.getAgentPosition(self.index)
        self.midline = [pos for pos in gameState.getWalls().asList(False) if pos[0] == gameState.data.layout.width // 2]
        self.safe_zones = self.getSafeZones(gameState)
        self.offensiveBuffer = deque(maxlen=5)
        self.ignoredFood = set()
        self.foodChoices = []
        
        # Identify teammate
        self.teammates = self.getTeam(gameState)
        self.teammate_index = [i for i in self.teammates if i != self.index][0]
        
    def getTeammatePosition(self, gameState):
        """Get the position of the teammate agent if visible."""
        teammate_state = gameState.getAgentState(self.teammate_index)
        if teammate_state.getPosition():
            return teammate_state.getPosition()
        return None

    def isTeammateOffensive(self, gameState):
        """Check if teammate is in offensive mode (Pacman)."""
        teammate_state = gameState.getAgentState(self.teammate_index)
        return teammate_state.isPacman

    def chooseAction(self, gameState):
        self.updateFoodCollected(gameState)
        
        # Check if we should switch roles
        role = self.shouldSwitchRole(gameState)
        if role == "defensive":
            # Use defensive-like strategy but don't create a new agent
            actions = gameState.getLegalActions(self.index)
            actions.remove('Stop')
            
            # Define defensive features and weights
            def defensiveEvaluate(gameState, action):
                features = Counter()
                successor = self.getSuccessor(gameState, action)
                myState = successor.getAgentState(self.index)
                myPos = myState.getPosition()
                
                # Defensive features similar to DefensiveAgent
                features['onDefense'] = 1
                if myState.isPacman: features['onDefense'] = 0
                
                enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
                invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
                features['numInvaders'] = len(invaders)
                
                if len(invaders) > 0:
                    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                    features['invaderDistance'] = min(dists)
                
                weights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10}
                return features * weights
            
            # Choose best action using defensive evaluation
            best_val = float("-inf")
            best_action = None
            for action in actions:
                val = defensiveEvaluate(gameState, action)
                if val > best_val:
                    best_val = val
                    best_action = action
            
            return best_action
        
        # Continue with offensive strategy
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        
        action_dict = {}
        for a in actions:
            action_dict[a] = self.evaluate(gameState, a)
        
        best_action = max(action_dict, key=action_dict.get)
        
        # Update the buffer to detect loops
        myPos = gameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()   
        foodList = list(set(foodList) - self.ignoredFood)
        
        if len(foodList) > 0:
            foodDistances = [self.getMazeDistance(myPos, food) for food in foodList]
            foodPos = foodList[foodDistances.index(min(foodDistances))]
            
            # Add to buffer to detect loops
            if (myPos, foodPos) in self.offensiveBuffer:
                self.ignoredFood.add(foodPos)
            
            self.offensiveBuffer.append((myPos, foodPos))
        
        return best_action
    
    
            
    def shouldSwitchRole(self, gameState):
        """Determine if the agent should switch between offense and defense."""
        # By default, offensive agents stay offensive and defensive agents stay defensive
        if isinstance(self, OffensiveAgent):
            # Switch to defense if:
            # 1. We're carrying a lot of food and close to enemies
            myState = gameState.getAgentState(self.index)
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            
            if self.food_collected > 4 and any(self.getMazeDistance(myState.getPosition(), g.getPosition()) < 5 for g in ghosts):
                return "defensive"
            
            # 2. Very little food remains to collect
            remaining_food = len(self.getFood(gameState).asList())
            if remaining_food < 3:
                return "defensive"
            
            return "offensive"
            
        elif isinstance(self, DefensiveAgent):
            # Switch to offense if:
            # 1. No invaders detected and teammate is not offensive
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            
            if len(invaders) == 0 and not self.isTeammateOffensive(gameState):
                # Check if we're far behind in score and need to be aggressive
                score = self.getScore(gameState)
                if score < -5:  # We're behind by 5 or more
                    return "offensive"
            
            return "defensive"
        
    def updateFoodCollected(self, gameState):
        """ Updates the food count based on what has been collected. """
        # Check if Pac-Man is back at spawn (indicating death)
        if not gameState.getAgentState(self.index).isPacman:
            self.food_collected = 0  # Reset food count upon death

        previousGameState = self.getPreviousObservation()
        if previousGameState:  # Only compare if there is a previous state
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
    def planAhead(self, gameState, depth=2):
        """Look ahead a few steps to make better decisions."""
        if depth == 0:
            return None, 0
        
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        if not actions:
            return None, 0

        
        best_score = float('-inf')
        best_action = None
        
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            
            # Evaluate immediate action
            immediate_score = self.evaluate(gameState, action)
            
            # Look ahead recursively
            _, future_score = self.planAhead(successor, depth-1)
            
            # Combine scores with discount factor
            total_score = immediate_score + 0.7 * future_score
            
            if total_score > best_score:
                best_score = total_score
                best_action = action
        
        return best_action, best_score

    # To use this in your agent, modify chooseAction:
    def chooseAction(self, gameState):
        # Use planning for complex situations
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        
        # If there are ghosts nearby or we're carrying food, use look-ahead planning
        myPos = gameState.getAgentState(self.index).getPosition()
        if self.food_collected > 0 or any(self.getMazeDistance(myPos, g.getPosition()) < 5 for g in ghosts):
            best_action, _ = self.planAhead(gameState)
            if best_action:
                return best_action
        
        # Otherwise use regular action selection
        return super().chooseAction(gameState)
    
    def getFeatures(self, gameState, action):
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()   
        features['successorScore'] = -len(foodList)#self.getScore(successor)
        

        # Compute distance to the nearest food
        foodList = list(set(foodList) - self.ignoredFood)
        if len(foodList) == 0:
            self.ignoredFood = set()
            foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        if len(foodList) > 0:
            # Find the nearest food
            foodDistances = [self.getMazeDistance(myPos, food) for food in foodList]
            minDistance = min(foodDistances)
            closestFoodPos = foodList[foodDistances.index(minDistance)]
        
            # Check if we're stuck in a loop
            if (myPos, closestFoodPos) in self.offensiveBuffer:
                features['distanceToFood'] = 999
            else:
                # Count food in the vicinity of the closest food (food density)
                foodDensity = sum(1 for food in foodList if self.getMazeDistance(closestFoodPos, food) < 5)
                
                # Reward paths toward dense food clusters
                features['distanceToFood'] = minDistance / (1 + 0.2 * foodDensity)
    
        # Ghost avoidance logic (same as before)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        features['numGhosts'] = len(ghosts)
        features['ghostDistance'] = 0
        features['safeZone'] = 0
        
        # Return to safe zone when carrying food
        if self.food_collected > 0:
            safe_zone = self.findNearestSafeZone(gameState)
            features['safeZone'] = self.getMazeDistance(myPos, safe_zone)
            # Increase urgency to return as we collect more food
            features['returnUrgency'] = self.food_collected
        
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            if min(dists) < 10:
                features['ghostDistance'] = 2 * min(dists) + 2
                if min(dists) < 4:
                    features['ghostDistance'] = 3 * min(dists) + 3
                    features['distanceToFood'] = 999
        
        return features

    def getWeights(self, gameState, action):
        # Adjust weights to prioritize safety when carrying food
        base_weights = {'successorScore': 100, 'distanceToFood': -1, 'ghostDistance': -1, 'safeZone': -1}
        
        # If carrying a lot of food, prioritize returning to safety
        if self.food_collected > 3:
            base_weights['safeZone'] = -5
            base_weights['returnUrgency'] = -2
        
        return base_weights

class DefensiveAgent(BaseAgent):
    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        # Create a map of patrol points near the boundary
        self.patrol_points = self.getPatrolPoints(gameState)
        self.current_patrol_target = None
    
    def getPatrolPoints(self, gameState):
        """Get strategic defensive positions along the boundary."""
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        
        # Determine boundary x-coordinate based on team
        boundary_x = width // 2 - (1 if self.red else 0)
        
        # Find patrol points 1-2 steps away from boundary in our territory
        patrol_x = boundary_x - (1 if self.red else -1)
        
        patrol_points = []
        for y in range(1, height - 1):
            if not gameState.hasWall(patrol_x, y):
                patrol_points.append((patrol_x, y))
        
        # Add a few strategic points further back
        for y in range(1, height - 1, 4):
            back_x = patrol_x - (2 if self.red else -2)
            if 0 <= back_x < width and not gameState.hasWall(back_x, y):
                patrol_points.append((back_x, y))
        
        return patrol_points
    
    def chooseAction(self, gameState):
        # Use parent's action selection if invaders are present
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        
        if len(invaders) > 0:
            return super().chooseAction(gameState)
        
        # If no invaders, patrol the boundary
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        
        # Choose a patrol target if none exists or we're close to current target
        myPos = gameState.getAgentState(self.index).getPosition()
        if (not self.current_patrol_target or 
            self.getMazeDistance(myPos, self.current_patrol_target) < 2):
            self.current_patrol_target = self.choosePatrolTarget(gameState)
        
        # Choose action that gets closest to patrol target
        best_dist = float('inf')
        best_action = None
        
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos = successor.getAgentState(self.index).getPosition()
            dist = self.getMazeDistance(pos, self.current_patrol_target)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
        if not best_action:
            best_action = 'Stop'
        
        return best_action
    
    def choosePatrolTarget(self, gameState):
        """Choose a strategic patrol point."""
        myPos = gameState.getAgentState(self.index).getPosition()
        
        # Get the food we're defending
        defending_food = self.getFoodYouAreDefending(gameState).asList()
        
        # If we have little food left, focus on defending it
        if len(defending_food) <= 4:
            # Find the patrol point closest to our remaining food
            food_distances = {}
            for patrol_point in self.patrol_points:
                avg_dist = sum(self.getMazeDistance(patrol_point, food) for food in defending_food) / len(defending_food)
                food_distances[patrol_point] = avg_dist
            
            return min(food_distances, key=food_distances.get)
        
        # Otherwise, choose a patrol point that's not too close to teammate
        teammate_pos = self.getTeammatePosition(gameState)
        
        if teammate_pos and not self.isTeammateOffensive(gameState):
            # Avoid patrolling near teammate if they're also defending
            valid_points = [p for p in self.patrol_points if self.getMazeDistance(p, teammate_pos) > 5]
            if valid_points:
                return min(valid_points, key=lambda p: self.getMazeDistance(myPos, p))
        
        # Default: choose the closest patrol point
        return min(self.patrol_points, key=lambda p: self.getMazeDistance(myPos, p))
    
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
            
            # Check if we can get to invader before they escape
            closest_invader = invaders[dists.index(min(dists))]
            closest_invader_pos = closest_invader.getPosition()
            
            # Calculate distance from invader to safe zone
            invader_to_safety = min([self.getMazeDistance(closest_invader_pos, safe) 
                                    for safe in self.safe_zones])
            
            # If we can intercept, prioritize that path
            if min(dists) < invader_to_safety:
                features['canIntercept'] = 1
        
        # Add patrol feature when no invaders present
        if len(invaders) == 0 and self.current_patrol_target:
            patrol_dist = self.getMazeDistance(myPos, self.current_patrol_target)
            features['patrolDistance'] = patrol_dist
        
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        if gameState.getAgentState(self.index).scaredTimer > 0:
            # If scared, avoid invaders instead of chasing them
            return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 10, 
                    'stop': -100, 'reverse': -2, 'patrolDistance': -1}
        else:
            return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 
                    'canIntercept': 500, 'stop': -100, 'reverse': -2, 'patrolDistance': -1}
