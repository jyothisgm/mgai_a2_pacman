from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from captureAgents import CaptureAgent
import random, time, util as util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='HeuristicAgent', second='HeuristicAgent'):
    """
    This function returns a team of two agents.
    """
    return [HeuristicAgent(firstIndex), HeuristicAgent(secondIndex)]

##########
# Agents #
##########

class HeuristicAgent(CaptureAgent):
    """
    A heuristic-based agent that makes smarter decisions 
    based on food, ghosts, capsules, and enemy movement.
    """

    def registerInitialState(self, gameState):
        """
        Initializes the agent at the start of the game.
        """
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)  # Store start position
        print(self.index, self.red, self.start)
    def chooseAction1(self, gameState):
        """
        Chooses an action based on heuristic evaluation.
        """
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP  # No valid actions available
        
        # Evaluate all actions using heuristics
        action_values = [(action, self.evaluate(gameState, action)) for action in actions]
        
        # Choose the best action based on heuristic score
        best_action = max(action_values, key=lambda x: x[1])[0]
        return best_action

    def evaluate(self, gameState, action):
        """
        Computes a heuristic score for a given action.
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights  # Dot product of features and weights

    def chooseAction(self, gameState):
        """
        Uses Monte Carlo Tree Search (MCTS) to simulate multiple moves ahead.
        """
        actions = gameState.getLegalActions(self.index)
        if not actions:
            return Directions.STOP  # No valid actions available

        # Simulate each action using rollouts
        action_scores = {}
        num_simulations = 10  # Number of simulations per action

        for action in actions:
            total_score = 0
            for _ in range(num_simulations):
                total_score += self.rollout(gameState.generateSuccessor(self.index, action), depth=3)
            action_scores[action] = total_score / num_simulations
            print(action_scores, action)
        # Choose the best action based on MCTS rollouts
        best_action = max(action_scores, key=action_scores.get)
        return best_action

    def rollout(self, gameState, depth):
        """
        Simulates random rollouts to estimate long-term reward.
        """
        if depth == 0 or gameState.isOver():
            return self.evaluate(gameState, Directions.STOP)

        legal_actions = gameState.getLegalActions(self.index)
        random_action = random.choice(legal_actions)
        return self.rollout(gameState.generateSuccessor(self.index, random_action), depth - 1)

    def getFeatures(self, gameState, action):
        """
        Extracts relevant features for smarter decision-making.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        print("Successor is ", successor)
        myPos = successor.getAgentState(self.index).getPosition()
        print("My current position is ", myPos) 
        # 1️⃣ Score Change After Action
        features['successorScore'] = self.getScore(successor)

        # 2️⃣ Distance to the Nearest Food
        foodList = self.getFood(successor).asList()
        if foodList:
            min_food_distance = min(self.getMazeDistance(myPos, food) for food in foodList)
            features['distanceToFood'] = min_food_distance

            # Find high-density food areas (clusters)
            cluster_foods = sum(1 for food in foodList if self.getMazeDistance(myPos, food) <= 3)
            features['foodCluster'] = cluster_foods  # More food nearby = higher value

        # 3️⃣ Distance to the Nearest Capsule (Power Pellet)
        capsules = self.getCapsules(successor)
        if capsules:
            min_capsule_distance = min(self.getMazeDistance(myPos, cap) for cap in capsules)
            features['distanceToCapsule'] = min_capsule_distance

        # 4️⃣ Ghost Detection
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        
        if ghosts:
            ghost_distances = [self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]
            min_ghost_distance = min(ghost_distances)

            # Count number of nearby ghosts (within danger range)
            close_ghosts = sum(1 for d in ghost_distances if d < 4)

            # Update features for ghost avoidance
            features['distanceToGhost'] = min_ghost_distance
            features['numCloseGhosts'] = close_ghosts  # More ghosts = higher risk

            # **Danger Zone Detection**
            if min_ghost_distance < 3:
                features['ghostThreat'] = 1
                if close_ghosts > 1:
                    features['ghostDangerZone'] = 1  # Multiple ghosts nearby

        # 5️⃣ Safe Escape Route Calculation
        safe_routes = sum(1 for a in successor.getLegalActions(self.index) if a != Directions.STOP)
        features['safeRoutes'] = safe_routes  # More routes = safer movement

        # 6️⃣ Encourage Returning Home if Carrying Food
        agent_state = successor.getAgentState(self.index)
        if agent_state.numCarrying > 3:
            border_x = (gameState.data.layout.width // 2) - 1 if self.red else (gameState.data.layout.width // 2)
            border_positions = [(border_x, y) for y in range(gameState.data.layout.height)
                                if not gameState.hasWall(border_x, y)]
            min_border_distance = min(self.getMazeDistance(myPos, border) for border in border_positions)
            features['distanceToBorder'] = min_border_distance

        return features

    def getWeights(self, gameState, action):
        """
        Assigns importance to each feature.
        """
        return {
            'successorScore': 100,  # Prioritize increasing score
            'distanceToFood': -5,  # Prefer shorter paths to food
            'foodCluster': 7,  # Prefer high-density food areas
            'distanceToCapsule': -2,  # Prefer shorter paths to capsules
            'distanceToGhost': 15,  # Avoid ghosts
            'numCloseGhosts': -30,  # Heavy penalty for multiple ghosts nearby
            'ghostThreat': -50,  # Severe penalty for being close to ghosts
            'ghostDangerZone': -100,  # Extreme penalty for being trapped by ghosts
            'safeRoutes': 20,  # Reward open paths
            'distanceToBorder': -10  # Encourage returning home when carrying food
        }

    def getSuccessor(self, gameState, action):
        """
        Finds the next state after taking an action.
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        return successor
