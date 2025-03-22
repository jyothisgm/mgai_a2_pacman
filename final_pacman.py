from captureAgents import CaptureAgent
import random, math, heapq
from game import Directions
import util

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    return [OffensiveAgent(firstIndex), DefensiveAgent(secondIndex)]

class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getAgentPosition(self.index)
        self.positionHistory = []
        self.historyLimit = 3

    def chooseAction(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions = [a for a in legal_actions if a != Directions.STOP]

        my_pos = gameState.getAgentPosition(self.index)
        food_list = self.getFood(gameState).asList()
        capsule_list = self.getCapsules(gameState)
        ghost_positions = self.getVisibleGhosts(gameState)
        carrying_food = gameState.getAgentState(self.index).numCarrying

        is_powered_up = any(
            gameState.getAgentState(enemy).scaredTimer > 2
            for enemy in self.getOpponents(gameState)
            if gameState.getAgentState(enemy).getPosition() in ghost_positions
        )

        best_score = float('-inf')
        best_action = Directions.STOP

        for action in legal_actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            score = self.evaluateState(
                legal_actions, new_pos, food_list, capsule_list, ghost_positions,
                carrying_food, is_powered_up, gameState, action
            )
            if score > best_score:
                best_score = score
                best_action = action

        next_state = gameState.generateSuccessor(self.index, best_action)
        self.positionHistory.append(next_state.getAgentPosition(self.index))
        if len(self.positionHistory) > self.historyLimit:
            self.positionHistory.pop(0)

        return best_action

    def evaluateState(self, legal_actions, my_pos, food_list, capsule_list,
                      ghost_positions, carrying_food, is_powered_up, gameState, action):
        features = self.getFeatures(
            legal_actions, my_pos, food_list, capsule_list, ghost_positions,
            carrying_food, is_powered_up, gameState, action
        )
        weights = self.getWeights(my_pos, carrying_food, ghost_positions, is_powered_up)
        return features * weights

    def getFeatures(self, legal_actions, my_pos, food_list, capsule_list,
                    ghost_positions, carrying_food, is_powered_up, gameState, action):
        situation = self.classifySituation(my_pos, carrying_food, ghost_positions, is_powered_up, gameState)

        if situation == "danger":
            return self.getFeaturesDanger(my_pos, ghost_positions, legal_actions, capsule_list, gameState)
        elif situation == "return":
            return self.getFeaturesReturn(my_pos, ghost_positions, gameState)
        elif situation == "power":
            return self.getFeaturesPower(my_pos, ghost_positions, food_list, gameState)
        else:
            return self.getFeaturesForage(my_pos, food_list, ghost_positions, capsule_list, gameState, action)

    def getSafeDirections(self, my_pos, ghost_positions, gameState):
        """
        Returns directions from current position that are away from nearby ghosts.
        """
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        safe_dirs = []

        for action in directions:
            next_pos = self.getNextPosition(my_pos, action)
            if not gameState.hasWall(int(next_pos[0]), int(next_pos[1])):
                if all(self.getMazeDistance(next_pos, ghost) > 2 for ghost in ghost_positions):
                    safe_dirs.append(action)

        return safe_dirs


    def classifySituation(self, my_pos, carrying_food, ghost_positions, is_powered_up, gameState):
        if self.isAcrossBorder(my_pos):
            if ghost_positions:
                min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
                if min_ghost_dist <= 3 and carrying_food >= 2 and not is_powered_up:
                    return "danger"
                elif min_ghost_dist <= 5 and not is_powered_up:
                    return "danger"
        if carrying_food >= 3:
            return "return"
        if is_powered_up:
            return "power"
        return "forage"

    def getFeaturesDanger(self, my_pos, ghost_positions, legal_actions, capsule_list, gameState):
        features = util.Counter()
        min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions]) if ghost_positions else float('inf')
 
        features["closeGhost"] = 1.0 / (min_ghost_dist + 1)
        if min_ghost_dist <= 1:
            features["deathZone"] = 1.0

        if len(legal_actions) <= 2:
            features["escapePotential"] = self.evaluateEscapePotential(my_pos, ghost_positions, legal_actions)

        if capsule_list:
            min_capsule_dist = min([self.getMazeDistance(my_pos, c) for c in capsule_list])
            if min_capsule_dist <= 1:
                features["capsulePriority"] = 1.0
            elif min_capsule_dist <= 5:
                features["capsuleDistance"] = 1.0 / (min_capsule_dist + 1)

        border_positions = self.getFriendlyBorders()
        if border_positions:
            min_border_distance = min([self.getMazeDistance(my_pos, b) for b in border_positions])
            features["invBorderDistance"] = 1.0 / (min_border_distance + 1)
            for border in border_positions:
                for ghost in ghost_positions:
                    if self.getMazeDistance(ghost, border) <= 2:
                        features["borderBlocked"] = 1.0
                        break

        if my_pos in self.positionHistory:
            features["repeatPenalty"] = 1.0
       
        return features

    def getFeaturesReturn(self, my_pos, ghost_positions, gameState):
        features = util.Counter()
        border_positions = self.getFriendlyBorders()
        if border_positions:
            min_border_distance = min([self.getMazeDistance(my_pos, b) for b in border_positions])
            features["invBorderDistance"] = 1.0 / (min_border_distance + 1)

        if ghost_positions:
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            features["ghostThreatWhileReturning"] = 1.0 / (min_ghost_dist + 1)

        return features

    def getFeaturesPower(self, my_pos, ghost_positions, food_list, gameState):
        features = util.Counter()
        if ghost_positions:
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            features["chaseGhost"] = 1.0 / (min_ghost_dist + 1)
        if food_list:
            dist = self.aStarSearch(my_pos, food_list, gameState)
            features["invFoodDistance"] = 1.0 / (dist + 1)
        return features

    def getFeaturesForage(self, my_pos, food_list, ghost_positions, capsule_list, gameState, action):
        features = util.Counter()
        if food_list:
            dist = self.aStarSearch(my_pos, food_list, gameState)
            features["invFoodDistance"] = 1.0 / (dist + 1)

        if ghost_positions:
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            #if min_ghost_dist <= 5:
                # safe_dirs = self.getSafeDirections(my_pos, ghost_positions, gameState)
                # features["safeDirectionBonus"] = len(safe_dirs) / 4.0
            features["closeGhost"] = 1.0 / (min_ghost_dist + 1)
            

        if not self.isAcrossBorder(my_pos):
            for entry in self.getFriendlyBorders():
                for ghost in ghost_positions:
                    if self.getMazeDistance(entry, ghost) <= 2:
                        features["entryBlocked"] = 1.0

        if my_pos in self.positionHistory:
            features["repeatPenalty"] = 1.0
            # Encourage open directions even when foraging
        
        return features

    def getWeights(self, my_pos, carrying_food, ghost_positions, is_powered_up):
        min_ghost_distance = min([self.getMazeDistance(my_pos, g) for g in ghost_positions]) if ghost_positions else float('inf')
        weights = util.Counter()

        if is_powered_up:
            return {"chaseGhost": 100, "invFoodDistance": 20}

        if min_ghost_distance <= 5:
            weights.update({
                "closeGhost": -300,
                "deathZone": -10000,
                "capsulePriority": 100,
                "capsuleDistance": 60,
                "escapePotential": 10,
                "invBorderDistance": 70,
                "borderBlocked": -150,
                "repeatPenalty": -100,
                "safeDirectionBonus": -5
            })
        elif carrying_food >= 3:
            weights.update({
                "invFoodDistance": 5,
                "closeGhost": -100,
                "invBorderDistance": 400,
                "ghostThreatWhileReturning": -150,
                "repeatPenalty": -100
            })
        else:
            weights.update({
                "invFoodDistance": 50,
                "closeGhost": -80,
                "capsuleDistance": 40,
                "invBorderDistance": 15,
                "borderBlocked": -150,
                "entryBlocked": -200,
                "repeatPenalty": -100,
                "safeDirectionBonus": 2
            })

        return weights

    def aStarSearch(self, start_pos, food_list, gameState):
        walls = gameState.getWalls()
        frontier = [(0, start_pos)]
        visited = set()
        cost_so_far = {start_pos: 0}

        while frontier:
            cost, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)
            if current in food_list:
                return cost
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = current[0] + dx, current[1] + dy
                if not walls[nx][ny]:
                    next_pos = (nx, ny)
                    new_cost = cost + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        heuristic = min([abs(nx - fx) + abs(ny - fy) for fx, fy in food_list]) if food_list else 0
                        heapq.heappush(frontier, (new_cost + heuristic, next_pos))
        return float('inf')

    def evaluateEscapePotential(self, my_pos, ghost_positions, legal_actions):
        """
        Evaluates how well Pacman can escape by maximizing distance from ghosts.
        """
        escape_score = 0
        try:
            for action in legal_actions:
                next_pos = self.getNextPosition(my_pos, action)

                # **Find the minimum distance to any ghost after taking this action**
                
                min_ghost_distance = min([self.getMazeDistance(next_pos, ghost) for ghost in ghost_positions])

                # **Assign higher scores to moves that increase distance from ghosts**
                escape_score += 100 / (min_ghost_distance + 1)  # Higher distance = better escape move
        except:
            pass
        return escape_score / len(legal_actions)  # Average out escape scores
    

    def getNextPosition(self, pos, action):
        x, y = pos
        dx, dy = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST:  (1, 0),
            Directions.WEST:  (-1, 0)
        }.get(action, (0, 0))
        return (x + dx, y + dy)

    def getVisibleGhosts(self, gameState):
        ghosts = []
        for enemy in self.getOpponents(gameState):
            state = gameState.getAgentState(enemy)
            if not state.isPacman and state.getPosition():
                ghosts.append(state.getPosition())
        return ghosts

    def isAcrossBorder(self, pos):
        mid_x = self.getCurrentObservation().data.layout.width // 2
        return pos[0] >= mid_x if self.red else pos[0] < mid_x

    def getFriendlyBorders(self):
        width = self.getCurrentObservation().data.layout.width
        height = self.getCurrentObservation().data.layout.height
        border_x = (width // 2) - 1 if self.red else width // 2
        walls = self.getCurrentObservation().getWalls().asList()
        return [(border_x, y) for y in range(height) if (border_x, y) not in walls]

##########
# DEFENSIVE AGENT #
##########

class DefensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getAgentPosition(self.index)
        self.positionHistory = []
        self.historyLimit = 3

    def chooseAction(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        best_score = float('-inf')
        best_action = Directions.STOP
        my_pos = gameState.getAgentPosition(self.index)
        invaders = self.getEnemyPacman(gameState)
        border_positions = self.getStrategicBorderPositions(gameState, invaders)

        for action in legal_actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            features = self.getFeatures(new_pos, invaders, border_positions, gameState)
            weights = self.getWeights()
            score = features * weights

            if score > best_score:
                best_score = score
                best_action = action

        next_state = gameState.generateSuccessor(self.index, best_action)
        self.positionHistory.append(next_state.getAgentPosition(self.index))
        if len(self.positionHistory) > self.historyLimit:
            self.positionHistory.pop(0)

        return best_action

    def getFeatures(self, my_pos, invaders, border_positions, gameState):
        features = util.Counter()

        # 1️⃣ On Defense
        features["onDefense"] = 1 if not self.isAcrossBorder(my_pos) else 0

        # 2️⃣ Invader Handling
        if invaders:
            features["numInvaders"] = len(invaders)
            min_invader_dist = min([self.getMazeDistance(my_pos, inv) for inv in invaders])
            features["invaderDistance"] = min_invader_dist
        else:
            # 3️⃣ Patrol: Stay near border
            if border_positions:
                min_border_dist = min([self.getMazeDistance(my_pos, b) for b in border_positions])
                features["borderProximity"] = min_border_dist

        # 4️⃣ Capsule defense (prevent enemy from reaching capsules)
        enemy_capsules = set(gameState.getCapsules()) - set(self.getCapsulesYouAreDefending(gameState))
        if enemy_capsules:
            min_capsule_dist = min([self.getMazeDistance(my_pos, c) for c in enemy_capsules])
            features["capsuleGuard"] = min_capsule_dist

        # 5️⃣ Penalize loops
        if my_pos in self.positionHistory:
            features["repeatPenalty"] = 1

        return features

    def getWeights(self):
        return util.Counter({
            "onDefense": 100,
            "numInvaders": -1000,
            "invaderDistance": -20,
            "borderProximity": -2,
            "capsuleGuard": -5,
            "repeatPenalty": -50
        })

    def getEnemyPacman(self, gameState):
        return [
            gameState.getAgentState(i).getPosition()
            for i in self.getOpponents(gameState)
            if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() is not None
        ]

    def isAcrossBorder(self, pos):
        width = self.getCurrentObservation().data.layout.width
        return pos[0] >= width // 2 if self.red else pos[0] < width // 2

    def getStrategicBorderPositions(self, gameState, invaders):
        width = self.getCurrentObservation().data.layout.width
        height = self.getCurrentObservation().data.layout.height
        walls = self.getCurrentObservation().getWalls().asList()
        border_x = (width // 2) - 1 if self.red else width // 2

        border_positions = [(border_x, y) for y in range(height) if (border_x, y) not in walls]

        if invaders:
            closest_invader = min(invaders, key=lambda inv: self.getMazeDistance(inv, self.start_position))
            closest_border = min(border_positions, key=lambda b: self.getMazeDistance(closest_invader, b))
            return [closest_border]

        return border_positions
